import base64
import io
import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory, send_file
from PIL import Image

from ai_services import (
    annotate_image,
    detect_emotions,
    detect_objects,
    generate_llm_description,
    save_emotion_history,
    summarize_emotion_history,
    get_music_recommendations,
    evaluate_llm_response,
    save_prompt_evaluation,
    summarize_prompt_evaluations,
    save_analysis_results,
    clear_analysis_results,
    count_saved_analysis_rows,
    RESULTS_CSV_FILE,
)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_FILE_SIZE_MB = 8

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_image_from_request():
    """
    Supports either:
      1. multipart/form-data with field name 'image'
      2. form/JSON field 'camera_image' containing a base64 data URL
    """
    try:
        if "image" in request.files and request.files["image"].filename:
            file = request.files["image"]

            if not allowed_file(file.filename):
                raise ValueError(
                    "Μη υποστηριζόμενος τύπος αρχείου. "
                    "Χρησιμοποίησε PNG, JPG, JPEG ή WEBP."
                )

            image = Image.open(file.stream)
            image.verify()

            file.stream.seek(0)
            image = Image.open(file.stream).convert("RGB")
            return image

        camera_image = request.form.get("camera_image") or (
            request.json.get("camera_image")
            if request.is_json and request.json
            else None
        )

        if camera_image:
            try:
                if "," in camera_image:
                    camera_image = camera_image.split(",", 1)[1]
                raw = base64.b64decode(camera_image, validate=True)
                image = Image.open(io.BytesIO(raw))
                image.verify()
                return Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as exc:
                raise ValueError(
                    "Η εικόνα από την κάμερα δεν είναι έγκυρη. "
                    "Δοκίμασε να πραγματοποιήσεις νέα λήψη."
                ) from exc

        raise ValueError("Δεν δόθηκε εικόνα.")

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(
            "Το αρχείο δεν μπόρεσε να διαβαστεί ως έγκυρη εικόνα."
        ) from exc


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        image = load_image_from_request()

        if "image" not in request.files or not request.files["image"].filename:
            raise ValueError("Επίλεξε μια εικόνα για ανάλυση.")
        original_image_name = request.files["image"].filename

        # ----------------------------------------------------
        # Emotion recognition
        # ----------------------------------------------------
        try:
            emotions = detect_emotions(image)
        except Exception:
            app.logger.exception("Emotion recognition failed")
            return jsonify(
                {
                    "ok": False,
                    "error_type": "emotion_model_error",
                    "error": (
                        "Η αναγνώριση συναισθήματος απέτυχε. "
                        "Έλεγξε ότι το DeepFace και τα απαιτούμενα model files "
                        "έχουν εγκατασταθεί σωστά."
                    ),
                }
            ), 500

        # ----------------------------------------------------
        # Object detection
        # Object detection failure does not stop emotion analysis.
        # ----------------------------------------------------
        try:
            objects = detect_objects(image)
        except Exception:
            app.logger.exception("Object detection failed")
            objects = []

        if not emotions:
            return jsonify(
                {
                    "ok": False,
                    "error_type": "no_face",
                    "error": (
                        "Δεν εντοπίστηκε πρόσωπο στην εικόνα. "
                        "Δοκίμασε μια καθαρότερη εικόνα όπου το πρόσωπο είναι "
                        "ορατό και επαρκώς φωτισμένο."
                    ),
                }
            ), 422

        # ----------------------------------------------------
        # Prompt Engineering
        # The same CV detections are sent automatically through
        # all three prompt styles. The user uploads the image once.
        # ----------------------------------------------------
        prompt_styles = ("strict", "structured", "natural")
        prompt_results = {}

        for prompt_style in prompt_styles:
            try:
                description = generate_llm_description(
                    objects=objects,
                    emotions=emotions,
                    prompt_style=prompt_style,
                )
            except Exception:
                app.logger.exception(
                    "LLM generation failed for prompt style %s",
                    prompt_style,
                )
                description = (
                    "The facial-expression analysis was completed successfully, "
                    "but the Llama 3.2 description could not be generated for "
                    f"the {prompt_style} prompt."
                )

            if not description or not description.strip():
                description = (
                    "The facial-expression analysis was completed successfully, "
                    "but the language model returned an empty response."
                )

            evaluation = evaluate_llm_response(
                text=description,
                objects=objects,
                emotions=emotions,
            )

            save_prompt_evaluation(
                prompt_style=prompt_style,
                evaluation=evaluation,
            )

            prompt_results[prompt_style] = {
                "description": description,
                "evaluation": evaluation,
            }

        # The strict prompt remains the primary description shown
        # in the main AI Description panel.
        llm_text = prompt_results["strict"]["description"]
        llm_evaluation = prompt_results["strict"]["evaluation"]

        # ----------------------------------------------------
        # Annotated image
        # ----------------------------------------------------
        annotated = annotate_image(
            image,
            emotions=emotions,
            objects=objects,
        )

        filename = f"{uuid.uuid4().hex}.jpg"
        output_path = UPLOAD_DIR / filename

        try:
            annotated.save(output_path, "JPEG", quality=92)
        except Exception:
            app.logger.exception("Could not save annotated image")
            return jsonify(
                {
                    "ok": False,
                    "error_type": "output_error",
                    "error": (
                        "Η ανάλυση ολοκληρώθηκε, αλλά δεν ήταν δυνατή η "
                        "αποθήκευση της επεξεργασμένης εικόνας."
                    ),
                }
            ), 500

        save_emotion_history(emotions)

        # ----------------------------------------------------
        # Music recommendations
        # ----------------------------------------------------
        dominant_face = max(
            emotions,
            key=lambda item: float(item.get("confidence", 0.0)),
        )
        dominant_emotion = str(
            dominant_face.get("label", "neutral")
        ).lower()

        music_recommendations = get_music_recommendations(
            dominant_emotion
        )

        analysis_id = uuid.uuid4().hex

        try:
            save_analysis_results(
                analysis_id=analysis_id,
                image_name=original_image_name,
                emotions=emotions,
                objects=objects,
                prompt_results=prompt_results,
            )
        except Exception:
            app.logger.exception("Could not save analysis results to CSV")

        return jsonify(
            {
                "ok": True,
                "analysis_id": analysis_id,
                "annotated_image_url": f"/static/uploads/{filename}",
                "emotions": emotions,
                "objects": objects,
                "description": llm_text,
                "llm_evaluation": llm_evaluation,
                "prompt_results": prompt_results,
                "dominant_emotion": dominant_emotion,
                "music_recommendations": music_recommendations,
            }
        )

    except ValueError as exc:
        return jsonify(
            {
                "ok": False,
                "error_type": "invalid_input",
                "error": str(exc),
            }
        ), 400

    except Exception:
        app.logger.exception("Unexpected analysis failure")
        return jsonify(
            {
                "ok": False,
                "error_type": "unexpected_error",
                "error": (
                    "Παρουσιάστηκε μη αναμενόμενο σφάλμα κατά την ανάλυση. "
                    "Δοκίμασε ξανά."
                ),
            }
        ), 500


@app.route("/emotion-summary", methods=["GET"])
def emotion_summary():
    return jsonify({"ok": True, **summarize_emotion_history()})


@app.route("/prompt-summary", methods=["GET"])
def prompt_summary():
    return jsonify(
        {
            "ok": True,
            **summarize_prompt_evaluations(),
        }
    )


@app.route("/export-results", methods=["GET"])
def export_results():
    if not RESULTS_CSV_FILE.exists():
        return jsonify(
            {"ok": False, "error": "Δεν υπάρχουν ακόμη αποθηκευμένα αποτελέσματα."}
        ), 404

    return send_file(
        RESULTS_CSV_FILE,
        as_attachment=True,
        download_name="emotion_ai_analysis_results.csv",
        mimetype="text/csv",
    )


@app.route("/results-status", methods=["GET"])
def results_status():
    return jsonify(
        {"ok": True, "saved_rows": count_saved_analysis_rows()}
    )


@app.route("/clear-results", methods=["POST"])
def clear_results():
    clear_analysis_results()
    return jsonify(
        {"ok": True, "message": "Τα αποθηκευμένα αποτελέσματα διαγράφηκαν."}
    )


@app.errorhandler(413)
def too_large(_):
    return jsonify(
        {
            "ok": False,
            "error": f"Η εικόνα είναι πολύ μεγάλη. Το μέγιστο επιτρεπόμενο μέγεθος είναι {MAX_FILE_SIZE_MB} MB.",
        }
    ), 413


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=debug_mode)
