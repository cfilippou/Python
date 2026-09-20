import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
HISTORY_FILE = BASE_DIR / "emotion_history.json"
PROMPT_EVALUATION_FILE = BASE_DIR / "prompt_evaluation_history.json"
RESULTS_CSV_FILE = BASE_DIR / "analysis_results.csv"

# Lazily initialized models keep application startup manageable.
_YOLO_MODEL = None


EMOTION_TRANSLATIONS = {
    "happy": "χαρούμενος",
    "sad": "λυπημένος",
    "angry": "θυμωμένος",
    "fear": "φοβισμένος",
    "disgust": "αηδιασμένος",
    "surprise": "έκπληκτος",
    "neutral": "ουδέτερος",
    "unknown": "άγνωστο",
}


def translate_emotion(label: str) -> str:
    return EMOTION_TRANSLATIONS.get(str(label).lower(), str(label))


def _get_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        from ultralytics import YOLO
        _YOLO_MODEL = YOLO("yolov8n.pt")
    return _YOLO_MODEL


def detect_objects(image: Image.Image) -> List[Dict]:
    """
    YOLOv8 object detection.
    Returns: [{"label": "person", "confidence": 0.94, "box": [x1,y1,x2,y2]}, ...]
    """
    model = _get_yolo()
    image_np = np.array(image)

    results = model.predict(
        source=image_np,
        conf=0.25,
        verbose=False,
    )

    detected = []
    if not results:
        return detected

    result = results[0]
    names = result.names

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

        detected.append(
            {
                "label": str(names[cls_id]),
                "confidence": round(confidence, 4),
                "box": [x1, y1, x2, y2],
            }
        )

    # Keep the response concise for the UI/LLM.
    detected.sort(key=lambda x: x["confidence"], reverse=True)
    return detected[:20]


def detect_emotions(image: Image.Image) -> List[Dict]:
    """
    DeepFace emotion analysis.
    Required emotion classes:
    angry, disgust, fear, happy, sad, surprise, neutral.

    Each result includes dominant label, confidence score, all class scores,
    and the face bounding box.
    """
    from deepface import DeepFace

    image_np = np.array(image)

    analysis = DeepFace.analyze(
        img_path=image_np,
        actions=["emotion"],
        enforce_detection=False,
        detector_backend=os.getenv("DEEPFACE_DETECTOR", "opencv"),
        silent=True,
    )

    if isinstance(analysis, dict):
        analysis = [analysis]

    output = []
    for face in analysis:
        region = face.get("region", {}) or {}
        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        w = int(region.get("w", 0))
        h = int(region.get("h", 0))

        scores = {
            key: round(float(value), 3)
            for key, value in (face.get("emotion", {}) or {}).items()
        }

        dominant = str(face.get("dominant_emotion", "unknown"))
        dominant_score = float(scores.get(dominant, 0.0))

        # DeepFace may return a full-image pseudo-region when no face is found.
        # Filter extremely weak/invalid face outputs.
        if w <= 0 or h <= 0:
            continue

        output.append(
            {
                "label": dominant,
                "confidence": round(dominant_score / 100.0, 4),
                "scores": scores,
                "box": [x, y, x + w, y + h],
            }
        )

    return output


def _compact_objects(objects: List[Dict]) -> str:
    labels = [item["label"] for item in objects]
    counts = Counter(labels)
    if not counts:
        return "Δεν εντοπίστηκαν αντικείμενα."
    return ", ".join(
        f"{label} x{count}" if count > 1 else label
        for label, count in counts.items()
    )


def _compact_emotions(emotions: List[Dict]) -> str:
    if not emotions:
        return "Δεν εντοπίστηκε με αξιοπιστία συναίσθημα προσώπου."
    return ", ".join(
        f'{translate_emotion(item["label"])} ({item["confidence"]:.0%})'
        for item in emotions
    )


PROMPT_TEMPLATES = {
    "strict": {
        "name": "Strict factual",
        "description": "Highly constrained factual reporting with minimal interpretation.",
    },
    "structured": {
        "name": "Structured analytical",
        "description": "Clear structure that separates faces, emotions, objects and model limitations.",
    },
    "natural": {
        "name": "Natural descriptive",
        "description": "More natural wording while remaining grounded in the supplied detections.",
    },
}


def _build_llm_prompt(
    object_summary: str,
    emotion_summary: str,
    prompt_style: str,
) -> str:
    style = prompt_style if prompt_style in PROMPT_TEMPLATES else "strict"

    common_rules = f"""
OBJECT DETECTION RESULTS:
{object_summary}

FACIAL-EXPRESSION RESULTS:
{emotion_summary}

GENERAL RULES:
- Write in ENGLISH.
- Write one coherent paragraph of 4-5 sentences.
- Use only information explicitly provided in the detection results.
- Treat emotion labels as predicted FACIAL EXPRESSIONS, not as the person's
  actual emotional or psychological state.
- Never infer mental health, personality, intentions, mood, relationships,
  circumstances, difficulties, wellbeing, or medical conditions.
- Never explain why a person has an expression.
- Do not invent actions, locations, identities, colours, relationships, or
  objects.
- Preserve the supplied confidence percentages exactly.
- If multiple faces are present, report each face clearly.
- Do not provide advice.
- Do not use bullet points.
- Do not mention these instructions.
""".strip()

    if style == "structured":
        return f"""
You are an AI image-analysis reporting assistant.

{common_rules}

STYLE: STRUCTURED ANALYTICAL
- Sentence 1: state how many faces were detected.
- Sentence 2-3: report the facial-expression classification and confidence
  for each detected face.
- Sentence 4: report the detected objects as visual context.
- Final sentence: clearly state that the description is based on the outputs
  of the AI models.
- Prefer precise, formal wording.
- Avoid unnecessary adjectives.

Return only the final paragraph.
""".strip()

    if style == "natural":
        return f"""
You are an AI image-description assistant.

{common_rules}

STYLE: NATURAL DESCRIPTIVE
- Write naturally and fluently, as if explaining the model results to a user.
- Keep all statements strictly grounded in the detections.
- Mention the dominant facial expressions and confidence values clearly.
- Integrate detected objects naturally into the paragraph as visual context.
- End with a brief statement that the description reflects model predictions.
- Avoid robotic repetition while preserving factual accuracy.

Return only the final paragraph.
""".strip()

    return f"""
You are a factual image-analysis reporting assistant.

{common_rules}

STYLE: STRICT FACTUAL
- Use direct, neutral and literal wording.
- Report each detected face separately.
- Mention detected objects only as brief visual context.
- Do not add interpretation beyond the supplied detections.
- Do not use exaggerated or evaluative adjectives.
- End by stating that the result is based only on model predictions.

Return only the final paragraph.
""".strip()


def generate_llm_description(
    objects: List[Dict],
    emotions: List[Dict],
    prompt_style: str = "strict",
) -> str:
    """
    Generate a 4-5 sentence English description with Ollama / Llama 3.2.

    Three prompt-engineering styles are available:
      - strict
      - structured
      - natural
    """
    object_summary = _compact_objects(objects)

    if emotions:
        lines = []
        for index, item in enumerate(emotions, start=1):
            label = str(item.get("label", "unknown")).lower()
            confidence = float(item.get("confidence", 0.0))
            lines.append(
                f"Face {index}: predicted facial expression = {label}; "
                f"confidence = {confidence:.0%}"
            )
        emotion_summary = "\n".join(lines)
    else:
        emotion_summary = "No reliable facial expression was detected."

    prompt = _build_llm_prompt(
        object_summary=object_summary,
        emotion_summary=emotion_summary,
        prompt_style=prompt_style,
    )

    base_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.2")

    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0 if prompt_style == "strict" else 0.2,
                    "top_p": 0.2 if prompt_style == "strict" else 0.5,
                    "num_predict": 220,
                    "repeat_penalty": 1.1,
                },
            },
            timeout=90,
        )
        response.raise_for_status()
        generated = str(response.json().get("response", "")).strip()

        if not generated:
            raise RuntimeError("Llama 3.2 returned an empty response.")

        return generated

    except requests.exceptions.ConnectionError:
        return (
            "The application could not connect to Llama 3.2 through Ollama. "
            "Make sure Ollama is running and llama3.2 is installed."
        )
    except requests.exceptions.Timeout:
        return "Llama 3.2 took too long to respond. Please try again."
    except Exception as exc:
        return f"An error occurred while communicating with Llama 3.2: {exc}"

def annotate_image(
    image: Image.Image,
    emotions: List[Dict],
    objects: List[Dict],
) -> Image.Image:
    """
    Draw annotations only for detected faces/emotions.

    Object detection still runs and its results can be used by the LLM,
    but object bounding boxes are intentionally not rendered on the image.
    """
    output = image.copy()
    draw = ImageDraw.Draw(output)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    # Draw ONLY face/emotion bounding boxes.
    for item in emotions:
        x1, y1, x2, y2 = item["box"]
        label = f'Expression: {item["label"]} {item["confidence"]:.0%}'

        draw.rectangle([x1, y1, x2, y2], outline="black", width=4)
        draw.rectangle(
            [x1 + 3, y1 + 3, x2 - 3, y2 - 3],
            outline="white",
            width=2
        )

        # Label placed near the bottom of the face box.
        draw.text(
            (x1 + 4, max(0, y2 - 22)),
            label,
            font=font,
            fill="white"
        )

    return output

MUSIC_RECOMMENDATIONS = {
    "happy": [
        ("Pharrell Williams", "Happy"),
        ("Katrina and the Waves", "Walking on Sunshine"),
        ("Justin Timberlake", "Can't Stop the Feeling!"),
        ("Queen", "Don't Stop Me Now"),
        ("Dua Lipa", "Levitating"),
    ],
    "sad": [
        ("Adele", "Someone Like You"),
        ("Lewis Capaldi", "Someone You Loved"),
        ("Billie Eilish", "when the party's over"),
        ("Coldplay", "Fix You"),
        ("The Script", "Breakeven"),
    ],
    "angry": [
        ("Linkin Park", "Numb"),
        ("Eminem", "Lose Yourself"),
        ("Imagine Dragons", "Believer"),
        ("Green Day", "Boulevard of Broken Dreams"),
        ("Foo Fighters", "The Pretender"),
    ],
    "fear": [
        ("Coldplay", "Paradise"),
        ("OneRepublic", "Counting Stars"),
        ("Sia", "Unstoppable"),
        ("Imagine Dragons", "Demons"),
        ("Florence + The Machine", "Shake It Out"),
    ],
    "disgust": [
        ("Kelly Clarkson", "Stronger"),
        ("P!nk", "So What"),
        ("Dua Lipa", "IDGAF"),
        ("Taylor Swift", "Shake It Off"),
        ("Miley Cyrus", "Flowers"),
    ],
    "surprise": [
        ("Avicii", "Wake Me Up"),
        ("Daft Punk", "Get Lucky"),
        ("The Weeknd", "Blinding Lights"),
        ("Bruno Mars", "24K Magic"),
        ("Calvin Harris", "Feel So Close"),
    ],
    "neutral": [
        ("Norah Jones", "Don't Know Why"),
        ("Jack Johnson", "Better Together"),
        ("Coldplay", "Yellow"),
        ("Ed Sheeran", "Photograph"),
        ("John Mayer", "Gravity"),
    ],
}


def get_music_recommendations(emotion: str) -> List[Dict]:
    """
    Return five lightweight YouTube recommendations for the dominant emotion.

    No YouTube API key is required. Each link opens a YouTube search for the
    artist and song, which avoids relying on a specific video ID.
    """
    from urllib.parse import quote_plus

    normalized = str(emotion or "neutral").lower()
    songs = MUSIC_RECOMMENDATIONS.get(
        normalized,
        MUSIC_RECOMMENDATIONS["neutral"],
    )

    return [
        {
            "artist": artist,
            "song": song,
            "youtube_url": (
                "https://www.youtube.com/results?search_query="
                + quote_plus(f"{artist} {song}")
            ),
        }
        for artist, song in songs
    ]


def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter for English LLM output."""
    import re

    sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _count_syllables(word: str) -> int:
    """
    Lightweight English syllable estimator used for Flesch Reading Ease.
    It is intentionally dependency-free and suitable for coursework evaluation.
    """
    import re

    word = re.sub(r"[^a-z]", "", str(word).lower())
    if not word:
        return 0

    vowels = "aeiouy"
    count = 0
    previous_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_vowel:
            count += 1
        previous_vowel = is_vowel

    if word.endswith("e") and count > 1:
        count -= 1

    return max(1, count)


def _clarity_score(text: str) -> Dict:
    """
    Clarity metric based on Flesch Reading Ease.

    The raw Flesch score is normalized to the 0-100 range.
    Higher values indicate text that is easier to read.
    """
    import re

    sentences = _split_sentences(text)
    words = re.findall(r"\b[A-Za-z']+\b", str(text))

    if not sentences or not words:
        return {
            "score": 0,
            "metric": "Flesch Reading Ease",
            "details": "No readable English text was available.",
        }

    syllables = sum(_count_syllables(word) for word in words)
    word_count = len(words)
    sentence_count = len(sentences)

    flesch = (
        206.835
        - 1.015 * (word_count / sentence_count)
        - 84.6 * (syllables / word_count)
    )

    normalized = max(0.0, min(100.0, flesch))

    return {
        "score": round(normalized, 1),
        "metric": "Flesch Reading Ease",
        "details": (
            f"{sentence_count} sentences, {word_count} words; "
            f"raw Flesch score {flesch:.1f}."
        ),
    }


def _coherence_score(text: str) -> Dict:
    """
    A transparent structural coherence heuristic.

    It rewards:
      - the requested 4-5 sentence paragraph,
      - reasonable sentence length,
      - sentence-to-sentence lexical continuity,
      - low repetition.
    """
    import re
    from collections import Counter

    sentences = _split_sentences(text)

    if not sentences:
        return {
            "score": 0,
            "metric": "Structural Coherence Heuristic",
            "details": "No sentences were available for evaluation.",
        }

    score = 100.0

    # Requested output is 4-5 sentences.
    sentence_count = len(sentences)
    if sentence_count < 4:
        score -= (4 - sentence_count) * 15
    elif sentence_count > 5:
        score -= (sentence_count - 5) * 10

    tokenized = []
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "are",
        "was", "were", "with", "as", "by", "this", "that", "it", "for",
        "on", "from", "also", "has", "have", "been"
    }

    for sentence in sentences:
        words = [
            word.lower()
            for word in re.findall(r"\b[A-Za-z']+\b", sentence)
            if word.lower() not in stopwords
        ]
        tokenized.append(words)

        # Very short or excessively long sentences are harder to follow.
        length = len(re.findall(r"\b[A-Za-z']+\b", sentence))
        if length < 5:
            score -= 5
        elif length > 35:
            score -= 7

    # Reward some continuity between adjacent sentences.
    transition_pairs = max(0, len(tokenized) - 1)
    if transition_pairs:
        connected = 0
        for left, right in zip(tokenized, tokenized[1:]):
            if set(left) & set(right):
                connected += 1

        continuity_ratio = connected / transition_pairs
        if continuity_ratio < 0.34:
            score -= 12
        elif continuity_ratio < 0.67:
            score -= 5

    # Penalize excessive repeated content words.
    all_words = [word for sentence in tokenized for word in sentence]
    if all_words:
        counts = Counter(all_words)
        repeated_excess = sum(max(0, count - 4) for count in counts.values())
        score -= min(15, repeated_excess * 2)

    score = max(0.0, min(100.0, score))

    return {
        "score": round(score, 1),
        "metric": "Structural Coherence Heuristic",
        "details": (
            f"{sentence_count} sentences; score reflects sentence structure, "
            "continuity and repetition."
        ),
    }


def _factual_accuracy_score(
    text: str,
    objects: List[Dict],
    emotions: List[Dict],
) -> Dict:
    """
    Grounding-based factual accuracy score.

    The description is checked against the machine-detected evidence supplied
    to the LLM. This is not a human truth judgement: it measures whether the
    generated text stays consistent with the supplied detections.
    """
    import re
    from collections import Counter

    lowered = str(text).lower()

    expected_facts = []
    matched_facts = []
    contradictions = []

    # Expected face-expression facts.
    for index, face in enumerate(emotions, start=1):
        label = str(face.get("label", "")).lower()
        confidence_pct = round(float(face.get("confidence", 0.0)) * 100)

        expected_facts.append(f"face {index}: {label} {confidence_pct}%")

        label_present = label and label in lowered

        # Accept integer confidence or the exact one-decimal representation.
        pct_pattern = rf"\b{confidence_pct}(?:\.0)?\s*%"
        confidence_present = bool(re.search(pct_pattern, lowered))

        if label_present and confidence_present:
            matched_facts.append(f"face {index}: {label} {confidence_pct}%")

    # Expected object labels; duplicates are counted once in the description
    # because the prompt only asks for contextual object reporting.
    object_labels = []
    for item in objects:
        label = str(item.get("label", "")).strip().lower()
        if label:
            object_labels.append(label)

    for label in sorted(set(object_labels)):
        expected_facts.append(f"object: {label}")
        if label in lowered:
            matched_facts.append(f"object: {label}")

    # Penalize emotion labels that were not detected as dominant expressions.
    known_emotions = {
        "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"
    }
    detected_emotions = {
        str(face.get("label", "")).lower()
        for face in emotions
    }

    for emotion in sorted(known_emotions - detected_emotions):
        if re.search(rf"\b{re.escape(emotion)}\b", lowered):
            contradictions.append(f"unsupported emotion: {emotion}")

    # Penalize percentages in the LLM output that do not correspond to
    # supplied dominant-expression confidences.
    allowed_percentages = {
        round(float(face.get("confidence", 0.0)) * 100)
        for face in emotions
    }

    mentioned_percentages = {
        int(match)
        for match in re.findall(r"\b(\d{1,3})(?:\.0)?\s*%", lowered)
        if 0 <= int(match) <= 100
    }

    for percentage in sorted(mentioned_percentages - allowed_percentages):
        contradictions.append(f"unsupported confidence: {percentage}%")

    if not expected_facts:
        return {
            "score": 100.0,
            "metric": "Detection Grounding Accuracy",
            "details": "No expected visual facts were available for comparison.",
            "matched": [],
            "missing": [],
            "contradictions": [],
        }

    coverage = len(matched_facts) / len(expected_facts)
    contradiction_penalty = min(0.5, len(contradictions) * 0.15)

    score = max(0.0, (coverage - contradiction_penalty) * 100.0)

    missing = [
        fact
        for fact in expected_facts
        if fact not in matched_facts
    ]

    return {
        "score": round(score, 1),
        "metric": "Detection Grounding Accuracy",
        "details": (
            f"Matched {len(matched_facts)} of {len(expected_facts)} expected "
            f"detection facts; {len(contradictions)} unsupported claims found."
        ),
        "matched": matched_facts,
        "missing": missing,
        "contradictions": contradictions,
    }


def evaluate_llm_response(
    text: str,
    objects: List[Dict],
    emotions: List[Dict],
) -> Dict:
    """
    Evaluate the generated LLM description using three transparent metrics:
    clarity, coherence and grounding-based factual accuracy.
    """
    clarity = _clarity_score(text)
    coherence = _coherence_score(text)
    factual = _factual_accuracy_score(text, objects, emotions)

    overall = round(
        (
            clarity["score"]
            + coherence["score"]
            + factual["score"]
        ) / 3,
        1,
    )

    return {
        "clarity": clarity,
        "coherence": coherence,
        "factual_accuracy": factual,
        "overall_score": overall,
        "note": (
            "These are automatic coursework-oriented metrics. "
            "Factual accuracy measures consistency with the supplied model "
            "detections rather than independent ground-truth truthfulness."
        ),
    }


def save_prompt_evaluation(
    prompt_style: str,
    evaluation: Dict,
) -> None:
    """Store prompt-style evaluation results for comparison across tests."""
    data = []

    if PROMPT_EVALUATION_FILE.exists():
        try:
            data = json.loads(
                PROMPT_EVALUATION_FILE.read_text(encoding="utf-8")
            )
        except Exception:
            data = []

    data.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prompt_style": prompt_style,
            "clarity": evaluation["clarity"]["score"],
            "coherence": evaluation["coherence"]["score"],
            "factual_accuracy": evaluation["factual_accuracy"]["score"],
            "overall_score": evaluation["overall_score"],
        }
    )

    data = data[-300:]
    PROMPT_EVALUATION_FILE.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def summarize_prompt_evaluations() -> Dict:
    """
    Return average metric scores for each prompt-engineering style.
    """
    if not PROMPT_EVALUATION_FILE.exists():
        return {"styles": {}, "total_runs": 0}

    try:
        data = json.loads(
            PROMPT_EVALUATION_FILE.read_text(encoding="utf-8")
        )
    except Exception:
        data = []

    grouped = {}

    for item in data:
        style = item.get("prompt_style", "strict")
        grouped.setdefault(
            style,
            {
                "runs": 0,
                "clarity": 0.0,
                "coherence": 0.0,
                "factual_accuracy": 0.0,
                "overall_score": 0.0,
            },
        )

        grouped[style]["runs"] += 1
        for key in (
            "clarity",
            "coherence",
            "factual_accuracy",
            "overall_score",
        ):
            grouped[style][key] += float(item.get(key, 0.0))

    for style, values in grouped.items():
        runs = max(1, values["runs"])
        for key in (
            "clarity",
            "coherence",
            "factual_accuracy",
            "overall_score",
        ):
            values[key] = round(values[key] / runs, 1)

        values["name"] = PROMPT_TEMPLATES.get(
            style,
            PROMPT_TEMPLATES["strict"],
        )["name"]

    return {
        "styles": grouped,
        "total_runs": len(data),
    }



RESULTS_CSV_COLUMNS = [
    "analysis_id", "timestamp", "image_name", "face_index",
    "dominant_emotion", "dominant_confidence",
    "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral",
    "detected_objects",
    "strict_response", "strict_clarity", "strict_coherence",
    "strict_factual_accuracy", "strict_overall",
    "structured_response", "structured_clarity", "structured_coherence",
    "structured_factual_accuracy", "structured_overall",
    "natural_response", "natural_clarity", "natural_coherence",
    "natural_factual_accuracy", "natural_overall",
]


def _objects_for_export(objects: List[Dict]) -> str:
    from collections import Counter

    labels = [
        str(item.get("label", "")).strip()
        for item in objects
        if str(item.get("label", "")).strip()
    ]
    if not labels:
        return ""

    counts = Counter(labels)
    return "; ".join(
        f"{label} x{count}" if count > 1 else label
        for label, count in sorted(counts.items())
    )


def save_analysis_results(
    analysis_id: str,
    image_name: str,
    emotions: List[Dict],
    objects: List[Dict],
    prompt_results: Dict,
) -> None:
    """Append one CSV row per detected face."""
    file_exists = RESULTS_CSV_FILE.exists()
    object_summary = _objects_for_export(objects)
    timestamp = datetime.now(timezone.utc).isoformat()

    def metric_value(style: str, metric: str):
        evaluation = (
            prompt_results.get(style, {}) or {}
        ).get("evaluation", {}) or {}
        return (evaluation.get(metric, {}) or {}).get("score", "")

    def overall_value(style: str):
        evaluation = (
            prompt_results.get(style, {}) or {}
        ).get("evaluation", {}) or {}
        return evaluation.get("overall_score", "")

    def response_value(style: str):
        text = str(
            (prompt_results.get(style, {}) or {}).get("description", "")
        )
        return text.replace("\n", " ").strip()

    rows = []

    for face_index, face in enumerate(emotions, start=1):
        scores = face.get("scores", {}) or {}

        rows.append({
            "analysis_id": analysis_id,
            "timestamp": timestamp,
            "image_name": image_name,
            "face_index": face_index,
            "dominant_emotion": str(face.get("label", "")),
            "dominant_confidence": round(
                float(face.get("confidence", 0.0)) * 100, 2
            ),
            "angry": round(float(scores.get("angry", 0.0)), 3),
            "disgust": round(float(scores.get("disgust", 0.0)), 3),
            "fear": round(float(scores.get("fear", 0.0)), 3),
            "happy": round(float(scores.get("happy", 0.0)), 3),
            "sad": round(float(scores.get("sad", 0.0)), 3),
            "surprise": round(float(scores.get("surprise", 0.0)), 3),
            "neutral": round(float(scores.get("neutral", 0.0)), 3),
            "detected_objects": object_summary,

            "strict_response": response_value("strict"),
            "strict_clarity": metric_value("strict", "clarity"),
            "strict_coherence": metric_value("strict", "coherence"),
            "strict_factual_accuracy": metric_value("strict", "factual_accuracy"),
            "strict_overall": overall_value("strict"),

            "structured_response": response_value("structured"),
            "structured_clarity": metric_value("structured", "clarity"),
            "structured_coherence": metric_value("structured", "coherence"),
            "structured_factual_accuracy": metric_value("structured", "factual_accuracy"),
            "structured_overall": overall_value("structured"),

            "natural_response": response_value("natural"),
            "natural_clarity": metric_value("natural", "clarity"),
            "natural_coherence": metric_value("natural", "coherence"),
            "natural_factual_accuracy": metric_value("natural", "factual_accuracy"),
            "natural_overall": overall_value("natural"),
        })

    if not rows:
        return

    with RESULTS_CSV_FILE.open("a", encoding="utf-8-sig", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=RESULTS_CSV_COLUMNS)
        if not file_exists or RESULTS_CSV_FILE.stat().st_size == 0:
            writer.writeheader()
        writer.writerows(rows)


def clear_analysis_results() -> None:
    if RESULTS_CSV_FILE.exists():
        RESULTS_CSV_FILE.unlink()


def count_saved_analysis_rows() -> int:
    if not RESULTS_CSV_FILE.exists():
        return 0

    try:
        with RESULTS_CSV_FILE.open(
            "r", encoding="utf-8-sig", newline=""
        ) as csvfile:
            return max(0, sum(1 for _ in csvfile) - 1)
    except Exception:
        return 0

def save_emotion_history(emotions: List[Dict]) -> None:
    if not emotions:
        return

    data = []
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = []

    now = datetime.now(timezone.utc).isoformat()
    for item in emotions:
        data.append(
            {
                "timestamp": now,
                "emotion": item["label"],
                "confidence": item["confidence"],
            }
        )

    # Prevent unlimited file growth during demos.
    data = data[-500:]
    HISTORY_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def summarize_emotion_history() -> Dict:
    if not HISTORY_FILE.exists():
        return {"counts": {}, "timeline": []}

    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        data = []

    counts = Counter(item.get("emotion", "unknown") for item in data)
    timeline = [
        {
            "timestamp": item.get("timestamp"),
            "emotion": item.get("emotion"),
            "confidence": item.get("confidence"),
        }
        for item in data[-60:]
    ]

    return {"counts": dict(counts), "timeline": timeline}
