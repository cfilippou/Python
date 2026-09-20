# Emotion-Based AI Image Assistant

A Flask web application implementing the assignment requirements:

- Facial emotion recognition with a pre-trained DeepFace model.
- Seven basic emotion categories: happy, sad, angry, fear, disgust, surprise, neutral.
- Emotion labels, confidence scores and face bounding boxes.
- YOLOv8 object detection with labels, confidence scores and bounding boxes.
- Annotated output image.
- Τοπική ενσωμάτωση Llama 3.2 μέσω Ollama.
- Chat-style Flask interface.
- Emotion distribution visualization with Chart.js.
- Error handling for invalid/large uploads and failed model/API responses.
- Suitable for GitHub submission and adaptable for Render / Hugging Face Spaces / PythonAnywhere.

## 1. Recommended Python version

Use Python 3.10 or 3.11.

## 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The first run may download model weights for YOLO and DeepFace.

## 4. Configure the free LLM API

Copy `.env.example` to `.env` and place your Hugging Face token in `HF_TOKEN`.

On Windows PowerShell:

```powershell
$env:HF_TOKEN="your_token_here"
```

macOS/Linux:

```bash
export HF_TOKEN="your_token_here"
```

Without a token, the application still runs, but it uses a deterministic fallback description instead of the external LLM.

## 5. Run

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## 6. Main workflow

2. DeepFace detects a face and predicts facial emotion.
3. YOLOv8 detects image objects.
4. Detected information is sent to an open-source LLM through Hugging Face Inference API.
5. The application generates a concise textual description.
6. Bounding boxes and labels are rendered onto an output image.
7. Emotion predictions are stored locally and visualized as a cumulative bar chart.

## 7. Suggested assignment testing

Use at least five diverse test images including variations in:

- lighting
- age
- pose
- occlusion
- emotion intensity

For each image, record:

- expected emotion
- predicted emotion
- confidence
- detected objects
- visual output
- LLM response
- observed failure cases

## 8. Notes for deployment

AI packages can exceed the memory limits of some free hosting plans.
For coursework, GitHub submission is usually the most reliable fallback if the selected free host cannot accommodate TensorFlow + YOLO.

For production optimization, possible improvements include:

- replacing heavyweight models with TensorFlow Lite or ONNX
- temporal smoothing of emotion predictions
- image resizing before inference
- caching model instances
- background workers for long inference requests


## Llama 3.2 μέσω Ollama

Η εφαρμογή χρησιμοποιεί το Llama 3.2 τοπικά μέσω Ollama.

```bash
ollama pull llama3.2
ollama run llama3.2
```

Το Flask backend συνδέεται από προεπιλογή στο `http://127.0.0.1:11434`.
Δεν απαιτείται Hugging Face token. Οι μεταβλητές `OLLAMA_URL` και
`OLLAMA_MODEL` μπορούν να αλλάξουν μέσω environment variables.


## Music recommendations

After each successful facial-expression analysis, the application selects the
dominant predicted expression and displays five song recommendations in the
format `Artist, Song`. Each recommendation links to a YouTube search page.
No YouTube API key is required.

The music recommendation component is intentionally lightweight: it demonstrates
how emotion-recognition output can drive a downstream music-selection feature
without implementing a full streaming music player.

## Error handling

The application now provides specific responses for invalid image files,
missing faces, DeepFace failures, Ollama/LLM failures, empty LLM responses,
oversized uploads and unexpected processing errors.


## LLM response evaluation

The application automatically evaluates each generated Llama 3.2 description
using three coursework-oriented metrics:

1. **Clarity — Flesch Reading Ease**  
   Measures how easy the English text is to read. The score is normalized to
   a 0-100 range.

2. **Coherence — Structural Coherence Heuristic**  
   Evaluates whether the response follows the requested 4-5 sentence structure,
   uses reasonable sentence lengths, maintains some lexical continuity between
   adjacent sentences and avoids excessive repetition.

3. **Factual Accuracy — Detection Grounding Accuracy**  
   Compares the generated description with the facial-expression and object
   detections that were supplied to the LLM. It rewards correctly reported
   dominant emotions, confidence values and object labels, while penalizing
   unsupported emotion labels or confidence percentages.

The interface displays the three scores in a table together with an overall
mean score. These automatic scores support the coursework evaluation, but the
grounding metric should not be interpreted as independent real-world ground
truth; it measures consistency with the computer-vision outputs supplied to
the LLM.


## Prompt engineering comparison

The application includes three prompt-engineering styles for Llama 3.2:

1. **Strict factual**  
   Uses highly constrained, neutral wording and minimizes interpretation.

2. **Structured analytical**  
   Forces a consistent sentence structure: detected faces, facial-expression
   results, detected objects, and a final model-grounding statement.

3. **Natural descriptive**  
   Allows more natural phrasing while retaining the same factual constraints.

The selected prompt style is sent with each image analysis. After the LLM
response is generated, the application records its clarity, coherence,
factual-accuracy and overall scores in `prompt_evaluation_history.json`.

The Prompt Engineering panel calculates the average scores for each prompt
style across repeated tests. This provides direct evidence of how changes in
prompt design affect response quality, which can be reported in the coursework.


## Automatic three-prompt comparison per image

The user now uploads each image only once. After the computer-vision models
produce the face/emotion and object detections, the same evidence is sent
automatically to Llama 3.2 three times:

1. Strict factual
2. Structured analytical
3. Natural descriptive

Each generated response is evaluated independently for clarity, coherence and
detection-grounded factual accuracy. The interface displays three separate
evaluation tables for the same image, allowing direct prompt-style comparison
without requiring the image to be uploaded three times.


## Experimental results CSV export

Every successful image analysis is automatically appended to
`analysis_results.csv`.

One row is created for each detected face. The file contains the image name,
face number, dominant expression, all seven DeepFace emotion scores, detected
objects, all three Llama 3.2 prompt responses, and the clarity, coherence,
factual-accuracy and overall evaluation scores for each prompt style.

The sidebar contains **Export Results (CSV)** and **Clear Saved Results**
controls. The CSV uses UTF-8 with BOM so it can be opened conveniently in
Microsoft Excel and can also be analysed with Python, R, SPSS or similar tools.
