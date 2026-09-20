# Emotion-Based Music Player with AI

An intelligent web application that combines **Computer Vision**, **Facial Emotion Recognition**, **Large Language Models (LLMs)** and **music recommendation**.

The application analyses an uploaded image, detects facial expressions, identifies visual objects, generates natural-language descriptions using an LLM, and recommends music according to the detected dominant facial expression.

This project was developed as part of the **Artificial Intelligence (AI109)** module.

---

## Features

- Image upload through a Flask web interface
- Face detection and facial emotion recognition
- Seven emotion categories:
  - Angry
  - Disgust
  - Fear
  - Happy
  - Sad
  - Surprise
  - Neutral
- Confidence scores for all emotion categories
- Face bounding-box visualization
- Object detection using YOLOv8
- Image descriptions generated with Llama 3.2
- Three prompt-engineering strategies
- Automatic LLM response evaluation
- Emotion-based music recommendations
- Emotion distribution visualization
- Experimental results stored and exported as CSV
- Responsive web interface
- Error handling for invalid images and model failures

---

## System Architecture

The application combines several AI components in a single processing pipeline:

1. The user uploads an image through the Flask interface.
2. The image is validated and processed.
3. Faces are detected and analysed using DeepFace.
4. Each detected face is classified into one of seven facial-expression categories.
5. YOLOv8 detects objects and provides additional visual context.
6. The detected information is passed to Llama 3.2.
7. Three different prompts generate alternative image descriptions.
8. The generated descriptions are automatically evaluated.
9. Music recommendations are selected according to the dominant detected facial expression.
10. The results are displayed through the web interface and stored for experimental analysis.

---

## Technologies

### Backend
- Python
- Flask

### Computer Vision
- DeepFace
- OpenCV
- YOLOv8 / Ultralytics

### Large Language Model
- Llama 3.2
- Ollama

### Frontend
- HTML
- CSS
- JavaScript

### Additional Libraries
- TensorFlow
- Pillow
- NumPy
- Requests

---

## Facial Emotion Recognition

Facial-expression recognition is implemented using **DeepFace**.

The system analyses each detected face and produces confidence scores for seven categories:

`angry`, `disgust`, `fear`, `happy`, `sad`, `surprise`, `neutral`

For every detected face, the interface displays:

- Dominant predicted facial expression
- Confidence percentage
- Scores for all seven categories
- Face bounding box

The application intentionally describes these outputs as **predicted facial expressions** rather than assuming that they represent a person's true psychological or emotional state.

---

## Object Detection

**YOLOv8** is used to identify objects and visual context in the uploaded image.

Object detections are primarily used as additional input for the language model.

Bounding boxes displayed in the final interface are restricted to detected faces in order to keep the visual output clear.

---

## LLM Integration

The project uses **Llama 3.2** locally through **Ollama**.

The LLM receives structured information produced by the computer-vision pipeline and generates a short description of the analysed image.

Three prompt-engineering strategies are evaluated automatically:

### Prompt 1 – Strict Factual

Designed to produce a concise and conservative description based closely on the model detections.

### Prompt 2 – Structured Analytical

Produces a more structured explanation of detected faces, facial expressions, confidence scores and visual context.

### Prompt 3 – Natural Descriptive

Produces a more natural-language description while still attempting to remain grounded in the supplied detections.

The three outputs are displayed simultaneously so that their differences can be compared.

---

## LLM Evaluation

Each generated description is evaluated using three criteria:

### Clarity

Measured using a normalized **Flesch Reading Ease** score.

### Coherence

Measured using a structural coherence heuristic that considers factors such as sentence structure, sentence length, lexical continuity and repetition.

### Factual Accuracy

Measured using a **Detection Grounding Accuracy** metric.

This metric measures consistency between the generated description and the outputs supplied by the computer-vision models.

> Important: Detection Grounding Accuracy is not equivalent to ground-truth visual accuracy. If an upstream vision model produces an incorrect detection, an LLM may reproduce that incorrect detection faithfully and still receive a high grounding score.

An overall score is calculated from the three evaluation criteria.

---

## Music Recommendation

After facial-expression analysis, the application selects music recommendations according to the dominant detected facial expression.

Five songs are suggested for each emotion category.

The recommendations provide YouTube search links and demonstrate how facial-expression recognition can be connected to an Emotion-Based Music Player.

---

## Experimental Evaluation

The system was evaluated using **10 diverse test images**.

The test set included variations in:

- Lighting conditions
- Age
- Facial-expression intensity
- Multiple people
- Face pose
- Side-profile faces
- Partial facial occlusion
- Glasses
- Low-light conditions

The experiments demonstrated both successful detections and important failure cases.

A major limitation observed during testing was **false-positive face detection**. In some difficult images, particularly those involving unusual lighting, strong facial expressions or challenging poses, the face detector identified additional regions as faces.

This affected subsequent stages of the pipeline because incorrect face detections could produce high-confidence facial-expression predictions, influence LLM descriptions and change the final music recommendation.

This demonstrates an important characteristic of multi-stage AI systems: errors produced by an upstream component can propagate through downstream components.

---

## Experimental Results Export

Successful analyses are automatically recorded in:

`analysis_results.csv`

Each detected face is stored as an individual row.

The exported information includes:

- Analysis ID
- Timestamp
- Image name
- Face index
- Dominant facial expression
- Dominant confidence
- Scores for all seven facial expressions
- Detected objects
- LLM responses
- Clarity scores
- Coherence scores
- Detection Grounding Accuracy
- Overall evaluation scores

The CSV file can be exported directly through the application interface.
