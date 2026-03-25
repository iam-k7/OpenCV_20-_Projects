# OpenCV 20+ Projects

A collection of hands-on OpenCV (and related ML) examples in Python, organized by "Day". Each script is intended to be a small, focused demo.

## What is in this repo

This repo is organized by "Day". Each Python file is intended to be a small, focused demo.

## Scripts Overview

- `Day-01_to_03/Day_01.py` - Motion detection using frame differencing, thresholding, and contour filtering.
- `Day-01_to_03/Day_02.py` - Real-time face detection using OpenCV Haar cascades.
- `Day-01_to_03/Day_03.py` - Red object detection in HSV, contour center calculation, and simple direction logic.
- `Day-04/create_data.py` - Capture face images from webcam into a `datasets/` directory (for training).
- `Day-04/face_recognize.py` - Train an OpenCV FisherFace recognizer and run face recognition from webcam.
- `Day-05/main.py` - Real-time facial emotion detection from webcam using `facial_emotion_recognition`.
- `Day-05/mobile.py` - Facial emotion detection from an IP Webcam stream URL.
- `Day-06/train.py` - Train a small Keras CNN for diabetes classification using `pima-indians-diabetes.csv`.
- `Day-06/test.py` - Load the saved diabetes model and print predictions.
- `Day-07/main.py` - Object detection using MobileNet-SSD (Caffe) with OpenCV DNN and webcam input.
- `Day-08/img_create.py` - Download images using `bing-image-downloader` (creates `dataset/train`).
- `Day-08/train.py` - Train a binary CNN classifier and save `model.json` and `model.h5`.
- `Day-08/test.py` - Load `model.json`/`model.h5` and classify images under `Day-08/dataset/test`.
- `Day-09/train.py` - Train a grayscale hand-gesture CNN for classes `NONE` to `FIVE` and save `model.json`/`model.h5`.
- `Day-09/test.py` - Load the trained gesture model and classify images under `Day-09/dataset/test`.
- `Day-10/train.py` - Leaf disease classification CNN training (saves `model.json`/`model.h5`).
- `Day-10/test.py` - Load the leaf disease model and classify images under `Day-10/dataset/test`.
- `Day-11/character.py` - PyQt5 GUI for Gujarati character recognition using a CNN. Expects datasets in `Day-11/dataset/train` and `Day-11/dataset/test`.
- `Day-11/updated.py` - PyQt5 UI/layout file (generated code).

## Requirements

Python 3.10+.

Install dependencies:

```bash
pip install -r requirements.txt
```
## How to run

Run a script from the repo root:

```bash
python "Day-01_to_03/Day_01.py"
python "Day-04/face_recognize.py"
python "Day-07/main.py"
```

If a script uses the webcam, it typically exits on `ESC` (some demos use `q`).

## Notes

- Some scripts use hard-coded Windows paths for models/datasets. If you get "file not found" errors, update those paths inside the corresponding script.
- Training scripts create model files like `model.json` and `model.h5`. Make sure your matching `test.py` loads the expected model outputs.
<!--
::contentReference[oaicite:4]{index=4}
