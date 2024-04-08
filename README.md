# Audio Sentiment Project

This repository contains code for building a model capable of detecting emotions from audio files. The project utilizes machine learning techniques and publicly available datasets to develop a robust system for audio sentiment analysis.

## Introduction

### Background
Traditionally, sentiment analysis has primarily focused on text data, analyzing written words to gauge emotions. However, recent advancements in machine learning have opened doors to a new frontier: exploring the emotional landscape embedded within human speech.

### Motivation
Understanding human emotions is a fundamental aspect of human interaction. It plays a critical role in numerous fields, including Customer Service, Mental Health Monitoring, and Human-Computer Interaction. This project aims to leverage publicly available datasets like RAVDESS, TESS, and CREMA-D, which capture human speech with corresponding emotional labels. By harnessing the power of parallel processing, the goal is to develop a robust and efficient system capable of accurately classifying emotions from audio data.

### Goals
1. **Enhance Speed and Efficiency:** Leverage parallel processing techniques to significantly speed up the system's performance, focusing on data preprocessing and potentially model training.
2. **Optimize for Deployment:** Explore TensorRT for optimized inference and convert the model to ONNX format, making it more versatile and adaptable for deployment across various platforms.
3. **Develop an Accurate System:** Build a robust system that effectively classifies emotions from audio data using machine learning models.

## Methodology

To build our audio sentiment detection system, we'll follow these key steps:

1. **Data Preprocessing:** Utilize libraries like Pandas and potentially Dask for parallel feature extraction, data cleaning, and normalization from audio data. Additionally, grid search will be employed to optimize the model using WandB.
2. **Feature Extraction:** Extract features from speech audio files using librosa.
3. **Model Training:** Select an appropriate machine learning model and explore parallel training techniques to accelerate training time.
4. **Optimization and Deployment:** Investigate using TensorRT for optimized inference on target hardware and convert the model to ONNX format for wider deployment across platforms.
5. **Evaluation and Refinement:** Continuously evaluate the system's performance and refine the approach based on the results, including parallelization strategies and model parameters.

## Dataset Description

For our audio sentiment detection system, we will leverage several publicly available datasets:

- **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song):** Provides recordings of actors expressing various emotions (e.g., neutral, happy, sad, angry) with corresponding labels.
- **TESS (Toronto Emotional Speech Set):** Offers recordings of spoken sentences conveying different emotions with labeled data.
- **CREMA-D (Crowd Sourced Emotional Multimodal Actors Dataset):** Includes audio recordings of acted speech along with emotional annotations from crowd-sourced labeling.

The combined dataset will have a size of 1.8 GB. These datasets offer a diverse range of audio samples with corresponding emotional labels, crucial for training and evaluating our sentiment detection model.

## Data Source

- [RAVDESS Emotional speech audio (Kaggle)](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)
- [CREMA-D (Kaggle)](https://www.kaggle.com/datasets/ejlok1/cremad)
- [Toronto Emotional Speech Set (TESS) (Kaggle)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

## Reference

- [Generating Real-Time Audio Sentiment Analysis With AI — Smashing Magazine](https://www.smashingmagazine.com/2023/09/generating-real-time-audio-sentiment-analysis-ai/)
- [RAVDESS | Kaggle](https://www.kaggle.com/code/khantalha07/ravdess)

## Installation

If using Windows, you will need `soundfile` as the backend. For Linux, `sox_io` is recommended. To help with handling of files, install `libsox-dev`:

```bash
sudo apt install libsox-dev
```

Other suggestions for a error message `RuntimeError: Couldn't find appropriate backend to handle uri dataset/data/RAVDEES/Actor_21/03-01-06-01-02-01-21.wav and format None.` you can lookup the following link:
- [cannot import torch audio ' No audio backend is available.'](https://stackoverflow.com/questions/62543843/cannot-import-torch-audio-no-audio-backend-is-available)