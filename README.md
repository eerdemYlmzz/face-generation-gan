

https://github.com/user-attachments/assets/4ce37928-547e-43db-b18e-a8e7ff85f86a


﻿face-generation-gan

A GAN that generates 64x64 RGB human faces from random noise, implemented in TensorFlow/Keras and trained on Google Colab T4 GPU.

**Author:** Erdem YILMAZ

## Generation Timelapse

A timelapse of faces forming over the course of training.

<video src="0723(1).mp4" controls width="512"></video>

If the player does not load, open the file directly: [0723(1).mp4](0723(1).mp4)

## Overview

The project takes a raw collection of face photographs, extracts and crops the faces with a Haar cascade detector, normalizes them into a fixed size tensor dataset, and trains a GAN to synthesize new faces. The trained generator and discriminator are exported as `.keras` models and published to the Hugging Face Hub [`virtuerdem/gan-human-faces`](https://huggingface.co/virtuerdem/gan-human-faces).

## Repository structure

| Path | Description |
|------|-------------|
| `keras3.ipynb` | End to end notebook: data collection notes, face extraction, preprocessing, model definition (generator + discriminator), and the initial training loop. |
| `gan_training.ipynb` | Standalone training notebook: loads the prebuilt `faces.pkl` dataset, restores/saves checkpoints, trains to 1000 epochs, exports the models, and uploads them to the Hugging Face Hub. |
| `haarcascade_frontalface_default.xml` | Pretrained OpenCV Haarcascade used for face detection during preprocessing. |
| `faces.pkl` | Preprocessed dataset: the cropped, resized, normalized faces serialized as a NumPy array. Not tracked in git (see `.gitignore`). |
| `directory/` | Raw source images (Kaggle dataset). Not tracked in git. |
| `requirements.txt` | Dependency list (currently empty). |
| `0_eeYJwyNgOzn4smHx.jpg` | Sample/reference image at repository root. Functional vs Sequential API. |

## Dataset

Source: Kaggle "Human Faces" dataset (`ashwingupta3012/human-faces`). An initial attempt to collect images via `bing-image-downloader` was abandoned due to insufficient data quality and volume.

The raw `directory/` folder contains **7219** images (6991 `.jpg`, 150 `.png`, 78 `.jpeg`). After running the Haar cascade face detector and keeping only the detected face regions, the pipeline produces **5082** cropped faces, which form the training set stored in `faces.pkl`.

## Method


**1. Face extraction.** Each raw image is read with OpenCV, converted to grayscale, and passed to `detectMultiScale(image, scaleFactor=1.25, minNeighbors=8)`. Every detected face region is cropped and resized to 64x64x3.

**2. Preprocessing.** The cropped faces are stacked into a NumPy array of shape `(5082, 64, 64, 3)`, cast to `float32`, and normalized from the `[0, 255]` range to `[-1, 1]` via `(x - 127.5) / 127.5`. The array is serialized to `faces.pkl` and loaded as a batched `tf.data.Dataset` with `BATCH_SIZE = 256`.

**3. Model architecture.**

*Generator* — maps a 100-dimensional noise vector to a 64x64x3 image:
- `Dense(4*4*512)` then `Reshape((4, 4, 512))`
- A stack of `Conv2DTranspose` layers (512, 256, 64, 64 filters) that progressively upsample to 64x64
- `BatchNormalization` + `LeakyReLU` after each block; final layer uses `tanh` activation to match the `[-1, 1]` data range
- Weights initialized with `RandomNormal(0.0, 0.02)`, `use_bias=False`

*Discriminator* — maps a 64x64x3 image to a single logit:
- A stack of `Conv2D` layers (64, 128, 256, 512 filters, stride 2) with `BatchNormalization`, `LeakyReLU`, and `Dropout(0.2)`
- `Flatten` then `Dense(1)` (logit output, no activation)

**4. Training.** Adversarial training with `BinaryCrossentropy(from_logits=True)`. Both networks use the `Adam` optimizer with a learning rate of `1e-4`. Each `train_step` samples a batch of noise, generates fake images, evaluates the discriminator on real and fake batches, and applies gradients to both networks via separate `GradientTape` contexts.

## Training details

| Parameter | Value |
|-----------|-------|
| Epochs | 1000 |
| Batch size | 256 |
| Latent dimension (noise size) | 100 |
| Output resolution | 64x64x3 |
| Optimizer | Adam, lr = 1e-4 (generator and discriminator) |
| Loss | Binary cross-entropy (from logits) |
| Hardware | NVIDIA T4 GPU (Google Colab) |
| Training time | 2.5 hours |

`gan_training.ipynb` uses `tf.train.CheckpointManager` to persist checkpoints every 5 epochs (keeping the last 5), so training can be resumed from the last saved epoch.

## Model artifacts

The final generator and discriminator are saved as `generator.keras` and `discriminator.keras` and pushed to the Hugging Face Hub at `virtuerdem/gan-human-faces`, together with a `config.json` describing the architecture. You can use models in your projects.


