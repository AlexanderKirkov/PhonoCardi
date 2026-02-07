# ML Model — Heart Sound Classifier

## Architecture

```
Input: 5s PCG audio @ 44.1kHz
         ↓
    Resample → 16kHz mono
         ↓
    Log-mel spectrogram (499 frames × 64 mel bins)
         ↓
    ┌─────────────────────────────────┐
    │  Conv1D(32, k=5) + BN + ReLU   │
    │  MaxPool(4) + Dropout(0.2)     │
    ├─────────────────────────────────┤
    │  Conv1D(64, k=5) + BN + ReLU   │
    │  MaxPool(4) + Dropout(0.2)     │
    ├─────────────────────────────────┤
    │  Conv1D(128, k=3) + BN + ReLU  │
    │  MaxPool(4) + Dropout(0.3)     │
    ├─────────────────────────────────┤
    │  Conv1D(128, k=3) + BN + ReLU  │
    │  GlobalAveragePooling1D        │
    ├─────────────────────────────────┤
    │  Dense(128) + ReLU             │
    │  Dropout(0.5)                  │
    │  Dense(5) + Softmax            │
    └─────────────────────────────────┘
         ↓
    Output: [normal, systolic_murmur, diastolic_murmur, extra_sound, noisy]
```

## Feature Extraction

Mel spectrogram parameters (must match between training and inference):

| Parameter       | Value    |
|-----------------|----------|
| Sample rate     | 16,000 Hz |
| FFT size        | 512      |
| Hop length      | 160 (10ms) |
| Mel bins        | 64       |
| Freq range      | 25–2000 Hz |
| Segment length  | 5.0s (80,000 samples) |
| Output shape    | [499, 64] |

These parameters are encoded identically in:
- `ml/training/train_heart_sound_classifier.py` (Python training)
- `shared/src/commonMain/.../ai/MelSpectrogramExtractor.kt` (Kotlin inference)

## Training

### Quick Start

```bash
cd ml/training
pip install -r requirements.txt
python train_heart_sound_classifier.py
```

This will:
1. Download PhysioNet/CinC 2016 Challenge training-a dataset (~350 recordings)
2. Extract log-mel spectrogram features with 50% overlapping 5s segments
3. Train a 1D CNN with data augmentation (SpecAugment) and class weighting
4. Evaluate on a held-out 15% test split
5. Export to TFLite (float16 quantized) and Core ML

### Dataset

**PhysioNet/CinC Challenge 2016**
- 3,240 heart sound recordings (5–120 seconds each)
- Binary labels: normal (-1) / abnormal (1)
- 6 databases from different stethoscopes and populations
- Link: https://physionet.org/content/challenge-2016/1.0.0/

To train on the full dataset, edit `train_heart_sound_classifier.py`:
```python
download_dataset(["training-a", "training-b", "training-c",
                  "training-d", "training-e", "training-f"])
```

### Expected Results

| Dataset     | Recordings | Accuracy | F1-Score |
|-------------|-----------|----------|----------|
| training-a  | ~400      | ~85%     | ~0.83    |
| All (a–f)   | ~3,240    | ~89%     | ~0.87    |

## Deployment

After training, copy the model files:

### Android (TFLite)
```bash
cp models/heart_sound_classifier.tflite androidApp/src/main/assets/
cp models/labels.txt androidApp/src/main/assets/
```

The Android classifier (`HeartSoundClassifier.kt`) uses:
- `org.tensorflow:tensorflow-lite:2.14.0`
- `org.tensorflow:tensorflow-lite-support:2.14.0`

### iOS (Core ML)
```bash
# Core ML model (compiled by Xcode automatically)
cp -r models/HeartSoundClassifier.mlpackage iosApp/PhonoCardi/

# Or if using compiled model:
xcrun coremlcompiler compile HeartSoundClassifier.mlpackage .
cp -r HeartSoundClassifier.mlmodelc iosApp/PhonoCardi/
```

The iOS classifier uses Kotlin/Native interop with CoreML framework.

## Model Size

| Format               | Size     |
|----------------------|----------|
| Keras (.keras)       | ~3.5 MB  |
| TFLite float32       | ~3.3 MB  |
| TFLite float16       | ~1.7 MB  |
| TFLite int8 (PTQ)    | ~0.9 MB  |
| Core ML (.mlpackage) | ~3.5 MB  |

## Preprocessing Pipeline

The same preprocessing is implemented in both languages:

```
┌─────────────────────────────────────────────────┐
│              Python (training)                   │
│  train_heart_sound_classifier.py                 │
│    load_wav() → extract_segments()               │
│    → compute_log_mel_spectrogram()               │
│                                                  │
│              Kotlin (inference)                   │
│  MelSpectrogramExtractor.kt                      │
│    extract() → resample() → computePowerSpectrum │
│    → applyMelFilterbank() → log()                │
└─────────────────────────────────────────────────┘
```

Both produce identical [499, 64] float32 tensors for a given input.

## Future Work

- [ ] Multi-class training with CinC 2022 dataset (systolic vs diastolic murmurs)
- [ ] On-device model update (federated learning)
- [ ] Model pruning for further size reduction
- [ ] INT8 quantization with calibration dataset
- [ ] NNAPI / ANE hardware acceleration benchmarks
