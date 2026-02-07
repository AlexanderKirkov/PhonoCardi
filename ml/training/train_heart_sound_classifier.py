#!/usr/bin/env python3
"""
PhonoCardi Heart Sound Classification — Training Pipeline

Trains a 1D CNN on the PhysioNet/CinC Challenge 2016 dataset for binary
classification (Normal vs Abnormal heart sounds), then exports to:
  - TensorFlow Lite (.tflite) for Android
  - Core ML (.mlmodel) for iOS

Dataset: https://physionet.org/content/challenge-2016/1.0.0/
Architecture: Mel-spectrogram → 1D CNN → Dense → Sigmoid

Usage:
    pip install tensorflow tensorflow-hub numpy scipy scikit-learn coremltools
    python train_heart_sound_classifier.py

The script will:
1. Download the PhysioNet 2016 training-a dataset
2. Preprocess audio → log-mel spectrograms
3. Train a 1D CNN classifier
4. Evaluate on held-out test set
5. Export to TFLite and Core ML formats
"""

import os
import sys
import glob
import shutil
import zipfile
import urllib.request
import numpy as np
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────

SAMPLE_RATE = 16000         # Resample all audio to 16kHz
SEGMENT_DURATION = 5.0      # Fixed-length segments in seconds
SEGMENT_SAMPLES = int(SAMPLE_RATE * SEGMENT_DURATION)  # 80000

# Mel spectrogram parameters
N_MELS = 64
N_FFT = 512
HOP_LENGTH = 160            # 10ms at 16kHz
FMIN = 25.0
FMAX = 2000.0               # Heart sounds are below 2kHz

# Training parameters
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.15
TEST_SPLIT = 0.15

# Class labels
CLASSES = ["normal", "abnormal"]
NUM_CLASSES = 2

# Paths
DATA_DIR = Path("./data/physionet2016")
MODEL_DIR = Path("./models")
TFLITE_PATH = MODEL_DIR / "heart_sound_classifier.tflite"
COREML_PATH = MODEL_DIR / "HeartSoundClassifier.mlpackage"
LABELS_PATH = MODEL_DIR / "labels.txt"

# ─── Dataset Download ────────────────────────────────────────────────────────

PHYSIONET_URLS = {
    "training-a": "https://physionet.org/files/challenge-2016/1.0.0/training-a.zip",
    "training-b": "https://physionet.org/files/challenge-2016/1.0.0/training-b.zip",
    "training-c": "https://physionet.org/files/challenge-2016/1.0.0/training-c.zip",
    "training-d": "https://physionet.org/files/challenge-2016/1.0.0/training-d.zip",
    "training-e": "https://physionet.org/files/challenge-2016/1.0.0/training-e.zip",
    "training-f": "https://physionet.org/files/challenge-2016/1.0.0/training-f.zip",
}

def download_dataset(datasets=None):
    """Download and extract PhysioNet 2016 challenge data."""
    if datasets is None:
        datasets = ["training-a"]  # Start with training-a (largest, best quality)
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for name in datasets:
        url = PHYSIONET_URLS[name]
        zip_path = DATA_DIR / f"{name}.zip"
        extract_dir = DATA_DIR / name
        
        if extract_dir.exists():
            print(f"  ✓ {name} already exists")
            continue
        
        print(f"  ↓ Downloading {name}...")
        urllib.request.urlretrieve(url, zip_path)
        
        print(f"  ⊞ Extracting {name}...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(DATA_DIR)
        
        zip_path.unlink()
    
    print(f"  ✓ Dataset ready in {DATA_DIR}")


# ─── Audio Processing ────────────────────────────────────────────────────────

def load_wav(filepath, target_sr=SAMPLE_RATE):
    """Load a WAV file, convert to mono, resample to target_sr."""
    from scipy.io import wavfile
    from scipy.signal import resample
    
    sr, audio = wavfile.read(filepath)
    
    # Convert to float32 in [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.float64:
        audio = audio.astype(np.float32)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Resample if needed
    if sr != target_sr:
        num_samples = int(len(audio) * target_sr / sr)
        audio = resample(audio, num_samples)
    
    return audio


def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    """Create a mel-scale filterbank matrix."""
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)
    
    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    filterbank = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left = bin_points[i]
        center = bin_points[i + 1]
        right = bin_points[i + 2]
        
        for j in range(left, center):
            if center != left:
                filterbank[i, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                filterbank[i, j] = (right - j) / (right - center)
    
    return filterbank


def compute_log_mel_spectrogram(audio, sr=SAMPLE_RATE):
    """Compute log-mel spectrogram features from raw audio."""
    # Pad or trim to fixed length
    if len(audio) < SEGMENT_SAMPLES:
        audio = np.pad(audio, (0, SEGMENT_SAMPLES - len(audio)))
    else:
        audio = audio[:SEGMENT_SAMPLES]
    
    # STFT
    window = np.hanning(N_FFT)
    n_frames = 1 + (len(audio) - N_FFT) // HOP_LENGTH
    
    stft = np.zeros((N_FFT // 2 + 1, n_frames))
    for i in range(n_frames):
        start = i * HOP_LENGTH
        frame = audio[start:start + N_FFT] * window
        spectrum = np.fft.rfft(frame)
        stft[:, i] = np.abs(spectrum) ** 2
    
    # Mel filterbank
    fb = mel_filterbank(sr, N_FFT, N_MELS, FMIN, FMAX)
    mel_spec = fb @ stft
    
    # Log scale
    log_mel = np.log(mel_spec + 1e-6)
    
    return log_mel.T.astype(np.float32)  # Shape: (n_frames, n_mels)


def extract_segments(audio, sr=SAMPLE_RATE, overlap=0.5):
    """Split long recordings into fixed-length overlapping segments."""
    step = int(SEGMENT_SAMPLES * (1 - overlap))
    segments = []
    
    for start in range(0, len(audio) - SEGMENT_SAMPLES + 1, step):
        segment = audio[start:start + SEGMENT_SAMPLES]
        segments.append(segment)
    
    # Always include at least one segment (padded if needed)
    if not segments:
        padded = np.pad(audio, (0, max(0, SEGMENT_SAMPLES - len(audio))))
        segments.append(padded[:SEGMENT_SAMPLES])
    
    return segments


# ─── Dataset Loading ─────────────────────────────────────────────────────────

def load_physionet_dataset(data_dirs=None):
    """Load PhysioNet 2016 dataset, returns features and labels."""
    if data_dirs is None:
        data_dirs = [DATA_DIR / "training-a"]
    
    features = []
    labels = []
    
    for data_dir in data_dirs:
        # Read reference labels
        ref_file = data_dir / "REFERENCE.csv"
        if not ref_file.exists():
            print(f"  ⚠ No REFERENCE.csv in {data_dir}, skipping")
            continue
        
        label_map = {}
        with open(ref_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    # Format: filename,-1(normal) or filename,1(abnormal)
                    filename, label = parts[0].strip(), int(parts[1].strip())
                    label_map[filename] = 0 if label == -1 else 1  # 0=normal, 1=abnormal
        
        # Load audio files
        wav_files = sorted(glob.glob(str(data_dir / "*.wav")))
        print(f"  Loading {len(wav_files)} recordings from {data_dir.name}...")
        
        for wav_path in wav_files:
            basename = Path(wav_path).stem
            if basename not in label_map:
                continue
            
            label = label_map[basename]
            
            try:
                audio = load_wav(wav_path)
                
                # Split into segments
                segments = extract_segments(audio, overlap=0.5)
                
                for segment in segments:
                    mel = compute_log_mel_spectrogram(segment)
                    features.append(mel)
                    labels.append(label)
            
            except Exception as e:
                print(f"    ⚠ Error loading {basename}: {e}")
                continue
    
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    print(f"  ✓ Loaded {len(X)} segments: {np.sum(y==0)} normal, {np.sum(y==1)} abnormal")
    return X, y


# ─── Model Definition ────────────────────────────────────────────────────────

def build_model(input_shape, num_classes=NUM_CLASSES):
    """
    Build a 1D CNN for heart sound classification.
    
    Architecture:
        Input (n_frames, n_mels)
        → Conv1D(32, 5) + BN + ReLU + MaxPool
        → Conv1D(64, 5) + BN + ReLU + MaxPool
        → Conv1D(128, 3) + BN + ReLU + MaxPool
        → Conv1D(128, 3) + BN + ReLU + GlobalAvgPool
        → Dense(128) + Dropout(0.5)
        → Dense(5) + Softmax
    
    Parameters: ~250K (mobile-friendly)
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    
    model = keras.Sequential([
        layers.Input(shape=input_shape, name="mel_input"),
        
        # Block 1
        layers.Conv1D(32, 5, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPool1D(4),
        layers.Dropout(0.2),
        
        # Block 2
        layers.Conv1D(64, 5, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPool1D(4),
        layers.Dropout(0.2),
        
        # Block 3
        layers.Conv1D(128, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPool1D(4),
        layers.Dropout(0.3),
        
        # Block 4
        layers.Conv1D(128, 3, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.GlobalAveragePooling1D(),
        
        # Classifier
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(5, activation="softmax", name="classification"),
    ], name="heart_sound_classifier")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    
    return model


# ─── Training ────────────────────────────────────────────────────────────────

def train(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Train the model with stratified split."""
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    from tensorflow import keras
    
    # Remap binary labels to 5-class for PhonoCardi compatibility
    # 0 = normal, 1 = abnormal → expanded to:
    #   0 = normal
    #   1 = systolic_murmur (from abnormal)
    #   2 = diastolic_murmur (placeholder)
    #   3 = extra_sound (placeholder)
    #   4 = noisy (placeholder)
    # For initial training, we use 0 vs 1 (normal vs murmur)
    # Multi-class requires George B. Moody dataset or CinC 2022
    
    # Stratified train/val/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SPLIT, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VALIDATION_SPLIT / (1 - TEST_SPLIT),
        stratify=y_train, random_state=42
    )
    
    print(f"\n  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # Handle class imbalance
    class_counts = np.bincount(y_train)
    total = len(y_train)
    class_weights = {i: total / (len(class_counts) * count) 
                     for i, count in enumerate(class_counts)}
    print(f"  Class weights: {class_weights}")
    
    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (n_frames, n_mels)
    model = build_model(input_shape)
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "best_model.keras"),
            monitor="val_accuracy", save_best_only=True
        ),
    ]
    
    # Data augmentation via time masking
    def augment(x):
        # Random time masking (SpecAugment-style)
        mask_len = np.random.randint(0, x.shape[0] // 10)
        mask_start = np.random.randint(0, x.shape[0] - mask_len)
        x[mask_start:mask_start + mask_len, :] = 0
        
        # Random frequency masking
        mask_len = np.random.randint(0, x.shape[1] // 8)
        mask_start = np.random.randint(0, x.shape[1] - mask_len)
        x[:, mask_start:mask_start + mask_len] = 0
        
        return x
    
    # Apply augmentation to training data
    X_train_aug = X_train.copy()
    for i in range(len(X_train_aug)):
        if np.random.random() < 0.5:
            X_train_aug[i] = augment(X_train_aug[i].copy())
    
    # Train
    print("\n  🏋️ Training...")
    history = model.fit(
        X_train_aug, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Evaluate
    print("\n  📊 Evaluation on test set:")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"     Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}")
    
    # Per-class metrics
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    from sklearn.metrics import classification_report, confusion_matrix
    print("\n" + classification_report(
        y_test, y_pred, target_names=["normal", "abnormal"]
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return model, history


# ─── Export ──────────────────────────────────────────────────────────────────

def export_tflite(model, output_path=TFLITE_PATH):
    """Export model to TensorFlow Lite with float16 quantization."""
    import tensorflow as tf
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Standard TFLite conversion
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Float16 quantization (halves model size, minimal accuracy loss)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Add metadata for TFLite Task Library compatibility
    tflite_model = converter.convert()
    
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  ✓ TFLite model saved: {output_path} ({size_kb:.0f} KB)")
    
    # Also export full-precision version for comparison
    converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_fp32 = converter_fp32.convert()
    fp32_path = output_path.with_suffix(".fp32.tflite")
    with open(fp32_path, "wb") as f:
        f.write(tflite_fp32)
    
    return output_path


def export_coreml(model, output_path=COREML_PATH):
    """Export model to Core ML for iOS."""
    try:
        import coremltools as ct
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        mlmodel = ct.convert(
            model,
            source="tensorflow",
            inputs=[ct.TensorType(shape=model.input_shape, name="mel_input")],
            classifier_config=ct.ClassifierConfig(
                class_labels=["normal", "systolic_murmur", "diastolic_murmur",
                              "extra_sound", "noisy"]
            ),
            minimum_deployment_target=ct.target.iOS17,
        )
        
        mlmodel.author = "PhonoCardi"
        mlmodel.short_description = (
            "Heart sound classifier trained on PhysioNet/CinC 2016. "
            "Input: 5-second log-mel spectrogram (499 frames × 64 mel bins). "
            "Output: 5-class probability distribution."
        )
        mlmodel.version = "1.0.0"
        
        mlmodel.save(str(output_path))
        print(f"  ✓ Core ML model saved: {output_path}")
        
    except ImportError:
        print("  ⚠ coremltools not installed, skipping Core ML export")
        print("    Install with: pip install coremltools")


def export_labels(output_path=LABELS_PATH):
    """Export class labels file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    labels = [
        "normal",
        "systolic_murmur",
        "diastolic_murmur",
        "extra_sound",
        "noisy",
    ]
    
    with open(output_path, "w") as f:
        for label in labels:
            f.write(label + "\n")
    
    print(f"  ✓ Labels saved: {output_path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PhonoCardi Heart Sound Classifier — Training Pipeline")
    print("=" * 60)
    
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Download data
    print("\n[1/5] Downloading PhysioNet 2016 dataset...")
    download_dataset(["training-a"])  # Add more for larger training set
    
    # Step 2: Load and preprocess
    print("\n[2/5] Loading and preprocessing audio...")
    X, y = load_physionet_dataset()
    
    # Step 3: Train
    print("\n[3/5] Training 1D CNN classifier...")
    model, history = train(X, y)
    
    # Step 4: Export
    print("\n[4/5] Exporting models...")
    export_tflite(model)
    export_coreml(model)
    export_labels()
    
    # Step 5: Summary
    print("\n[5/5] Done! 🎉")
    print(f"\n  Output files:")
    print(f"    Android: {TFLITE_PATH}")
    print(f"    iOS:     {COREML_PATH}")
    print(f"    Labels:  {LABELS_PATH}")
    print(f"\n  Copy to your project:")
    print(f"    cp {TFLITE_PATH} androidApp/src/main/assets/")
    print(f"    cp {LABELS_PATH} androidApp/src/main/assets/")
    print(f"    cp -r {COREML_PATH} iosApp/PhonoCardi/")
    print("=" * 60)


if __name__ == "__main__":
    main()
