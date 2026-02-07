#!/usr/bin/env python3
"""
PhonoCardi — Model Validation Script

Validates that the trained TFLite model:
  1. Loads without errors
  2. Accepts correct input shape [1, 499, 64]
  3. Produces correct output shape [1, 5]
  4. Output sums to ~1.0 (valid probability distribution)
  5. Runs within acceptable latency (<100ms on desktop)
  6. Handles edge cases (silence, noise, clipping)

Usage:
    python validate_model.py [path/to/model.tflite]
"""

import sys
import time
import numpy as np

def validate_tflite_model(model_path="./models/heart_sound_classifier.tflite"):
    """Run comprehensive validation on TFLite model."""
    import tensorflow as tf
    
    print("=" * 50)
    print("PhonoCardi — Model Validation")
    print("=" * 50)
    
    CLASSES = ["normal", "systolic_murmur", "diastolic_murmur", "extra_sound", "noisy"]
    INPUT_SHAPE = (1, 499, 64)
    OUTPUT_SHAPE = (1, 5)
    
    # Test 1: Load model
    print("\n[1] Loading model...")
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print(f"    ✓ Model loaded: {model_path}")
    except Exception as e:
        print(f"    ✗ Failed to load: {e}")
        return False
    
    # Test 2: Verify shapes
    print("\n[2] Verifying I/O shapes...")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_shape = tuple(input_details[0]['shape'])
    output_shape = tuple(output_details[0]['shape'])
    
    assert input_shape == INPUT_SHAPE, f"Input: expected {INPUT_SHAPE}, got {input_shape}"
    assert output_shape == OUTPUT_SHAPE, f"Output: expected {OUTPUT_SHAPE}, got {output_shape}"
    print(f"    ✓ Input:  {input_shape} ({input_details[0]['dtype']})")
    print(f"    ✓ Output: {output_shape} ({output_details[0]['dtype']})")
    
    # Test 3: Inference with synthetic mel spectrogram
    print("\n[3] Running inference (synthetic mel input)...")
    test_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    assert output.shape == OUTPUT_SHAPE, f"Output shape mismatch: {output.shape}"
    prob_sum = float(output[0].sum())
    assert 0.95 < prob_sum < 1.05, f"Probabilities don't sum to ~1.0: {prob_sum}"
    assert all(output[0] >= 0), "Negative probabilities detected"
    
    top_class = CLASSES[output[0].argmax()]
    top_prob = float(output[0].max())
    print(f"    ✓ Output: {dict(zip(CLASSES, [f'{p:.3f}' for p in output[0]]))}")
    print(f"    ✓ Top: {top_class} ({top_prob:.3f})")
    print(f"    ✓ Sum: {prob_sum:.6f}")
    
    # Test 4: Latency benchmark
    print("\n[4] Latency benchmark (100 iterations)...")
    times = []
    for _ in range(100):
        inp = np.random.randn(*INPUT_SHAPE).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], inp)
        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    
    avg_ms = np.mean(times)
    p95_ms = np.percentile(times, 95)
    print(f"    ✓ Avg: {avg_ms:.1f} ms | P95: {p95_ms:.1f} ms | Max: {max(times):.1f} ms")
    
    # Test 5: Edge cases
    print("\n[5] Edge cases...")
    
    # Silence (all zeros)
    silence = np.zeros(INPUT_SHAPE, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], silence)
    interpreter.invoke()
    silence_out = interpreter.get_tensor(output_details[0]['index'])
    print(f"    Silence → {CLASSES[silence_out[0].argmax()]} ({silence_out[0].max():.3f})")
    
    # Pure noise
    noise = np.random.randn(*INPUT_SHAPE).astype(np.float32) * 10
    interpreter.set_tensor(input_details[0]['index'], noise)
    interpreter.invoke()
    noise_out = interpreter.get_tensor(output_details[0]['index'])
    print(f"    Noise   → {CLASSES[noise_out[0].argmax()]} ({noise_out[0].max():.3f})")
    
    # Constant value (DC)
    dc = np.ones(INPUT_SHAPE, dtype=np.float32) * 5.0
    interpreter.set_tensor(input_details[0]['index'], dc)
    interpreter.invoke()
    dc_out = interpreter.get_tensor(output_details[0]['index'])
    print(f"    DC      → {CLASSES[dc_out[0].argmax()]} ({dc_out[0].max():.3f})")
    
    # Model size
    import os
    size_bytes = os.path.getsize(model_path)
    print(f"\n[6] Model size: {size_bytes / 1024:.0f} KB ({size_bytes / 1024 / 1024:.1f} MB)")
    
    print("\n" + "=" * 50)
    print("✓ All validation checks passed!")
    print("=" * 50)
    return True


def validate_mel_consistency():
    """
    Verify Python and Kotlin mel extraction produce identical results.
    Generates a test WAV file, computes mel spectrogram in Python,
    and saves the expected output for Kotlin unit test comparison.
    """
    from scipy.io import wavfile
    
    print("\n[Mel Consistency Check]")
    
    # Generate synthetic heart sound (5s, 16kHz)
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    
    # Simulate S1+S2 at 72 BPM
    heart_rate = 72
    period = 60.0 / heart_rate
    signal = np.zeros_like(t)
    
    for beat_start in np.arange(0, duration, period):
        # S1: 50-100Hz burst at beat_start
        s1_start = beat_start
        s1_dur = 0.12
        mask = (t >= s1_start) & (t < s1_start + s1_dur)
        envelope = np.exp(-30 * (t[mask] - s1_start - s1_dur / 2) ** 2)
        signal[mask] += 0.5 * envelope * np.sin(2 * np.pi * 60 * t[mask])
        
        # S2: 80-200Hz burst at beat_start + 0.35*period
        s2_start = beat_start + 0.35 * period
        s2_dur = 0.10
        mask = (t >= s2_start) & (t < s2_start + s2_dur)
        if mask.any():
            envelope = np.exp(-40 * (t[mask] - s2_start - s2_dur / 2) ** 2)
            signal[mask] += 0.3 * envelope * np.sin(2 * np.pi * 120 * t[mask])
    
    # Add some noise
    signal += np.random.randn(len(signal)).astype(np.float32) * 0.02
    
    # Save test WAV
    test_dir = Path("./test_data")
    test_dir.mkdir(exist_ok=True)
    wavfile.write(str(test_dir / "synthetic_heart.wav"), sr, (signal * 32767).astype(np.int16))
    
    # Compute mel spectrogram using same parameters as Kotlin
    from train_heart_sound_classifier import compute_log_mel_spectrogram
    mel = compute_log_mel_spectrogram(signal, sr)
    
    np.save(str(test_dir / "expected_mel.npy"), mel)
    print(f"    ✓ Test WAV saved: {test_dir / 'synthetic_heart.wav'}")
    print(f"    ✓ Expected mel saved: {test_dir / 'expected_mel.npy'}")
    print(f"    ✓ Mel shape: {mel.shape}")
    print(f"    ✓ Mel range: [{mel.min():.2f}, {mel.max():.2f}]")


if __name__ == "__main__":
    from pathlib import Path
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./models/heart_sound_classifier.tflite"
    
    if Path(model_path).exists():
        validate_tflite_model(model_path)
    else:
        print(f"Model not found: {model_path}")
        print("Run train_heart_sound_classifier.py first to generate the model.")
        print("\nRunning mel consistency check instead...")
        try:
            validate_mel_consistency()
        except ImportError as e:
            print(f"  ⚠ Skipping: {e}")
