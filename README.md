# PhonoCardi 🫀

**Digital Phonocardiogram Application for iOS & Android**

A cross-platform mobile application for digital auscultation using an electronic stethoscope. Built with Kotlin Multiplatform (KMP), the app captures heart sounds, applies medical-grade digital filters, processes signals according to phonocardiogram (PCG) principles, and provides real-time color-coded waveform visualization.

## Features

- **Real-time audio capture** from external stethoscope microphone (3.5mm / USB-C / Lightning)
- **Digital signal processing** — Butterworth bandpass (20–600 Hz), notch filter (50/60 Hz), adaptive gain
- **PCG analysis** — Shannon energy envelope, S1/S2 detection, heart rate calculation
- **Color-coded visualization** — Real-time scrolling waveform with S1 (red), S2 (cyan), murmur (yellow) markers
- **Post-recording review** — Full waveform navigation, zoom, playback with variable speed
- **AI heart sound classification** — On-device ML (Core ML / TensorFlow Lite) for anomaly detection
- **Export** — WAV audio files, PDF reports with PCG waveforms
- **Doctor sharing** — Share recordings via email, messaging, or cloud storage
- **Recording history** — Organized library with search, tags, and metadata

## Architecture

```
phonocardi/
├── shared/                    # KMP shared module (≈80% of code)
│   ├── commonMain/            # Cross-platform business logic
│   │   ├── audio/             # Audio capture interface
│   │   ├── dsp/               # DSP filters (Butterworth, notch, etc.)
│   │   ├── pcg/               # PCG processing (envelope, S1/S2, HR)
│   │   ├── ai/                # ML classifier interface
│   │   ├── recording/         # WAV writer, recording manager
│   │   ├── viewmodel/         # Shared ViewModels (KMP-ViewModel)
│   │   └── db/                # SQLDelight database
│   ├── iosMain/               # iOS actual implementations
│   └── androidMain/           # Android actual implementations
├── iosApp/                    # SwiftUI application
└── androidApp/                # Jetpack Compose application
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Shared Logic | Kotlin Multiplatform 2.1 |
| iOS UI | SwiftUI (iOS 26+) |
| Android UI | Jetpack Compose (Android 14+, API 34) |
| iOS Audio | AVAudioEngine |
| Android Audio | Oboe (C++ via JNI) |
| Database | SQLDelight 2.0 |
| DI | Koin 4.0 |
| iOS ML | Core ML |
| Android ML | TensorFlow Lite |
| Async | kotlinx-coroutines + Flow |

## Prerequisites

- **Android Studio Ladybug** (2024.2+) with KMP plugin
- **Xcode 26** (for iOS builds)
- **JDK 17+**
- **CocoaPods** or **SPM** (for iOS dependencies)

## Getting Started

### Clone & Build

```bash
git clone https://github.com/your-username/phonocardi.git
cd phonocardi
```

### Android

```bash
./gradlew :androidApp:assembleDebug
```

### iOS

```bash
cd iosApp
pod install    # if using CocoaPods
open PhonoCardi.xcworkspace
```

Build and run from Xcode on a physical device (simulator has no microphone input).

## Audio Pipeline

```
Stethoscope Mic → ADC (44.1kHz/16-bit) → DC Removal (HPF 5Hz)
    → Notch Filter (50/60Hz) → Bandpass (20–600Hz, 4th order Butterworth)
    → Adaptive Gain → Shannon Envelope → S1/S2 Detection
    → Visualization + WAV Recording + AI Analysis
```

### Cardiac Frequency Bands

| Sound | Frequency | Duration | Description |
|-------|-----------|----------|-------------|
| S1 (Lub) | 20–150 Hz | 100–170 ms | Mitral & tricuspid valve closure |
| S2 (Dub) | 50–250 Hz | 80–140 ms | Aortic & pulmonic valve closure |
| S3 | 20–70 Hz | ~40 ms | Ventricular filling |
| S4 | 20–70 Hz | ~40 ms | Atrial contraction |
| Murmurs | 100–600 Hz | Variable | Turbulent blood flow |

## AI Model

The heart sound classifier uses a 1D CNN architecture trained on the PhysioNet/CinC Challenge 2016 dataset. It runs entirely on-device for privacy and offline capability.

**Classification categories:** Normal, Systolic Murmur, Diastolic Murmur, Extra Sound (S3/S4), Noisy/Unclassifiable

> ⚠️ **Medical Disclaimer:** AI analysis is for informational purposes only and should not replace professional medical diagnosis.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgments

- [PhysioNet/CinC Challenge 2016](https://physionet.org/content/challenge-2016/) — Heart sound dataset
- [AudioKit](https://audiokit.io/) — Original iOS audio framework inspiration
- [Oboe](https://github.com/google/oboe) — Android low-latency audio
