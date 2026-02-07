import SwiftUI

// MARK: - Home Screen

struct HomeView: View {
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Quick Stats
                    HStack(spacing: 10) {
                        StatCardView(label: "Avg HR", value: "72", unit: "BPM", icon: "heart.fill", color: .red)
                        StatCardView(label: "Recordings", value: "24", unit: "total", icon: "waveform", color: .teal)
                        StatCardView(label: "Normal", value: "92", unit: "%", icon: "checkmark.circle", color: .green)
                    }
                    
                    // Waveform Preview
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Last Recording")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        WaveformView(heartRate: 72)
                            .frame(height: 140)
                    }
                    .padding()
                    .background(.regularMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 16))
                    
                    // Record Button
                    NavigationLink(destination: RecordingView()) {
                        HStack {
                            Image(systemName: "mic.fill")
                            Text("Start New Recording")
                                .fontWeight(.bold)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 16)
                        .background(Color.teal)
                        .foregroundColor(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                    }
                    
                    // Recent Recordings
                    HStack {
                        Text("Recent Recordings").font(.headline)
                        Spacer()
                        NavigationLink("View All") { HistoryView() }
                            .font(.subheadline)
                    }
                    
                    ForEach(sampleRecordings) { recording in
                        RecordingRow(recording: recording)
                    }
                }
                .padding()
            }
            .navigationTitle("PhonoCardi")
        }
    }
}

// MARK: - Recording Screen

struct RecordingView: View {
    @State private var isRecording = true
    @State private var elapsed = 0
    
    let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()
    
    var body: some View {
        VStack(spacing: 16) {
            // Status
            HStack(spacing: 6) {
                Circle()
                    .fill(.red)
                    .frame(width: 8, height: 8)
                    .opacity(isRecording ? 1 : 0.3)
                Text(isRecording ? "RECORDING" : "PAUSED")
                    .font(.caption.weight(.bold))
                    .foregroundColor(.red)
            }
            
            Text(String(format: "%02d:%02d", elapsed / 60, elapsed % 60))
                .font(.system(size: 42, weight: .bold, design: .monospaced))
            
            // Heart Rate
            HStack(alignment: .bottom, spacing: 6) {
                Image(systemName: "heart.fill")
                    .foregroundColor(.red)
                    .font(.title2)
                Text("72")
                    .font(.system(size: 48, weight: .bold, design: .monospaced))
                Text("BPM")
                    .foregroundColor(.secondary)
            }
            
            // Waveform
            WaveformView(heartRate: 72, isAnimating: isRecording)
                .frame(height: 200)
                .padding(.horizontal, 4)
                .background(.regularMaterial)
                .clipShape(RoundedRectangle(cornerRadius: 16))
            
            // Stats
            HStack(spacing: 10) {
                StatCardView(label: "S1-S2", value: "310", unit: "ms", icon: "waveform.path", color: .red)
                StatCardView(label: "S2-S1", value: "520", unit: "ms", icon: "waveform.path", color: .teal)
                StatCardView(label: "Quality", value: "96", unit: "%", icon: "checkmark.shield", color: .green)
            }
            
            Spacer()
            
            // Controls
            HStack(spacing: 24) {
                Button(action: { isRecording.toggle() }) {
                    Image(systemName: isRecording ? "pause.fill" : "play.fill")
                        .font(.title)
                        .frame(width: 72, height: 72)
                        .background(isRecording ? Color.red : Color.teal)
                        .foregroundColor(.white)
                        .clipShape(Circle())
                }
                
                Button(action: {}) {
                    Image(systemName: "stop.fill")
                        .font(.title3)
                        .frame(width: 56, height: 56)
                        .background(.ultraThinMaterial)
                        .clipShape(Circle())
                }
            }
        }
        .padding()
        .onReceive(timer) { _ in
            if isRecording { elapsed += 1 }
        }
    }
}

// MARK: - History Screen

struct HistoryView: View {
    var body: some View {
        NavigationStack {
            List {
                ForEach(sampleRecordings) { recording in
                    RecordingRow(recording: recording)
                }
            }
            .navigationTitle("History")
        }
    }
}

// MARK: - Analysis Screen

struct AnalysisView: View {
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Status
                    VStack(spacing: 12) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 64))
                            .foregroundColor(.teal)
                        Text("Normal Heart Sounds")
                            .font(.title2.bold())
                        Text("Last analyzed: Today, 14:30")
                            .foregroundColor(.secondary)
                        Text("94% Confidence")
                            .font(.subheadline.bold())
                            .padding(.horizontal, 16)
                            .padding(.vertical, 6)
                            .background(Color.teal)
                            .foregroundColor(.white)
                            .clipShape(Capsule())
                    }
                    .padding()
                    .frame(maxWidth: .infinity)
                    .background(.regularMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 20))
                    
                    // Results
                    VStack(alignment: .leading, spacing: 16) {
                        Text("Detection Results").font(.headline)
                        
                        ForEach([
                            ("Normal", 0.94, Color.teal),
                            ("Systolic Murmur", 0.03, Color.orange),
                            ("Diastolic Murmur", 0.01, Color.red),
                        ], id: \.0) { label, value, color in
                            HStack {
                                Text(label).font(.subheadline)
                                Spacer()
                                ProgressView(value: value)
                                    .frame(width: 100)
                                    .tint(color)
                                Text("\(Int(value * 100))%")
                                    .font(.caption.bold().monospaced())
                                    .frame(width: 36, alignment: .trailing)
                            }
                        }
                    }
                    
                    // Disclaimer
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundColor(.orange)
                        Text("AI analysis is for informational purposes only and should not replace professional medical diagnosis.")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    .background(Color.orange.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                }
                .padding()
            }
            .navigationTitle("AI Analysis")
        }
    }
}

// MARK: - Settings Screen

struct SettingsView: View {
    @AppStorage("darkMode") private var darkMode = false
    @AppStorage("notchFilter") private var notchFilter = true
    @AppStorage("autoGain") private var autoGain = true
    @AppStorage("showEnvelope") private var showEnvelope = true
    @AppStorage("autoAnalyze") private var autoAnalyze = false
    
    var body: some View {
        NavigationStack {
            Form {
                Section("Appearance") {
                    Toggle("Dark Mode", isOn: $darkMode)
                }
                Section("Audio & Filters") {
                    Toggle("50 Hz Notch Filter", isOn: $notchFilter)
                    Toggle("Auto Gain Control", isOn: $autoGain)
                    HStack {
                        Text("Bandpass: ")
                        Spacer()
                        Text("20 – 600 Hz")
                            .font(.subheadline.monospaced())
                            .foregroundColor(.secondary)
                    }
                }
                Section("Visualization") {
                    Toggle("Show Envelope", isOn: $showEnvelope)
                }
                Section("AI Analysis") {
                    Toggle("Auto-Analyze", isOn: $autoAnalyze)
                }
                Section("Data") {
                    Button("Export All Recordings") {}
                    Button("Clear History", role: .destructive) {}
                }
                Section {
                    Text("PhonoCardi v1.0 · KMP · iOS 26 / Android 14")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .frame(maxWidth: .infinity, alignment: .center)
                }
            }
            .navigationTitle("Settings")
        }
    }
}

// MARK: - Reusable Components

struct StatCardView: View {
    let label: String
    let value: String
    let unit: String
    let icon: String
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.caption2)
                    .foregroundColor(color)
                Text(label)
                    .font(.caption2.weight(.semibold))
                    .foregroundColor(.secondary)
                    .textCase(.uppercase)
            }
            HStack(alignment: .bottom, spacing: 2) {
                Text(value)
                    .font(.system(size: 24, weight: .bold, design: .monospaced))
                Text(unit)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.regularMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 14))
    }
}

struct RecordingRow: View {
    let recording: SampleRecording
    
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "waveform.path.ecg")
                .foregroundColor(recording.isNormal ? .teal : .red)
                .frame(width: 32)
            
            VStack(alignment: .leading) {
                Text(recording.date).font(.subheadline.weight(.semibold))
                Text(recording.detail).font(.caption).foregroundColor(.secondary)
            }
            
            Spacer()
            
            Text(recording.status)
                .font(.caption2.bold())
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(recording.isNormal ? Color.teal.opacity(0.15) : Color.red.opacity(0.15))
                .foregroundColor(recording.isNormal ? .teal : .red)
                .clipShape(Capsule())
            
            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.vertical, 4)
    }
}

// MARK: - Sample Data

struct SampleRecording: Identifiable {
    let id = UUID()
    let date: String
    let detail: String
    let status: String
    var isNormal: Bool { status == "Normal" }
}

let sampleRecordings = [
    SampleRecording(date: "Today, 14:30", detail: "0:45 · 72 BPM", status: "Normal"),
    SampleRecording(date: "Today, 09:15", detail: "1:20 · 68 BPM", status: "Normal"),
    SampleRecording(date: "Yesterday, 18:45", detail: "0:55 · 88 BPM", status: "Murmur"),
    SampleRecording(date: "Jan 29, 11:00", detail: "1:05 · 74 BPM", status: "Normal"),
    SampleRecording(date: "Jan 28, 16:20", detail: "0:38 · 71 BPM", status: "Normal"),
]
