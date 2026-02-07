import SwiftUI

/// Color-coded PCG waveform renderer.
/// S1 = Red, S2 = Cyan, Murmur = Yellow, Baseline = Green/Teal
struct WaveformView: View {
    let waveformData: [Float]
    let envelopeData: [Float]
    let heartRate: Int
    let showEnvelope: Bool
    let showMarkers: Bool
    let isAnimating: Bool
    
    @State private var offset: CGFloat = 0
    
    // Color scheme
    private let s1Color = Color(red: 1.0, green: 0.42, blue: 0.42)     // #FF6B6B
    private let s2Color = Color(red: 0.31, green: 0.80, blue: 0.77)    // #4ECDC4
    private let murmurColor = Color(red: 1.0, green: 0.85, blue: 0.24) // #FFD93D
    private let baselineColor = Color(red: 0.0, green: 0.83, blue: 0.67) // #00D4AA
    private let gridColor = Color.gray.opacity(0.15)
    private let gridMajorColor = Color.gray.opacity(0.25)
    
    init(
        waveformData: [Float] = [],
        envelopeData: [Float] = [],
        heartRate: Int = 0,
        showEnvelope: Bool = true,
        showMarkers: Bool = true,
        isAnimating: Bool = false
    ) {
        self.waveformData = waveformData
        self.envelopeData = envelopeData
        self.heartRate = heartRate
        self.showEnvelope = showEnvelope
        self.showMarkers = showMarkers
        self.isAnimating = isAnimating
    }
    
    var body: some View {
        ZStack(alignment: .topTrailing) {
            TimelineView(.animation(minimumInterval: 1.0/60.0, paused: !isAnimating)) { timeline in
                Canvas { context, size in
                    let w = size.width
                    let h = size.height
                    let midY = h / 2
                    
                    // Grid
                    drawGrid(context: context, w: w, h: h)
                    
                    guard !waveformData.isEmpty else { return }
                    
                    // Envelope
                    if showEnvelope && !envelopeData.isEmpty {
                        drawEnvelope(context: context, w: w, h: h, midY: midY)
                    }
                    
                    // Waveform
                    drawWaveform(context: context, w: w, h: h, midY: midY)
                }
            }
            .background(Color(.systemBackground).opacity(0.05))
            .clipShape(RoundedRectangle(cornerRadius: 12))
            
            // Heart rate overlay
            if heartRate > 0 {
                Text("\(heartRate) BPM")
                    .font(.system(size: 14, weight: .bold, design: .monospaced))
                    .foregroundColor(s1Color)
                    .padding(12)
            }
            
            // Legend
            VStack(alignment: .leading) {
                Spacer()
                HStack(spacing: 12) {
                    legendItem("S1", color: s1Color)
                    legendItem("S2", color: s2Color)
                    legendItem("Murmur", color: murmurColor)
                }
                .padding(8)
            }
        }
    }
    
    private func legendItem(_ label: String, color: Color) -> some View {
        HStack(spacing: 4) {
            RoundedRectangle(cornerRadius: 1)
                .fill(color)
                .frame(width: 12, height: 3)
            Text(label)
                .font(.system(size: 10))
                .foregroundColor(.secondary)
        }
    }
    
    private func drawGrid(context: GraphicsContext, w: CGFloat, h: CGFloat) {
        // Minor grid
        for i in 0...20 {
            let x = CGFloat(i) * w / 20
            context.stroke(Path { p in p.move(to: CGPoint(x: x, y: 0)); p.addLine(to: CGPoint(x: x, y: h)) },
                          with: .color(gridColor), lineWidth: 0.5)
        }
        for i in 0...8 {
            let y = CGFloat(i) * h / 8
            context.stroke(Path { p in p.move(to: CGPoint(x: 0, y: y)); p.addLine(to: CGPoint(x: w, y: y)) },
                          with: .color(gridColor), lineWidth: 0.5)
        }
        // Center line
        context.stroke(Path { p in p.move(to: CGPoint(x: 0, y: h/2)); p.addLine(to: CGPoint(x: w, y: h/2)) },
                      with: .color(gridMajorColor), lineWidth: 1)
    }
    
    private func drawEnvelope(context: GraphicsContext, w: CGFloat, h: CGFloat, midY: CGFloat) {
        var path = Path()
        let step = w / CGFloat(envelopeData.count)
        
        path.move(to: CGPoint(x: 0, y: midY))
        for i in envelopeData.indices {
            let x = CGFloat(i) * step
            let y = midY - CGFloat(envelopeData[i]) * h * 0.45
            path.addLine(to: CGPoint(x: x, y: y))
        }
        for i in envelopeData.indices.reversed() {
            let x = CGFloat(i) * step
            let y = midY + CGFloat(envelopeData[i]) * h * 0.45
            path.addLine(to: CGPoint(x: x, y: y))
        }
        path.closeSubpath()
        context.fill(path, with: .color(s1Color.opacity(0.1)))
    }
    
    private func drawWaveform(context: GraphicsContext, w: CGFloat, h: CGFloat, midY: CGFloat) {
        let step = w / CGFloat(waveformData.count)
        
        for i in 1..<waveformData.count {
            let x1 = CGFloat(i - 1) * step
            let y1 = midY - CGFloat(waveformData[i - 1]) * h * 0.42
            let x2 = CGFloat(i) * step
            let y2 = midY - CGFloat(waveformData[i]) * h * 0.42
            
            let color = colorForSample(index: i, total: waveformData.count)
            
            context.stroke(
                Path { p in p.move(to: CGPoint(x: x1, y: y1)); p.addLine(to: CGPoint(x: x2, y: y2)) },
                with: .color(color),
                lineWidth: 2
            )
        }
    }
    
    private func colorForSample(index: Int, total: Int) -> Color {
        // Simulate S1/S2 regions for demo
        let cycleLen = total / 8
        guard cycleLen > 0 else { return baselineColor }
        let phase = Double(index % cycleLen) / Double(cycleLen)
        
        if phase > 0.05 && phase < 0.15 { return s1Color }
        if phase > 0.35 && phase < 0.43 { return s2Color }
        return baselineColor
    }
}
