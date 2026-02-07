package com.phonocardi.android.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.phonocardi.android.ui.theme.WaveformColors
import com.phonocardi.pcg.HeartSound
import com.phonocardi.pcg.HeartSoundType

/**
 * Real-time PCG waveform renderer with color-coded heart sounds.
 *
 * Colors:
 * - S1 regions: Red (#FF6B6B)
 * - S2 regions: Cyan (#4ECDC4)
 * - Murmur: Yellow (#FFD93D)
 * - Baseline: Green (#00D4AA)
 */
@Composable
fun WaveformCanvas(
    waveformData: FloatArray,
    envelopeData: FloatArray = floatArrayOf(),
    heartSounds: List<HeartSound> = emptyList(),
    heartRate: Int = 0,
    showEnvelope: Boolean = true,
    showMarkers: Boolean = true,
    modifier: Modifier = Modifier
) {
    val colors = MaterialTheme.colorScheme

    Box(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(
                if (colors.background.luminance() < 0.5f)
                    Color(0xFF0D1321) else Color(0xFFF8FAFC)
            )
    ) {
        Canvas(
            modifier = Modifier
                .fillMaxSize()
                .padding(4.dp)
        ) {
            val w = size.width
            val h = size.height
            val midY = h / 2f

            // Draw grid
            drawGrid(w, h, colors)

            if (waveformData.isEmpty()) return@Canvas

            // Draw envelope
            if (showEnvelope && envelopeData.isNotEmpty()) {
                drawEnvelope(envelopeData, w, h, midY)
            }

            // Draw waveform with color coding
            drawColorCodedWaveform(waveformData, heartSounds, w, h, midY)

            // Draw S1/S2 markers
            if (showMarkers) {
                drawHeartSoundMarkers(heartSounds, waveformData.size, w, h)
            }
        }

        // Heart rate overlay
        if (heartRate > 0) {
            Text(
                text = "$heartRate BPM",
                color = WaveformColors.S1,
                fontSize = 14.sp,
                modifier = Modifier
                    .align(Alignment.TopEnd)
                    .padding(12.dp)
            )
        }

        // Legend
        Row(
            modifier = Modifier
                .align(Alignment.BottomStart)
                .padding(8.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            LegendItem("S1", WaveformColors.S1)
            LegendItem("S2", WaveformColors.S2)
            LegendItem("Murmur", WaveformColors.Murmur)
        }
    }
}

@Composable
private fun LegendItem(label: String, color: Color) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Canvas(modifier = Modifier.size(10.dp, 3.dp)) {
            drawRect(color)
        }
        Text(
            text = label,
            fontSize = 10.sp,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}

private fun DrawScope.drawGrid(w: Float, h: Float, colors: androidx.compose.material3.ColorScheme) {
    val gridColor = colors.outlineVariant.copy(alpha = 0.3f)
    val majorColor = colors.outline.copy(alpha = 0.4f)

    // Minor grid lines
    val xStep = w / 20f
    val yStep = h / 8f
    for (i in 0..20) {
        val x = i * xStep
        drawLine(gridColor, Offset(x, 0f), Offset(x, h), strokeWidth = 0.5f)
    }
    for (i in 0..8) {
        val y = i * yStep
        drawLine(gridColor, Offset(0f, y), Offset(w, y), strokeWidth = 0.5f)
    }

    // Major grid + center line
    for (i in 0..4) {
        val x = i * w / 4f
        drawLine(majorColor, Offset(x, 0f), Offset(x, h), strokeWidth = 1f)
    }
    drawLine(majorColor, Offset(0f, h / 2f), Offset(w, h / 2f), strokeWidth = 1f)
}

private fun DrawScope.drawEnvelope(
    envelope: FloatArray, w: Float, h: Float, midY: Float
) {
    if (envelope.isEmpty()) return
    val color = WaveformColors.S1.copy(alpha = 0.12f)
    val path = Path()
    val step = w / envelope.size

    path.moveTo(0f, midY)
    for (i in envelope.indices) {
        path.lineTo(i * step, midY - envelope[i] * h * 0.45f)
    }
    for (i in envelope.indices.reversed()) {
        path.lineTo(i * step, midY + envelope[i] * h * 0.45f)
    }
    path.close()
    drawPath(path, color)
}

private fun DrawScope.drawColorCodedWaveform(
    data: FloatArray,
    heartSounds: List<HeartSound>,
    w: Float, h: Float, midY: Float
) {
    if (data.size < 2) return

    val step = w / data.size
    val s1Regions = heartSounds.filter { it.type == HeartSoundType.S1 }
        .map { it.peak.sampleIndex }
    val s2Regions = heartSounds.filter { it.type == HeartSoundType.S2 }
        .map { it.peak.sampleIndex }

    // Draw in segments with appropriate colors
    var prevX = 0f
    var prevY = midY - data[0] * h * 0.42f

    for (i in 1 until data.size) {
        val x = i * step
        val y = midY - data[i] * h * 0.42f

        val color = when {
            isNearPeak(i, s1Regions, data.size / 20) -> WaveformColors.S1
            isNearPeak(i, s2Regions, data.size / 25) -> WaveformColors.S2
            else -> WaveformColors.Baseline
        }

        drawLine(
            color = color,
            start = Offset(prevX, prevY),
            end = Offset(x, y),
            strokeWidth = 2f,
            cap = StrokeCap.Round
        )

        prevX = x
        prevY = y
    }
}

private fun isNearPeak(index: Int, peaks: List<Int>, radius: Int): Boolean {
    return peaks.any { kotlin.math.abs(index - it) < radius }
}

private fun DrawScope.drawHeartSoundMarkers(
    heartSounds: List<HeartSound>,
    dataSize: Int, w: Float, h: Float
) {
    if (dataSize == 0) return
    val step = w / dataSize

    for (sound in heartSounds) {
        val x = sound.peak.sampleIndex * step
        val color = when (sound.type) {
            HeartSoundType.S1 -> WaveformColors.S1.copy(alpha = 0.4f)
            HeartSoundType.S2 -> WaveformColors.S2.copy(alpha = 0.4f)
            else -> WaveformColors.Murmur.copy(alpha = 0.4f)
        }

        // Dashed vertical line
        val dashLen = 6f
        var y = 0f
        while (y < h) {
            drawLine(
                color, Offset(x, y), Offset(x, minOf(y + dashLen, h)),
                strokeWidth = 1f
            )
            y += dashLen * 2
        }
    }
}
