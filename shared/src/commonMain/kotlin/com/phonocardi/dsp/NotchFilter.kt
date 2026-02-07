package com.phonocardi.dsp

import kotlin.math.*

/**
 * IIR Notch Filter for power line interference rejection.
 * Removes 50 Hz (EU) or 60 Hz (US) mains hum from the signal.
 * Uses a narrow rejection band (Q=30) to minimize distortion of cardiac sounds.
 */
class NotchFilter(
    private val sampleRate: Float,
    private val notchFrequency: Float = 50f,
    private val qualityFactor: Float = 30f
) : AudioFilter {

    private val section: BiquadSection

    init {
        val w0 = 2.0 * PI * notchFrequency / sampleRate
        val alpha = sin(w0) / (2.0 * qualityFactor)

        val b0 = 1.0f
        val b1 = (-2.0 * cos(w0)).toFloat()
        val b2 = 1.0f
        val a0 = (1.0 + alpha).toFloat()
        val a1 = (-2.0 * cos(w0)).toFloat()
        val a2 = (1.0 - alpha).toFloat()

        // Normalize coefficients
        section = BiquadSection(
            b0 / a0, b1 / a0, b2 / a0,
            a1 / a0, a2 / a0
        )
    }

    override fun process(samples: FloatArray): FloatArray {
        return section.process(samples)
    }

    override fun reset() {
        section.reset()
    }
}
