package com.phonocardi.dsp

import kotlin.math.*

/**
 * 2nd-order Butterworth High-Pass Filter.
 * Used for DC offset removal at 5 Hz cutoff.
 * Removes any DC bias from the ADC without affecting heart sounds (>20 Hz).
 */
class HighPassFilter(
    private val sampleRate: Float,
    private val cutoffFrequency: Float = 5f
) : AudioFilter {

    private val section: BiquadSection

    init {
        val w0 = 2.0 * PI * cutoffFrequency / sampleRate
        val alpha = sin(w0) / (2.0 * sqrt(2.0)) // Q = 1/sqrt(2) for Butterworth

        val cosW0 = cos(w0)
        val a0 = 1.0 + alpha
        val b0 = ((1.0 + cosW0) / 2.0 / a0).toFloat()
        val b1 = (-(1.0 + cosW0) / a0).toFloat()
        val b2 = ((1.0 + cosW0) / 2.0 / a0).toFloat()
        val a1 = (-2.0 * cosW0 / a0).toFloat()
        val a2 = ((1.0 - alpha) / a0).toFloat()

        section = BiquadSection(b0, b1, b2, a1, a2)
    }

    override fun process(samples: FloatArray): FloatArray {
        return section.process(samples)
    }

    override fun reset() {
        section.reset()
    }
}
