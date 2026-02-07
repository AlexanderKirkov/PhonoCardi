package com.phonocardi.dsp

import kotlin.math.*

/**
 * 4th-order Butterworth Bandpass Filter.
 * Implemented as cascade of 2nd-order sections (SOS) for numerical stability.
 * Uses bilinear transform (Tustin's method) for analog-to-digital conversion.
 *
 * Designed for cardiac sound isolation: 20–600 Hz passband captures
 * S1, S2, S3, S4, and murmur frequencies while rejecting external noise.
 */
class ButterworthBandpass(
    private val sampleRate: Float,
    private val lowCutoff: Float,
    private val highCutoff: Float,
    private val order: Int = 4
) : AudioFilter {

    private val sections: List<BiquadSection>

    init {
        sections = designFilter()
    }

    private fun designFilter(): List<BiquadSection> {
        val nyquist = sampleRate / 2.0
        val wLow = tan(PI * lowCutoff / sampleRate)
        val wHigh = tan(PI * highCutoff / sampleRate)
        val bw = wHigh - wLow
        val w0 = sqrt(wLow * wHigh)

        val numSections = order / 2
        val biquads = mutableListOf<BiquadSection>()

        for (k in 0 until numSections) {
            // Butterworth pole angles
            val theta = PI * (2.0 * k + 1) / (2.0 * order) + PI / 2.0
            val poleReal = cos(theta)
            val poleImag = sin(theta)

            // Bandpass transform: each LP pole becomes a conjugate pair
            val sigma = -bw * poleReal
            val omega = bw * poleImag

            // Bilinear transform coefficients
            val a = 1.0 + sigma + (sigma * sigma + omega * omega + w0 * w0)
            val b0 = (bw / a).toFloat()
            val b1 = 0f
            val b2 = (-bw / a).toFloat()
            val a1 = (2.0 * (w0 * w0 - 1.0) / a).toFloat()
            val a2 = ((1.0 - sigma + (sigma * sigma + omega * omega + w0 * w0)) / a - 1.0).toFloat()

            biquads.add(BiquadSection(b0, b1, b2, a1, a2))
        }

        return biquads
    }

    override fun process(samples: FloatArray): FloatArray {
        var output = samples.copyOf()
        for (section in sections) {
            output = section.process(output)
        }
        return output
    }

    override fun reset() {
        sections.forEach { it.reset() }
    }
}
