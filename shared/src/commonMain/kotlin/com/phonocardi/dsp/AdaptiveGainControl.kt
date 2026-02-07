package com.phonocardi.dsp

import kotlin.math.*

/**
 * Adaptive Gain Control (AGC).
 * Normalizes signal amplitude for consistent visualization regardless of
 * stethoscope placement pressure or mic sensitivity differences.
 *
 * Uses a slow attack / fast release envelope follower to avoid
 * amplifying noise during silent periods.
 */
class AdaptiveGainControl(
    private val sampleRate: Float,
    private val targetRms: Float = 0.3f,
    private val attackTimeMs: Float = 500f,
    private val releaseTimeMs: Float = 100f,
    private val maxGain: Float = 50f,
    private val minGain: Float = 0.1f
) : AudioFilter {

    private var currentGain: Float = 1f
    private var envelopeLevel: Float = 0f

    private val attackCoeff = exp(-1f / (attackTimeMs * sampleRate / 1000f))
    private val releaseCoeff = exp(-1f / (releaseTimeMs * sampleRate / 1000f))

    override fun process(samples: FloatArray): FloatArray {
        val output = FloatArray(samples.size)

        for (i in samples.indices) {
            val absVal = abs(samples[i])

            // Envelope follower with asymmetric attack/release
            envelopeLevel = if (absVal > envelopeLevel) {
                releaseCoeff * envelopeLevel + (1f - releaseCoeff) * absVal
            } else {
                attackCoeff * envelopeLevel + (1f - attackCoeff) * absVal
            }

            // Compute desired gain
            val desiredGain = if (envelopeLevel > 1e-6f) {
                (targetRms / envelopeLevel).coerceIn(minGain, maxGain)
            } else {
                currentGain
            }

            // Smooth gain transition
            currentGain += (desiredGain - currentGain) * 0.001f

            output[i] = samples[i] * currentGain
        }

        return output
    }

    override fun reset() {
        currentGain = 1f
        envelopeLevel = 0f
    }
}
