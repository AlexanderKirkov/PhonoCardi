package com.phonocardi.dsp

import kotlin.math.*

/**
 * Base interface for all audio filters.
 */
interface AudioFilter {
    fun process(samples: FloatArray): FloatArray
    fun reset()
}

/**
 * Single biquad (2nd-order IIR) section.
 * Building block for higher-order Butterworth filters.
 * Implements Direct Form II Transposed for numerical stability.
 */
class BiquadSection(
    private val b0: Float,
    private val b1: Float,
    private val b2: Float,
    private val a1: Float,
    private val a2: Float
) {
    private var z1: Float = 0f
    private var z2: Float = 0f

    fun process(input: Float): Float {
        val output = b0 * input + z1
        z1 = b1 * input - a1 * output + z2
        z2 = b2 * input - a2 * output
        return output
    }

    fun process(samples: FloatArray): FloatArray {
        return FloatArray(samples.size) { i -> process(samples[i]) }
    }

    fun reset() {
        z1 = 0f
        z2 = 0f
    }
}

/**
 * Filter configuration for the audio processing pipeline.
 */
data class FilterConfig(
    val highPassCutoff: Float = 20f,
    val lowPassCutoff: Float = 600f,
    val notchFrequency: Float = 50f,
    val filterOrder: Int = 4,
    val enableNotch: Boolean = true,
    val enableAdaptiveGain: Boolean = true,
    val sampleRate: Float = 44100f
)
