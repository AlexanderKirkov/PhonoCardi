package com.phonocardi.dsp

/**
 * Composable DSP filter chain.
 * Applies filters in order: DC removal → Notch → Bandpass → AGC.
 * Each stage can be individually enabled/disabled via FilterConfig.
 */
class FilterChain(config: FilterConfig) : AudioFilter {

    private val filters: List<AudioFilter>

    init {
        val chain = mutableListOf<AudioFilter>()

        // Stage 1: DC offset removal (always enabled)
        chain.add(HighPassFilter(config.sampleRate, cutoffFrequency = 5f))

        // Stage 2: Power line notch filter
        if (config.enableNotch) {
            chain.add(NotchFilter(config.sampleRate, config.notchFrequency))
            // Also add harmonic rejection (2nd harmonic)
            chain.add(NotchFilter(config.sampleRate, config.notchFrequency * 2))
        }

        // Stage 3: Bandpass filter for cardiac frequency isolation
        chain.add(
            ButterworthBandpass(
                sampleRate = config.sampleRate,
                lowCutoff = config.highPassCutoff,
                highCutoff = config.lowPassCutoff,
                order = config.filterOrder
            )
        )

        // Stage 4: Adaptive gain control
        if (config.enableAdaptiveGain) {
            chain.add(AdaptiveGainControl(config.sampleRate))
        }

        filters = chain
    }

    override fun process(samples: FloatArray): FloatArray {
        var output = samples
        for (filter in filters) {
            output = filter.process(output)
        }
        return output
    }

    override fun reset() {
        filters.forEach { it.reset() }
    }

    companion object {
        /** Default cardiac auscultation filter chain */
        fun cardiac(sampleRate: Float = 44100f): FilterChain {
            return FilterChain(
                FilterConfig(
                    highPassCutoff = 20f,
                    lowPassCutoff = 600f,
                    notchFrequency = 50f,
                    filterOrder = 4,
                    enableNotch = true,
                    enableAdaptiveGain = true,
                    sampleRate = sampleRate
                )
            )
        }

        /** Lung auscultation preset (wider bandwidth) */
        fun pulmonary(sampleRate: Float = 44100f): FilterChain {
            return FilterChain(
                FilterConfig(
                    highPassCutoff = 60f,
                    lowPassCutoff = 2000f,
                    notchFrequency = 50f,
                    filterOrder = 4,
                    enableNotch = true,
                    enableAdaptiveGain = true,
                    sampleRate = sampleRate
                )
            )
        }
    }
}
