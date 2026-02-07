package com.phonocardi.ai

import kotlin.math.*

/**
 * Mel-spectrogram feature extractor for heart sound classification.
 *
 * Matches the Python training pipeline exactly:
 *   - Sample rate: 16 kHz
 *   - FFT size: 512
 *   - Hop length: 160 (10ms)
 *   - Mel bins: 64
 *   - Frequency range: 25–2000 Hz
 *   - Segment duration: 5 seconds → 499 frames
 *
 * The output [499 × 64] log-mel spectrogram is fed directly to the
 * TFLite/CoreML classifier.
 */
class MelSpectrogramExtractor(
    private val sampleRate: Int = 16000,
    private val nFft: Int = 512,
    private val hopLength: Int = 160,
    private val nMels: Int = 64,
    private val fMin: Float = 25f,
    private val fMax: Float = 2000f,
    private val segmentDuration: Float = 5.0f
) {
    val segmentSamples = (sampleRate * segmentDuration).toInt()  // 80000
    val numFrames = 1 + (segmentSamples - nFft) / hopLength      // 499
    
    // Pre-computed mel filterbank
    private val melFilterbank: Array<FloatArray> = buildMelFilterbank()
    
    // Pre-computed Hanning window
    private val window: FloatArray = FloatArray(nFft) { i ->
        (0.5f * (1f - cos(2f * PI.toFloat() * i / nFft))).toFloat()
    }

    /**
     * Extract log-mel spectrogram from raw audio samples.
     *
     * @param samples Audio samples at 44100 Hz (will be resampled to 16kHz)
     * @param inputSampleRate Sample rate of the input audio
     * @return Float array of shape [numFrames * nMels] (flattened [499, 64])
     */
    fun extract(samples: FloatArray, inputSampleRate: Int = 44100): FloatArray {
        // Step 1: Resample to 16kHz if needed
        val resampled = if (inputSampleRate != sampleRate) {
            resample(samples, inputSampleRate, sampleRate)
        } else {
            samples
        }

        // Step 2: Pad or trim to exactly segmentSamples
        val segment = FloatArray(segmentSamples)
        val copyLen = minOf(resampled.size, segmentSamples)
        resampled.copyInto(segment, 0, 0, copyLen)

        // Step 3: STFT → power spectrum
        val powerSpec = computePowerSpectrum(segment)

        // Step 4: Apply mel filterbank
        val melSpec = applyMelFilterbank(powerSpec)

        // Step 5: Log scale
        val logMel = FloatArray(numFrames * nMels)
        for (i in melSpec.indices) {
            logMel[i] = ln(melSpec[i] + 1e-6f)
        }

        return logMel
    }

    /**
     * Extract features suitable for the model input tensor.
     * Returns shape [1, numFrames, nMels] as a flat array.
     */
    fun extractForModel(samples: FloatArray, inputSampleRate: Int = 44100): FloatArray {
        return extract(samples, inputSampleRate)
    }

    // ── Internal methods ──

    private fun computePowerSpectrum(signal: FloatArray): Array<FloatArray> {
        val specSize = nFft / 2 + 1
        val spec = Array(numFrames) { FloatArray(specSize) }

        for (frame in 0 until numFrames) {
            val start = frame * hopLength

            // Windowed frame
            val windowed = FloatArray(nFft)
            for (i in 0 until nFft) {
                windowed[i] = if (start + i < signal.size) {
                    signal[start + i] * window[i]
                } else 0f
            }

            // Real FFT (Cooley-Tukey)
            val fftReal = FloatArray(nFft)
            val fftImag = FloatArray(nFft)
            windowed.copyInto(fftReal)
            fft(fftReal, fftImag, nFft)

            // Power spectrum: |X(f)|²
            for (k in 0 until specSize) {
                spec[frame][k] = fftReal[k] * fftReal[k] + fftImag[k] * fftImag[k]
            }
        }

        return spec
    }

    private fun applyMelFilterbank(powerSpec: Array<FloatArray>): FloatArray {
        val result = FloatArray(numFrames * nMels)
        val specSize = nFft / 2 + 1

        for (frame in 0 until numFrames) {
            for (mel in 0 until nMels) {
                var sum = 0f
                for (k in 0 until specSize) {
                    sum += melFilterbank[mel][k] * powerSpec[frame][k]
                }
                result[frame * nMels + mel] = sum
            }
        }

        return result
    }

    private fun buildMelFilterbank(): Array<FloatArray> {
        val specSize = nFft / 2 + 1

        fun hzToMel(f: Float): Float = 2595f * log10(1f + f / 700f)
        fun melToHz(m: Float): Float = 700f * (10f.pow(m / 2595f) - 1f)

        val melMin = hzToMel(fMin)
        val melMax = hzToMel(fMax)
        val melPoints = FloatArray(nMels + 2) { i ->
            melMin + i * (melMax - melMin) / (nMels + 1)
        }
        val hzPoints = melPoints.map { melToHz(it) }
        val binPoints = hzPoints.map { ((nFft + 1) * it / sampleRate).toInt() }

        return Array(nMels) { i ->
            val fb = FloatArray(specSize)
            val left = binPoints[i]
            val center = binPoints[i + 1]
            val right = binPoints[i + 2]

            for (j in left until center) {
                if (center != left && j < specSize) {
                    fb[j] = (j - left).toFloat() / (center - left)
                }
            }
            for (j in center until right) {
                if (right != center && j < specSize) {
                    fb[j] = (right - j).toFloat() / (right - center)
                }
            }
            fb
        }
    }

    /**
     * Simple linear interpolation resampler.
     * For production, use a polyphase filter (e.g., libsamplerate).
     */
    private fun resample(input: FloatArray, fromRate: Int, toRate: Int): FloatArray {
        val ratio = toRate.toFloat() / fromRate
        val outputSize = (input.size * ratio).toInt()
        val output = FloatArray(outputSize)

        for (i in 0 until outputSize) {
            val srcIdx = i / ratio
            val idx0 = srcIdx.toInt().coerceIn(0, input.size - 2)
            val frac = srcIdx - idx0
            output[i] = input[idx0] * (1f - frac) + input[idx0 + 1] * frac
        }

        return output
    }

    /**
     * In-place Cooley-Tukey radix-2 FFT.
     * n must be a power of 2.
     */
    private fun fft(real: FloatArray, imag: FloatArray, n: Int) {
        // Bit-reversal permutation
        var j = 0
        for (i in 0 until n) {
            if (i < j) {
                var tmp = real[i]; real[i] = real[j]; real[j] = tmp
                tmp = imag[i]; imag[i] = imag[j]; imag[j] = tmp
            }
            var m = n / 2
            while (m >= 1 && j >= m) {
                j -= m
                m /= 2
            }
            j += m
        }

        // FFT butterfly
        var step = 2
        while (step <= n) {
            val halfStep = step / 2
            val angle = -2.0 * PI / step

            for (k in 0 until halfStep) {
                val wr = cos(angle * k).toFloat()
                val wi = sin(angle * k).toFloat()

                var i = k
                while (i < n) {
                    val jj = i + halfStep
                    val tr = wr * real[jj] - wi * imag[jj]
                    val ti = wr * imag[jj] + wi * real[jj]

                    real[jj] = real[i] - tr
                    imag[jj] = imag[i] - ti
                    real[i] += tr
                    imag[i] += ti

                    i += step
                }
            }
            step *= 2
        }
    }

    companion object {
        /** Standard instance matching the training pipeline */
        fun default() = MelSpectrogramExtractor()

        /** Input tensor dimensions for the classifier model */
        const val MODEL_INPUT_FRAMES = 499
        const val MODEL_INPUT_MELS = 64
    }
}
