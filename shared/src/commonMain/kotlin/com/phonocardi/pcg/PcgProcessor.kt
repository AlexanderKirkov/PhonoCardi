package com.phonocardi.pcg

import kotlin.math.*

/**
 * Phonocardiogram (PCG) signal processor.
 * Extracts clinically meaningful data from filtered heart sound audio.
 *
 * Processing pipeline:
 * 1. Shannon energy envelope — enhances S1/S2 peaks
 * 2. Adaptive threshold peak detection — finds heart sound onsets
 * 3. S1/S2 classification — systolic/diastolic timing rule
 * 4. Heart rate calculation — from S1-S1 intervals
 */
class PcgProcessor(
    private val sampleRate: Float = 44100f
) {
    private val heartRateHistory = mutableListOf<Float>()
    private val maxHistorySize = 10

    /**
     * Shannon Energy Envelope.
     * -x² * log(x²) emphasizes medium-amplitude components typical of S1/S2
     * while suppressing both noise (low amplitude) and artifacts (high amplitude).
     */
    fun shannonEnvelope(samples: FloatArray): FloatArray {
        val normalized = normalize(samples)
        val shannon = FloatArray(normalized.size) { i ->
            val x = abs(normalized[i])
            if (x > 1e-6f) {
                (-x * x * ln(x * x)).coerceAtLeast(0f)
            } else {
                0f
            }
        }
        // Smooth with moving average (20ms window)
        val windowSize = (0.02f * sampleRate).toInt()
        return movingAverage(shannon, windowSize)
    }

    /**
     * Detect peaks in the envelope signal using adaptive thresholding.
     * Returns sample indices of detected peaks.
     */
    fun detectPeaks(
        envelope: FloatArray,
        minDistanceSec: Float = 0.2f,
        thresholdFactor: Float = 0.3f
    ): List<PeakInfo> {
        val minDistance = (minDistanceSec * sampleRate).toInt()
        val threshold = envelope.max() * thresholdFactor

        val peaks = mutableListOf<PeakInfo>()
        var lastPeakIdx = -minDistance

        for (i in 1 until envelope.size - 1) {
            if (envelope[i] > envelope[i - 1] &&
                envelope[i] >= envelope[i + 1] &&
                envelope[i] > threshold &&
                i - lastPeakIdx >= minDistance
            ) {
                peaks.add(
                    PeakInfo(
                        sampleIndex = i,
                        timeSeconds = i / sampleRate,
                        amplitude = envelope[i]
                    )
                )
                lastPeakIdx = i
            }
        }

        return peaks
    }

    /**
     * Classify detected peaks as S1 or S2 using the timing rule:
     * Systolic interval (S1→S2) < Diastolic interval (S2→S1) at normal heart rates.
     */
    fun classifyHeartSounds(peaks: List<PeakInfo>): List<HeartSound> {
        if (peaks.size < 3) return peaks.mapIndexed { i, p ->
            HeartSound(p, if (i % 2 == 0) HeartSoundType.S1 else HeartSoundType.S2)
        }

        val sounds = mutableListOf<HeartSound>()

        // Compute intervals between consecutive peaks
        val intervals = (1 until peaks.size).map { i ->
            peaks[i].timeSeconds - peaks[i - 1].timeSeconds
        }

        // Group into pairs and compare: shorter interval = systole
        for (i in intervals.indices) {
            val isShortInterval = if (i + 1 < intervals.size) {
                intervals[i] < intervals[i + 1]
            } else if (i > 0) {
                intervals[i] < intervals[i - 1]
            } else {
                true
            }

            if (i == 0) {
                // First peak: S1 if followed by short interval (systole)
                sounds.add(HeartSound(peaks[0], if (isShortInterval) HeartSoundType.S1 else HeartSoundType.S2))
            }

            // Alternating S1/S2 based on interval pattern
            val prevType = sounds.lastOrNull()?.type ?: HeartSoundType.S1
            val nextType = if (prevType == HeartSoundType.S1) HeartSoundType.S2 else HeartSoundType.S1
            sounds.add(HeartSound(peaks[i + 1], nextType))

            if (i + 1 >= peaks.size - 1) break
        }

        return sounds
    }

    /**
     * Calculate heart rate (BPM) from S1 peaks.
     * Uses a sliding window of the last 10 S1-S1 intervals for stability.
     */
    fun calculateHeartRate(heartSounds: List<HeartSound>): HeartRateResult {
        val s1Peaks = heartSounds.filter { it.type == HeartSoundType.S1 }

        if (s1Peaks.size < 2) {
            return HeartRateResult(
                bpm = heartRateHistory.lastOrNull()?.toInt() ?: 0,
                confidence = 0f,
                isStable = false
            )
        }

        // Calculate R-R intervals (S1 to S1)
        val intervals = (1 until s1Peaks.size).map { i ->
            s1Peaks[i].peak.timeSeconds - s1Peaks[i - 1].peak.timeSeconds
        }

        // Remove outliers (> 2 std deviations)
        val mean = intervals.average().toFloat()
        val std = sqrt(intervals.map { (it - mean) * (it - mean) }.average()).toFloat()
        val filtered = intervals.filter { abs(it - mean) < 2 * std }

        if (filtered.isEmpty()) {
            return HeartRateResult(0, 0f, false)
        }

        val avgInterval = filtered.average().toFloat()
        val bpm = (60f / avgInterval).toInt()

        // Update history
        heartRateHistory.add(bpm.toFloat())
        if (heartRateHistory.size > maxHistorySize) {
            heartRateHistory.removeAt(0)
        }

        // Calculate HRV (SDNN approximation)
        val hrv = if (filtered.size > 1) {
            val rrMean = filtered.average()
            sqrt(filtered.map { (it - rrMean).pow(2) }.average()).toFloat() * 1000f
        } else 0f

        return HeartRateResult(
            bpm = bpm,
            confidence = minOf(1f, filtered.size / 5f),
            isStable = std < mean * 0.15f,
            hrv = hrv
        )
    }

    /**
     * Full PCG analysis pipeline.
     * Takes filtered audio and returns complete analysis.
     */
    fun analyze(filteredSamples: FloatArray): PcgAnalysis {
        val envelope = shannonEnvelope(filteredSamples)
        val peaks = detectPeaks(envelope)
        val heartSounds = classifyHeartSounds(peaks)
        val heartRate = calculateHeartRate(heartSounds)

        val s1Sounds = heartSounds.filter { it.type == HeartSoundType.S1 }
        val s2Sounds = heartSounds.filter { it.type == HeartSoundType.S2 }

        // Calculate systolic/diastolic durations
        var systolicSum = 0f
        var systolicCount = 0
        var diastolicSum = 0f
        var diastolicCount = 0

        for (i in heartSounds.indices) {
            if (i + 1 < heartSounds.size) {
                val interval = heartSounds[i + 1].peak.timeSeconds - heartSounds[i].peak.timeSeconds
                if (heartSounds[i].type == HeartSoundType.S1) {
                    systolicSum += interval
                    systolicCount++
                } else {
                    diastolicSum += interval
                    diastolicCount++
                }
            }
        }

        // Signal quality estimation
        val signalQuality = estimateSignalQuality(filteredSamples, envelope, peaks)

        return PcgAnalysis(
            heartRate = heartRate,
            heartSounds = heartSounds,
            envelope = envelope,
            s1Count = s1Sounds.size,
            s2Count = s2Sounds.size,
            avgSystolicDuration = if (systolicCount > 0) (systolicSum / systolicCount * 1000f) else 0f,
            avgDiastolicDuration = if (diastolicCount > 0) (diastolicSum / diastolicCount * 1000f) else 0f,
            signalQuality = signalQuality
        )
    }

    private fun estimateSignalQuality(
        samples: FloatArray,
        envelope: FloatArray,
        peaks: List<PeakInfo>
    ): Float {
        if (peaks.size < 2) return 0f

        // Quality factors:
        // 1. Signal-to-noise ratio estimation
        val peakAmplitudes = peaks.map { it.amplitude }
        val avgPeak = peakAmplitudes.average().toFloat()
        val noiseLevel = envelope.filter { it < avgPeak * 0.1f }.average().toFloat()
        val snrFactor = if (noiseLevel > 0) minOf(1f, avgPeak / noiseLevel / 10f) else 1f

        // 2. Regularity of peak intervals
        val intervals = (1 until peaks.size).map { i ->
            peaks[i].timeSeconds - peaks[i - 1].timeSeconds
        }
        val intervalStd = if (intervals.size > 1) {
            val mean = intervals.average()
            sqrt(intervals.map { (it - mean).pow(2) }.average()).toFloat()
        } else 0f
        val regularityFactor = 1f - minOf(1f, intervalStd * 5f)

        // 3. Clipping detection
        val maxSample = samples.maxOrNull() ?: 0f
        val clippingFactor = if (maxSample > 0.95f) 0.5f else 1f

        return (snrFactor * 0.4f + regularityFactor * 0.4f + clippingFactor * 0.2f)
            .coerceIn(0f, 1f)
    }

    // --- Utility functions ---

    private fun normalize(samples: FloatArray): FloatArray {
        val maxVal = samples.maxOfOrNull { abs(it) } ?: 1f
        return if (maxVal > 0) {
            FloatArray(samples.size) { samples[it] / maxVal }
        } else samples
    }

    private fun movingAverage(data: FloatArray, windowSize: Int): FloatArray {
        val result = FloatArray(data.size)
        var sum = 0f
        val halfWindow = windowSize / 2

        for (i in data.indices) {
            val start = maxOf(0, i - halfWindow)
            val end = minOf(data.size - 1, i + halfWindow)

            if (i == 0) {
                for (j in start..end) sum += data[j]
            } else {
                val oldStart = maxOf(0, i - 1 - halfWindow)
                val newEnd = minOf(data.size - 1, i + halfWindow)
                if (oldStart > 0 && i - 1 - halfWindow >= 0) sum -= data[i - 1 - halfWindow]
                if (newEnd < data.size) sum += data[minOf(data.size - 1, i + halfWindow)]
            }

            result[i] = sum / (end - start + 1)
        }

        return result
    }

    fun reset() {
        heartRateHistory.clear()
    }
}

// --- Data classes ---

data class PeakInfo(
    val sampleIndex: Int,
    val timeSeconds: Float,
    val amplitude: Float
)

enum class HeartSoundType { S1, S2, S3, S4, MURMUR }

data class HeartSound(
    val peak: PeakInfo,
    val type: HeartSoundType
)

data class HeartRateResult(
    val bpm: Int,
    val confidence: Float,
    val isStable: Boolean,
    val hrv: Float = 0f
)

data class PcgAnalysis(
    val heartRate: HeartRateResult,
    val heartSounds: List<HeartSound>,
    val envelope: FloatArray,
    val s1Count: Int,
    val s2Count: Int,
    val avgSystolicDuration: Float,
    val avgDiastolicDuration: Float,
    val signalQuality: Float
)
