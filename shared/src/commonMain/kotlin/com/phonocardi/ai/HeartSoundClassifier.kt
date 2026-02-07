package com.phonocardi.ai

import kotlinx.serialization.Serializable

/**
 * Platform-specific heart sound classifier.
 * iOS: Core ML model (HeartSoundClassifier.mlpackage)
 * Android: TensorFlow Lite (heart_sound_classifier.tflite)
 *
 * Both platforms share the same preprocessing pipeline:
 *   Raw PCG audio → Resample to 16kHz → Log-mel spectrogram → Model → Probabilities
 *
 * Model architecture (1D CNN, ~835K params):
 *   Input: [1, 499, 64] log-mel spectrogram (5s segment)
 *   Conv1D(32, k=5) + BN + ReLU + MaxPool(4)
 *   Conv1D(64, k=5) + BN + ReLU + MaxPool(4)
 *   Conv1D(128, k=3) + BN + ReLU + MaxPool(4)
 *   Conv1D(128, k=3) + BN + ReLU + GAP
 *   Dense(128) + ReLU + Dropout(0.5)
 *   Dense(5) + Softmax
 *
 * Trained on: PhysioNet/CinC Challenge 2016 dataset
 * Classes:    normal, systolic_murmur, diastolic_murmur, extra_sound, noisy
 */
expect class HeartSoundClassifier {
    suspend fun classify(pcgSegment: FloatArray, sampleRate: Float): AiAnalysisResult
    fun isModelLoaded(): Boolean
    fun loadModel()
    fun close()
}

@Serializable
data class AiAnalysisResult(
    val classification: HeartSoundClass,
    val confidence: Float,
    val probabilities: Map<String, Float>,
    val details: String,
    val modelVersion: String = "1.0.0"
) {
    companion object {
        val EMPTY = AiAnalysisResult(
            HeartSoundClass.NOISY, 0f, emptyMap(), "No analysis performed", "none"
        )

        fun fromProbabilities(probs: FloatArray): AiAnalysisResult {
            require(probs.size == 5) { "Expected 5 probabilities, got ${probs.size}" }
            val classes = HeartSoundClass.entries
            val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
            val classification = classes[maxIdx]
            val confidence = probs[maxIdx]
            val probMap = classes.mapIndexed { i, cls -> cls.modelLabel to probs[i] }.toMap()
            val details = when {
                classification == HeartSoundClass.NORMAL && confidence > 0.9f ->
                    "Clear S1 and S2 sounds with regular rhythm. High confidence."
                classification == HeartSoundClass.NORMAL ->
                    "Heart sounds appear normal. Moderate confidence."
                classification == HeartSoundClass.NOISY ->
                    "Recording quality too low. Ensure proper stethoscope placement."
                else ->
                    "Potential ${classification.displayName} detected (${(confidence * 100).toInt()}%). Consider evaluation."
            }
            return AiAnalysisResult(classification, confidence, probMap, details)
        }
    }
}

@Serializable
enum class HeartSoundClass(val modelLabel: String) {
    NORMAL("normal"),
    SYSTOLIC_MURMUR("systolic_murmur"),
    DIASTOLIC_MURMUR("diastolic_murmur"),
    EXTRA_SOUND("extra_sound"),
    NOISY("noisy");

    val displayName: String
        get() = when (this) {
            NORMAL -> "Normal"
            SYSTOLIC_MURMUR -> "Systolic Murmur"
            DIASTOLIC_MURMUR -> "Diastolic Murmur"
            EXTRA_SOUND -> "Extra Sound (S3/S4)"
            NOISY -> "Noisy / Unclassifiable"
        }

    val isAbnormal: Boolean
        get() = this != NORMAL && this != NOISY

    companion object {
        fun fromLabel(label: String) = entries.firstOrNull { it.modelLabel == label } ?: NOISY
    }
}
