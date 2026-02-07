package com.phonocardi.ai

import kotlinx.cinterop.*
import platform.CoreML.*
import platform.Foundation.*

/**
 * iOS heart sound classifier using Core ML.
 *
 * Loads HeartSoundClassifier.mlmodelc from the app bundle and runs inference
 * on log-mel spectrogram features extracted by MelSpectrogramExtractor.
 *
 * Model input:  [1, 499, 64] float32 — log-mel spectrogram
 * Model output: [1, 5] float32 — class probabilities
 */
actual class HeartSoundClassifier {
    companion object {
        private const val MODEL_NAME = "HeartSoundClassifier"
        private const val NUM_CLASSES = 5
        private const val INPUT_FRAMES = 499
        private const val INPUT_MELS = 64
        private const val MODEL_VERSION = "1.0.0"
    }

    private var model: MLModel? = null
    private val melExtractor = MelSpectrogramExtractor()

    actual fun isModelLoaded(): Boolean = model != null

    actual fun loadModel() {
        if (model != null) return

        // Load compiled Core ML model from app bundle
        val modelUrl = NSBundle.mainBundle.URLForResource(MODEL_NAME, withExtension = "mlmodelc")
            ?: throw RuntimeException("Model $MODEL_NAME.mlmodelc not found in bundle")

        val error: ObjCObjectVar<NSError?> = alloc()
        val compiledModel = MLModel.modelWithContentsOfURL(modelUrl, error = error.ptr)
        if (error.value != null) {
            throw RuntimeException("Failed to load Core ML model: ${error.value?.localizedDescription}")
        }
        model = compiledModel
    }

    actual suspend fun classify(pcgSegment: FloatArray, sampleRate: Float): AiAnalysisResult {
        if (!isModelLoaded()) loadModel()
        val mlModel = model
            ?: return AiAnalysisResult.EMPTY.copy(details = "Core ML model not loaded")

        // Extract mel spectrogram features (resamples internally to 16kHz)
        val melFeatures = melExtractor.extract(pcgSegment, sampleRate.toInt())

        // Create MLMultiArray input [1, 499, 64]
        val shape = NSArray.arrayWithObjects(
            NSNumber(int = 1),
            NSNumber(int = INPUT_FRAMES),
            NSNumber(int = INPUT_MELS)
        )

        val inputArrayError: ObjCObjectVar<NSError?> = alloc()
        val inputArray = MLMultiArray(
            shape = shape as List<NSNumber>,
            dataType = MLMultiArrayDataType.MLMultiArrayDataTypeFloat32,
            error = inputArrayError.ptr
        ) ?: return AiAnalysisResult.EMPTY.copy(details = "Failed to create input array")

        // Copy mel features to MLMultiArray
        for (frame in 0 until INPUT_FRAMES) {
            for (mel in 0 until INPUT_MELS) {
                val idx = frame * INPUT_MELS + mel
                val key = NSArray.arrayWithObjects(
                    NSNumber(int = 0),
                    NSNumber(int = frame),
                    NSNumber(int = mel)
                )
                inputArray.setObject(NSNumber(float = melFeatures[idx]), forKeyedSubscript = key)
            }
        }

        // Create feature provider
        val featureProvider = try {
            MLDictionaryFeatureProvider(
                dictionary = mapOf("mel_input" to MLFeatureValue.featureValueWithMultiArray(inputArray)),
                error = null
            )
        } catch (e: Exception) {
            return AiAnalysisResult.EMPTY.copy(details = "Feature provider error: ${e.message}")
        }

        // Run prediction
        val predictionError: ObjCObjectVar<NSError?> = alloc()
        val prediction = mlModel.predictionFromFeatures(featureProvider!!, error = predictionError.ptr)
            ?: return AiAnalysisResult.EMPTY.copy(
                details = "Prediction failed: ${predictionError.value?.localizedDescription}"
            )

        // Extract output probabilities
        val outputFeature = prediction.featureValueForName("classification")
        val outputArray = outputFeature?.multiArrayValue
            ?: return AiAnalysisResult.EMPTY.copy(details = "Missing output tensor")

        val probs = FloatArray(NUM_CLASSES)
        for (i in 0 until NUM_CLASSES) {
            probs[i] = outputArray.objectAtIndexedSubscript(i.toLong()).floatValue
        }

        return AiAnalysisResult.fromProbabilities(probs).copy(modelVersion = MODEL_VERSION)
    }

    actual fun close() {
        model = null
    }
}
