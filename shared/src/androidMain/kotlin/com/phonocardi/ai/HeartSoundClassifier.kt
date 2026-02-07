package com.phonocardi.ai

import android.content.Context
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Android heart sound classifier using TensorFlow Lite.
 *
 * Loads heart_sound_classifier.tflite from assets and runs inference
 * on log-mel spectrogram features extracted by MelSpectrogramExtractor.
 *
 * Model input:  [1, 499, 64] float32 — log-mel spectrogram
 * Model output: [1, 5] float32 — class probabilities
 */
actual class HeartSoundClassifier(
    private val context: Context
) {
    companion object {
        private const val MODEL_FILE = "heart_sound_classifier.tflite"
        private const val LABELS_FILE = "labels.txt"
        private const val NUM_CLASSES = 5
        private const val INPUT_FRAMES = 499
        private const val INPUT_MELS = 64
        private const val MODEL_VERSION = "1.0.0"
        private const val NUM_THREADS = 2
    }

    private var interpreter: Interpreter? = null
    private var labels: List<String> = emptyList()
    private val melExtractor = MelSpectrogramExtractor()

    actual fun isModelLoaded(): Boolean = interpreter != null

    actual fun loadModel() {
        if (interpreter != null) return
        try {
            val modelBuffer = FileUtil.loadMappedFile(context, MODEL_FILE)
            val options = Interpreter.Options().apply {
                setNumThreads(NUM_THREADS)
            }
            interpreter = Interpreter(modelBuffer, options)

            labels = try {
                FileUtil.loadLabels(context, LABELS_FILE)
            } catch (e: Exception) {
                listOf("normal", "systolic_murmur", "diastolic_murmur", "extra_sound", "noisy")
            }

            // Verify shapes
            val inShape = interpreter!!.getInputTensor(0).shape()
            val outShape = interpreter!!.getOutputTensor(0).shape()
            require(inShape.contentEquals(intArrayOf(1, INPUT_FRAMES, INPUT_MELS))) {
                "Input shape mismatch: ${inShape.contentToString()}"
            }
            require(outShape.contentEquals(intArrayOf(1, NUM_CLASSES))) {
                "Output shape mismatch: ${outShape.contentToString()}"
            }
        } catch (e: Exception) {
            interpreter = null
            throw RuntimeException("Failed to load TFLite model: ${e.message}", e)
        }
    }

    actual suspend fun classify(pcgSegment: FloatArray, sampleRate: Float): AiAnalysisResult {
        if (!isModelLoaded()) loadModel()
        val interp = interpreter
            ?: return AiAnalysisResult.EMPTY.copy(details = "Model not loaded")

        // Extract mel spectrogram features (resamples internally to 16kHz)
        val melFeatures = melExtractor.extract(pcgSegment, sampleRate.toInt())

        // Input tensor [1, 499, 64]
        val inputBuffer = ByteBuffer.allocateDirect(INPUT_FRAMES * INPUT_MELS * 4)
            .order(ByteOrder.nativeOrder())
        for (v in melFeatures) inputBuffer.putFloat(v)
        inputBuffer.rewind()

        // Output tensor [1, 5]
        val outputBuffer = ByteBuffer.allocateDirect(NUM_CLASSES * 4)
            .order(ByteOrder.nativeOrder())

        // Run inference
        try {
            interp.run(inputBuffer, outputBuffer)
        } catch (e: Exception) {
            return AiAnalysisResult.EMPTY.copy(details = "Inference error: ${e.message}")
        }

        // Parse probabilities
        outputBuffer.rewind()
        val probs = FloatArray(NUM_CLASSES) { outputBuffer.float }

        return AiAnalysisResult.fromProbabilities(probs).copy(modelVersion = MODEL_VERSION)
    }

    actual fun close() {
        interpreter?.close()
        interpreter = null
    }
}
