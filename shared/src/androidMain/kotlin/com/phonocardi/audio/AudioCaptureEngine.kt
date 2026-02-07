package com.phonocardi.audio

import android.Manifest
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.*

/**
 * Android audio capture using AudioRecord.
 * For production, replace with Oboe (C++ via JNI) for lower latency.
 * AudioRecord is used here for prototype simplicity.
 */
actual class AudioCaptureEngine actual constructor() {

    private var audioRecord: AudioRecord? = null
    private var captureJob: Job? = null
    private var running = false
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    actual fun start(config: AudioConfig, callback: AudioCallback) {
        val channelConfig = if (config.channelCount == 1)
            AudioFormat.CHANNEL_IN_MONO else AudioFormat.CHANNEL_IN_STEREO

        val encoding = AudioFormat.ENCODING_PCM_FLOAT

        val minBufferSize = AudioRecord.getMinBufferSize(
            config.sampleRate, channelConfig, encoding
        )

        val bufferSize = maxOf(config.bufferSizeFrames * 4, minBufferSize)

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.UNPROCESSED, // Raw mic input, no AGC/NS
            config.sampleRate,
            channelConfig,
            encoding,
            bufferSize
        )

        audioRecord?.startRecording()
        running = true

        captureJob = scope.launch {
            val buffer = FloatArray(config.bufferSizeFrames)

            while (isActive && running) {
                val read = audioRecord?.read(
                    buffer, 0, buffer.size, AudioRecord.READ_BLOCKING
                ) ?: 0

                if (read > 0) {
                    callback.onAudioData(buffer.copyOf(read), read)
                }
            }
        }
    }

    actual fun stop() {
        running = false
        captureJob?.cancel()
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
    }

    actual fun isRunning(): Boolean = running

    actual fun availableInputs(): List<AudioInputDevice> {
        // Android doesn't easily enumerate audio inputs like iOS
        // Return a default built-in mic entry
        return listOf(
            AudioInputDevice(
                id = "default",
                name = "Device Microphone",
                type = AudioInputType.BUILT_IN_MIC
            )
        )
    }

    actual fun selectInput(deviceId: String) {
        // On Android, input selection is handled via AudioRecord source
        // For USB/Bluetooth selection, use AudioManager.setCommunicationDevice (API 31+)
    }
}
