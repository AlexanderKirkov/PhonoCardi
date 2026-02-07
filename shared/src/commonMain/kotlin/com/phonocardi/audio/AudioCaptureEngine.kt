package com.phonocardi.audio

/**
 * Platform-specific audio capture engine.
 * iOS: AVAudioEngine with installTap
 * Android: Oboe (C++ via JNI) for low-latency capture
 */
expect class AudioCaptureEngine() {
    fun start(config: AudioConfig, callback: AudioCallback)
    fun stop()
    fun isRunning(): Boolean
    fun availableInputs(): List<AudioInputDevice>
    fun selectInput(deviceId: String)
}

data class AudioConfig(
    val sampleRate: Int = 44100,
    val channelCount: Int = 1,
    val bitsPerSample: Int = 16,
    val bufferSizeFrames: Int = 4096,
    val preferExternalInput: Boolean = true
)

fun interface AudioCallback {
    fun onAudioData(samples: FloatArray, frameCount: Int)
}

data class AudioInputDevice(
    val id: String,
    val name: String,
    val type: AudioInputType
)

enum class AudioInputType {
    BUILT_IN_MIC,
    HEADSET_MIC,
    USB_AUDIO,
    BLUETOOTH,
    UNKNOWN
}
