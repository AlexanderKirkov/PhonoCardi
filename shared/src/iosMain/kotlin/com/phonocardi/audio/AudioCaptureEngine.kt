package com.phonocardi.audio

/**
 * iOS audio capture using AVAudioEngine.
 *
 * Implementation note: The actual AVAudioEngine calls must be made
 * via Kotlin/Native interop with the AVFoundation framework.
 * This file provides the expect/actual structure.
 */
actual class AudioCaptureEngine actual constructor() {

    private var running = false

    actual fun start(config: AudioConfig, callback: AudioCallback) {
        // TODO: Implement via AVAudioEngine Kotlin/Native interop
        // val engine = AVAudioEngine()
        // val inputNode = engine.inputNode
        // val format = inputNode.outputFormatForBus(0u)
        // inputNode.installTapOnBus(0u, bufferSize, format) { buffer, _ ->
        //     val samples = buffer.toFloatArray()
        //     callback.onAudioData(samples, samples.size)
        // }
        // engine.startAndReturnError(null)
        running = true
    }

    actual fun stop() {
        // engine.stop()
        // engine.inputNode.removeTapOnBus(0u)
        running = false
    }

    actual fun isRunning(): Boolean = running

    actual fun availableInputs(): List<AudioInputDevice> {
        // TODO: Query AVAudioSession.sharedInstance().availableInputs
        return listOf(
            AudioInputDevice("default", "iPhone Microphone", AudioInputType.BUILT_IN_MIC)
        )
    }

    actual fun selectInput(deviceId: String) {
        // TODO: AVAudioSession.sharedInstance().setPreferredInput()
    }
}
