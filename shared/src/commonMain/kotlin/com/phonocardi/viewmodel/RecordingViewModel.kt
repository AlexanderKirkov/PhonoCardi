package com.phonocardi.viewmodel

import com.phonocardi.audio.*
import com.phonocardi.dsp.FilterChain
import com.phonocardi.dsp.FilterConfig
import com.phonocardi.pcg.*
import com.phonocardi.recording.Recording
import com.phonocardi.recording.WavFileWriter
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.datetime.Clock

/**
 * Shared ViewModel for live audio capture and PCG processing.
 * Manages the real-time pipeline: capture → filter → analyze → visualize.
 */
class RecordingViewModel(
    private val audioEngine: AudioCaptureEngine,
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
) {
    // --- State ---
    private val _state = MutableStateFlow(RecordingState())
    val state: StateFlow<RecordingState> = _state.asStateFlow()

    private val _waveformBuffer = MutableStateFlow(FloatArray(0))
    val waveformBuffer: StateFlow<FloatArray> = _waveformBuffer.asStateFlow()

    private val _envelopeBuffer = MutableStateFlow(FloatArray(0))
    val envelopeBuffer: StateFlow<FloatArray> = _envelopeBuffer.asStateFlow()

    private val _heartSounds = MutableStateFlow<List<HeartSound>>(emptyList())
    val heartSounds: StateFlow<List<HeartSound>> = _heartSounds.asStateFlow()

    // Processing components
    private var filterChain: FilterChain = FilterChain.cardiac()
    private val pcgProcessor = PcgProcessor()
    private val ringBuffer = AudioRingBuffer(44100 * 60) // 60 sec max
    private val recordingBuffer = mutableListOf<Float>()

    private var timerJob: Job? = null

    // --- Actions ---

    fun startRecording(config: FilterConfig = FilterConfig()) {
        filterChain = FilterChain(config)
        pcgProcessor.reset()
        ringBuffer.clear()
        recordingBuffer.clear()

        _state.update {
            it.copy(
                isRecording = true,
                isPaused = false,
                elapsedSeconds = 0,
                heartRate = HeartRateResult(0, 0f, false),
                signalQuality = 0f,
                error = null
            )
        }

        // Start audio capture
        audioEngine.start(AudioConfig(preferExternalInput = true)) { samples, frameCount ->
            processAudioBuffer(samples)
        }

        // Start elapsed timer
        timerJob = scope.launch {
            while (isActive) {
                delay(1000)
                if (_state.value.isRecording && !_state.value.isPaused) {
                    _state.update { it.copy(elapsedSeconds = it.elapsedSeconds + 1) }
                }
            }
        }
    }

    fun pauseRecording() {
        _state.update { it.copy(isPaused = true) }
    }

    fun resumeRecording() {
        _state.update { it.copy(isPaused = false) }
    }

    fun stopRecording(): RecordingResult? {
        timerJob?.cancel()
        audioEngine.stop()

        val elapsed = _state.value.elapsedSeconds
        _state.update {
            it.copy(isRecording = false, isPaused = false)
        }

        if (recordingBuffer.isEmpty()) return null

        val samples = recordingBuffer.toFloatArray()
        val wavWriter = WavFileWriter()
        val wavData = wavWriter.encode(samples)

        // Run final analysis on complete recording
        val analysis = pcgProcessor.analyze(samples)

        return RecordingResult(
            wavData = wavData,
            durationMs = elapsed * 1000L,
            analysis = analysis,
            samples = samples
        )
    }

    fun updateFilterConfig(config: FilterConfig) {
        filterChain = FilterChain(config)
    }

    private fun processAudioBuffer(samples: FloatArray) {
        if (_state.value.isPaused) return

        // Apply filter chain
        val filtered = filterChain.process(samples)

        // Store for recording
        recordingBuffer.addAll(filtered.toList())

        // Write to ring buffer for visualization
        ringBuffer.write(filtered)

        // PCG analysis (every ~100ms worth of samples)
        val analysisWindowSize = (44100 * 5) // 5-second analysis window
        if (ringBuffer.available >= analysisWindowSize) {
            val analysisBuffer = FloatArray(analysisWindowSize)
            ringBuffer.peek(analysisBuffer, analysisWindowSize)

            val analysis = pcgProcessor.analyze(analysisBuffer)

            _state.update {
                it.copy(
                    heartRate = analysis.heartRate,
                    signalQuality = analysis.signalQuality,
                    systolicDurationMs = analysis.avgSystolicDuration,
                    diastolicDurationMs = analysis.avgDiastolicDuration
                )
            }

            _heartSounds.value = analysis.heartSounds
            _envelopeBuffer.value = analysis.envelope
        }

        // Update waveform display (last 2 seconds)
        val displaySize = minOf(88200, ringBuffer.available) // 2 sec
        val displayBuffer = FloatArray(displaySize)
        ringBuffer.peek(displayBuffer, displaySize)
        _waveformBuffer.value = displayBuffer
    }

    fun dispose() {
        timerJob?.cancel()
        if (audioEngine.isRunning()) {
            audioEngine.stop()
        }
        scope.cancel()
    }
}

data class RecordingState(
    val isRecording: Boolean = false,
    val isPaused: Boolean = false,
    val elapsedSeconds: Int = 0,
    val heartRate: HeartRateResult = HeartRateResult(0, 0f, false),
    val signalQuality: Float = 0f,
    val systolicDurationMs: Float = 0f,
    val diastolicDurationMs: Float = 0f,
    val error: String? = null
) {
    val elapsedFormatted: String
        get() {
            val min = elapsedSeconds / 60
            val sec = elapsedSeconds % 60
            return "${min.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}"
        }
}

data class RecordingResult(
    val wavData: ByteArray,
    val durationMs: Long,
    val analysis: PcgAnalysis,
    val samples: FloatArray
)
