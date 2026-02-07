package com.phonocardi.recording

import com.phonocardi.ai.AiAnalysisResult
import kotlinx.coroutines.flow.Flow
import kotlinx.serialization.Serializable

@Serializable
data class Recording(
    val id: String,
    val filename: String,
    val filepath: String,
    val createdAt: Long,
    val durationMs: Long,
    val sampleRate: Int = 44100,
    val heartRate: Int? = null,
    val patientId: String? = null,
    val notes: String? = null,
    val aiResult: AiAnalysisResult? = null,
    val tags: List<String> = emptyList(),
    val signalQuality: Float? = null
) {
    val durationFormatted: String
        get() {
            val totalSeconds = durationMs / 1000
            val minutes = totalSeconds / 60
            val seconds = totalSeconds % 60
            return "${minutes}:${seconds.toString().padStart(2, '0')}"
        }
}

interface RecordingRepository {
    fun getAllRecordings(): Flow<List<Recording>>
    fun getRecordingById(id: String): Flow<Recording?>
    suspend fun insertRecording(recording: Recording)
    suspend fun updateRecording(recording: Recording)
    suspend fun deleteRecording(id: String)
    suspend fun searchByDateRange(startMs: Long, endMs: Long): List<Recording>
    suspend fun searchByTag(tag: String): List<Recording>
}
