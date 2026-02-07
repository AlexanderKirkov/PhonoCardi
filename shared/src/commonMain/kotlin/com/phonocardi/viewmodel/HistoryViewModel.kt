package com.phonocardi.viewmodel

import com.phonocardi.recording.Recording
import com.phonocardi.recording.RecordingRepository
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

/**
 * ViewModel for recording history list and search.
 */
class HistoryViewModel(
    private val repository: RecordingRepository,
    private val scope: CoroutineScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
) {
    private val _state = MutableStateFlow(HistoryState())
    val state: StateFlow<HistoryState> = _state.asStateFlow()

    init {
        // Observe all recordings
        scope.launch {
            repository.getAllRecordings().collect { recordings ->
                _state.update {
                    it.copy(
                        recordings = recordings,
                        filteredRecordings = applyFilter(recordings, it.activeFilter),
                        isLoading = false
                    )
                }
            }
        }
    }

    fun setFilter(filter: RecordingFilter) {
        _state.update {
            it.copy(
                activeFilter = filter,
                filteredRecordings = applyFilter(it.recordings, filter)
            )
        }
    }

    fun search(query: String) {
        _state.update {
            it.copy(
                searchQuery = query,
                filteredRecordings = if (query.isBlank()) {
                    applyFilter(it.recordings, it.activeFilter)
                } else {
                    it.recordings.filter { r ->
                        r.filename.contains(query, ignoreCase = true) ||
                        r.notes?.contains(query, ignoreCase = true) == true ||
                        r.tags.any { t -> t.contains(query, ignoreCase = true) }
                    }
                }
            )
        }
    }

    fun deleteRecording(id: String) {
        scope.launch {
            repository.deleteRecording(id)
        }
    }

    private fun applyFilter(recordings: List<Recording>, filter: RecordingFilter): List<Recording> {
        return when (filter) {
            RecordingFilter.ALL -> recordings
            RecordingFilter.NORMAL -> recordings.filter {
                it.aiResult?.classification?.isAbnormal == false
            }
            RecordingFilter.ABNORMAL -> recordings.filter {
                it.aiResult?.classification?.isAbnormal == true
            }
            RecordingFilter.THIS_WEEK -> {
                val weekAgo = System.currentTimeMillis() - 7 * 24 * 3600 * 1000
                recordings.filter { it.createdAt >= weekAgo }
            }
            RecordingFilter.THIS_MONTH -> {
                val monthAgo = System.currentTimeMillis() - 30L * 24 * 3600 * 1000
                recordings.filter { it.createdAt >= monthAgo }
            }
        }
    }

    fun dispose() {
        scope.cancel()
    }
}

data class HistoryState(
    val recordings: List<Recording> = emptyList(),
    val filteredRecordings: List<Recording> = emptyList(),
    val activeFilter: RecordingFilter = RecordingFilter.ALL,
    val searchQuery: String = "",
    val isLoading: Boolean = true
)

enum class RecordingFilter {
    ALL, NORMAL, ABNORMAL, THIS_WEEK, THIS_MONTH
}
