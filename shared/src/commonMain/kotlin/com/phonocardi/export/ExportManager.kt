package com.phonocardi.export

import com.phonocardi.recording.Recording
import com.phonocardi.pcg.PcgAnalysis
import com.phonocardi.ai.AiAnalysisResult

/**
 * Platform-specific PDF report generator.
 * iOS: UIGraphicsPDFRenderer
 * Android: android.graphics.pdf.PdfDocument
 */
expect class PdfReportGenerator {
    suspend fun generateReport(
        recording: Recording,
        analysis: PcgAnalysis?,
        aiResult: AiAnalysisResult?,
        waveformSamples: FloatArray,
        outputPath: String
    ): String
}

/**
 * Platform-specific sharing.
 * iOS: UIActivityViewController
 * Android: ShareSheet + FileProvider
 */
expect class ShareManager {
    fun shareFile(filePath: String, mimeType: String, title: String)
    fun shareMultipleFiles(filePaths: List<String>, title: String)
}

enum class ExportFormat {
    WAV,
    PDF,
    CSV;

    val mimeType: String
        get() = when (this) {
            WAV -> "audio/wav"
            PDF -> "application/pdf"
            CSV -> "text/csv"
        }

    val extension: String
        get() = when (this) {
            WAV -> ".wav"
            PDF -> ".pdf"
            CSV -> ".csv"
        }
}
