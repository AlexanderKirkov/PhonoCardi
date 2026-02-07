package com.phonocardi.export

import com.phonocardi.recording.Recording
import com.phonocardi.pcg.PcgAnalysis
import com.phonocardi.ai.AiAnalysisResult

actual class PdfReportGenerator {
    actual suspend fun generateReport(
        recording: Recording,
        analysis: PcgAnalysis?,
        aiResult: AiAnalysisResult?,
        waveformSamples: FloatArray,
        outputPath: String
    ): String {
        // TODO: Implement using android.graphics.pdf.PdfDocument
        // val document = PdfDocument()
        // val pageInfo = PdfDocument.PageInfo.Builder(595, 842, 1).create()
        // val page = document.startPage(pageInfo)
        // ... draw waveform, text, etc on page.canvas
        // document.writeTo(FileOutputStream(outputPath))
        return outputPath
    }
}

actual class ShareManager {
    actual fun shareFile(filePath: String, mimeType: String, title: String) {
        // TODO: Implement with Intent.ACTION_SEND + FileProvider
    }

    actual fun shareMultipleFiles(filePaths: List<String>, title: String) {
        // TODO: Implement with Intent.ACTION_SEND_MULTIPLE
    }
}
