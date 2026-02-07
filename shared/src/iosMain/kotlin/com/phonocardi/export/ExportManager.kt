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
        // TODO: Implement via UIGraphicsPDFRenderer
        return outputPath
    }
}

actual class ShareManager {
    actual fun shareFile(filePath: String, mimeType: String, title: String) {
        // TODO: Implement via UIActivityViewController
    }

    actual fun shareMultipleFiles(filePaths: List<String>, title: String) {
        // TODO: Implement via UIActivityViewController
    }
}
