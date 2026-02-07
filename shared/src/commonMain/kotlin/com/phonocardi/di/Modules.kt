package com.phonocardi.di

import com.phonocardi.dsp.FilterChain
import com.phonocardi.dsp.FilterConfig
import com.phonocardi.pcg.PcgProcessor
import com.phonocardi.recording.WavFileWriter
import org.koin.core.module.Module
import org.koin.dsl.module

/**
 * Shared Koin module — platform-agnostic dependencies.
 */
val sharedModule = module {
    single { PcgProcessor() }
    single { WavFileWriter() }
    factory { FilterChain.cardiac() }
    factory { FilterConfig() }
}

/**
 * Platform modules must provide:
 * - AudioCaptureEngine
 * - HeartSoundClassifier
 * - PdfReportGenerator
 * - ShareManager
 * - DatabaseDriverFactory
 * - RecordingRepository
 */
expect fun platformModule(): Module
