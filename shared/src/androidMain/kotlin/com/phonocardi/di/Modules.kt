package com.phonocardi.di

import com.phonocardi.ai.HeartSoundClassifier
import com.phonocardi.audio.AudioCaptureEngine
import com.phonocardi.db.DatabaseDriverFactory
import com.phonocardi.export.PdfReportGenerator
import com.phonocardi.export.ShareManager
import org.koin.core.module.Module
import org.koin.dsl.module

actual fun platformModule(): Module = module {
    single { AudioCaptureEngine() }
    single { HeartSoundClassifier(get()) }
    single { PdfReportGenerator() }
    single { ShareManager() }
    single { DatabaseDriverFactory(get()) }
}
