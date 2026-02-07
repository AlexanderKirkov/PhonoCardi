package com.phonocardi.android

import android.app.Application
import com.phonocardi.di.platformModule
import com.phonocardi.di.sharedModule
import org.koin.android.ext.koin.androidContext
import org.koin.core.context.startKoin

class PhonoCardiApp : Application() {
    override fun onCreate() {
        super.onCreate()

        startKoin {
            androidContext(this@PhonoCardiApp)
            modules(sharedModule, platformModule())
        }
    }
}
