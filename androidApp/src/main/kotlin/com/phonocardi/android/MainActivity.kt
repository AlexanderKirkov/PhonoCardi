package com.phonocardi.android

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.phonocardi.android.ui.screens.*
import com.phonocardi.android.ui.theme.PhonoCardiTheme

class MainActivity : ComponentActivity() {

    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            // Permission granted, user can start recording
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Request microphone permission
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermission.launch(Manifest.permission.RECORD_AUDIO)
        }

        setContent {
            PhonoCardiTheme {
                PhonoCardiNavigation()
            }
        }
    }
}

@Composable
fun PhonoCardiNavigation() {
    val navController = rememberNavController()

    Scaffold(
        bottomBar = {
            BottomNavBar(navController = navController)
        }
    ) { paddingValues ->
        NavHost(
            navController = navController,
            startDestination = "home",
            modifier = Modifier.padding(paddingValues)
        ) {
            composable("home") { HomeScreen(navController) }
            composable("recording") { RecordingScreen(navController) }
            composable("history") { HistoryScreen(navController) }
            composable("analysis") { AnalysisScreen(navController) }
            composable("settings") { SettingsScreen(navController) }
            composable("playback/{recordingId}") { backStackEntry ->
                val id = backStackEntry.arguments?.getString("recordingId") ?: ""
                PlaybackScreen(navController, recordingId = id)
            }
        }
    }
}
