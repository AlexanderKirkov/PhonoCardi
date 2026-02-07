package com.phonocardi.android.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

// Color-coded waveform colors
object WaveformColors {
    val S1 = Color(0xFFFF6B6B)       // Red for S1 (Lub)
    val S2 = Color(0xFF4ECDC4)       // Cyan for S2 (Dub)
    val Murmur = Color(0xFFFFD93D)   // Yellow for murmurs
    val Baseline = Color(0xFF00D4AA) // Green for baseline
}

private val DarkColorScheme = darkColorScheme(
    primary = Color(0xFF00D4AA),
    onPrimary = Color(0xFF003329),
    primaryContainer = Color(0xFF004D3E),
    onPrimaryContainer = Color(0xFF6EFFD8),
    secondary = Color(0xFF4ECDC4),
    onSecondary = Color(0xFF00332F),
    background = Color(0xFF0A0E17),
    onBackground = Color(0xFFF0F4F8),
    surface = Color(0xFF111827),
    onSurface = Color(0xFFF0F4F8),
    surfaceVariant = Color(0xFF1E293B),
    onSurfaceVariant = Color(0xFF8899AA),
    error = Color(0xFFFF4757),
    onError = Color(0xFF690005),
    outline = Color(0xFF1E3A5F),
    outlineVariant = Color(0xFF152238),
)

private val LightColorScheme = lightColorScheme(
    primary = Color(0xFF0891B2),
    onPrimary = Color(0xFFFFFFFF),
    primaryContainer = Color(0xFFCDE7EF),
    onPrimaryContainer = Color(0xFF001F26),
    secondary = Color(0xFF0891B2),
    onSecondary = Color(0xFFFFFFFF),
    background = Color(0xFFF4F7FB),
    onBackground = Color(0xFF1A2332),
    surface = Color(0xFFFFFFFF),
    onSurface = Color(0xFF1A2332),
    surfaceVariant = Color(0xFFEEF2F7),
    onSurfaceVariant = Color(0xFF5A6B7D),
    error = Color(0xFFDC2626),
    onError = Color(0xFFFFFFFF),
    outline = Color(0xFFD0DAE8),
    outlineVariant = Color(0xFFE2EAF3),
)

@Composable
fun PhonoCardiTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme

    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography(),
        content = content
    )
}
