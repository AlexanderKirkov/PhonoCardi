package com.phonocardi.android.ui.screens

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material.icons.outlined.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.navigation.NavController
import androidx.navigation.compose.currentBackStackEntryAsState
import com.phonocardi.android.ui.components.WaveformCanvas

// ─── Bottom Navigation ───

data class NavItem(val route: String, val label: String, val icon: ImageVector, val selectedIcon: ImageVector)

val navItems = listOf(
    NavItem("home", "Home", Icons.Outlined.Home, Icons.Filled.Home),
    NavItem("recording", "Record", Icons.Outlined.Mic, Icons.Filled.Mic),
    NavItem("history", "History", Icons.Outlined.List, Icons.Filled.List),
    NavItem("analysis", "Analysis", Icons.Outlined.Analytics, Icons.Filled.Analytics),
    NavItem("settings", "Settings", Icons.Outlined.Settings, Icons.Filled.Settings),
)

@Composable
fun BottomNavBar(navController: NavController) {
    val backStackEntry by navController.currentBackStackEntryAsState()
    val currentRoute = backStackEntry?.destination?.route

    NavigationBar {
        navItems.forEach { item ->
            val selected = currentRoute == item.route
            NavigationBarItem(
                selected = selected,
                onClick = {
                    if (currentRoute != item.route) {
                        navController.navigate(item.route) {
                            popUpTo("home") { saveState = true }
                            launchSingleTop = true
                            restoreState = true
                        }
                    }
                },
                icon = {
                    Icon(
                        if (selected) item.selectedIcon else item.icon,
                        contentDescription = item.label
                    )
                },
                label = { Text(item.label, fontSize = 10.sp) }
            )
        }
    }
}

// ─── Home / Dashboard Screen ───

@Composable
fun HomeScreen(navController: NavController) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(20.dp)
    ) {
        // Header
        Text("PhonoCardi", fontSize = 24.sp, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(4.dp))
        Text("Welcome back", color = MaterialTheme.colorScheme.onSurfaceVariant, fontSize = 14.sp)
        Spacer(Modifier.height(20.dp))

        // Quick Stats
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            StatCard("Avg HR", "72", "BPM", Modifier.weight(1f))
            StatCard("Recordings", "24", "total", Modifier.weight(1f))
            StatCard("Normal", "92", "%", Modifier.weight(1f))
        }
        Spacer(Modifier.height(20.dp))

        // Waveform Preview
        Card(
            modifier = Modifier.fillMaxWidth(),
            shape = MaterialTheme.shapes.large
        ) {
            Column(Modifier.padding(12.dp)) {
                Text("Last Recording", fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
                Spacer(Modifier.height(8.dp))
                WaveformCanvas(
                    waveformData = floatArrayOf(), // Placeholder — filled at runtime
                    modifier = Modifier.fillMaxWidth().height(140.dp)
                )
            }
        }
        Spacer(Modifier.height(16.dp))

        // Record Button
        Button(
            onClick = { navController.navigate("recording") },
            modifier = Modifier.fillMaxWidth().height(56.dp),
            shape = MaterialTheme.shapes.large
        ) {
            Icon(Icons.Default.Mic, contentDescription = null)
            Spacer(Modifier.width(8.dp))
            Text("Start New Recording", fontWeight = FontWeight.Bold, fontSize = 16.sp)
        }
        Spacer(Modifier.height(24.dp))

        // Recent recordings header
        Row(
            Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Text("Recent Recordings", fontWeight = FontWeight.Bold, fontSize = 16.sp)
            TextButton(onClick = { navController.navigate("history") }) {
                Text("View All")
            }
        }

        // Sample recording items
        listOf(
            Triple("Today, 14:30", "0:45 · 72 BPM", "Normal"),
            Triple("Today, 09:15", "1:20 · 68 BPM", "Normal"),
            Triple("Yesterday, 18:45", "0:55 · 88 BPM", "Murmur"),
        ).forEach { (date, detail, status) ->
            RecordingListItem(
                date = date,
                detail = detail,
                status = status,
                onClick = { navController.navigate("playback/sample") }
            )
            Spacer(Modifier.height(8.dp))
        }
    }
}

@Composable
fun StatCard(label: String, value: String, unit: String, modifier: Modifier = Modifier) {
    Card(modifier = modifier, shape = MaterialTheme.shapes.medium) {
        Column(Modifier.padding(14.dp)) {
            Text(label, fontSize = 11.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.height(4.dp))
            Row(verticalAlignment = Alignment.Bottom) {
                Text(value, fontSize = 26.sp, fontWeight = FontWeight.Bold)
                Spacer(Modifier.width(4.dp))
                Text(unit, fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        }
    }
}

@Composable
fun RecordingListItem(date: String, detail: String, status: String, onClick: () -> Unit) {
    Card(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth(),
        shape = MaterialTheme.shapes.medium
    ) {
        Row(
            Modifier.padding(14.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                Icons.Default.GraphicEq,
                contentDescription = null,
                tint = if (status == "Normal") MaterialTheme.colorScheme.primary
                       else MaterialTheme.colorScheme.error,
                modifier = Modifier.size(28.dp)
            )
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text(date, fontWeight = FontWeight.SemiBold, fontSize = 14.sp)
                Text(detail, fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
            AssistChip(
                onClick = {},
                label = { Text(status, fontSize = 11.sp) },
                colors = AssistChipDefaults.assistChipColors(
                    containerColor = if (status == "Normal")
                        MaterialTheme.colorScheme.primaryContainer
                    else MaterialTheme.colorScheme.errorContainer
                )
            )
        }
    }
}

// ─── Recording Screen ───

@Composable
fun RecordingScreen(navController: NavController) {
    var isRecording by remember { mutableStateOf(true) }
    var elapsed by remember { mutableIntStateOf(0) }

    LaunchedEffect(isRecording) {
        if (isRecording) {
            while (true) {
                kotlinx.coroutines.delay(1000)
                elapsed++
            }
        }
    }

    Column(
        modifier = Modifier.fillMaxSize().padding(20.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        Spacer(Modifier.height(16.dp))

        // Recording indicator
        Text(
            if (isRecording) "● RECORDING" else "PAUSED",
            color = MaterialTheme.colorScheme.error,
            fontWeight = FontWeight.Bold,
            fontSize = 13.sp
        )
        Spacer(Modifier.height(4.dp))
        Text(
            "%02d:%02d".format(elapsed / 60, elapsed % 60),
            fontSize = 42.sp,
            fontWeight = FontWeight.Bold
        )
        Spacer(Modifier.height(12.dp))

        // Heart rate
        Row(verticalAlignment = Alignment.Bottom) {
            Icon(Icons.Default.Favorite, tint = MaterialTheme.colorScheme.error, contentDescription = null)
            Spacer(Modifier.width(8.dp))
            Text("72", fontSize = 48.sp, fontWeight = FontWeight.Bold)
            Spacer(Modifier.width(4.dp))
            Text("BPM", fontSize = 16.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Spacer(Modifier.height(20.dp))

        // Waveform
        Card(Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.large) {
            WaveformCanvas(
                waveformData = floatArrayOf(),
                heartRate = 72,
                modifier = Modifier.fillMaxWidth().height(200.dp).padding(4.dp)
            )
        }
        Spacer(Modifier.height(16.dp))

        // Stats
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            StatCard("S1-S2", "310", "ms", Modifier.weight(1f))
            StatCard("S2-S1", "520", "ms", Modifier.weight(1f))
            StatCard("Quality", "96", "%", Modifier.weight(1f))
        }
        Spacer(Modifier.weight(1f))

        // Controls
        Row(
            horizontalArrangement = Arrangement.spacedBy(20.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            FloatingActionButton(
                onClick = { isRecording = !isRecording },
                containerColor = if (isRecording)
                    MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(72.dp)
            ) {
                Icon(
                    if (isRecording) Icons.Default.Pause else Icons.Default.PlayArrow,
                    contentDescription = null,
                    modifier = Modifier.size(32.dp)
                )
            }
            OutlinedIconButton(
                onClick = { navController.popBackStack() },
                modifier = Modifier.size(56.dp)
            ) {
                Icon(Icons.Default.Stop, contentDescription = "Stop")
            }
        }
        Spacer(Modifier.height(20.dp))
    }
}

// ─── Playback Screen ───

@Composable
fun PlaybackScreen(navController: NavController, recordingId: String) {
    var playing by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp)
    ) {
        Text("Recording Review", fontSize = 20.sp, fontWeight = FontWeight.Bold)
        Text("Today, 14:30 · 0:45", fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.height(16.dp))

        // Waveform
        Card(Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.large) {
            WaveformCanvas(
                waveformData = floatArrayOf(),
                modifier = Modifier.fillMaxWidth().height(220.dp).padding(4.dp)
            )
        }
        Spacer(Modifier.height(12.dp))

        // Progress bar
        LinearProgressIndicator(
            progress = { 0.35f },
            modifier = Modifier.fillMaxWidth(),
        )
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text("0:16", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text("0:45", fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Spacer(Modifier.height(12.dp))

        // Play button
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
            FloatingActionButton(onClick = { playing = !playing }) {
                Icon(
                    if (playing) Icons.Default.Pause else Icons.Default.PlayArrow,
                    contentDescription = null
                )
            }
        }
        Spacer(Modifier.height(20.dp))

        // Stats
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            StatCard("Heart Rate", "72", "BPM", Modifier.weight(1f))
            StatCard("HRV", "48", "ms", Modifier.weight(1f))
        }
        Spacer(Modifier.height(16.dp))

        // AI Result card
        Card(Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.large) {
            Column(Modifier.padding(16.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(Icons.Default.Psychology, tint = MaterialTheme.colorScheme.primary, contentDescription = null)
                    Spacer(Modifier.width(10.dp))
                    Column {
                        Text("AI Analysis", fontWeight = FontWeight.SemiBold)
                        Text(
                            "Normal heart sounds detected",
                            fontSize = 13.sp,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                }
                Spacer(Modifier.height(10.dp))
                Text(
                    "Clear S1 and S2 sounds with regular rhythm. No murmurs detected. Signal quality: 96%.",
                    fontSize = 13.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Spacer(Modifier.height(10.dp))
                AssistChip(
                    onClick = {},
                    label = { Text("Confidence: 94%") }
                )
            }
        }
        Spacer(Modifier.height(16.dp))

        // Export / Share buttons
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            OutlinedButton(
                onClick = {},
                modifier = Modifier.weight(1f)
            ) {
                Icon(Icons.Default.Download, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(6.dp))
                Text("Export")
            }
            Button(
                onClick = {},
                modifier = Modifier.weight(1f)
            ) {
                Icon(Icons.Default.Share, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(Modifier.width(6.dp))
                Text("Share")
            }
        }
    }
}

// ─── History Screen ───

@Composable
fun HistoryScreen(navController: NavController) {
    Column(
        modifier = Modifier.fillMaxSize().padding(20.dp)
    ) {
        Text("Recording History", fontSize = 24.sp, fontWeight = FontWeight.Bold)
        Text("7 recordings", fontSize = 13.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.height(16.dp))

        // Filter chips
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf("All", "Normal", "Murmur").forEachIndexed { i, label ->
                FilterChip(
                    selected = i == 0,
                    onClick = {},
                    label = { Text(label) }
                )
            }
        }
        Spacer(Modifier.height(16.dp))

        // Recordings list
        Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
            listOf(
                Triple("Today, 14:30", "0:45 · 72 BPM", "Normal"),
                Triple("Today, 09:15", "1:20 · 68 BPM", "Normal"),
                Triple("Yesterday, 18:45", "0:55 · 88 BPM", "Murmur"),
                Triple("Jan 29, 11:00", "1:05 · 74 BPM", "Normal"),
                Triple("Jan 28, 16:20", "0:38 · 71 BPM", "Normal"),
                Triple("Jan 27, 09:45", "0:52 · 90 BPM", "Murmur"),
                Triple("Jan 26, 14:10", "1:15 · 66 BPM", "Normal"),
            ).forEach { (date, detail, status) ->
                RecordingListItem(date, detail, status) {
                    navController.navigate("playback/sample")
                }
            }
        }
    }
}

// ─── Analysis Screen ───

@Composable
fun AnalysisScreen(navController: NavController) {
    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp)
    ) {
        Text("AI Analysis", fontSize = 24.sp, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(20.dp))

        // Status card
        Card(Modifier.fillMaxWidth(), shape = MaterialTheme.shapes.large) {
            Column(Modifier.padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally) {
                Icon(
                    Icons.Default.CheckCircle,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(64.dp)
                )
                Spacer(Modifier.height(12.dp))
                Text("Normal Heart Sounds", fontSize = 22.sp, fontWeight = FontWeight.Bold)
                Spacer(Modifier.height(4.dp))
                Text("Last analyzed: Today, 14:30", color = MaterialTheme.colorScheme.onSurfaceVariant)
                Spacer(Modifier.height(14.dp))
                AssistChip(onClick = {}, label = { Text("94% Confidence", fontWeight = FontWeight.Bold) })
            }
        }
        Spacer(Modifier.height(20.dp))

        Text("Detection Results", fontWeight = FontWeight.Bold, fontSize = 16.sp)
        Spacer(Modifier.height(12.dp))

        listOf(
            Triple("Normal", 0.94f, MaterialTheme.colorScheme.primary),
            Triple("Systolic Murmur", 0.03f, MaterialTheme.colorScheme.tertiary),
            Triple("Diastolic Murmur", 0.01f, MaterialTheme.colorScheme.error),
        ).forEach { (label, value, color) ->
            Row(
                Modifier.fillMaxWidth().padding(vertical = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(label, Modifier.weight(1f), fontSize = 14.sp)
                LinearProgressIndicator(
                    progress = { value },
                    modifier = Modifier.width(100.dp),
                    color = color
                )
                Spacer(Modifier.width(8.dp))
                Text("${(value * 100).toInt()}%", fontWeight = FontWeight.Bold, fontSize = 13.sp)
            }
        }

        Spacer(Modifier.height(24.dp))

        // Disclaimer
        Card(
            Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.errorContainer.copy(alpha = 0.3f)
            )
        ) {
            Row(Modifier.padding(14.dp)) {
                Icon(Icons.Default.Warning, tint = MaterialTheme.colorScheme.error, contentDescription = null)
                Spacer(Modifier.width(10.dp))
                Text(
                    "AI analysis is for informational purposes only and should not replace professional medical diagnosis.",
                    fontSize = 12.sp,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

// ─── Settings Screen ───

@Composable
fun SettingsScreen(navController: NavController) {
    var darkMode by remember { mutableStateOf(true) }
    var notchFilter by remember { mutableStateOf(true) }
    var autoGain by remember { mutableStateOf(true) }
    var showEnvelope by remember { mutableStateOf(true) }
    var autoAnalyze by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp)
    ) {
        Text("Settings", fontSize = 24.sp, fontWeight = FontWeight.Bold)
        Spacer(Modifier.height(20.dp))

        Text("APPEARANCE", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onSurfaceVariant)
        SettingToggle("Dark Mode", "Follow system theme", darkMode) { darkMode = it }
        HorizontalDivider(Modifier.padding(vertical = 4.dp))

        Spacer(Modifier.height(12.dp))
        Text("AUDIO & FILTERS", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onSurfaceVariant)
        SettingToggle("50 Hz Notch Filter", "Remove power line interference", notchFilter) { notchFilter = it }
        SettingToggle("Auto Gain Control", "Normalize signal amplitude", autoGain) { autoGain = it }
        HorizontalDivider(Modifier.padding(vertical = 4.dp))

        Spacer(Modifier.height(12.dp))
        Text("VISUALIZATION", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onSurfaceVariant)
        SettingToggle("Show Envelope", "Display signal envelope overlay", showEnvelope) { showEnvelope = it }
        HorizontalDivider(Modifier.padding(vertical = 4.dp))

        Spacer(Modifier.height(12.dp))
        Text("AI ANALYSIS", fontSize = 12.sp, fontWeight = FontWeight.Bold, color = MaterialTheme.colorScheme.onSurfaceVariant)
        SettingToggle("Auto-Analyze", "Run AI after each recording", autoAnalyze) { autoAnalyze = it }

        Spacer(Modifier.height(24.dp))
        Text(
            "PhonoCardi v1.0 · KMP · iOS 26 / Android 14",
            fontSize = 12.sp,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.align(Alignment.CenterHorizontally)
        )
    }
}

@Composable
fun SettingToggle(title: String, subtitle: String, checked: Boolean, onCheckedChange: (Boolean) -> Unit) {
    Row(
        Modifier.fillMaxWidth().padding(vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(Modifier.weight(1f)) {
            Text(title, fontWeight = FontWeight.SemiBold, fontSize = 14.sp)
            Text(subtitle, fontSize = 12.sp, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}
