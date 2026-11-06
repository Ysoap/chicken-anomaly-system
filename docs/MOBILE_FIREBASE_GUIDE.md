# ?? Mobile App + Firebase Integration Guide

## ?? Firebase Project Setup

### Step 1: Create Firebase Project

1. Go to https://console.firebase.google.com
2. Click **"Add project"**
3. Project name: chicken-anomaly-detection
4. Enable Google Analytics (optional, recommended for monitoring)
5. Choose default account and create project

---

### Step 2: Add Android App

1. In Firebase Console, click **"Add app"** ? Select Android
2. **Android package name:** com.chickendetect
3. **App nickname:** Chicken Detect (optional)
4. **Debug signing certificate SHA-1:** Leave blank for now (add later for Auth)
5. Click **"Register app"**
6. **Download google-services.json**
7. Place file in: mobile-app/app/google-services.json

?? **Important:** Never commit google-services.json to public repos (.gitignore already configured)

---

### Step 3: Enable Firebase Services

#### 3.1 Realtime Database

1. In Firebase Console ? **Build** ? **Realtime Database**
2. Click **"Create Database"**
3. Location: us-central1 (or closest to your region)
4. Security rules: Start in **test mode** (change later)
5. Database URL: https://chicken-anomaly-detection.firebaseio.com

**Security Rules (Test Mode):**
```json
{
  "rules": {
    ".read": true,
    ".write": true
  }
}
```

**Production Rules (After development):**
```json
{
  "rules": {
    "anomalies": {
      ".read": "auth != null",
      ".write": "auth != null && auth.uid == 'backend-service'"
    }
  }
}
```

---

#### 3.2 Cloud Messaging (Push Notifications)

1. In Firebase Console ? **Build** ? **Cloud Messaging**
2. Already enabled by default
3. Click **"Cloud Messaging API (Legacy)"** ? Enable if prompted
4. **Server key:** Copy this (needed for backend Firebase Admin SDK)
   - Go to **Project Settings** ? **Cloud Messaging** ? **Server key**

---

#### 3.3 Storage (Optional - for video thumbnails)

1. In Firebase Console ? **Build** ? **Storage**
2. Click **"Get Started"**
3. Security rules: Start in test mode
4. Location: Same as Realtime Database

---

### Step 4: Download Admin SDK Credentials (Backend)

1. In Firebase Console ? **Project Settings** (gear icon)
2. **Service accounts** tab
3. Click **"Generate new private key"**
4. Save file as: ackend/api/firebase-adminsdk.json

?? **Critical:** Never commit this file (already in .gitignore)

---

## ?? Android Project Setup

### Build Configuration

**File:** mobile-app/build.gradle.kts (Project-level)

```kotlin
buildscript {
    dependencies {
        classpath("com.google.gms:google-services:4.4.0")
    }
}

plugins {
    id("com.android.application") version "8.1.0" apply false
    id("org.jetbrains.kotlin.android") version "1.9.0" apply false
}
```

**File:** mobile-app/app/build.gradle.kts (App-level)

```kotlin
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.google.gms.google-services")
}

android {
    namespace = "com.chickendetect"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.chickendetect"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
    }

    buildFeatures {
        compose = true
    }

    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.3"
    }
}

dependencies {
    // Firebase
    implementation(platform("com.google.firebase:firebase-bom:32.5.0"))
    implementation("com.google.firebase:firebase-auth-ktx")
    implementation("com.google.firebase:firebase-database-ktx")
    implementation("com.google.firebase:firebase-storage-ktx")
    implementation("com.google.firebase:firebase-messaging-ktx")

    // Jetpack Compose
    implementation("androidx.compose.ui:ui:1.5.4")
    implementation("androidx.compose.material3:material3:1.1.2")
    implementation("androidx.compose.ui:ui-tooling-preview:1.5.4")
    implementation("androidx.activity:activity-compose:1.8.0")

    // ViewModel
    implementation("androidx.lifecycle:lifecycle-viewmodel-ktx:2.6.2")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.6.2")
    implementation("androidx.lifecycle:lifecycle-livedata-ktx:2.6.2")

    // Charts
    implementation("com.github.PhilJay:MPAndroidChart:v3.1.0")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-play-services:1.7.3")
}
```

---

## ??? Android Architecture (MVVM)

### Project Structure

```
app/src/main/java/com/chickendetect/
+-- ui/
¦   +-- MainActivity.kt
¦   +-- screens/
¦   ¦   +-- DashboardScreen.kt
¦   ¦   +-- HistoryScreen.kt
¦   ¦   +-- SettingsScreen.kt
¦   +-- components/
¦       +-- AnomalyCard.kt
¦       +-- LineChart.kt
¦       +-- AlertList.kt
+-- viewmodel/
¦   +-- DashboardViewModel.kt
¦   +-- HistoryViewModel.kt
+-- data/
¦   +-- repository/
¦   ¦   +-- AnomalyRepository.kt
¦   +-- model/
¦   ¦   +-- AnomalyData.kt
¦   ¦   +-- ChartData.kt
¦   +-- firebase/
¦       +-- RealtimeDBManager.kt
¦       +-- FCMService.kt
+-- utils/
    +-- NotificationHelper.kt
    +-- DateFormatter.kt
```

---

## ?? Firebase Realtime Database Integration

### Data Model

**File:** data/model/AnomalyData.kt

```kotlin
data class AnomalyData(
    val anomaly_score: Double = 0.0,
    val is_anomaly: Boolean = false,
    val timestamp: Long = 0L,
    val diagnosis: String = "",
    val severity: String = "low",
    val top_features: List<Feature> = emptyList()
)

data class Feature(
    val name: String = "",
    val value: Double = 0.0,
    val zscore: Double = 0.0,
    val contribution: Double = 0.0
)
```

---

### Realtime Database Manager

**File:** data/firebase/RealtimeDBManager.kt

```kotlin
import com.google.firebase.database.*

class RealtimeDBManager {
    private val database = FirebaseDatabase.getInstance()
    private val anomalyRef = database.getReference("anomalies")

    // Listen to real-time updates (last 1 minute)
    fun listenToLiveData(callback: (AnomalyData) -> Unit) {
        val today = SimpleDateFormat("yyyy-MM-dd", Locale.US).format(Date())
        
        anomalyRef.child(today)
            .orderByKey()
            .limitToLast(60)  // Last 60 seconds
            .addChildEventListener(object : ChildEventListener {
                override fun onChildAdded(snapshot: DataSnapshot, previousChildName: String?) {
                    val data = snapshot.getValue(AnomalyData::class.java)
                    data?.let { callback(it) }
                }

                override fun onChildChanged(snapshot: DataSnapshot, previousChildName: String?) {
                    val data = snapshot.getValue(AnomalyData::class.java)
                    data?.let { callback(it) }
                }

                override fun onChildRemoved(snapshot: DataSnapshot) {}
                override fun onChildMoved(snapshot: DataSnapshot, previousChildName: String?) {}
                override fun onCancelled(error: DatabaseError) {
                    Log.e("Firebase", "Listen failed: ")
                }
            })
    }

    // Fetch historical data
    suspend fun fetchHistoricalData(date: String, hours: Int): List<AnomalyData> {
        return suspendCoroutine { continuation ->
            anomalyRef.child(date)
                .get()
                .addOnSuccessListener { snapshot ->
                    val dataList = snapshot.children.mapNotNull { 
                        it.getValue(AnomalyData::class.java) 
                    }
                    continuation.resume(dataList)
                }
                .addOnFailureListener { e ->
                    continuation.resumeWithException(e)
                }
        }
    }
}
```

---

## ?? Push Notifications (Firebase Cloud Messaging)

### FCM Service

**File:** data/firebase/FCMService.kt

```kotlin
import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage

class FCMService : FirebaseMessagingService() {

    override fun onNewToken(token: String) {
        super.onNewToken(token)
        Log.d("FCM", "New token: ")
        // TODO: Send token to backend if needed
    }

    override fun onMessageReceived(remoteMessage: RemoteMessage) {
        super.onMessageReceived(remoteMessage)

        // Handle notification payload
        remoteMessage.notification?.let {
            val title = it.title ?: "Anomaly Alert"
            val body = it.body ?: ""
            showNotification(title, body)
        }

        // Handle data payload
        remoteMessage.data.let { data ->
            val anomalyScore = data["anomaly_score"]?.toDoubleOrNull()
            val diagnosis = data["diagnosis"]
            val severity = data["severity"]
            
            // Show high-priority notification
            if (severity == "high") {
                showHighPriorityNotification(diagnosis ?: "Anomaly detected")
            }
        }
    }

    private fun showNotification(title: String, body: String) {
        val notificationManager = getSystemService(NOTIFICATION_SERVICE) as NotificationManager

        // Create notification channel (Android 8+)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                "anomaly_alerts",
                "Anomaly Alerts",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Alerts for chicken farm anomalies"
                enableLights(true)
                lightColor = Color.RED
                enableVibration(true)
            }
            notificationManager.createNotificationChannel(channel)
        }

        val notification = NotificationCompat.Builder(this, "anomaly_alerts")
            .setSmallIcon(R.drawable.ic_alert)
            .setContentTitle(title)
            .setContentText(body)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .setSound(RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION))
            .build()

        notificationManager.notify(System.currentTimeMillis().toInt(), notification)
    }

    private fun showHighPriorityNotification(diagnosis: String) {
        // Implementation similar to showNotification but with PRIORITY_MAX
    }
}
```

---

### AndroidManifest.xml Configuration

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android">
    
    <uses-permission android:name="android.permission.INTERNET" />
    <uses-permission android:name="android.permission.POST_NOTIFICATIONS" />

    <application
        android:name=".ChickenDetectApp"
        android:allowBackup="true"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:theme="@style/Theme.ChickenDetect">

        <activity
            android:name=".ui.MainActivity"
            android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>

        <!-- FCM Service -->
        <service
            android:name=".data.firebase.FCMService"
            android:exported="false">
            <intent-filter>
                <action android:name="com.google.firebase.MESSAGING_EVENT" />
            </intent-filter>
        </service>

    </application>
</manifest>
```

---

## ?? UI Components (Jetpack Compose)

### Dashboard Screen

**File:** ui/screens/DashboardScreen.kt

```kotlin
@Composable
fun DashboardScreen(viewModel: DashboardViewModel = viewModel()) {
    val anomalyData by viewModel.currentAnomaly.observeAsState()

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        // Current Anomaly Card
        AnomalyCard(data = anomalyData)

        Spacer(modifier = Modifier.height(16.dp))

        // Line Chart (last 1 hour)
        LineChartView(data = viewModel.historicalData.value)

        Spacer(modifier = Modifier.height(16.dp))

        // Recent Alerts
        AlertList(alerts = viewModel.recentAlerts.value)
    }
}

@Composable
fun AnomalyCard(data: AnomalyData?) {
    val backgroundColor = when {
        data == null -> Color.Gray
        data.anomaly_score > 0.75 -> Color.Red
        data.anomaly_score > 0.5 -> Color.Yellow
        else -> Color.Green
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = backgroundColor)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Text("Current Anomaly Score", style = MaterialTheme.typography.titleMedium)
            Text(
                text = String.format("%.2f", data?.anomaly_score ?: 0.0),
                style = MaterialTheme.typography.displayLarge
            )
            if (data?.is_anomaly == true) {
                Text("?? ALERT: ", style = MaterialTheme.typography.bodyMedium)
            }
        }
    }
}
```

---

## ? Testing Checklist

### Firebase Connection
- [ ] google-services.json placed correctly
- [ ] Firebase initialized (check Logcat for "Firebase initialized")
- [ ] Realtime DB connection works (check Logcat for successful read)

### Push Notifications
- [ ] FCM token generated (check Logcat)
- [ ] Test notification from Firebase Console ? Cloud Messaging ? Send test message
- [ ] Notification displays on device
- [ ] Notification click opens app

### Real-time Data
- [ ] Dashboard updates when backend pushes data
- [ ] Historical data loads correctly
- [ ] Charts render without errors

### Edge Cases
- [ ] App handles network loss gracefully
- [ ] Offline mode shows cached data
- [ ] Firebase quota limits not exceeded

---

## ?? Common Issues

**Issue 1: google-services.json not found**
- Solution: Ensure file is in mobile-app/app/google-services.json, not mobile-app/google-services.json

**Issue 2: FCM token null**
- Solution: Check AndroidManifest.xml has INTERNET permission and FCMService declared

**Issue 3: Realtime DB permission denied**
- Solution: Change security rules to test mode temporarily

**Issue 4: Charts not rendering**
- Solution: Add maven { url 'https://jitpack.io' } to settings.gradle.kts

---

**For questions, tag Pandu in GitHub Issues!**