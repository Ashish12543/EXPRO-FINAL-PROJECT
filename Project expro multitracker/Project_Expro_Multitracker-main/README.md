# Elderly Fall Detection and Activity Monitoring System

Real-time elderly monitoring system built around YOLO pose estimation, persistent identity tracking, Telegram alerts, a Flask dashboard, offline video annotation, and daily/weekly activity reports.

## Overview

This project watches a live camera feed or an uploaded video and tries to understand what each person is doing. It can detect normal activities like walking, standing, sitting, and sleeping, and it can also detect minor and major fall events.

The system is designed for caregiver support and elder-care environments where fast fall detection, clear summaries, and a simple dashboard are useful.

## Key Features

- Real-time pose-based activity detection
- `WALKING`, `STANDING`, `SITTING`, `SLEEPING`
- `MINOR FALL` and `MAJOR FALL` detection
- Persistent person IDs across frames
- Face recognition, ReID, and manual name registration
- Live Flask dashboard with dark mode
- Telegram alerts and Telegram summary messages
- Primary-resident Telegram summaries
- Low-power mode with motion-based wake-up
- Uploaded video annotation with progress bar
- Annotated video playback at `1x`, `2x`, and `4x`
- Daily and weekly activity reports
- CSV and PDF export
- AI-style activity recommendations

## How It Works

1. The app opens a webcam or processes an uploaded video.
2. YOLO pose estimation detects body keypoints.
3. The system classifies each person’s current state based on pose, body angle, and motion.
4. Activity time is accumulated for walking, standing, sitting, and sleeping.
5. Fall states are tracked separately so a person does not get mislabeled as just lying down.
6. The dashboard updates live, Telegram alerts are sent when needed, and the database stores activity history.

## Fall Logic

The fall detection logic is designed to distinguish between:

- normal lying or sitting on the floor,
- a minor fall,
- and a major fall.

The project uses:

- body angle
- vertical velocity
- overall movement
- posture collapse from upright to horizontal
- confirmation windows for escalation

Major falls are escalated when a person remains down for the configured time window.

## Telegram Behavior

Telegram is split into two behaviors:

- Activity summaries are bundled into one message for a single primary resident.
- Minor and major fall alerts are still sent separately.

If a major fall is detected, the system can send a fast burst of three Telegram notifications.

You can configure the primary resident from the settings page so Telegram focuses on one person instead of listing all tracked residents.

## Dashboard Features

The Flask dashboard includes:

- live camera preview
- active alerts
- fall history
- people cards with activity durations
- daily activity summary
- AI recommendations
- activity analytics charts
- dark mode toggle
- upload video workflow
- report download page

The analytics section shows:

- a time-based trend chart
- a donut chart for activity breakdown

## Video Upload and Annotation

You can upload a video of a person walking, sitting, or falling and the system will:

- process the video frame by frame,
- annotate activities on the frames,
- show processing progress,
- and generate a downloadable annotated MP4.

The annotated video page also supports playback speeds of:

- `1x`
- `2x`
- `4x`

## Activity Reports

The project can generate:

- daily reports
- weekly reports

Each report includes:

- walking, standing, sitting, and sleeping time
- monitored time
- fall count
- last fall type
- last fall time
- AI recommendations

Reports can be exported as:

- CSV
- PDF

## AI Recommendations

The recommendation engine analyzes the activity pattern and gives practical suggestions based on:

- how much the person walked
- how long they sat
- how much they stood
- how much they slept
- whether falls were detected

The recommendations are presented with:

- severity
- confidence score
- short explanation text

## Low-Power Mode

If the person leaves the frame for long enough, the system enters low-power mode.

In low-power mode:

- the OpenCV preview window closes
- the overlay is suppressed
- the camera polls for motion again after a short delay

When motion returns, the system resumes live monitoring automatically.

## Project Structure

- `smart_fall_activity_report.py` - main application
- `requirements.txt` - Python dependencies
- `system_settings.example.json` - safe configuration template
- `system_settings.json` - local settings file
- `run.ps1` - Windows launcher script
- `yolo11n-pose.pt` - pose model used by the app

## Requirements

- Python `3.10` to `3.12`
- Windows webcam or any OpenCV-compatible camera
- Enough disk space for:
  - uploaded videos
  - annotated videos
  - local database files
- NVIDIA GPU recommended, but the app can run on CPU

## Quick Start

### 1. Open the project folder

```powershell
cd "C:\Project expro multitracker\Project_Expro_Multitracker-main"
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate it

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
```

### 5. Create your local settings file

```powershell
Copy-Item system_settings.example.json system_settings.json
```

### 6. Start the app

```powershell
.\run.ps1
```

### 7. Open the dashboard

```text
http://127.0.0.1:5000
```

## Configuration

Important settings:

- `enable_telegram` - turn Telegram alerts on or off
- `bot_token` - Telegram bot token
- `chat_id` - Telegram chat ID
- `telegram_primary_person` - the one resident Telegram should summarize
- `fall_confirm_window_sec` - how long a person must remain down before major-fall escalation
- `max_people_to_track` - how many people the tracker tries to follow
- `video_output_mode` - dashboard, window, or both
- `preferred_camera` - camera index

## Runtime Notes

- The app creates the SQLite database automatically.
- The live loop uses asynchronous notification sending so alerts do not block frame processing.
- The dashboard and reports use the same activity totals and fall history stored in the database.
- The app is designed for Windows and uses OpenCV camera access.

## Troubleshooting

- If the camera does not open, check your camera index in the settings page.
- If Telegram says it is disabled, make sure:
  - `enable_telegram` is checked
  - `bot_token` is filled in
  - `chat_id` is filled in
- If you see multiple people in Telegram summaries, set `telegram_primary_person` to the resident you want to follow.
- If you do not want the webcam preview window, use dashboard-only output mode.

## Security

Do not commit your real Telegram bot token, chat ID, or machine-specific settings. Keep those only in your local `system_settings.json`.

## Notes For Presentation

If you need to explain the project quickly:

- It is a pose-based elderly monitoring system.
- It detects activities and falls in real time.
- It keeps identity persistent across frames.
- It sends Telegram alerts and summaries.
- It generates reports and annotated videos for review.

