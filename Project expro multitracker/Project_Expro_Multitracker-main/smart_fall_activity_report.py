import cv2
import time
import csv
import io
import numpy as np
import requests
import sys
import os
import socket
import argparse
import textwrap
import torch
import torch.nn as nn
from ultralytics import YOLO
from collections import defaultdict
import sqlite3
import pickle
import json
import uuid
from datetime import datetime, date, timedelta
from contextlib import nullcontext
from flask import Flask, jsonify, request, Response, send_from_directory, send_file
import threading
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANNOTATED_VIDEO_DIR = os.path.join(BASE_DIR, "annotated_videos")
VIDEO_UPLOAD_DIR = os.path.join(BASE_DIR, "video_uploads")
os.makedirs(ANNOTATED_VIDEO_DIR, exist_ok=True)
os.makedirs(VIDEO_UPLOAD_DIR, exist_ok=True)

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except Exception:
    face_recognition = None
    FACE_RECOGNITION_AVAILABLE = False

try:
    import torchreid
    from torchreid.utils import FeatureExtractor
    REID_AVAILABLE = True
except Exception:
    torchreid = None
    FeatureExtractor = None
    REID_AVAILABLE = False

data_lock = threading.Lock()

# ==================== Person Re-Identification (ReID) ====================
class ReIDManager:
    def __init__(self, threshold=0.75):
        self.enabled = REID_AVAILABLE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = None
        if self.enabled:
            # Using OSNet (osnet_x1_0) - specialized for person re-identification
            # FeatureExtractor handles model loading and preprocessing (resize/norm)
            self.extractor = FeatureExtractor(
                model_name='osnet_x1_0',
                device=self.device,
                verbose=False
            )
        
        self.threshold = threshold
        # Identity Store: persistent_id -> {'embedding': tensor, 'last_seen': timestamp}
        self.identity_bank = {}
        self.next_persistent_id = 1
        self.bank_file = os.path.join(BASE_DIR, "reid_bank.pickle")
        self.load_bank()

    def load_bank(self):
        if os.path.exists(self.bank_file):
            try:
                with open(self.bank_file, "rb") as f:
                    data = pickle.load(f)
                    
                if isinstance(data, dict):
                    if 'bank' in data:
                        self.identity_bank = data['bank']
                        self.next_persistent_id = data.get('next_id', 1)
                    else:
                        # Legacy format: the whole pickle was the bank
                        self.identity_bank = data
                        # Estimate next_id from keys like "Person_N"
                        pids = [int(k.split('_')[1]) for k in data.keys() if isinstance(k, str) and k.startswith("Person_")]
                        self.next_persistent_id = max(pids) + 1 if pids else 1
                
                # Validation: Remove invalid entries that would cause KeyError
                valid_bank = {}
                for pid, entry in self.identity_bank.items():
                    if isinstance(entry, dict) and 'embedding' in entry:
                        valid_bank[pid] = entry
                    elif isinstance(entry, np.ndarray):
                        # Very old format where entry WAS the embedding
                        valid_bank[pid] = {'embedding': entry, 'last_seen': time.time()}
                
                self.identity_bank = valid_bank
                print(f"Loaded {len(self.identity_bank)} valid identities from ReID bank.")
            except Exception as e:
                print(f"Error loading ReID bank: {e}")

    def save_bank(self):
        try:
            with open(self.bank_file, "wb") as f:
                pickle.dump({'bank': self.identity_bank, 'next_id': self.next_persistent_id}, f)
        except Exception as e:
            print(f"Error saving ReID bank: {e}")

    @torch.no_grad()
    def get_embedding(self, person_crop):
        if not self.enabled or self.extractor is None:
            return None
        if person_crop is None or person_crop.size == 0: return None
        # Convert BGR (OpenCV) to RGB (expected by torchreid/PIL)
        rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
        # extractor returns a torch tensor
        features = self.extractor([rgb_crop])
        # L2 Normalize for cosine similarity matching
        features = nn.functional.normalize(features, p=2, dim=1)
        return features.cpu().numpy().flatten()

    def match_identity(self, current_embedding, current_sig=None, blocked_ids=None):
        if current_embedding is None and current_sig is None: return None
        
        blocked_ids = set(blocked_ids or [])
        best_id = None
        max_sim = -1
        
        for pid, data in self.identity_bank.items():
            if pid in blocked_ids:
                continue
            # 1. Try ReID Embedding Match
            person_sim = -1
            if 'embedding' in data:
                # Moving Average format
                gallery_emb = data['embedding']
                if gallery_emb.shape == current_embedding.shape:
                    person_sim = np.dot(current_embedding, gallery_emb)
            elif 'embeddings' in data and data['embeddings']:
                # Legacy Gallery format: Find best match
                for gallery_emb in data['embeddings']:
                    if gallery_emb.shape == current_embedding.shape:
                        sim = np.dot(current_embedding, gallery_emb)
                        if sim > person_sim: person_sim = sim
            
            if person_sim > max_sim:
                max_sim = person_sim
                best_id = pid
        
        # Threshold check for ReID
        if max_sim > self.threshold:
            # Update Identity Bank with Moving Average (Stability Patch)
            # Formula: 0.8 * old + 0.2 * new, then re-normalize
            if 'embedding' not in self.identity_bank[best_id]:
                # Convert legacy to moving average format
                self.identity_bank[best_id]['embedding'] = self.identity_bank[best_id]['embeddings'][0]
            
            old_emb = self.identity_bank[best_id]['embedding']
            updated_emb = 0.8 * old_emb + 0.2 * current_embedding
            # L2 Normalize the updated embedding
            norm = np.linalg.norm(updated_emb)
            if norm > 0:
                self.identity_bank[best_id]['embedding'] = updated_emb / norm
            
            self.identity_bank[best_id]['last_seen'] = time.time()
            return best_id
        
        # 2. Fallback to Clothing Color (Stability Patch)
        if current_sig is not None:
            for pid, data in self.identity_bank.items():
                if pid in blocked_ids:
                    continue
                if 'color_sig' in data:
                    if compare_signatures(current_sig, data['color_sig']) > 0.65:
                        # Found a match via color! Update its last seen
                        self.identity_bank[pid]['last_seen'] = time.time()
                        return pid

        # 3. New identity
        new_id = f"Person_{self.next_persistent_id}"
        self.next_persistent_id += 1
        self.identity_bank[new_id] = {
            'embedding': current_embedding,
            'color_sig': current_sig,
            'last_seen': time.time(),
            'first_seen': time.time()
        }
        return new_id

    def add_to_gallery(self, pid, embedding):
        """Add a new angle to a person's signature if it's sufficiently different"""
        if pid not in self.identity_bank or embedding is None: return
        
        gallery = self.identity_bank[pid].setdefault('embeddings', [])
        
        # Only add if it's a 'new' angle (sim < 0.90 compared to existing ones)
        # and gallery is not too large (max 10 angles)
        is_new_angle = True
        for gallery_emb in gallery:
            if np.dot(embedding, gallery_emb) > 0.90:
                is_new_angle = False
                break
        
        if is_new_angle and len(gallery) < 10:
            gallery.append(embedding)
            print(f"📸 Captured new body angle for {pid} (Total: {len(gallery)})")

    def prune_bank(self, max_idle=86400, min_duration=5, protected_ids=None):
        """Remove short-lived 'ghost' IDs to prevent bank bloat. Default idle increased to 24h."""
        now = time.time()
        to_delete = []
        protected_ids = protected_ids or []
        for pid, data in self.identity_bank.items():
            if pid in protected_ids:
                continue # Never prune named individuals
                
            idle_time = now - data['last_seen']
            duration = data['last_seen'] - data.get('first_seen', data['last_seen'])
            
            # If seen once and never again for 24 hours, or tracked for < 5s then gone
            if idle_time > max_idle or (idle_time > 300 and duration < min_duration):
                to_delete.append(pid)
        
        for pid in to_delete:
            del self.identity_bank[pid]
        if to_delete:
            print(f"🧹 Pruned {len(to_delete)} ghost identities from ReID bank.")
        return to_delete

reid_manager = ReIDManager(threshold=0.5)
tracker_to_persistent = {} # Maps YOLO tracker_id -> persistent_id
IDENTITY_MODE = "reid" if REID_AVAILABLE else "tracker"
SINGLE_PERSON_MODE = False
SINGLE_PERSON_LABEL = "Resident"
SETTINGS_FILE = os.path.join(BASE_DIR, "system_settings.json")

DEFAULT_SETTINGS = {
    "bot_token": "",
    "chat_id": "",
    "telegram_primary_person": "",
    "message_cooldown_sec": 90,
    "fall_confirm_window_sec": 10.0,
    "preferred_camera": "0",
    "max_people_to_track": 4,
    "enable_telegram": False,
    "enable_detection": True,
    "enable_voice_alert": False,
    "enable_wellness_monitoring": True,
    "display_metrics_overlay": True,
    "video_output_mode": "both",
    "enable_local_preview": False,
    "prefer_gpu": True,
    "camera_width": 640,
    "camera_height": 480,
    "camera_fps": 15,
    "camera_fourcc": "MJPG",
    "camera_frame_flush_count": 4,
    "stream_jpeg_quality": 70,
    "stream_max_fps": 8,
    "stream_output_width": 480,
    "sleep_camera_width": 320,
    "sleep_camera_height": 240,
    "sleep_poll_interval_sec": 1.0,
    "sleep_motion_min_pixels": 350,
    "normal_motion_min_pixels": 1200,
    "yolo_imgsz": 416,
    "yolo_confidence": 0.65,
    "yolo_tracker": "bytetrack.yaml",
    "reid_scan_interval_frames": 45,
    "face_name_interval_frames": 90,
    "deployment_mode": "server",
    "node_id": socket.gethostname(),
    "central_server_url": "http://127.0.0.1:5000",
    "server_bind_host": "0.0.0.0",
    "server_port": 5000
}

MAJOR_FALL_TELEGRAM_BURST_COUNT = 3
MAJOR_FALL_TELEGRAM_BURST_DELAY_SEC = 0.12
LOW_POWER_IDLE_TIMEOUT_SEC = 6.0
SLEEPING_LYING_CONFIRM_SEC = 8.0
SLEEPING_FROM_SITTING_SEC = 2.5
TRACK_ASSOCIATION_MAX_AGE_FRAMES = 20
TRACK_ASSOCIATION_MAX_DISTANCE_PX = 120
TRACK_ASSOCIATION_MIN_SCORE = 0.55
TRACK_ASSOCIATION_MATCH_MARGIN = 0.06
TRACK_ASSOCIATION_REMAP_MARGIN = 0.09
STANDING_SITTING_CONFIRM_FRAMES = 8
SITTING_STANDING_CONFIRM_FRAMES = 5
WALKING_CONFIRM_FRAMES = 3
STANDING_CONFIRM_FRAMES = 4
PREVIEW_WINDOW_NAME = "Elderly Monitor Live Overlay"
PREVIEW_WINDOW_WIDTH = 1280
PREVIEW_WINDOW_HEIGHT = 720

settings = DEFAULT_SETTINGS.copy()
system_events = []
last_telegram_sent_at = {}
last_notified_activity = {}
latest_stream_frame = None
latest_stream_frame_id = 0
last_stream_encode_at = 0.0
pending_stream_frame = None
pending_stream_frame_seq = 0
last_camera_read_at = 0.0
last_sleep_peek_at = 0.0
camera_reconfigure_pending = False
preview_window_enabled = False
preview_window_initialized = False
camera_active = True  # Camera on/off control
remote_nodes = {}
remote_edge_reports = {}
model = None
model_lock = threading.Lock()
http_task_lock = threading.Lock()
pending_http_posts = {}
last_http_error_log = {}

def add_system_event(message, level="info"):
    entry = {
        "message": str(message),
        "level": level,
        "timestamp": time.time(),
        "time_str": time.strftime("%H:%M:%S", time.localtime())
    }
    with data_lock:
        system_events.append(entry)
        del system_events[:-100]
    print(f"[{level.upper()}] {message}")

def parse_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    return default

def clamp_int(value, default, minimum=None, maximum=None):
    try:
        parsed = int(float(value))
    except Exception:
        parsed = int(default)
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed

def clamp_float(value, default, minimum=None, maximum=None):
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if minimum is not None:
        parsed = max(minimum, parsed)
    if maximum is not None:
        parsed = min(maximum, parsed)
    return parsed

def load_settings():
    global settings
    settings = DEFAULT_SETTINGS.copy()
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                settings.update({k: data[k] for k in DEFAULT_SETTINGS if k in data})
        except Exception as e:
            print(f"Error loading settings: {e}")
    settings["message_cooldown_sec"] = clamp_int(settings.get("message_cooldown_sec", 90), 90, minimum=0)
    settings["fall_confirm_window_sec"] = clamp_float(settings.get("fall_confirm_window_sec", 10.0), 10.0, minimum=1.0)
    settings["preferred_camera"] = str(settings.get("preferred_camera", "0") or "0")
    settings["max_people_to_track"] = clamp_int(settings.get("max_people_to_track", DEFAULT_SETTINGS["max_people_to_track"]), DEFAULT_SETTINGS["max_people_to_track"], minimum=1, maximum=10)
    settings["deployment_mode"] = str(settings.get("deployment_mode", DEFAULT_SETTINGS["deployment_mode"]) or DEFAULT_SETTINGS["deployment_mode"]).strip().lower()
    if settings["deployment_mode"] not in {"server", "edge", "standalone"}:
        settings["deployment_mode"] = DEFAULT_SETTINGS["deployment_mode"]
    settings["video_output_mode"] = str(settings.get("video_output_mode", DEFAULT_SETTINGS["video_output_mode"]) or DEFAULT_SETTINGS["video_output_mode"]).strip().lower()
    if settings["video_output_mode"] not in {"dashboard", "window", "both"}:
        settings["video_output_mode"] = DEFAULT_SETTINGS["video_output_mode"]
    settings["node_id"] = str(settings.get("node_id", DEFAULT_SETTINGS["node_id"]) or DEFAULT_SETTINGS["node_id"]).strip()
    settings["central_server_url"] = str(settings.get("central_server_url", DEFAULT_SETTINGS["central_server_url"]) or DEFAULT_SETTINGS["central_server_url"]).strip().rstrip("/")
    settings["server_bind_host"] = str(settings.get("server_bind_host", DEFAULT_SETTINGS["server_bind_host"]) or DEFAULT_SETTINGS["server_bind_host"]).strip()
    settings["server_port"] = clamp_int(settings.get("server_port", DEFAULT_SETTINGS["server_port"]), DEFAULT_SETTINGS["server_port"], minimum=1)
    settings["telegram_primary_person"] = str(settings.get("telegram_primary_person", DEFAULT_SETTINGS["telegram_primary_person"]) or "").strip()
    settings["prefer_gpu"] = parse_bool(settings.get("prefer_gpu", DEFAULT_SETTINGS["prefer_gpu"]), DEFAULT_SETTINGS["prefer_gpu"])
    settings["camera_width"] = clamp_int(settings.get("camera_width", DEFAULT_SETTINGS["camera_width"]), DEFAULT_SETTINGS["camera_width"], minimum=320)
    settings["camera_height"] = clamp_int(settings.get("camera_height", DEFAULT_SETTINGS["camera_height"]), DEFAULT_SETTINGS["camera_height"], minimum=240)
    settings["camera_fps"] = clamp_int(settings.get("camera_fps", DEFAULT_SETTINGS["camera_fps"]), DEFAULT_SETTINGS["camera_fps"], minimum=1, maximum=60)
    settings["camera_fourcc"] = str(settings.get("camera_fourcc", DEFAULT_SETTINGS["camera_fourcc"]) or DEFAULT_SETTINGS["camera_fourcc"]).strip().upper()[:4] or DEFAULT_SETTINGS["camera_fourcc"]
    settings["camera_frame_flush_count"] = clamp_int(settings.get("camera_frame_flush_count", DEFAULT_SETTINGS["camera_frame_flush_count"]), DEFAULT_SETTINGS["camera_frame_flush_count"], minimum=0, maximum=6)
    settings["stream_jpeg_quality"] = clamp_int(settings.get("stream_jpeg_quality", DEFAULT_SETTINGS["stream_jpeg_quality"]), DEFAULT_SETTINGS["stream_jpeg_quality"], minimum=40, maximum=95)
    settings["stream_max_fps"] = clamp_int(settings.get("stream_max_fps", DEFAULT_SETTINGS["stream_max_fps"]), DEFAULT_SETTINGS["stream_max_fps"], minimum=1, maximum=30)
    settings["stream_output_width"] = clamp_int(settings.get("stream_output_width", DEFAULT_SETTINGS["stream_output_width"]), DEFAULT_SETTINGS["stream_output_width"], minimum=320, maximum=1920)
    settings["sleep_camera_width"] = clamp_int(settings.get("sleep_camera_width", DEFAULT_SETTINGS["sleep_camera_width"]), DEFAULT_SETTINGS["sleep_camera_width"], minimum=160, maximum=640)
    settings["sleep_camera_height"] = clamp_int(settings.get("sleep_camera_height", DEFAULT_SETTINGS["sleep_camera_height"]), DEFAULT_SETTINGS["sleep_camera_height"], minimum=120, maximum=480)
    settings["sleep_poll_interval_sec"] = clamp_float(settings.get("sleep_poll_interval_sec", DEFAULT_SETTINGS["sleep_poll_interval_sec"]), DEFAULT_SETTINGS["sleep_poll_interval_sec"], minimum=0.05, maximum=2.0)
    settings["sleep_motion_min_pixels"] = clamp_int(settings.get("sleep_motion_min_pixels", DEFAULT_SETTINGS["sleep_motion_min_pixels"]), DEFAULT_SETTINGS["sleep_motion_min_pixels"], minimum=25, maximum=20000)
    settings["normal_motion_min_pixels"] = clamp_int(settings.get("normal_motion_min_pixels", DEFAULT_SETTINGS["normal_motion_min_pixels"]), DEFAULT_SETTINGS["normal_motion_min_pixels"], minimum=25, maximum=50000)
    settings["yolo_imgsz"] = clamp_int(settings.get("yolo_imgsz", DEFAULT_SETTINGS["yolo_imgsz"]), DEFAULT_SETTINGS["yolo_imgsz"], minimum=320, maximum=1280)
    settings["yolo_confidence"] = clamp_float(settings.get("yolo_confidence", DEFAULT_SETTINGS["yolo_confidence"]), DEFAULT_SETTINGS["yolo_confidence"], minimum=0.1, maximum=0.95)
    settings["yolo_tracker"] = str(settings.get("yolo_tracker", DEFAULT_SETTINGS["yolo_tracker"]) or DEFAULT_SETTINGS["yolo_tracker"]).strip() or DEFAULT_SETTINGS["yolo_tracker"]
    settings["reid_scan_interval_frames"] = clamp_int(settings.get("reid_scan_interval_frames", DEFAULT_SETTINGS["reid_scan_interval_frames"]), DEFAULT_SETTINGS["reid_scan_interval_frames"], minimum=15)
    settings["face_name_interval_frames"] = clamp_int(settings.get("face_name_interval_frames", DEFAULT_SETTINGS["face_name_interval_frames"]), DEFAULT_SETTINGS["face_name_interval_frames"], minimum=30)
    for key in ["enable_telegram", "enable_detection", "enable_voice_alert", "enable_wellness_monitoring", "display_metrics_overlay", "enable_local_preview"]:
        settings[key] = parse_bool(settings.get(key, DEFAULT_SETTINGS[key]), DEFAULT_SETTINGS[key])
    settings["enable_local_preview"] = False

def save_settings():
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        add_system_event("Settings file updated")
    except Exception as e:
        print(f"Error saving settings: {e}")
        add_system_event(f"Settings save failed: {e}", level="error")
        raise

def update_settings_from_payload(payload):
    global settings
    new_settings = DEFAULT_SETTINGS.copy()
    new_settings.update(settings)
    new_settings["bot_token"] = str(payload.get("bot_token", "")).strip()
    new_settings["chat_id"] = str(payload.get("chat_id", "")).strip()
    new_settings["message_cooldown_sec"] = clamp_int(payload.get("message_cooldown_sec", settings["message_cooldown_sec"]), settings["message_cooldown_sec"], minimum=0)
    new_settings["fall_confirm_window_sec"] = clamp_float(payload.get("fall_confirm_window_sec", settings["fall_confirm_window_sec"]), settings["fall_confirm_window_sec"], minimum=1.0)
    new_settings["preferred_camera"] = str(payload.get("preferred_camera", settings["preferred_camera"]) or "0")
    new_settings["max_people_to_track"] = clamp_int(payload.get("max_people_to_track", settings["max_people_to_track"]), settings["max_people_to_track"], minimum=1, maximum=10)
    new_settings["deployment_mode"] = str(payload.get("deployment_mode", settings["deployment_mode"]) or settings["deployment_mode"]).strip().lower()
    if new_settings["deployment_mode"] not in {"server", "edge", "standalone"}:
        new_settings["deployment_mode"] = settings["deployment_mode"]
    new_settings["video_output_mode"] = str(payload.get("video_output_mode", settings.get("video_output_mode", DEFAULT_SETTINGS["video_output_mode"])) or settings.get("video_output_mode", DEFAULT_SETTINGS["video_output_mode"])).strip().lower()
    if new_settings["video_output_mode"] not in {"dashboard", "window", "both"}:
        new_settings["video_output_mode"] = settings.get("video_output_mode", DEFAULT_SETTINGS["video_output_mode"])
    new_settings["node_id"] = str(payload.get("node_id", settings["node_id"]) or settings["node_id"]).strip()
    new_settings["central_server_url"] = str(payload.get("central_server_url", settings["central_server_url"]) or settings["central_server_url"]).strip().rstrip("/")
    new_settings["server_bind_host"] = str(payload.get("server_bind_host", settings["server_bind_host"]) or settings["server_bind_host"]).strip()
    new_settings["server_port"] = clamp_int(payload.get("server_port", settings["server_port"]), settings["server_port"], minimum=1)
    new_settings["telegram_primary_person"] = str(payload.get("telegram_primary_person", settings.get("telegram_primary_person", "")) or "").strip()
    for key in ["enable_telegram", "enable_detection", "enable_voice_alert", "enable_wellness_monitoring", "display_metrics_overlay", "enable_local_preview"]:
        new_settings[key] = parse_bool(payload.get(key, False), False)
    new_settings["enable_local_preview"] = False
    settings = new_settings
    save_settings()
    add_system_event("Settings saved")

def parse_runtime_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--mode", choices=["server", "edge", "standalone"])
    parser.add_argument("--node-id")
    parser.add_argument("--central-server-url")
    parser.add_argument("--server-host")
    parser.add_argument("--server-port", type=int)
    args, _ = parser.parse_known_args()
    return args

def apply_runtime_overrides():
    args = parse_runtime_args()
    if args.mode:
        settings["deployment_mode"] = args.mode
    if args.node_id:
        settings["node_id"] = args.node_id.strip()
    if args.central_server_url:
        settings["central_server_url"] = args.central_server_url.strip().rstrip("/")
    if args.server_host:
        settings["server_bind_host"] = args.server_host.strip()
    if args.server_port:
        settings["server_port"] = int(args.server_port)

def is_edge_mode():
    return settings.get("deployment_mode") == "edge"

def is_server_mode():
    return settings.get("deployment_mode") in {"server", "standalone"}

def get_central_server_url():
    return str(settings.get("central_server_url", DEFAULT_SETTINGS["central_server_url"]) or DEFAULT_SETTINGS["central_server_url"]).rstrip("/")

def get_public_settings():
    with data_lock:
        data = dict(settings)
    return data

def use_dashboard_stream():
    return settings.get("video_output_mode", DEFAULT_SETTINGS["video_output_mode"]) in {"dashboard", "both"}

def use_local_preview():
    return settings.get("video_output_mode", DEFAULT_SETTINGS["video_output_mode"]) in {"window", "both"}

def get_effective_stream_max_fps():
    if use_dashboard_stream() and use_local_preview():
        return max(1, min(3, STREAM_MAX_FPS))
    return max(1, STREAM_MAX_FPS)

def telegram_ready():
    return bool(settings.get("enable_telegram") and settings.get("bot_token") and settings.get("chat_id"))

def telegram_api_request(method, payload, timeout=5):
    url = f"https://api.telegram.org/bot{settings['bot_token']}/{method}"
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()

def send_telegram_message(message, category="general", force=False):
    if not telegram_ready():
        return False, "Telegram is disabled or not configured"
    now = time.time()
    cooldown = int(settings.get("message_cooldown_sec", 90))
    last_sent = last_telegram_sent_at.get(category, 0)
    if not force and cooldown > 0 and (now - last_sent) < cooldown:
        return False, "Cooldown active"
    payload = {"chat_id": settings["chat_id"], "text": message}
    try:
        telegram_api_request("sendMessage", payload, timeout=5)
        last_telegram_sent_at[category] = now
        add_system_event(f"Telegram message sent: {category}")
        return True, "Message sent"
    except Exception as e:
        add_system_event(f"Telegram send failed: {e}", level="error")
        return False, str(e)

def pin_telegram_message(message_id, disable_notification=True):
    if not telegram_ready():
        return False, "Telegram is disabled or not configured"
    if not message_id:
        return False, "Missing Telegram message id"
    payload = {
        "chat_id": settings["chat_id"],
        "message_id": int(message_id),
        "disable_notification": bool(disable_notification)
    }
    try:
        telegram_api_request("pinChatMessage", payload, timeout=5)
        add_system_event("Telegram daily summary pinned")
        return True, "Pinned"
    except Exception as e:
        add_system_event(f"Telegram pin failed: {e}", level="warning")
        return False, str(e)

def edit_telegram_message(message_id, message):
    if not telegram_ready():
        return False, "Telegram is disabled or not configured"
    if not message_id:
        return False, "Missing Telegram message id"
    payload = {
        "chat_id": settings["chat_id"],
        "message_id": int(message_id),
        "text": message
    }
    try:
        telegram_api_request("editMessageText", payload, timeout=5)
        return True, "Edited"
    except Exception as e:
        add_system_event(f"Telegram edit failed: {e}", level="warning")
        return False, str(e)

def unpin_telegram_message(message_id=None):
    if not telegram_ready():
        return False, "Telegram is disabled or not configured"
    payload = {"chat_id": settings["chat_id"]}
    if message_id is not None:
        payload["message_id"] = int(message_id)
    try:
        telegram_api_request("unpinChatMessage", payload, timeout=5)
        return True, "Unpinned"
    except Exception as e:
        add_system_event(f"Telegram unpin failed: {e}", level="warning")
        return False, str(e)

def send_telegram_burst(message, category, count=1, delay_sec=0.0, force=False):
    """Send repeated Telegram notifications, mainly for urgent alerts."""
    ok_count = 0
    last_result = "Not sent"
    for idx in range(max(1, int(count))):
        burst_category = f"{category}:burst:{idx}" if count > 1 else category
        ok, last_result = send_telegram_message(message, category=burst_category, force=force)
        if ok:
            ok_count += 1
        if idx < count - 1 and delay_sec > 0:
            time.sleep(delay_sec)
    return ok_count > 0, f"Sent {ok_count}/{count} notifications"

def send_telegram_burst_async(message, category, count=1, delay_sec=0.0, force=False):
    threading.Thread(
        target=send_telegram_burst,
        args=(message, category, count, delay_sec, force),
        daemon=True
    ).start()

def send_telegram_message_async(message, category="general", force=False):
    threading.Thread(
        target=send_telegram_message,
        args=(message, category, force),
        daemon=True
    ).start()

def queue_http_post(key, url, payload, timeout=1.0, error_label="HTTP request"):
    with http_task_lock:
        pending_http_posts[key] = {
            "url": url,
            "payload": payload,
            "timeout": timeout,
            "error_label": error_label
        }
    return True

def http_post_worker():
    while True:
        task = None
        with http_task_lock:
            if pending_http_posts:
                task_key = next(iter(pending_http_posts))
                task = pending_http_posts.pop(task_key)
        if task is None:
            time.sleep(0.02)
            continue

        try:
            response = requests.post(task["url"], json=task["payload"], timeout=task["timeout"])
            response.raise_for_status()
        except Exception as e:
            label = task.get("error_label", "HTTP request")
            now = time.time()
            last_logged = last_http_error_log.get(label, 0.0)
            if (now - last_logged) >= 15.0:
                last_http_error_log[label] = now
                add_system_event(f"{label} failed: {e}", level="warning")

def notify_activity_change(person_key, activity):
    global activity_summary_dirty
    if not settings.get("enable_wellness_monitoring", True):
        return
    if "FALL" in activity or activity == "RECOVERED":
        return
    prior = last_notified_activity.get(person_key)
    if prior == activity:
        return
    last_notified_activity[person_key] = activity
    activity_summary_dirty = True

load_settings()
apply_runtime_overrides()
add_system_event("Settings loaded")

def resolve_runtime_device():
    if not settings.get("prefer_gpu", True):
        return "cpu", "GPU disabled in settings", False
    try:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda:0", torch.cuda.get_device_name(0), True
    except Exception as e:
        return "cpu", f"CUDA probe failed: {e}", False
    return "cpu", "CUDA unavailable", False

YOLO_DEVICE, YOLO_DEVICE_NAME, CUDA_ACTIVE = resolve_runtime_device()
YOLO_HALF = CUDA_ACTIVE
YOLO_IMGSZ = int(settings.get("yolo_imgsz", DEFAULT_SETTINGS["yolo_imgsz"]))
YOLO_CONFIDENCE = float(settings.get("yolo_confidence", DEFAULT_SETTINGS["yolo_confidence"]))
YOLO_TRACKER = str(settings.get("yolo_tracker", DEFAULT_SETTINGS["yolo_tracker"]) or DEFAULT_SETTINGS["yolo_tracker"])
CAMERA_WIDTH = int(settings.get("camera_width", DEFAULT_SETTINGS["camera_width"]))
CAMERA_HEIGHT = int(settings.get("camera_height", DEFAULT_SETTINGS["camera_height"]))
CAMERA_FPS = int(settings.get("camera_fps", DEFAULT_SETTINGS["camera_fps"]))
CAMERA_FOURCC = str(settings.get("camera_fourcc", DEFAULT_SETTINGS["camera_fourcc"]) or DEFAULT_SETTINGS["camera_fourcc"])
CAMERA_FRAME_FLUSH_COUNT = int(settings.get("camera_frame_flush_count", DEFAULT_SETTINGS["camera_frame_flush_count"]))
YOLO_MAX_DET = max(1, min(10, int(settings.get("max_people_to_track", DEFAULT_SETTINGS["max_people_to_track"]))))
PROCESSING_WIDTH = min(480, CAMERA_WIDTH)
PROCESSING_HEIGHT = max(1, int(round(CAMERA_HEIGHT * (PROCESSING_WIDTH / max(CAMERA_WIDTH, 1)))))
STREAM_JPEG_QUALITY = int(settings.get("stream_jpeg_quality", DEFAULT_SETTINGS["stream_jpeg_quality"]))
STREAM_MAX_FPS = int(settings.get("stream_max_fps", DEFAULT_SETTINGS["stream_max_fps"]))
STREAM_OUTPUT_WIDTH = int(settings.get("stream_output_width", DEFAULT_SETTINGS["stream_output_width"]))
SLEEP_CAMERA_WIDTH = int(settings.get("sleep_camera_width", DEFAULT_SETTINGS["sleep_camera_width"]))
SLEEP_CAMERA_HEIGHT = int(settings.get("sleep_camera_height", DEFAULT_SETTINGS["sleep_camera_height"]))
SLEEP_POLL_INTERVAL_SEC = float(settings.get("sleep_poll_interval_sec", DEFAULT_SETTINGS["sleep_poll_interval_sec"]))
SLEEP_MOTION_MIN_PIXELS = int(settings.get("sleep_motion_min_pixels", DEFAULT_SETTINGS["sleep_motion_min_pixels"]))
NORMAL_MOTION_MIN_PIXELS = int(settings.get("normal_motion_min_pixels", DEFAULT_SETTINGS["normal_motion_min_pixels"]))
REID_SCAN_INTERVAL_FRAMES = int(settings.get("reid_scan_interval_frames", DEFAULT_SETTINGS["reid_scan_interval_frames"]))
FACE_NAME_INTERVAL_FRAMES = int(settings.get("face_name_interval_frames", DEFAULT_SETTINGS["face_name_interval_frames"]))
preview_window_enabled = use_local_preview()

cv2.setUseOptimized(True)
try:
    cv2.setNumThreads(min(4, os.cpu_count() or 4))
except Exception:
    pass

if CUDA_ACTIVE:
    try:
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ==================== Face Recognition Setup ====================
FACES_DIR = os.path.join(BASE_DIR, "registered_faces")
ENCODINGS_FILE = os.path.join(BASE_DIR, "encodings.pickle")
if not os.path.exists(FACES_DIR): os.makedirs(FACES_DIR)

known_face_encodings = []
known_face_names = []

def load_encodings():
    global known_face_encodings, known_face_names
    if not FACE_RECOGNITION_AVAILABLE:
        print("Face recognition unavailable. Registration and auto-naming are disabled.")
        known_face_encodings = []
        known_face_names = []
        return
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
            known_face_encodings = data["encodings"]
            known_face_names = data["names"]
    print(f"Loaded {len(known_face_names)} registered faces.")

load_encodings()

def register_face(image_or_list, name, yolo_model=None):
    global known_face_encodings, known_face_names
    if not FACE_RECOGNITION_AVAILABLE:
        return False
    if image_or_list is None: return False
    
    images = image_or_list if isinstance(image_or_list, list) else [image_or_list]
    success_count = 0
    
    for image in images:
        try:
            if isinstance(image, np.ndarray):
                # If we have a YOLO model, try to crop people out first for better accuracy
                if yolo_model:
                    inference_lock = model_lock if yolo_model is model else nullcontext()
                    with inference_lock, torch.inference_mode():
                        results = yolo_model(
                            image,
                            imgsz=YOLO_IMGSZ,
                            conf=YOLO_CONFIDENCE,
                            device=YOLO_DEVICE,
                            half=YOLO_HALF,
                            verbose=False
                        )
                    if results[0].boxes:
                        for box in results[0].boxes.xyxy:
                            x1, y1, x2, y2 = map(int, box.cpu().numpy())
                            crop = image[y1:y2, x1:x2]
                            if process_single_image(crop, name):
                                success_count += 1
                        continue # Already processed crops for this frame
                
                if process_single_image(image, name):
                    success_count += 1
            else: # Flask FileStorage
                file_bytes = np.frombuffer(image.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None: continue
                if process_single_image(img, name):
                    success_count += 1
        except Exception as e:
            print(f"Face Registration Error: {e}")
            
    if success_count > 0:
        with data_lock:
            with open(ENCODINGS_FILE, "wb") as f:
                pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)
        return True
    return False

def process_single_image(img, name):
    global known_face_encodings, known_face_names
    if not FACE_RECOGNITION_AVAILABLE:
        return False
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, number_of_times_to_upsample=1)
        encodings = face_recognition.face_encodings(rgb, boxes)
        if len(encodings) > 0:
            with data_lock:
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
            return True
    except: pass
    return False

# ==================== Database Setup ====================
DB_PATH = os.path.join(BASE_DIR, "monitor_data.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Table for daily activity summaries
    c.execute('''CREATE TABLE IF NOT EXISTS activity
                 (date TEXT, person_id TEXT, walking REAL, standing REAL, sitting REAL, sleeping REAL, PRIMARY KEY(date, person_id))''')
    
    # Migration: Add standing column if it doesn't exist
    try:
        c.execute("ALTER TABLE activity ADD COLUMN standing REAL DEFAULT 0")
    except sqlite3.OperationalError:
        pass # Already exists

    # Table for fall events
    c.execute('''CREATE TABLE IF NOT EXISTS falls
                 (timestamp DATETIME, person_id TEXT, type TEXT)''')
    
    # Migration: Add unix_timestamp column if it doesn't exist
    try:
        c.execute("ALTER TABLE falls ADD COLUMN unix_timestamp REAL")
    except sqlite3.OperationalError:
        pass # Already exists
    conn.commit()
    conn.close()

init_db()

def log_activity_to_db():
    """Sync current in-memory stats to DB and save ReID banks every minute"""
    while True:
        time.sleep(60)
        rollover_summary = rollover_daily_stats_if_needed()
        if rollover_summary is not None:
            send_daily_telegram_summary(rollover_summary)

        # Snapshot the data while holding lock to minimize contention
        with data_lock:
            stats_snapshot = []
            if person_state or any(walking_time.values()) or any(standing_time.values()): # Only if there is active data
                for pid in list(all_tracked_people):
                    stats_snapshot.append((str(pid), walking_time.get(pid, 0), standing_time.get(pid, 0),
                                         sitting_time.get(pid, 0), sleeping_time.get(pid, 0)))
        
        if not stats_snapshot:
            # Just save the banks and continue if no activity stats to log
            with data_lock:
                reid_manager.save_bank()
                save_manual_id_map()
            continue

        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            today = str(date.today())
            for pid, w, st, s, sl in stats_snapshot:
                c.execute('''INSERT OR REPLACE INTO activity (date, person_id, walking, standing, sitting, sleeping)
                             VALUES (?, ?, ?, ?, ?, ?)''', (today, pid, w, st, s, sl))
            conn.commit()
            conn.close()
            
            # Save ReID and Identity mapping
            with data_lock:
                protected = list(manual_id_map.keys())
                pruned_ids = reid_manager.prune_bank(protected_ids=protected)
                for pid in pruned_ids:
                    all_tracked_people.discard(pid)
                    # Also remove from stats if they have very little time (ghosts)
                    total_time = walking_time.get(pid, 0) + standing_time.get(pid, 0) + sitting_time.get(pid, 0) + sleeping_time.get(pid, 0)
                    if total_time < 5:
                        walking_time.pop(pid, None)
                        standing_time.pop(pid, None)
                        sitting_time.pop(pid, None)
                        sleeping_time.pop(pid, None)
                
                reid_manager.save_bank()
                save_manual_id_map()
            print("✓ Database and ReID banks synchronized.")
        except Exception as e:
            print(f"Sync Error: {e}")

# Start DB sync thread (MOVED TO END OF INITIALIZATION)

# ==================== Flask Server ====================
app = Flask(__name__)

@app.route("/favicon.svg")
def favicon_svg():
    return send_from_directory(BASE_DIR, "favicon.svg", mimetype="image/svg+xml")

fall = False
active_alerts = []  # List of unacknowledged falls

# Track fall events with timestamps (history)
fall_events = []

# Track walking/sleeping/sitting/standing durations
walking_time = defaultdict(float)
standing_time = defaultdict(float)
sleeping_time = defaultdict(float)
sitting_time = defaultdict(float)

# Track current state per person
person_state = {}
person_last_time = {}
activity_transition_candidate = {}
activity_transition_count = defaultdict(int)
lying_start_time = {} # Track when a person started lying down
minor_fall_start_time = {} # Track duration of minor fall for escalation
recovery_mode = {} # pid -> expiry_time (suppress minor fall alerts while getting up)
recovery_confirm_count = {} # persistent_id -> count (frames of sustained upright activity)
active_fall_event = {} # pid -> True (prevent multiple alerts for the same fall)

# Movement tracking for static object filtering and activity refinement
person_start_pos = {}
person_last_pos = {} # Track last frame position for velocity
person_velocity = defaultdict(float) # Rolling average velocity
person_vertical_velocity = defaultdict(float) # Rolling average vertical velocity
person_frames_seen = {}
person_is_confirmed = {}

# Squelch logic for ghost IDs/phantom bodies
last_global_alert_time = 0
last_alert_coords = {} # type -> (x, y)
last_alert_pid = {}   # type -> pid
all_tracked_people = set()  # Persistent list of all detected IDs
manual_id_map = {} # Manual link: YOLO_ID -> Registered Name
activity_label_map = {} # Frozen activity label shown in the UI for each persistent ID
person_signatures = {} # Store color histograms: Name -> Histogram

ID_MAP_FILE = os.path.join(BASE_DIR, "manual_id_map.pickle")

def load_manual_id_map():
    global manual_id_map
    if os.path.exists(ID_MAP_FILE):
        try:
            with open(ID_MAP_FILE, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                manual_id_map = data
                print(f"Loaded {len(manual_id_map)} manual ID mappings.")
            else:
                print(f"Warning: {ID_MAP_FILE} had invalid format. Starting fresh.")
                manual_id_map = {}
        except Exception as e:
            print(f"Error loading manual ID map: {e}")
            manual_id_map = {}

def save_manual_id_map():
    try:
        # Prevent overwriting with empty map if it's likely a load failure
        # (Only save if we have mappings or if the file didn't exist)
        with open(ID_MAP_FILE, "wb") as f:
            pickle.dump(manual_id_map, f)
    except Exception as e:
        print(f"Error saving manual ID map: {e}")

load_manual_id_map()

def get_display_id(persistent_id):
    """Resolve how a tracked person should appear in stats and the dashboard."""
    if SINGLE_PERSON_MODE:
        return SINGLE_PERSON_LABEL
    if persistent_id in manual_id_map:
        return manual_id_map[persistent_id]
    return persistent_id

def get_activity_label(persistent_id):
    """Return a stable activity label that does not change when the person is renamed."""
    if SINGLE_PERSON_MODE:
        return SINGLE_PERSON_LABEL
    if persistent_id is None:
        return "Unknown"
    persistent_id = str(persistent_id)
    if persistent_id not in activity_label_map:
        activity_label_map[persistent_id] = get_display_id(persistent_id)
    return activity_label_map[persistent_id]

def resolve_person_key(person_id):
    """Resolve DB/display labels back to the stable internal person key when possible."""
    if person_id is None:
        return None
    person_id = str(person_id)
    if person_id in manual_id_map:
        return person_id
    for persistent_id, display_name in manual_id_map.items():
        if display_name == person_id:
            return persistent_id
    return person_id

def score_identity_candidate(center_coords, candidate_pid, current_embedding=None, current_sig=None, mapped_pid=None):
    """Score how well a candidate persistent ID matches the current detection."""
    with data_lock:
        last_pos = person_last_pos.get(candidate_pid)
        last_seen_frame = last_detection.get(candidate_pid, None)
        bank_entry = dict(reid_manager.identity_bank.get(candidate_pid, {}))

    if last_pos is None or last_seen_frame is None:
        return None

    age = frame_count - last_seen_frame
    if age > TRACK_ASSOCIATION_MAX_AGE_FRAMES:
        return None

    dist = float(np.sqrt((center_coords[0] - last_pos[0]) ** 2 + (center_coords[1] - last_pos[1]) ** 2))
    if dist > TRACK_ASSOCIATION_MAX_DISTANCE_PX:
        return None

    spatial_score = max(0.0, 1.0 - (dist / TRACK_ASSOCIATION_MAX_DISTANCE_PX))
    recency_score = max(0.0, 1.0 - (age / TRACK_ASSOCIATION_MAX_AGE_FRAMES))
    score = 0.65 * spatial_score + 0.35 * recency_score

    appearance_score = None
    candidate_emb = bank_entry.get("embedding")
    if current_embedding is not None and candidate_emb is not None and getattr(candidate_emb, "shape", None) == getattr(current_embedding, "shape", None):
        appearance_score = float(np.dot(current_embedding, candidate_emb))
    elif current_sig is not None and bank_entry.get("color_sig") is not None:
        appearance_score = float(compare_signatures(current_sig, bank_entry["color_sig"]))

    if appearance_score is not None:
        score = 0.55 * max(appearance_score, 0.0) + 0.30 * spatial_score + 0.15 * recency_score

    if mapped_pid is not None and candidate_pid == mapped_pid:
        score += 0.05

    return score

def find_recent_identity_match(center_coords, current_embedding=None, current_sig=None, occupied_pids=None, mapped_pid=None, return_score=False):
    """
    Prefer an already-active identity when YOLO emits a fresh track id for the same person.
    This avoids rapid identity churn when tracker ids are unstable.
    """
    occupied_pids = set(occupied_pids or [])
    best_pid = None
    best_score = -1.0
    second_best_score = -1.0

    with data_lock:
        last_positions = dict(person_last_pos)
        last_seen_frames = dict(last_detection)

    for candidate_pid, last_pos in last_positions.items():
        if candidate_pid in occupied_pids:
            continue

        score = score_identity_candidate(
            center_coords,
            candidate_pid,
            current_embedding=current_embedding,
            current_sig=current_sig,
            mapped_pid=mapped_pid
        )
        if score is None:
            continue

        if score > best_score:
            second_best_score = best_score
            best_score = score
            best_pid = candidate_pid
        elif score > second_best_score:
            second_best_score = score

    if best_score < TRACK_ASSOCIATION_MIN_SCORE:
        return (None, best_score) if return_score else None

    if second_best_score >= 0 and (best_score - second_best_score) < TRACK_ASSOCIATION_MATCH_MARGIN and best_score < (TRACK_ASSOCIATION_MIN_SCORE + 0.12):
        return (None, best_score) if return_score else None

    return (best_pid, best_score) if return_score else best_pid

def validate_tracker_identity(yolo_id_str, current_pid, center_coords, current_embedding=None, current_sig=None, occupied_pids=None):
    """Re-check an existing YOLO-to-persistent mapping before we trust it.

    Keep the previously assigned persistent ID stable whenever it still looks
    plausible. Only drop the mapping when the match is clearly stale.
    """
    if current_pid is None or IDENTITY_MODE == "tracker":
        return current_pid

    current_score = score_identity_candidate(
        center_coords,
        current_pid,
        current_embedding=current_embedding,
        current_sig=current_sig,
        mapped_pid=current_pid
    )
    if current_score is None:
        return None

    if current_score < TRACK_ASSOCIATION_MIN_SCORE:
        return None

    return current_pid

def get_color_signature(image):
    """Calculate color histogram for Re-Identification"""
    try:
        if image is None or image.size == 0: return None
        # Focus on the torso (center of the crop)
        h, w = image.shape[:2]
        torso = image[int(h*0.2):int(h*0.7), int(w*0.2):int(w*0.8)]
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    except: return None

def compare_signatures(sig1, sig2):
    """Compare two color histograms (0.0 to 1.0)"""
    if sig1 is None or sig2 is None: return 0
    return cv2.compareHist(sig1, sig2, cv2.HISTCMP_CORREL)

status_message = ""
status_expiry = 0
camera_available = False
activity_pause_notice = ""
activity_timing_paused = False
multi_person_count = 0
daily_stats_day = str(date.today())
last_daily_summary_pinned_date = None
last_daily_summary_message_id = None
latest_daily_summary_text = ""
latest_daily_summary_date = str(date.today())
last_activity_summary_message_id = None
last_activity_summary_text = ""
last_activity_summary_sent_at = 0.0
activity_summary_dirty = True

def load_stats_from_db():
    global walking_time, standing_time, sleeping_time, sitting_time, all_tracked_people
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        today = str(date.today())
        c.execute("SELECT person_id, walking, standing, sitting, sleeping FROM activity WHERE date=?", (today,))
        rows = c.fetchall()
        with data_lock:
            for r in rows:
                pid, w, st, s, sl = r
                person_key = resolve_person_key(pid)
                walking_time[person_key] += w
                standing_time[person_key] += st
                sitting_time[person_key] += s
                sleeping_time[person_key] += sl
                all_tracked_people.add(person_key)
        conn.close()
        print(f"Loaded stats for {len(rows)} people from database.")
    except Exception as e:
        print(f"Error loading stats: {e}")

load_stats_from_db()

def load_fall_history():
    global fall_events
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT timestamp, person_id, type, unix_timestamp FROM falls ORDER BY unix_timestamp DESC LIMIT 50")
        rows = c.fetchall()
        for r in rows:
            ts, pid, ftype, uts = r
            
            # Robust timestamp parsing
            if isinstance(ts, str):
                try:
                    # SQLite datetime.now() usually looks like '2026-01-28 14:25:10.123456'
                    # or '2026-01-28 14:25:10'
                    dt_obj = datetime.strptime(ts.split('.')[0], "%Y-%m-%d %H:%M:%S")
                    display_time = dt_obj.strftime("%H:%M:%S")
                    derived_uts = dt_obj.timestamp()
                except:
                    display_time = ts # Fallback
                    derived_uts = uts if uts else time.time()
            else:
                display_time = ts.strftime("%H:%M:%S")
                derived_uts = uts if uts else ts.timestamp()

            person_key = resolve_person_key(pid)
            with data_lock:
                fall_events.append({
                    "person": get_activity_label(person_key),
                    "type": ftype,
                    "timestamp": uts if uts else derived_uts,
                    "time_str": display_time
                })
        # Keep them in chronological order for the list (appends happened in reverse order from SELECT)
        with data_lock:
            fall_events.sort(key=lambda x: x['timestamp'])
        conn.close()
        print(f"Loaded {len(fall_events)} fall events from history.")
    except Exception as e:
        print(f"Error loading fall history: {e}")

load_fall_history()

# Start DB sync thread (Wait for all initializations to complete)
threading.Thread(target=log_activity_to_db, daemon=True).start()

@app.route("/trigger", methods=["POST"])
def trigger():
    global fall, active_alerts
    data = request.get_json(silent=True) or {}
    person_id = data.get("person_id")
    msg = data.get("message", "Unknown")
    fall_type = data.get("type", "FALL")
    
    if not person_id:
        # Backward compatibility or fallback
        person_id = msg.split(' (')[0] if ' (' in msg else msg

    now = time.time()
    
    with data_lock:
        # Update existing alert for this person to prevent duplicates (e.g. Minor -> Major)
        found = False
        for alert in active_alerts:
            # Match by the clean person_id
            if str(alert['person_id']) == str(person_id):
                alert['type'] = fall_type
                alert['message'] = msg
                alert['time_str'] = time.strftime("%H:%M:%S", time.localtime(now))
                alert['timestamp'] = now
                found = True
                break
        
        if not found:
            active_alerts.append({
                "person_id": person_id,
                "message": msg,
                "type": fall_type,
                "time_str": time.strftime("%H:%M:%S", time.localtime(now)),
                "timestamp": now
            })
    
    fall = True
    return "OK"

@app.route("/api/acknowledge/<pid>", methods=["POST"])
def acknowledge(pid):
    global active_alerts
    with data_lock:
        # Match by person_id for precise removal
        active_alerts = [a for a in active_alerts if str(a['person_id']) != str(pid) and str(a['message']) != str(pid)]
    return jsonify({"status": "success"})

@app.route("/fall")
def check():
    with data_lock:
        is_fall = len(active_alerts) > 0
    return jsonify({"fall": is_fall})

last_frame = None

def rename_person(old_id, new_name):
    """Safely rename a person and transfer all their stats and mappings."""
    with data_lock:
        _rename_person_internal(old_id, new_name)

def _rename_person_internal(old_id, new_name):
    # Resolve to the stable internal person key if old_id is already a display name.
    target_persistent_id = resolve_person_key(old_id)

    manual_id_map[str(target_persistent_id)] = new_name

    # Keep activity counters on the persistent key; only merge legacy labels if they exist.
    for legacy_id in {str(old_id), str(new_name)}:
        if legacy_id != str(target_persistent_id):
            walking_time[target_persistent_id] += walking_time.pop(legacy_id, 0)
            standing_time[target_persistent_id] += standing_time.pop(legacy_id, 0)
            sitting_time[target_persistent_id] += sitting_time.pop(legacy_id, 0)
            sleeping_time[target_persistent_id] += sleeping_time.pop(legacy_id, 0)

    all_tracked_people.add(str(target_persistent_id))
    if str(old_id) != str(target_persistent_id):
        all_tracked_people.discard(str(old_id))
    all_tracked_people.discard(str(new_name))
    
    # Update active alerts and history in memory
    for alert in active_alerts:
        if str(alert.get('person_id')) == str(old_id) or str(alert.get('person_id')) == str(target_persistent_id):
            alert['person_id'] = new_name
            alert['message'] = alert['message'].replace(str(old_id), new_name).replace(str(target_persistent_id), new_name)
            
    for event in fall_events:
        if str(event.get('person')) == str(old_id) or str(event.get('person')) == str(target_persistent_id):
            event['person'] = new_name

    # Save mappings AND bank immediately to ensure persistence
    save_manual_id_map()
    reid_manager.save_bank()
    
    # Update database: Delete old entries (stats were transferred to new name)
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("DELETE FROM activity WHERE person_id=?", (str(old_id),))
        if target_persistent_id != old_id:
            c.execute("DELETE FROM activity WHERE person_id=?", (str(target_persistent_id),))
        
        # Also update fall history to link to the new name
        c.execute("UPDATE falls SET person_id=? WHERE person_id=?", (new_name, str(old_id)))
        if target_persistent_id != old_id:
            c.execute("UPDATE falls SET person_id=? WHERE person_id=?", (new_name, str(target_persistent_id)))
            
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error updating DB for rename: {e}")
        
    print(f"👤 Renamed {old_id} (ID: {target_persistent_id}) to {new_name} and saved.")

@app.route("/register", methods=["POST"])
def register():
    global last_frame, status_message, status_expiry
    name = request.form.get("name")
    target_id = request.form.get("yolo_id") # Register by active ID (can be persistent_id)
    
    if not name:
        return jsonify({"status": "error", "message": "Missing name"})
    
    # CASE 1: Manual ID naming (No face needed)
    with data_lock:
        if target_id and (str(target_id) in person_state or str(target_id) in all_tracked_people):
            _rename_person_internal(str(target_id), name)
            status_message = f"ID {target_id} is now {name}"
            status_expiry = time.time() + 5
            return jsonify({"status": "success", "message": f"Successfully named body {target_id} as {name}"})

        # NEW: Fallback for Navbar button - If only one person is visible or unnamed, name them
        if not target_id:
            unnamed = [pid for pid in person_state if pid not in manual_id_map]
            if len(unnamed) == 1:
                target_id = unnamed[0]
                _rename_person_internal(str(target_id), name)
                status_message = f"Auto-linked {name} to ID {target_id}"
                status_expiry = time.time() + 5
                return jsonify({"status": "success", "message": f"Successfully named visible body {target_id} as {name}"})
            elif len(unnamed) > 1:
                 return jsonify({"status": "error", "message": "Multiple unnamed bodies. Click the ID button next to the body instead."})

    # CASE 2: Uploaded files or Live frame
    front_img = request.files.get("front")
    back_img = request.files.get("back")
    
    to_process = []
    if front_img: to_process.append(front_img)
    if back_img: to_process.append(back_img)
    
    if not to_process:
        if last_frame is not None:
            to_process = [last_frame]
        else:
            return jsonify({"status": "error", "message": "No photos uploaded or live frame available"})
    
    if register_face(to_process, name, yolo_model=model if not (front_img or back_img) else None):
        status_message = f"Registered: {name}"
        status_expiry = time.time() + 5
        return jsonify({"status": "success", "message": f"Successfully registered {name}"})
    return jsonify({"status": "error", "message": "No face detected in provided images"})

@app.route("/api/history")
def activity_history():
    """Get history for charts"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if SINGLE_PERSON_MODE:
        c.execute("""SELECT date, ? as person_id, SUM(walking), SUM(standing), SUM(sitting), SUM(sleeping)
                     FROM activity
                     GROUP BY date
                     ORDER BY date DESC LIMIT 50""", (SINGLE_PERSON_LABEL,))
    else:
        c.execute("SELECT date, person_id, walking, standing, sitting, sleeping FROM activity ORDER BY date DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    return jsonify([
        {
            "date": r[0],
            "pid": get_activity_label(r[1]),
            "walk": r[2],
            "stand": r[3],
            "sit": r[4],
            "sleep": r[5]
        }
        for r in rows
    ])

@app.route("/api/history/monthly")
def monthly_history():
    """Get monthly aggregated history for charts"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if SINGLE_PERSON_MODE:
        c.execute("""SELECT SUBSTR(date, 1, 7) as month, ? as person_id, SUM(walking), SUM(standing), SUM(sitting), SUM(sleeping) 
                     FROM activity
                     GROUP BY month
                     ORDER BY month DESC LIMIT 24""", (SINGLE_PERSON_LABEL,))
    else:
        c.execute("""SELECT SUBSTR(date, 1, 7) as month, person_id, SUM(walking), SUM(standing), SUM(sitting), SUM(sleeping) 
                     FROM activity 
                     GROUP BY month, person_id 
                     ORDER BY month DESC LIMIT 24""")
    rows = c.fetchall()
    conn.close()
    return jsonify([
        {
            "date": r[0],
            "pid": get_activity_label(r[1]),
            "walk": r[2],
            "stand": r[3],
            "sit": r[4],
            "sleep": r[5]
        }
        for r in rows
    ])

@app.route("/api/settings", methods=["GET", "POST"])
def api_settings():
    if request.method == "POST":
        try:
            payload = request.get_json(silent=True) or {}
            update_settings_from_payload(payload)
            return jsonify({"status": "success", "settings": get_public_settings()})
        except Exception as e:
            add_system_event(f"Settings API error: {e}", level="error")
            return jsonify({"status": "error", "message": str(e)}), 500
    return jsonify(get_public_settings())

@app.route("/api/settings/test-telegram", methods=["POST"])
def api_test_telegram():
    ok, message = send_telegram_message("Test alert from ElderlyCare Pro settings page.", category="test", force=True)
    status = "success" if ok else "error"
    return jsonify({"status": status, "message": message})

@app.route("/api/events")
def api_events():
    with data_lock:
        events = list(reversed(system_events[-30:]))
    return jsonify(events)

@app.route("/api/events/clear", methods=["POST"])
def api_clear_events():
    with data_lock:
        system_events.clear()
    add_system_event("System events cleared")
    return jsonify({"status": "success"})

def generate_video_stream():
    last_sent_frame_id = -1
    while True:
        if not use_dashboard_stream():
            time.sleep(0.25)
            continue
        with data_lock:
            frame_bytes = latest_stream_frame
            frame_id = latest_stream_frame_id
        if frame_bytes is None or frame_id == last_sent_frame_id:
            time.sleep(0.01)
            continue
        last_sent_frame_id = frame_id
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/video_feed")
def video_feed():
    return Response(
        generate_video_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache"
        }
    )

@app.route("/api/camera-status")
def api_camera_status():
    with data_lock:
        has_frame = latest_stream_frame is not None
    return jsonify({
        "camera_available": camera_available,
        "has_frame": has_frame,
        "camera_active": camera_active,
        "video_output_mode": settings.get("video_output_mode", DEFAULT_SETTINGS["video_output_mode"]),
        "system_sleeping": system_sleeping,
        "multiple_people_detected": activity_timing_paused,
        "activity_timing_paused": activity_timing_paused,
        "activity_pause_notice": activity_pause_notice,
        "multi_person_count": multi_person_count
    })

@app.route("/api/camera/close", methods=["POST"])
def api_camera_close():
    """Close the camera"""
    global camera_active, cap
    camera_active = False
    if cap is not None:
        cap.release()
        cap = None
    clear_stream_frame()
    add_system_event("Camera closed via API", level="info")
    return jsonify({"status": "success", "camera_active": camera_active})

@app.route("/api/camera/open", methods=["POST"])
def api_camera_open():
    """Open/Restart the camera"""
    global camera_active, cap, camera_available
    if not camera_active or cap is None:
        cap = open_camera()
        camera_available = cap is not None
        camera_active = True
        if camera_available:
            add_system_event("Camera opened via API", level="info")
        else:
            add_system_event("Failed to open camera via API", level="warning")
    return jsonify({"status": "success", "camera_active": camera_active, "camera_available": camera_available})

@app.route("/api/node-heartbeat", methods=["POST"])
def api_node_heartbeat():
    data = request.get_json(silent=True) or {}
    node_id = str(data.get("node_id", "")).strip()
    if not node_id:
        return jsonify({"status": "error", "message": "Missing node_id"}), 400

    with data_lock:
        remote_nodes[node_id] = {
            "node_id": node_id,
            "camera_available": bool(data.get("camera_available", False)),
            "has_frame": bool(data.get("has_frame", False)),
            "deployment_mode": str(data.get("deployment_mode", "edge")),
            "address": request.remote_addr,
            "last_seen": time.time(),
            "time_str": time.strftime("%H:%M:%S", time.localtime())
        }
    return jsonify({"status": "success"})

@app.route("/api/edge/report", methods=["POST"])
def api_edge_report():
    data = request.get_json(silent=True) or {}
    node_id = str(data.get("node_id", "")).strip()
    if not node_id:
        return jsonify({"status": "error", "message": "Missing node_id"}), 400

    now = time.time()
    with data_lock:
        remote_edge_reports[node_id] = {
            "node_id": node_id,
            "people": list(data.get("people", [])),
            "falls": list(data.get("falls", [])),
            "active_alerts": list(data.get("active_alerts", [])),
            "unnamed_ids": list(data.get("unnamed_ids", [])),
            "updated_at": now
        }
        remote_nodes[node_id] = {
            "node_id": node_id,
            "camera_available": bool(data.get("camera_available", False)),
            "has_frame": bool(data.get("has_frame", False)),
            "deployment_mode": "edge",
            "address": request.remote_addr,
            "last_seen": now,
            "time_str": time.strftime("%H:%M:%S", time.localtime(now))
        }
    return jsonify({"status": "success"})

@app.route("/api/nodes")
def api_nodes():
    now = time.time()
    with data_lock:
        nodes = [{
            **node,
            "seconds_since_seen": round(now - node["last_seen"], 1)
        } for node in remote_nodes.values()]
    nodes.sort(key=lambda item: item["node_id"])
    return jsonify(nodes)

@app.route("/settings")
def settings_page():
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>System Settings</title>
        <link rel="icon" href="/favicon.svg" type="image/svg+xml">
        <link rel="shortcut icon" href="/favicon.svg" type="image/svg+xml">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {
                --bg: #070b1c;
                --panel: #0d142b;
                --panel-border: #1b2442;
                --text: #f1f5f9;
                --muted: #94a3b8;
                --accent: #1d4ed8;
                --accent-soft: #123454;
                --success: #16a34a;
            }
            body { background: radial-gradient(circle at top left, #0e1533, var(--bg) 45%); color: var(--text); min-height: 100vh; font-family: 'Segoe UI', sans-serif; }
            .shell { max-width: 1240px; margin: 0 auto; padding: 28px 22px 44px; }
            .title { font-size: 2.2rem; font-weight: 800; }
            .subtitle { color: var(--muted); margin-bottom: 28px; }
            .panel { background: rgba(13, 20, 43, 0.95); border: 1px solid var(--panel-border); border-radius: 28px; box-shadow: 0 20px 60px rgba(0,0,0,0.25); }
            .main-panel { padding: 22px; }
            .side-panel { padding: 22px; }
            .form-label { color: #cbd5e1; font-weight: 600; margin-bottom: 8px; }
            .form-control, .form-select { background: #0a1024; border: 1px solid #202b4a; color: var(--text); border-radius: 16px; padding: 14px 16px; }
            .form-control:focus, .form-select:focus { background: #0a1024; color: var(--text); border-color: #3558c8; box-shadow: 0 0 0 .2rem rgba(29,78,216,.2); }
            .setting-check { background: #0a1024; border: 1px solid #202b4a; border-radius: 18px; padding: 16px 18px; }
            .btn-primary { background: linear-gradient(135deg, #18498e, #10365f); border: none; border-radius: 16px; padding: 14px 24px; font-weight: 700; }
            .btn-outline-light { border-color: #253457; color: #e2e8f0; border-radius: 16px; }
            .tool-btn { width: 100%; text-align: left; margin-bottom: 12px; padding: 14px 16px; background: #0a1024; border: 1px solid #202b4a; color: var(--text); border-radius: 16px; }
            .events-box { background: #0a1024; border: 1px solid #202b4a; border-radius: 18px; min-height: 260px; max-height: 340px; overflow-y: auto; padding: 14px; }
            .event-item { border-bottom: 1px solid #1b2442; padding: 10px 0; }
            .event-item:last-child { border-bottom: none; }
            .event-time { color: var(--muted); font-size: 0.8rem; }
            .helper { color: var(--muted); font-size: 0.92rem; }
            .status-pill { display:inline-block; padding: 6px 12px; border-radius: 999px; background: rgba(22,163,74,0.15); color: #86efac; font-size: 0.82rem; }
            a.top-link { color: #bfdbfe; text-decoration: none; }
        </style>
    </head>
    <body>
        <div class="shell">
            <div class="d-flex justify-content-between align-items-start flex-wrap gap-3 mb-4">
                <div>
                    <div class="title">System Settings</div>
                    <div class="subtitle">Configuration for alerts, detection, and wellness monitoring.</div>
                </div>
                <div class="d-flex gap-2 align-items-center">
                    <span id="save-state" class="status-pill">Ready</span>
                    <a class="top-link" href="/">Back to Dashboard</a>
                </div>
            </div>

            <div class="row g-4">
                <div class="col-lg-8">
                    <div class="panel main-panel">
                        <div class="row g-3">
                            <div class="col-md-6">
                                <label class="form-label">Bot Token</label>
                                <input id="bot_token" class="form-control" type="password" placeholder="Telegram bot token">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Chat ID</label>
                                <input id="chat_id" class="form-control" type="text" placeholder="Telegram chat ID">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Telegram Primary Resident</label>
                                <input id="telegram_primary_person" class="form-control" type="text" placeholder="e.g. Resident, 1, John">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Message Cooldown (sec)</label>
                                <input id="message_cooldown_sec" class="form-control" type="number" min="0">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Fall Confirm Window (sec)</label>
                                <input id="fall_confirm_window_sec" class="form-control" type="number" min="1" step="0.5">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Preferred Camera</label>
                                <select id="preferred_camera" class="form-select">
                                    <option value="0">Auto / Camera 0</option>
                                    <option value="1">Camera 1</option>
                                    <option value="2">Camera 2</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Max People to Track</label>
                                <input id="max_people_to_track" class="form-control" type="number" min="1" max="10">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Deployment Mode</label>
                                <select id="deployment_mode" class="form-select">
                                    <option value="server">Server</option>
                                    <option value="edge">Edge</option>
                                    <option value="standalone">Standalone</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Video Output</label>
                                <select id="video_output_mode" class="form-select">
                                    <option value="both">Window + Dashboard Preview</option>
                                    <option value="window">Low-Latency Window</option>
                                    <option value="dashboard">Dashboard Feed</option>
                                </select>
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Node ID</label>
                                <input id="node_id" class="form-control" type="text" placeholder="living-room-laptop">
                            </div>
                            <div class="col-md-6">
                                <label class="form-label">Central Server URL</label>
                                <input id="central_server_url" class="form-control" type="text" placeholder="http://192.168.1.10:5000">
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Server Bind Host</label>
                                <input id="server_bind_host" class="form-control" type="text" placeholder="0.0.0.0">
                            </div>
                            <div class="col-md-3">
                                <label class="form-label">Server Port</label>
                                <input id="server_port" class="form-control" type="number" min="1" max="65535">
                            </div>
                            <div class="col-md-6">
                                <label class="setting-check d-flex align-items-center gap-2">
                                    <input id="enable_telegram" class="form-check-input mt-0" type="checkbox">
                                    <span>Enable Telegram</span>
                                </label>
                            </div>
                            <div class="col-md-6">
                                <label class="setting-check d-flex align-items-center gap-2">
                                    <input id="enable_detection" class="form-check-input mt-0" type="checkbox">
                                    <span>Enable Detection</span>
                                </label>
                            </div>
                            <div class="col-md-6">
                                <label class="setting-check d-flex align-items-center gap-2">
                                    <input id="enable_voice_alert" class="form-check-input mt-0" type="checkbox">
                                    <span>Enable Voice Alert</span>
                                </label>
                            </div>
                            <div class="col-md-6">
                                <label class="setting-check d-flex align-items-center gap-2">
                                    <input id="enable_wellness_monitoring" class="form-check-input mt-0" type="checkbox">
                                    <span>Enable Wellness Monitoring</span>
                                </label>
                            </div>
                            <div class="col-12">
                                <label class="setting-check d-flex align-items-center gap-2">
                                    <input id="display_metrics_overlay" class="form-check-input mt-0" type="checkbox">
                                    <span>Display Metrics Overlay (on video)</span>
                                </label>
                            </div>
                            <div class="col-12 d-flex align-items-center gap-3 pt-2">
                                <button class="btn btn-primary" onclick="saveSettings()">Save Settings</button>
                                <span class="helper">Use BotFather to create the bot, then paste your bot token and target chat ID here.</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="col-lg-4">
                    <div class="panel side-panel mb-4">
                        <h4 class="mb-3">Camera Control</h4>
                        <button class="tool-btn" onclick="openCamera()"><i class="fas fa-video me-2"></i>Open Camera</button>
                        <button class="tool-btn" onclick="closeCamera()"><i class="fas fa-video-slash me-2"></i>Close Camera</button>
                        <div id="camera-status" class="mt-2 text-muted small"></div>
                    </div>
                    <div class="panel side-panel mb-4">
                        <h4 class="mb-3">Tools</h4>
                        <button class="tool-btn" onclick="sendTestAlert()"><i class="fas fa-paper-plane me-2"></i>Send Test Telegram Alert</button>
                        <button class="tool-btn" onclick="clearLogs()"><i class="fas fa-trash me-2"></i>Clear Logs</button>
                    </div>
                    <div class="panel side-panel">
                        <h4 class="mb-3">System Events</h4>
                        <div id="events-box" class="events-box"><div class="helper">No events yet.</div></div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            const fields = [
                'bot_token', 'chat_id', 'telegram_primary_person', 'message_cooldown_sec', 'fall_confirm_window_sec',
                'preferred_camera', 'max_people_to_track', 'deployment_mode', 'video_output_mode', 'node_id',
                'central_server_url', 'server_bind_host', 'server_port', 'enable_telegram',
                'enable_detection', 'enable_voice_alert', 'enable_wellness_monitoring',
                'display_metrics_overlay'
            ];

            function setSaveState(text, good=true) {
                const el = document.getElementById('save-state');
                el.textContent = text;
                el.style.background = good ? 'rgba(22,163,74,0.15)' : 'rgba(220,38,38,0.15)';
                el.style.color = good ? '#86efac' : '#fca5a5';
            }

            function applySettings(data) {
                fields.forEach(id => {
                    const el = document.getElementById(id);
                    if (!el) return;
                    if (el.type === 'checkbox') el.checked = !!data[id];
                    else el.value = data[id] ?? '';
                });
            }

            async function loadSettings() {
                try {
                    setSaveState('Loading...');
                    const res = await fetch('/api/settings');
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.message || 'Failed to load settings');
                    applySettings(data);
                    setSaveState('Loaded');
                } catch (err) {
                    console.error(err);
                    setSaveState('Load Failed', false);
                    alert(`Could not load settings: ${err.message}`);
                }
            }

            function collectSettings() {
                const payload = {};
                fields.forEach(id => {
                    const el = document.getElementById(id);
                    payload[id] = el.type === 'checkbox' ? el.checked : el.value;
                });
                return payload;
            }

            async function saveSettings() {
                try {
                    setSaveState('Saving...');
                    const res = await fetch('/api/settings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify(collectSettings())
                    });
                    const data = await res.json();
                    if (!res.ok || data.status !== 'success') {
                        throw new Error(data.message || 'Failed to save settings');
                    }
                    applySettings(data.settings || collectSettings());
                    setSaveState('Saved');
                    refreshEvents();
                } catch (err) {
                    console.error(err);
                    setSaveState('Save Failed', false);
                    alert(`Could not save settings: ${err.message}`);
                }
            }

            async function sendTestAlert() {
                try {
                    const res = await fetch('/api/settings/test-telegram', { method: 'POST' });
                    const data = await res.json();
                    if (!res.ok) throw new Error(data.message || 'Test alert failed');
                    alert(data.message);
                    refreshEvents();
                } catch (err) {
                    console.error(err);
                    alert(`Could not send test alert: ${err.message}`);
                }
            }

            async function closeCamera() {
                try {
                    const res = await fetch('/api/camera/close', { method: 'POST' });
                    const data = await res.json();
                    document.getElementById('camera-status').textContent = 'Camera closed';
                    refreshEvents();
                } catch (err) {
                    console.error(err);
                }
            }

            async function openCamera() {
                try {
                    const res = await fetch('/api/camera/open', { method: 'POST' });
                    const data = await res.json();
                    document.getElementById('camera-status').textContent = data.camera_available ? 'Camera opened' : 'Failed to open camera';
                    refreshEvents();
                } catch (err) {
                    console.error(err);
                }
            }

            async function clearLogs() {
                try {
                    await fetch('/api/events/clear', { method: 'POST' });
                    refreshEvents();
                } catch (err) {
                    console.error(err);
                }
            }

            async function refreshEvents() {
                const box = document.getElementById('events-box');
                try {
                    const res = await fetch('/api/events');
                    const events = await res.json();
                    if (!res.ok) throw new Error('Failed to load events');
                    if (!events.length) {
                        box.innerHTML = '<div class="helper">No events yet.</div>';
                        return;
                    }
                    box.innerHTML = events.map(e => `
                        <div class="event-item">
                            <div>${e.message}</div>
                            <div class="event-time">${e.time_str} • ${e.level}</div>
                        </div>
                    `).join('');
                } catch (err) {
                    console.error(err);
                    box.innerHTML = '<div class="helper">Unable to load events.</div>';
                }
            }

            loadSettings();
            refreshEvents();
            setInterval(refreshEvents, 5000);
        </script>
    </body>
    </html>
    """
    return html

@app.route("/")
def home():
    """Enhanced modern dashboard with Monthly Analytics and Fall Messaging"""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Elderly Monitor Pro</title>
        <link rel="icon" href="/favicon.svg" type="image/svg+xml">
        <link rel="shortcut icon" href="/favicon.svg" type="image/svg+xml">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            :root {
                --primary-color: #4361ee;
                --secondary-color: #3f37c9;
                --danger-color: #ef233c;
                --success-color: #2ecc71;
                --warning-color: #f39c12;
                --app-bg: #f0f2f5;
                --app-text: #1f2937;
                --nav-bg: #ffffff;
                --card-bg: #ffffff;
                --sidebar-bg: #f8f9fa;
                --surface: #ffffff;
                --surface-alt: #f8f9fa;
                --border-color: #e0e0e0;
                --muted-text: #666666;
                --shadow-color: rgba(0,0,0,0.05);
                --chart-grid: #f0f0f0;
                --chart-tooltip-bg: #1e1e2f;
            }
            body {
                background-color: var(--app-bg);
                font-family: 'Inter', sans-serif;
                color: var(--app-text);
                transition: background-color 0.25s ease, color 0.25s ease;
            }
            body[data-theme="dark"] {
                --app-bg: #0b1020;
                --app-text: #e5e7eb;
                --nav-bg: #111827;
                --card-bg: #111827;
                --sidebar-bg: #0f172a;
                --surface: #111827;
                --surface-alt: #0b1220;
                --border-color: #24324a;
                --muted-text: #9ca3af;
                --shadow-color: rgba(0,0,0,0.35);
                --chart-grid: #24324a;
                --chart-tooltip-bg: #0f172a;
            }
            .navbar { background: var(--nav-bg); border-bottom: 1px solid var(--border-color); box-shadow: 0 2px 4px var(--shadow-color); }
            .sidebar { background: var(--sidebar-bg); border-right: 1px solid var(--border-color); min-height: 100vh; padding: 20px; }
            .card { border: none; border-radius: 12px; box-shadow: 0 4px 6px var(--shadow-color); margin-bottom: 24px; transition: transform 0.2s, background-color 0.25s ease, color 0.25s ease; background: var(--card-bg); color: var(--app-text); }
            .card:hover { transform: translateY(-2px); }
            .status-badge { padding: 6px 12px; border-radius: 8px; font-weight: 600; font-size: 0.8rem; }
            
            .bg-walking { background-color: #d1fae5; color: #065f46; }
            .bg-standing { background-color: #dbeafe; color: #1e40af; }
            .bg-sleeping { background-color: #f3e8ff; color: #5b21b6; }
            .bg-sitting { background-color: #fef3c7; color: #92400e; }
            .bg-major-fall { background-color: #fee2e2; color: #991b1b; animation: pulse 2s infinite; }
            .bg-minor-fall { background-color: #ffedd5; color: #9a3412; }
            .bg-recovered { background-color: #d1fae5; color: #065f46; border: 1px solid #10b981; }
            .bg-away { background-color: #f3f4f6; color: #4b5563; }
            
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(239, 35, 60, 0.4); }
                70% { box-shadow: 0 0 0 10px rgba(239, 35, 60, 0); }
                100% { box-shadow: 0 0 0 0 rgba(239, 35, 60, 0); }
            }

            .alert-item { 
                background: var(--card-bg); border-radius: 10px; padding: 16px; margin-bottom: 12px; color: var(--app-text);
                border-left: 6px solid var(--danger-color); box-shadow: 0 2px 8px rgba(239, 35, 60, 0.1);
            }
            .alert-item.recovered { border-left-color: var(--success-color); box-shadow: 0 2px 8px rgba(46, 204, 113, 0.1); }
            
            .stat-icon { width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; margin-bottom: 10px; }
            .icon-walk { background: #d1fae5; color: #10b981; }
            .icon-stand { background: #dbeafe; color: #3b82f6; }
            .icon-sit { background: #fef3c7; color: #f59e0b; }
            .icon-sleep { background: #f3e8ff; color: #8b5cf6; }

            #fall-message-display { position: fixed; top: 20px; right: 20px; z-index: 9999; width: 320px; }
            .toast-fall { 
                background: var(--danger-color); color: white; padding: 16px; border-radius: 8px; margin-bottom: 10px;
                box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); display: flex; align-items: center; animation: slideIn 0.3s ease-out;
            }
            @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
            
            .nav-link { color: var(--muted-text); padding: 10px 15px; border-radius: 8px; margin-bottom: 5px; font-weight: 500; }
            .nav-link:hover, .nav-link.active { background: var(--surface); color: var(--primary-color); box-shadow: 0 2px 4px var(--shadow-color); }
            
            .chart-container { height: 300px; }
            .chart-container-lg { height: 320px; }
            .chart-container-sm { height: 320px; }
            .video-shell { background: linear-gradient(135deg, #101827, #0f172a); border-radius: 16px; overflow: hidden; min-height: 320px; position: relative; }
            .video-feed { width: 100%; height: 100%; min-height: 320px; object-fit: cover; display: block; background: #0b1220; }
            .video-empty { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: #cbd5e1; font-weight: 600; background: radial-gradient(circle at top, rgba(67,97,238,0.12), rgba(15,23,42,0.98)); }
            .video-meta { position: absolute; left: 16px; bottom: 16px; background: rgba(15,23,42,0.72); color: white; padding: 8px 12px; border-radius: 999px; font-size: 0.85rem; }
            .summary-strip { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }
            .summary-pill { background: var(--surface-alt); border-radius: 14px; padding: 14px; color: var(--app-text); }
            .summary-label { font-size: 0.78rem; text-transform: uppercase; color: var(--muted-text); font-weight: 700; letter-spacing: .04em; }
            .summary-value { font-size: 1rem; font-weight: 800; margin-top: 6px; }
            .recommend-list { margin: 0; padding-left: 18px; color: var(--app-text); }
            .recommend-list li { margin-bottom: 8px; }
            .recommend-item { list-style: none; padding: 12px 14px; border-radius: 12px; border: 1px solid var(--border-color); background: var(--surface-alt); }
            .recommend-high { border-left: 5px solid #dc2626; }
            .recommend-medium { border-left: 5px solid #d97706; }
            .recommend-low { border-left: 5px solid #2563eb; }
            .recommend-info { border-left: 5px solid #0ea5e9; }
            body[data-theme="dark"] .bg-white,
            body[data-theme="dark"] .card-header,
            body[data-theme="dark"] .dropdown-menu,
            body[data-theme="dark"] .bg-light,
            body[data-theme="dark"] .bg-light-subtle,
            body[data-theme="dark"] .list-group-item,
            body[data-theme="dark"] .form-control,
            body[data-theme="dark"] .form-select,
            body[data-theme="dark"] .input-group-text {
                background-color: var(--surface) !important;
                color: var(--app-text) !important;
                border-color: var(--border-color) !important;
            }
            body[data-theme="dark"] .bg-light,
            body[data-theme="dark"] .bg-light-subtle {
                background-color: var(--surface-alt) !important;
            }
            body[data-theme="dark"] .text-dark,
            body[data-theme="dark"] .text-muted,
            body[data-theme="dark"] .small,
            body[data-theme="dark"] .fw-normal {
                color: var(--muted-text) !important;
            }
            body[data-theme="dark"] .border,
            body[data-theme="dark"] .border-bottom {
                border-color: var(--border-color) !important;
            }
            body[data-theme="dark"] .btn-outline-dark {
                color: var(--app-text);
                border-color: var(--border-color);
            }
            body[data-theme="dark"] .btn-outline-dark:hover {
                background: var(--surface-alt);
                color: var(--app-text);
            }
            body[data-theme="dark"] .btn-light {
                background: var(--surface-alt);
                color: var(--app-text);
                border-color: var(--border-color);
            }
            body[data-theme="dark"] .navbar .btn-outline-secondary {
                color: var(--app-text);
                border-color: var(--border-color);
            }
            body[data-theme="dark"] .navbar .btn-outline-secondary:hover {
                background: var(--surface-alt);
            }
        </style>
    </head>
    <body>
        <div id="fall-message-display"></div>

        <nav class="navbar px-4 py-2 sticky-top">
            <div class="container-fluid">
                <a class="navbar-brand fw-bold d-flex align-items-center" href="#">
                    <div class="bg-primary text-white p-2 rounded-3 me-2" style="width: 35px; height: 35px; display: flex; align-items: center; justify-content: center;">
                        <i class="fas fa-heartbeat"></i>
                    </div>
                    ElderlyCare <span class="text-primary ms-1">Pro</span>
                </a>
                <div class="d-flex align-items-center gap-3">
                    <a class="btn btn-outline-primary btn-sm px-3" href="/annotate-video">
                        <i class="fas fa-film me-2"></i>Upload Video
                    </a>
                    <a class="btn btn-outline-primary btn-sm px-3" href="/reports">
                        <i class="fas fa-file-export me-2"></i>Reports
                    </a>
                    <button id="theme-toggle" class="btn btn-outline-secondary btn-sm px-3" type="button" onclick="toggleTheme()">
                        <i class="fas fa-moon me-2"></i> Dark Mode
                    </button>
                    <div class="input-group input-group-sm shadow-sm" style="width: 250px;">
                        <input type="text" id="reg-name" class="form-control border-0 px-3" placeholder="Register Name">
                        <button class="btn btn-primary px-3" onclick="registerPerson()">
                            <i class="fas fa-user-plus"></i>
                        </button>
                    </div>
                    <div id="live-time" class="badge bg-light text-dark border p-2 fw-normal"></div>
                </div>
            </div>
        </nav>

        <div class="container-fluid">
            <div class="row">
                <!-- Sidebar -->
                <div class="col-md-2 d-none d-md-block sidebar">
                    <div class="nav flex-column">
                        <a href="/" class="nav-link active"><i class="fas fa-chart-line me-2"></i> Dashboard</a>
                        <a href="#fall-history" class="nav-link"><i class="fas fa-history me-2"></i> Event History</a>
                        <a href="#people-grid" class="nav-link"><i class="fas fa-users me-2"></i> Managed People</a>
                        <a href="/settings" class="nav-link"><i class="fas fa-cog me-2"></i> Settings</a>
                    </div>
                    <hr>
                    <div class="p-3 bg-white rounded-3 shadow-sm mt-4">
                        <div class="small text-muted mb-2">System Status</div>
                        <div class="d-flex align-items-center">
                            <div class="spinner-grow spinner-grow-sm text-success me-2"></div>
                            <span class="small fw-bold">Live Monitoring</span>
                        </div>
                    </div>
                </div>

                <!-- Main Content -->
                <div class="col-md-10 p-4">
                    <div id="alert-container"></div>

                    <div class="row g-4">
                        <!-- Activity Grid -->
                        <div class="col-lg-8">
                            <div class="d-flex justify-content-between align-items-center mb-4">
                                <h5 class="fw-bold mb-0">Daily Activity Tracker</h5>
                                <div id="unnamed-container" class="d-none">
                                     <div id="unnamed-list" class="d-flex gap-2"></div>
                                </div>
                            </div>

                            <div class="card p-3 mb-4">
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <div>
                                        <h5 class="fw-bold mb-0">Live Camera Feed</h5>
                                        <p class="text-muted small mb-0">Use low-latency window mode for the fastest live view, or dashboard mode if you need browser streaming.</p>
                                    </div>
                                </div>
                                <div id="activity-pause-banner" class="alert alert-warning py-2 px-3 small mb-3 d-none"></div>
                                <div class="video-shell">
                                    <img id="video-feed" class="video-feed d-none" alt="Live monitoring feed">
                                    <div id="video-empty" class="video-empty d-none">Camera feed unavailable</div>
                                    <div class="video-meta" id="video-meta">Connecting to camera...</div>
                                </div>
                            </div>

                            <div class="card p-4 mb-4">
                                <div class="d-flex justify-content-between align-items-center mb-3">
                                    <div>
                                        <h5 class="fw-bold mb-0">Daily Summary</h5>
                                        <p class="text-muted small mb-0">Saved activity totals and movement recommendations for today</p>
                                    </div>
                                    <div id="summary-date" class="text-muted small"></div>
                                </div>
                                <div id="summary-strip" class="summary-strip mb-4"></div>
                                <div class="card border-0 bg-light-subtle mb-4">
                                    <div class="card-body">
                                        <div class="d-flex justify-content-between align-items-center mb-2">
                                            <div class="fw-bold small text-uppercase text-muted">Pinned Telegram Summary</div>
                                            <span id="telegram-summary-badge" class="badge text-bg-secondary">Waiting</span>
                                        </div>
                                        <pre id="telegram-summary-text" class="mb-0 small text-wrap" style="white-space: pre-wrap;">Waiting for the latest pinned summary.</pre>
                                    </div>
                                </div>
                                <div>
                                    <h6 class="fw-bold mb-3">AI Recommendations</h6>
                                    <ul id="recommendations-list" class="recommend-list"></ul>
                                </div>
                            </div>
                            
                            <div id="people-grid" class="row"></div>

                            <!-- Analytics Section -->
                            <div class="card p-4">
                                <div class="d-flex justify-content-between align-items-center mb-4">
                                    <div>
                                        <h5 class="fw-bold mb-0" id="chart-title">Activity Analytics</h5>
                                <p class="text-muted small mb-0">Trend over time and breakdown by activity</p>
                                    </div>
                                    <div class="btn-group p-1 bg-light rounded-3">
                                        <button class="btn btn-sm px-3 rounded-2" id="btn-daily" onclick="setViewMode('daily')">Daily</button>
                                        <button class="btn btn-sm px-3 rounded-2" id="btn-monthly" onclick="setViewMode('monthly')">Monthly</button>
                                    </div>
                                </div>
                                <div class="row g-4">
                                    <div class="col-lg-8">
                                        <div class="chart-container chart-container-lg">
                                            <canvas id="activityTrendChart"></canvas>
                                        </div>
                                    </div>
                                    <div class="col-lg-4">
                                        <div class="chart-container chart-container-sm">
                                            <canvas id="activityBreakdownChart"></canvas>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- Side Panel: Fall Monitor -->
                        <div class="col-lg-4">
                            <h5 class="fw-bold mb-4">Fall Monitor Feed</h5>
                            <div class="card shadow-sm h-100">
                                <div class="card-header bg-white py-3">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <span class="small fw-bold text-uppercase text-muted letter-spacing-1">Recent Events</span>
                                        <i class="fas fa-bell text-muted"></i>
                                    </div>
                                </div>
                                <div class="card-body p-0" style="max-height: 700px; overflow-y: auto;">
                                    <div id="fall-history" class="list-group list-group-flush"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            let activityTrendChart = null;
            let activityBreakdownChart = null;
            let currentViewMode = 'daily';
            let seenAlerts = new Set();

            const THEME_STORAGE_KEY = 'elderly-monitor-theme';

            function getPreferredTheme() {
                const stored = localStorage.getItem(THEME_STORAGE_KEY);
                if (stored === 'light' || stored === 'dark') {
                    return stored;
                }
                return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
            }

            function updateThemeButton(theme) {
                const button = document.getElementById('theme-toggle');
                if (!button) return;
                button.innerHTML = theme === 'dark'
                    ? '<i class="fas fa-sun me-2"></i> Light Mode'
                    : '<i class="fas fa-moon me-2"></i> Dark Mode';
                button.setAttribute('aria-pressed', theme === 'dark' ? 'true' : 'false');
            }

            function applyTheme(theme) {
                const resolved = theme === 'dark' ? 'dark' : 'light';
                document.body.setAttribute('data-theme', resolved);
                localStorage.setItem(THEME_STORAGE_KEY, resolved);
                updateThemeButton(resolved);
                window.lastChartUpdate = 0;
                if (activityTrendChart) {
                    activityTrendChart.destroy();
                    activityTrendChart = null;
                }
                if (activityBreakdownChart) {
                    activityBreakdownChart.destroy();
                    activityBreakdownChart = null;
                }
            }

            function toggleTheme() {
                const current = document.body.getAttribute('data-theme') || 'light';
                applyTheme(current === 'dark' ? 'light' : 'dark');
                update();
            }

            function initTheme() {
                applyTheme(getPreferredTheme());
            }

            function getChartPalette() {
                const isDark = document.body.getAttribute('data-theme') === 'dark';
                return {
                    tooltipBg: isDark ? '#0f172a' : '#1e1e2f',
                    grid: isDark ? '#24324a' : '#f0f0f0',
                    text: isDark ? '#e5e7eb' : '#374151',
                    muted: isDark ? '#94a3b8' : '#6b7280'
                };
            }

            function setViewMode(mode) {
                currentViewMode = mode;
                document.getElementById('btn-daily').classList.toggle('bg-white', mode === 'daily');
                document.getElementById('btn-daily').classList.toggle('shadow-sm', mode === 'daily');
                document.getElementById('btn-monthly').classList.toggle('bg-white', mode === 'monthly');
                document.getElementById('btn-monthly').classList.toggle('shadow-sm', mode === 'monthly');
                document.getElementById('chart-title').textContent = mode === 'daily'
                    ? 'Daily Activity Analytics'
                    : 'Monthly Activity Analytics';
                window.lastChartUpdate = 0;
                update();
            }

            function registerPerson(yoloId = null) {
                const name = document.getElementById('reg-name').value;
                if (!name) { alert('Enter a name first'); return; }
                const formData = new FormData();
                formData.append('name', name);
                if (yoloId) formData.append('yolo_id', yoloId);
                fetch('/register', { method: 'POST', body: formData })
                    .then(r => r.json())
                    .then(data => {
                        if(data.status === 'success') {
                            document.getElementById('reg-name').value = '';
                        }
                        alert(data.message);
                    });
            }

            function groupHistoryByDate(data) {
                const buckets = {};
                data.forEach(row => {
                    const key = row.date;
                    if (!buckets[key]) {
                        buckets[key] = { walk: 0, stand: 0, sit: 0, sleep: 0 };
                    }
                    buckets[key].walk += Number(row.walk || 0);
                    buckets[key].stand += Number(row.stand || 0);
                    buckets[key].sit += Number(row.sit || 0);
                    buckets[key].sleep += Number(row.sleep || 0);
                });
                return Object.keys(buckets)
                    .sort()
                    .map(date => ({
                        date,
                        ...buckets[date],
                        total: buckets[date].walk + buckets[date].stand + buckets[date].sit + buckets[date].sleep
                    }));
            }

            function formatChartValue(seconds) {
                const num = Number(seconds || 0);
                return currentViewMode === 'monthly'
                    ? (num / 3600).toFixed(1)
                    : (num / 60).toFixed(0);
            }

            function initChart(data) {
                const grouped = groupHistoryByDate(data).reverse();
                const labels = grouped.map(row => row.date);
                const palette = getChartPalette();
                const colors = {
                    walk: '#10b981',
                    stand: '#3b82f6',
                    sit: '#f59e0b',
                    sleep: '#8b5cf6',
                    total: '#ef4444'
                };

                const useHours = currentViewMode === 'monthly';
                const unitLabel = useHours ? 'Hours' : 'Minutes';
                const scaleFactor = useHours ? 3600 : 60;
                const datasetValues = type => grouped.map(row => Number((row[type] / scaleFactor).toFixed(2)));
                const totalValues = grouped.map(row => Number((row.total / scaleFactor).toFixed(2)));
                const totalByType = {
                    walk: grouped.reduce((acc, row) => acc + row.walk, 0),
                    stand: grouped.reduce((acc, row) => acc + row.stand, 0),
                    sit: grouped.reduce((acc, row) => acc + row.sit, 0),
                    sleep: grouped.reduce((acc, row) => acc + row.sleep, 0)
                };

                if (activityTrendChart) activityTrendChart.destroy();
                if (activityBreakdownChart) activityBreakdownChart.destroy();

                const trendCtx = document.getElementById('activityTrendChart').getContext('2d');
                activityTrendChart = new Chart(trendCtx, {
                    type: 'bar',
                    data: {
                        labels,
                        datasets: [
                            {
                                label: 'Walking',
                                data: datasetValues('walk'),
                                backgroundColor: colors.walk,
                                borderColor: colors.walk,
                                borderWidth: 1,
                                stack: 'activity'
                            },
                            {
                                label: 'Standing',
                                data: datasetValues('stand'),
                                backgroundColor: colors.stand,
                                borderColor: colors.stand,
                                borderWidth: 1,
                                stack: 'activity'
                            },
                            {
                                label: 'Sitting',
                                data: datasetValues('sit'),
                                backgroundColor: colors.sit,
                                borderColor: colors.sit,
                                borderWidth: 1,
                                stack: 'activity'
                            },
                            {
                                label: 'Sleeping',
                                data: datasetValues('sleep'),
                                backgroundColor: colors.sleep,
                                borderColor: colors.sleep,
                                borderWidth: 1,
                                stack: 'activity'
                            },
                            {
                                label: 'Total',
                                data: totalValues,
                                type: 'line',
                                yAxisID: 'y1',
                                borderColor: colors.total,
                                backgroundColor: colors.total,
                                tension: 0.35,
                                pointRadius: 4,
                                pointHoverRadius: 6,
                                borderWidth: 3,
                                fill: false
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: { intersect: false, mode: 'index' },
                        plugins: {
                            legend: {
                                position: 'top',
                                labels: { usePointStyle: true, boxWidth: 6, color: palette.text }
                            },
                            tooltip: {
                                backgroundColor: palette.tooltipBg,
                                padding: 12,
                                callbacks: {
                                    label: function(context) {
                                        const label = context.dataset.label || '';
                                        const value = Number(context.parsed.y || context.parsed || 0);
                                        return `${label}: ${value}${useHours ? ' h' : ' min'}`;
                                    }
                                }
                            }
                        },
                        scales: {
                            x: {
                                stacked: true,
                                grid: { display: false },
                                ticks: { color: palette.text }
                            },
                            y: {
                                stacked: true,
                                beginAtZero: true,
                                title: { display: true, text: unitLabel },
                                grid: { color: palette.grid },
                                ticks: { color: palette.text }
                            },
                            y1: {
                                beginAtZero: true,
                                position: 'right',
                                grid: { drawOnChartArea: false },
                                title: { display: true, text: `Total ${unitLabel}` },
                                ticks: { color: palette.muted }
                            }
                        }
                    }
                });

                const breakdownCtx = document.getElementById('activityBreakdownChart').getContext('2d');
                activityBreakdownChart = new Chart(breakdownCtx, {
                    type: 'doughnut',
                    data: {
                        labels: ['Walking', 'Standing', 'Sitting', 'Sleeping'],
                        datasets: [{
                            data: [
                                Number((totalByType.walk / scaleFactor).toFixed(2)),
                                Number((totalByType.stand / scaleFactor).toFixed(2)),
                                Number((totalByType.sit / scaleFactor).toFixed(2)),
                                Number((totalByType.sleep / scaleFactor).toFixed(2))
                            ],
                            backgroundColor: [colors.walk, colors.stand, colors.sit, colors.sleep],
                            borderColor: document.body.getAttribute('data-theme') === 'dark' ? '#0f172a' : '#ffffff',
                            borderWidth: 2,
                            hoverOffset: 8
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        cutout: '62%',
                        plugins: {
                            legend: {
                                position: 'bottom',
                                labels: { color: palette.text, usePointStyle: true, boxWidth: 8 }
                            },
                            tooltip: {
                                backgroundColor: palette.tooltipBg,
                                padding: 12,
                                callbacks: {
                                    label: function(context) {
                                        const label = context.label || '';
                                        const value = Number(context.parsed || 0);
                                        const total = context.dataset.data.reduce((acc, v) => acc + Number(v || 0), 0) || 1;
                                        const percent = ((value / total) * 100).toFixed(1);
                                        return `${label}: ${value}${useHours ? ' h' : ' min'} (${percent}%)`;
                                    }
                                }
                            }
                        }
                    }
                });
            }

            function showFallToast(msg, type) {
                const container = document.getElementById('fall-message-display');
                const toast = document.createElement('div');
                toast.className = 'toast-fall';
                if (type === 'RECOVERED') {
                    toast.style.background = '#10b981';
                    toast.innerHTML = `<i class="fas fa-check-circle me-3"></i> <div><strong>RECOVERY</strong><br><small>${msg}</small></div>`;
                } else {
                    toast.innerHTML = `<i class="fas fa-exclamation-triangle me-3"></i> <div><strong>FALL ALERT</strong><br><small>${msg}</small></div>`;
                }
                container.appendChild(toast);
                setTimeout(() => {
                    toast.style.opacity = '0';
                    setTimeout(() => toast.remove(), 300);
                }, 6000);
            }

            function ackFall(pid) {
                fetch(`/api/acknowledge/${pid}`, { method: 'POST' }).then(() => update());
            }

            function renderDailySummary(summary) {
                document.getElementById('summary-date').textContent = summary.date || '';
                const items = [
                    ['Walking', summary.walking_dur],
                    ['Standing', summary.standing_dur],
                    ['Sitting', summary.sitting_dur],
                    ['Sleeping', summary.sleeping_dur],
                    ['Monitored', summary.monitored_dur]
                ];
                document.getElementById('summary-strip').innerHTML = items.map(([label, value]) => `
                    <div class="summary-pill">
                        <div class="summary-label">${label}</div>
                        <div class="summary-value">${value}</div>
                    </div>
                `).join('');

                document.getElementById('recommendations-list').innerHTML = (summary.recommendations || []).map(item => {
                    const sev = (item.severity || 'info').toLowerCase();
                    const pct = Math.round((item.confidence || 0) * 100);
                    const badgeClass = sev === 'high' ? 'text-bg-danger' : sev === 'medium' ? 'text-bg-warning' : 'text-bg-info';
                    return `
                        <li class="recommend-item recommend-${sev}">
                            <div class="d-flex justify-content-between align-items-start gap-3">
                                <div>
                                    <span class="badge ${badgeClass} me-2">${(item.title || 'Info').toUpperCase()}</span>
                                    <span>${item.text || ''}</span>
                                </div>
                                <span class="badge rounded-pill text-bg-light border text-dark">${pct}%</span>
                            </div>
                        </li>`;
                }).join('');

                const telegramText = summary.telegram_summary_text || 'Waiting for the latest pinned summary.';
                const telegramSummaryText = document.getElementById('telegram-summary-text');
                if (telegramSummaryText) {
                    telegramSummaryText.textContent = telegramText;
                }
                const badge = document.getElementById('telegram-summary-badge');
                if (badge) {
                    if (summary.telegram_summary_pinned) {
                        badge.className = 'badge text-bg-success';
                        badge.textContent = 'Pinned';
                    } else {
                        badge.className = 'badge text-bg-secondary';
                        badge.textContent = 'Pending';
                    }
                }
            }

            function update() {
                fetch('/api/report')
                    .then(r => r.json())
                    .then(data => {
                        document.getElementById('live-time').innerHTML = `<i class="far fa-clock me-1 text-muted"></i> ${new Date().toLocaleTimeString()}`;
                        
                        // Active Alerts
                        let alertHtml = '';
                        for (let l of data.active_alerts) {
                            const isRecovered = l.type === 'RECOVERED';
                            const alertKey = `${l.person_id}-${l.type}-${l.timestamp}`;
                            if (!seenAlerts.has(alertKey)) {
                                showFallToast(`${l.message}`, l.type);
                                seenAlerts.add(alertKey);
                            }
                            
                            alertHtml += `
                                <div class="alert-item shadow-sm ${isRecovered ? 'recovered' : ''}">
                                    <div class="d-flex justify-content-between align-items-center">
                                        <div>
                                            <h6 class="mb-1 ${isRecovered ? 'text-success' : 'text-danger'} fw-bold">
                                                ${isRecovered ? '<i class="fas fa-check-circle"></i> RECOVERY' : '<i class="fas fa-exclamation-circle"></i> FALL ALERT'}
                                            </h6>
                                            <p class="mb-0 small text-muted"><strong>${l.message}</strong> - Detected at ${l.time_str}</p>
                                        </div>
                                        <button class="btn btn-outline-dark btn-sm rounded-pill px-3" onclick="ackFall('${l.person_id}')">Dismiss</button>
                                    </div>
                                </div>`;
                        }
                        document.getElementById('alert-container').innerHTML = alertHtml;
                        
                        // Unnamed IDs
                        const unnamedContainer = document.getElementById('unnamed-container');
                        if (data.unnamed_ids && data.unnamed_ids.length > 0) {
                            unnamedContainer.classList.remove('d-none');
                            document.getElementById('unnamed-list').innerHTML = data.unnamed_ids.map(id => 
                                `<button class="btn btn-warning btn-sm border-0 rounded-2 px-3 shadow-sm" onclick="registerPerson('${id}')">Name ID ${id}</button>`
                            ).join('');
                        } else {
                            unnamedContainer.classList.add('d-none');
                        }

                        // People Grid
                        document.getElementById('people-grid').innerHTML = data.people.map(p => {
                            let activity = p.current_activity;
                            let badgeCls = 'bg-' + activity.toLowerCase().replace(' ', '-');
                            const rec = p.recommendations && p.recommendations.length ? p.recommendations[0] : null;
                            const recSeverity = rec ? (rec.severity || 'info').toLowerCase() : 'info';
                            const recBadge = recSeverity === 'high' ? 'text-bg-danger' : recSeverity === 'medium' ? 'text-bg-warning' : 'text-bg-info';
                            const recText = rec ? `${rec.text} (${Math.round((rec.confidence || 0) * 100)}%)` : 'No AI recommendation yet';
                            return `
                                <div class="col-md-6 mb-4" style="opacity: ${p.is_active ? '1.0' : '0.6'}">
                                    <div class="card p-4 h-100">
                                        <div class="d-flex justify-content-between align-items-center mb-4">
                                            <div class="d-flex align-items-center">
                                                <div class="bg-light p-2 rounded-circle me-3" style="width: 45px; height: 45px; display: flex; align-items: center; justify-content: center;">
                                                    <i class="fas fa-user text-muted"></i>
                                                </div>
                                                <h6 class="fw-bold mb-0">${p.person}</h6>
                                            </div>
                                            <span class="status-badge ${badgeCls}">${activity}</span>
                                        </div>
                                        <div class="row g-2">
                                            <div class="col-3 text-center">
                                                <div class="stat-icon icon-walk mx-auto"><i class="fas fa-walking"></i></div>
                                                <div class="fw-bold small">${p.walking_dur}</div>
                                            </div>
                                            <div class="col-3 text-center">
                                                <div class="stat-icon icon-stand mx-auto"><i class="fas fa-male"></i></div>
                                                <div class="fw-bold small">${p.standing_dur}</div>
                                            </div>
                                            <div class="col-3 text-center">
                                                <div class="stat-icon icon-sit mx-auto"><i class="fas fa-chair"></i></div>
                                                <div class="fw-bold small">${p.sitting_dur}</div>
                                            </div>
                                            <div class="col-3 text-center">
                                                <div class="stat-icon icon-sleep mx-auto"><i class="fas fa-bed"></i></div>
                                                <div class="fw-bold small">${p.sleeping_dur}</div>
                                            </div>
                                        </div>
                                        <div class="mt-3">
                                            <span class="badge ${recBadge} mb-2">${rec ? (rec.title || 'Insight') : 'Info'}</span>
                                            <div class="small text-muted">${recText}</div>
                                        </div>
                                    </div>
                                </div>`;
                        }).join('') || '<div class="col-12"><div class="card p-5 text-center text-muted">No active detections...</div></div>';
                        
                        // Fall History
                        document.getElementById('fall-history').innerHTML = data.falls.map(f => {
                            let icon = f.type === 'MAJOR FALL' ? 'fa-exclamation-circle text-danger' : 
                                      (f.type === 'RECOVERED' ? 'fa-check-circle text-success' : 'fa-exclamation-triangle text-warning');
                            return `
                                <div class="list-group-item px-4 py-3 border-0 border-bottom">
                                    <div class="d-flex align-items-center">
                                        <i class="fas ${icon} fs-5 me-3"></i>
                                        <div class="flex-grow-1">
                                            <div class="fw-bold small">${f.person}</div>
                                            <div class="text-muted" style="font-size: 0.75rem;">${f.type} • ${f.time_str}</div>
                                        </div>
                                    </div>
                                </div>`;
                        }).join('') || '<div class="p-5 text-center text-muted small">No fall history yet.</div>';
                    });

                fetch('/api/daily-summary')
                    .then(r => r.json())
                    .then(summary => renderDailySummary(summary));

                fetch('/api/camera-status')
                    .then(r => r.json())
                    .then(status => {
                        const pauseBanner = document.getElementById('activity-pause-banner');
                        const feed = document.getElementById('video-feed');
                        const empty = document.getElementById('video-empty');
                        const meta = document.getElementById('video-meta');
                        if (status.system_sleeping) {
                            feed.classList.add('d-none');
                            feed.removeAttribute('src');
                            empty.classList.remove('d-none');
                            empty.textContent = 'Low power mode active';
                            meta.textContent = 'Camera sleeping; peeking for movement every 1 second';
                        } else if (status.video_output_mode === 'window') {
                            feed.classList.add('d-none');
                            feed.removeAttribute('src');
                            empty.classList.remove('d-none');
                            empty.textContent = 'Low-latency local preview window is active';
                            meta.textContent = 'Dashboard stream disabled';
                        } else if (status.has_frame) {
                            if (!feed.getAttribute('src')) {
                                feed.setAttribute('src', '/video_feed');
                            }
                            feed.classList.remove('d-none');
                            empty.classList.add('d-none');
                            meta.textContent = status.video_output_mode === 'both'
                                ? 'Local preview active; dashboard preview updated at low rate'
                                : 'Live dashboard feed active';
                        } else if (!status.camera_available) {
                            feed.classList.add('d-none');
                            feed.removeAttribute('src');
                            empty.classList.remove('d-none');
                            meta.textContent = 'No camera detected';
                        } else {
                            if (!feed.getAttribute('src')) {
                                feed.setAttribute('src', '/video_feed');
                            }
                            feed.classList.remove('d-none');
                            empty.classList.remove('d-none');
                            meta.textContent = 'Waiting for frames...';
                        }

                        if (pauseBanner) {
                            if (status.activity_timing_paused) {
                                pauseBanner.textContent = status.activity_pause_notice || 'MULTIPLE PEOPLE DETECTED - ACTIVITY TIMING PAUSED';
                                pauseBanner.classList.remove('d-none');
                                if (meta && !meta.textContent.toLowerCase().includes('paused')) {
                                    meta.textContent = `${meta.textContent} | Activity timing paused`;
                                }
                            } else {
                                pauseBanner.textContent = '';
                                pauseBanner.classList.add('d-none');
                            }
                        }
                    });

                // Update charts
                if (!window.lastChartUpdate || (Date.now() - window.lastChartUpdate > 30000)) {
                    const endpoint = currentViewMode === 'daily' ? '/api/history' : '/api/history/monthly';
                    fetch(endpoint).then(r => r.json()).then(data => {
                        initChart(data);
                        window.lastChartUpdate = Date.now();
                    });
                }
            }
            initTheme();
            setInterval(update, 2000);
            update();
            setViewMode('daily');
        </script>
    </body>
    </html>
    """
    return html

def format_duration(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"

def recommendation_item(text, severity="info", confidence=0.5, title=None):
    severity = str(severity or "info").lower()
    confidence = max(0.0, min(1.0, float(confidence or 0.0)))
    return {
        "text": str(text),
        "severity": severity,
        "confidence": confidence,
        "title": title or severity.title(),
    }

def recommendation_to_text(item):
    if isinstance(item, dict):
        severity = str(item.get("severity", "info")).upper()
        confidence = int(round(float(item.get("confidence", 0.0)) * 100))
        title = item.get("title") or severity.title()
        text = item.get("text", "")
        return f"[{title} {confidence}%] {text}"
    return str(item)

def build_recommendations(walk_s, stand_s, sit_s, sleep_s, monitored_s, fall_count=0, person_name=None):
    recommendations = []
    if monitored_s <= 0:
        return [recommendation_item("Not enough activity data yet. Keep the camera on during the day to build recommendations.", "info", 0.35, "Info")]

    walk_ratio = walk_s / monitored_s
    stand_ratio = stand_s / monitored_s
    sit_ratio = sit_s / monitored_s
    sleep_ratio = sleep_s / monitored_s

    insight = []
    confidence = 0.55
    if sit_ratio >= 0.5:
        insight.append(f"sitting-heavy day ({sit_ratio * 100:.0f}% sitting)")
        confidence = 0.9
    elif walk_ratio >= 0.35:
        insight.append(f"movement-friendly day ({walk_ratio * 100:.0f}% walking)")
        confidence = 0.88
    elif sleep_ratio >= 0.45:
        insight.append(f"rest-heavy day ({sleep_ratio * 100:.0f}% sleeping)")
        confidence = 0.84
    else:
        insight.append("fairly balanced activity mix")
        confidence = 0.72

    if person_name:
        recommendations.append(recommendation_item(
            f"{person_name} looks like they had a {insight[0]}.",
            "info",
            confidence,
            "Insight"
        ))
    else:
        recommendations.append(recommendation_item(
            f"Today looks like a {insight[0]}.",
            "info",
            confidence,
            "Insight"
        ))

    if fall_count > 0:
        recommendations.append(recommendation_item(
            "Recent falls were detected. Keep walkways clear, use non-slip footwear, and avoid rushing when standing up.",
            "high",
            0.96,
            "Safety"
        ))

    if walk_ratio < 0.18 and monitored_s >= 30 * 60:
        recommendations.append(recommendation_item(
            "Walking time is low today. Try two or three short 5 to 10 minute walks after meals or during TV breaks.",
            "medium",
            0.84,
            "Activity"
        ))
    elif walk_ratio > 0.40:
        recommendations.append(recommendation_item(
            "Walking is strong today. Keep the same routine and add light stretching after longer walks.",
            "low",
            0.76,
            "Activity"
        ))

    if sit_ratio > 0.45:
        recommendations.append(recommendation_item(
            "Sitting is dominating the day. Break up long sitting periods every 30 to 60 minutes with a short stand-up or hallway walk.",
            "medium",
            0.87,
            "Posture"
        ))
    if sit_ratio > walk_ratio * 2.5:
        recommendations.append(recommendation_item(
            "Your sitting time is much higher than walking time. Add a few indoor laps, gentle leg lifts, or stretch breaks.",
            "medium",
            0.82,
            "Posture"
        ))

    if stand_ratio < 0.12 and monitored_s >= 20 * 60:
        recommendations.append(recommendation_item(
            "Standing time is low. Try standing during phone calls, light chores, or short reset breaks.",
            "low",
            0.74,
            "Activity"
        ))

    if sleep_ratio > 0.55 and monitored_s > 2 * 60 * 60:
        recommendations.append(recommendation_item(
            "Daytime sleeping is high. Keep naps shorter, stay hydrated, and keep a steady bedtime routine.",
            "medium",
            0.88,
            "Rest"
        ))
    elif sleep_ratio < 0.12 and monitored_s > 2 * 60 * 60:
        recommendations.append(recommendation_item(
            "Sleep time is very low. Make sure the resident is getting enough rest and calm breaks during the day.",
            "low",
            0.73,
            "Rest"
        ))

    if not recommendations:
        recommendations.append(recommendation_item("Today's activity balance looks healthy. Keep following the same routine.", "low", 0.8, "Insight"))
    return recommendations

def build_daily_summary_text(summary):
    lines = [
        f"Daily Activity Summary - {summary['date']}",
        f"Walking: {summary['walking_dur']}",
        f"Standing: {summary['standing_dur']}",
        f"Sitting: {summary['sitting_dur']}",
        f"Sleeping: {summary['sleeping_dur']}",
        f"Monitored: {summary['monitored_dur']}",
        "",
        "AI recommendations:"
    ]
    recommendations = summary.get("recommendations", [])
    if recommendations:
        for item in recommendations[:4]:
            lines.append(f"- {recommendation_to_text(item)}")
    else:
        lines.append("- No suggestions right now.")
    return "\n".join(lines)

def build_daily_summary_from_totals(day_label, walk_s, stand_s, sit_s, sleep_s, live_flags=None):
    monitored_s = walk_s + stand_s + sit_s + sleep_s
    recommendations = build_recommendations(walk_s, stand_s, sit_s, sleep_s, monitored_s, person_name="Resident")
    summary = {
        "date": str(day_label),
        "walking_seconds": float(walk_s),
        "standing_seconds": float(stand_s),
        "sitting_seconds": float(sit_s),
        "sleeping_seconds": float(sleep_s),
        "monitored_seconds": float(monitored_s),
        "walking_dur": format_duration(walk_s),
        "standing_dur": format_duration(stand_s),
        "sitting_dur": format_duration(sit_s),
        "sleeping_dur": format_duration(sleep_s),
        "monitored_dur": format_duration(monitored_s),
        "recommendations": recommendations
    }
    if live_flags:
        summary.update(live_flags)
    summary["telegram_summary_text"] = build_daily_summary_text(summary)
    return summary

def get_daily_summary():
    with data_lock:
        walk_s = float(sum(walking_time.values()))
        stand_s = float(sum(standing_time.values()))
        sit_s = float(sum(sitting_time.values()))
        sleep_s = float(sum(sleeping_time.values()))
        live_flags = {
            "multiple_people_detected": activity_timing_paused,
            "activity_pause_notice": activity_pause_notice,
            "activity_timing_paused": activity_timing_paused,
            "telegram_summary_pinned": bool(last_daily_summary_message_id),
            "telegram_summary_text": latest_daily_summary_text if last_daily_summary_message_id and latest_daily_summary_text else "",
            "telegram_summary_date": latest_daily_summary_date if latest_daily_summary_text else ""
        }

    summary = build_daily_summary_from_totals(str(date.today()), walk_s, stand_s, sit_s, sleep_s, live_flags=live_flags)
    if not summary.get("telegram_summary_text"):
        summary["telegram_summary_text"] = build_daily_summary_text(summary)
    return summary

def resolve_telegram_primary_person(people):
    if not people:
        return None

    configured = str(settings.get("telegram_primary_person", "") or "").strip()
    if configured:
        for person in people:
            person_name = str(person.get("person", "")).strip()
            current_activity = str(person.get("current_activity", "")).strip()
            if configured == person_name or configured == current_activity:
                return person

    active_people = [person for person in people if person.get("current_activity") != "AWAY"]
    candidate_pool = active_people or people

    def activity_seconds(person):
        return (
            float(person.get("walking_seconds", 0) or 0)
            + float(person.get("standing_seconds", 0) or 0)
            + float(person.get("sitting_seconds", 0) or 0)
            + float(person.get("sleeping_seconds", 0) or 0)
        )

    return max(candidate_pool, key=activity_seconds)

def parse_iso_date(value, fallback=None):
    if value:
        try:
            return date.fromisoformat(str(value))
        except Exception:
            pass
    return fallback or date.today()

def format_fall_timestamp(timestamp_value, unix_timestamp=None):
    if isinstance(timestamp_value, datetime):
        return timestamp_value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(timestamp_value, str) and timestamp_value.strip():
        text = timestamp_value.strip()
        try:
            return datetime.strptime(text.split(".")[0], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return text
    if unix_timestamp:
        try:
            return datetime.fromtimestamp(float(unix_timestamp)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return ""
    return ""

def build_activity_report_dataset(period="daily", anchor_date=None):
    period = str(period or "daily").strip().lower()
    anchor = parse_iso_date(anchor_date, date.today())
    if period == "weekly":
        start_date = anchor - timedelta(days=6)
        end_date = anchor
    else:
        start_date = anchor
        end_date = anchor

    start_str = start_date.isoformat()
    end_str = end_date.isoformat()

    person_rows = {}
    overall = {
        "walking_seconds": 0.0,
        "standing_seconds": 0.0,
        "sitting_seconds": 0.0,
        "sleeping_seconds": 0.0,
        "fall_count": 0,
        "last_fall_type": "",
        "last_fall_time": "",
    }

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        SELECT date, person_id, walking, standing, sitting, sleeping
        FROM activity
        WHERE date BETWEEN ? AND ?
        ORDER BY date ASC, person_id ASC
        """,
        (start_str, end_str)
    )
    activity_rows = c.fetchall()

    c.execute(
        """
        SELECT person_id, type, timestamp, unix_timestamp
        FROM falls
        WHERE date(timestamp) BETWEEN ? AND ?
        ORDER BY COALESCE(unix_timestamp, strftime('%s', timestamp)) DESC, timestamp DESC
        """,
        (start_str, end_str)
    )
    fall_rows = c.fetchall()
    conn.close()

    for _, raw_pid, walking, standing, sitting, sleeping in activity_rows:
        person_name = get_activity_label(resolve_person_key(raw_pid))
        row = person_rows.setdefault(person_name, {
            "person": person_name,
            "walking_seconds": 0.0,
            "standing_seconds": 0.0,
            "sitting_seconds": 0.0,
            "sleeping_seconds": 0.0,
            "fall_count": 0,
            "last_fall_type": "",
            "last_fall_time": "",
        })
        row["walking_seconds"] += float(walking or 0)
        row["standing_seconds"] += float(standing or 0)
        row["sitting_seconds"] += float(sitting or 0)
        row["sleeping_seconds"] += float(sleeping or 0)

    for raw_pid, fall_type, ts_value, unix_timestamp in fall_rows:
        if fall_type not in {"MINOR FALL", "MAJOR FALL"}:
            continue
        person_name = get_activity_label(resolve_person_key(raw_pid))
        row = person_rows.setdefault(person_name, {
            "person": person_name,
            "walking_seconds": 0.0,
            "standing_seconds": 0.0,
            "sitting_seconds": 0.0,
            "sleeping_seconds": 0.0,
            "fall_count": 0,
            "last_fall_type": "",
            "last_fall_time": "",
        })
        row["fall_count"] += 1
        if not row["last_fall_type"]:
            row["last_fall_type"] = fall_type
            row["last_fall_time"] = format_fall_timestamp(ts_value, unix_timestamp)
        overall["fall_count"] += 1
        if not overall["last_fall_type"]:
            overall["last_fall_type"] = fall_type
            overall["last_fall_time"] = format_fall_timestamp(ts_value, unix_timestamp)

    people = []
    for person_name, row in person_rows.items():
        monitored_seconds = row["walking_seconds"] + row["standing_seconds"] + row["sitting_seconds"] + row["sleeping_seconds"]
        row["monitored_seconds"] = monitored_seconds
        row["walking_dur"] = format_duration(row["walking_seconds"])
        row["standing_dur"] = format_duration(row["standing_seconds"])
        row["sitting_dur"] = format_duration(row["sitting_seconds"])
        row["sleeping_dur"] = format_duration(row["sleeping_seconds"])
        row["monitored_dur"] = format_duration(monitored_seconds)
        row["recommendations"] = build_recommendations(
            row["walking_seconds"],
            row["standing_seconds"],
            row["sitting_seconds"],
            row["sleeping_seconds"],
            monitored_seconds,
            fall_count=row["fall_count"],
            person_name=person_name
        )
        people.append(row)

    people.sort(key=lambda item: (item["monitored_seconds"], item["person"]), reverse=True)

    overall["monitored_seconds"] = sum(item["monitored_seconds"] for item in people)
    overall["walking_seconds"] = sum(item["walking_seconds"] for item in people)
    overall["standing_seconds"] = sum(item["standing_seconds"] for item in people)
    overall["sitting_seconds"] = sum(item["sitting_seconds"] for item in people)
    overall["sleeping_seconds"] = sum(item["sleeping_seconds"] for item in people)
    overall["walking_dur"] = format_duration(overall["walking_seconds"])
    overall["standing_dur"] = format_duration(overall["standing_seconds"])
    overall["sitting_dur"] = format_duration(overall["sitting_seconds"])
    overall["sleeping_dur"] = format_duration(overall["sleeping_seconds"])
    overall["monitored_dur"] = format_duration(overall["monitored_seconds"])
    overall["recommendations"] = build_recommendations(
        overall["walking_seconds"],
        overall["standing_seconds"],
        overall["sitting_seconds"],
        overall["sleeping_seconds"],
        overall["monitored_seconds"],
        fall_count=overall["fall_count"],
        person_name="Overall"
    )

    return {
        "period": period,
        "start_date": start_str,
        "end_date": end_str,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": overall,
        "people": people,
    }

def build_activity_report_csv(report):
    output = io.StringIO()
    writer = csv.writer(output)
    summary = report["summary"]

    writer.writerow(["Report Type", report["period"].title()])
    writer.writerow(["Start Date", report["start_date"]])
    writer.writerow(["End Date", report["end_date"]])
    writer.writerow(["Generated At", report["generated_at"]])
    writer.writerow([])
    writer.writerow(["Overall Summary"])
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Walking", summary["walking_dur"]])
    writer.writerow(["Standing", summary["standing_dur"]])
    writer.writerow(["Sitting", summary["sitting_dur"]])
    writer.writerow(["Sleeping", summary["sleeping_dur"]])
    writer.writerow(["Monitored", summary["monitored_dur"]])
    writer.writerow(["Fall Count", summary["fall_count"]])
    writer.writerow(["Last Fall Type", summary["last_fall_type"] or "None"])
    writer.writerow(["Last Fall Time", summary["last_fall_time"] or "None"])
    writer.writerow([])
    writer.writerow(["Resident Summary"])
    writer.writerow([
        "Resident",
        "Walking",
        "Standing",
        "Sitting",
        "Sleeping",
        "Monitored",
        "Fall Count",
        "Last Fall Type",
        "Last Fall Time",
        "AI recommendations"
    ])
    for person in report["people"]:
        writer.writerow([
            person["person"],
            person["walking_dur"],
            person["standing_dur"],
            person["sitting_dur"],
            person["sleeping_dur"],
            person["monitored_dur"],
            person["fall_count"],
            person["last_fall_type"] or "None",
            person["last_fall_time"] or "None",
            " | ".join(recommendation_to_text(item) for item in person["recommendations"])
        ])
    return output.getvalue().encode("utf-8")

def escape_pdf_text(text):
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

def build_pdf_bytes(title, lines):
    page_width = 612
    page_height = 792
    margin_x = 40
    margin_top = 44
    line_height = 14
    max_lines = int((page_height - margin_top * 2) / line_height) - 2

    expanded_lines = [title, ""]
    for line in lines:
        if line == "":
            expanded_lines.append("")
            continue
        wrapped = textwrap.wrap(str(line), width=92) or [""]
        expanded_lines.extend(wrapped)

    pages = [expanded_lines[i:i + max_lines] for i in range(0, len(expanded_lines), max_lines)]
    if not pages:
        pages = [[""]]

    objects = []

    def add_object(body):
        if isinstance(body, str):
            body = body.encode("latin-1", "replace")
        objects.append(body)

    add_object("<< /Type /Catalog /Pages 2 0 R >>")
    # Pages object is filled after page objects are known.
    add_object("")
    add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    page_obj_nums = []
    content_obj_nums = []
    for idx, page_lines in enumerate(pages):
        content_obj_num = 4 + idx * 2
        page_obj_num = 5 + idx * 2
        content_obj_nums.append(content_obj_num)
        page_obj_nums.append(page_obj_num)

        content_ops = [
            "BT",
            "/F1 10 Tf",
            f"{margin_x} {page_height - margin_top} Td",
        ]
        first_line = True
        for line in page_lines:
            if first_line:
                content_ops.append(f"({escape_pdf_text(line)}) Tj")
                first_line = False
            else:
                content_ops.append("T*")
                content_ops.append(f"({escape_pdf_text(line)}) Tj")
        content_ops.append("ET")
        content_stream = "\n".join(content_ops)
        content_body = f"<< /Length {len(content_stream.encode('latin-1', 'replace'))} >>\nstream\n{content_stream}\nendstream"
        add_object(content_body)

        page_body = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_obj_num} 0 R >>"
        )
        add_object(page_body)

    pages_kids = " ".join(f"{num} 0 R" for num in page_obj_nums)
    objects[1] = f"<< /Type /Pages /Kids [{pages_kids}] /Count {len(page_obj_nums)} >>".encode("latin-1")

    pdf = io.BytesIO()
    pdf.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(pdf.tell())
        pdf.write(f"{index} 0 obj\n".encode("ascii"))
        pdf.write(obj)
        pdf.write(b"\nendobj\n")

    xref_offset = pdf.tell()
    pdf.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.write(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.write(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.write(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    return pdf.getvalue()

def build_activity_report_pdf(report):
    summary = report["summary"]
    lines = [
        f"Report Type: {report['period'].title()}",
        f"Start Date: {report['start_date']}",
        f"End Date: {report['end_date']}",
        f"Generated At: {report['generated_at']}",
        "",
        "Overall Summary",
        f"Walking: {summary['walking_dur']}",
        f"Standing: {summary['standing_dur']}",
        f"Sitting: {summary['sitting_dur']}",
        f"Sleeping: {summary['sleeping_dur']}",
        f"Monitored: {summary['monitored_dur']}",
        f"Fall Count: {summary['fall_count']}",
        f"Last Fall Type: {summary['last_fall_type'] or 'None'}",
        f"Last Fall Time: {summary['last_fall_time'] or 'None'}",
        "",
        "Resident Summary",
    ]
    for person in report["people"]:
        lines.extend([
            f"Resident: {person['person']}",
            f"  Walking: {person['walking_dur']} | Standing: {person['standing_dur']} | Sitting: {person['sitting_dur']} | Sleeping: {person['sleeping_dur']}",
            f"  Monitored: {person['monitored_dur']} | Fall Count: {person['fall_count']}",
            f"  Last Fall: {person['last_fall_type'] or 'None'} at {person['last_fall_time'] or 'None'}",
            f"  AI recommendations: {'; '.join(recommendation_to_text(item) for item in person['recommendations'])}",
            "",
        ])
    return build_pdf_bytes(f"ElderlyCare {report['period'].title()} Activity Report", lines)

def build_reports_page():
    today = date.today()
    week_start = today - timedelta(days=6)
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Activity Reports</title>
        <link rel="icon" href="/favicon.svg" type="image/svg+xml">
        <link rel="shortcut icon" href="/favicon.svg" type="image/svg+xml">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            body {{ background: radial-gradient(circle at top left, #0f172a, #050816 60%); color: #e5e7eb; min-height: 100vh; }}
            .wrap {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 48px; }}
            .panel {{ background: rgba(15,23,42,0.92); border: 1px solid #22304a; border-radius: 24px; box-shadow: 0 24px 80px rgba(0,0,0,0.35); }}
            .panel-inner {{ padding: 24px; }}
            .muted {{ color: #94a3b8; }}
            .report-card {{ background: rgba(255,255,255,0.03); border: 1px solid #22304a; border-radius: 18px; padding: 18px; }}
        </style>
    </head>
    <body>
        <div class="wrap">
            <div class="d-flex justify-content-between align-items-center flex-wrap gap-3 mb-4">
                <div>
                    <h1 class="fw-bold mb-1">Activity Reports</h1>
                    <div class="muted">Export daily or weekly caregiver summaries as CSV or PDF.</div>
                </div>
                <a class="btn btn-outline-light" href="/">Back to Dashboard</a>
            </div>
            <div class="panel">
                <div class="panel-inner">
                    <div class="row g-4">
                        <div class="col-md-6">
                            <div class="report-card h-100">
                                <h4 class="mb-2">Daily Report</h4>
                                <div class="muted mb-3">Summary for today: {today.isoformat()}</div>
                                <div class="d-flex flex-wrap gap-2">
                                    <a class="btn btn-primary" href="/reports/export?period=daily&format=csv">Download CSV</a>
                                    <a class="btn btn-outline-light" href="/reports/export?period=daily&format=pdf">Download PDF</a>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="report-card h-100">
                                <h4 class="mb-2">Weekly Report</h4>
                                <div class="muted mb-3">Summary for {week_start.isoformat()} to {today.isoformat()}</div>
                                <div class="d-flex flex-wrap gap-2">
                                    <a class="btn btn-primary" href="/reports/export?period=weekly&format=csv">Download CSV</a>
                                    <a class="btn btn-outline-light" href="/reports/export?period=weekly&format=pdf">Download PDF</a>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

def persist_daily_activity_snapshot(day_label, walking_snapshot, standing_snapshot, sitting_snapshot, sleeping_snapshot):
    person_ids = set(walking_snapshot) | set(standing_snapshot) | set(sitting_snapshot) | set(sleeping_snapshot)
    if not person_ids:
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    for pid in person_ids:
        c.execute(
            '''INSERT OR REPLACE INTO activity (date, person_id, walking, standing, sitting, sleeping)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (
                str(day_label),
                str(pid),
                float(walking_snapshot.get(pid, 0)),
                float(standing_snapshot.get(pid, 0)),
                float(sitting_snapshot.get(pid, 0)),
                float(sleeping_snapshot.get(pid, 0))
            )
        )
    conn.commit()
    conn.close()

def rollover_daily_stats_if_needed():
    global daily_stats_day, latest_daily_summary_text, latest_daily_summary_date
    current_day = str(date.today())
    if current_day == daily_stats_day:
        return None

    with data_lock:
        if current_day == daily_stats_day:
            return None

        previous_day = daily_stats_day
        walking_snapshot = dict(walking_time)
        standing_snapshot = dict(standing_time)
        sitting_snapshot = dict(sitting_time)
        sleeping_snapshot = dict(sleeping_time)
        active_pids = list(person_state.keys())
        now = time.time()

        daily_stats_day = current_day
        walking_time.clear()
        standing_time.clear()
        sitting_time.clear()
        sleeping_time.clear()
        last_notified_activity.clear()
        activity_transition_candidate.clear()
        activity_transition_count.clear()
        for pid in active_pids:
            person_last_time[pid] = now

    summary = build_daily_summary_from_totals(
        previous_day,
        float(sum(walking_snapshot.values())),
        float(sum(standing_snapshot.values())),
        float(sum(sitting_snapshot.values())),
        float(sum(sleeping_snapshot.values()))
    )
    latest_daily_summary_text = summary["telegram_summary_text"]
    latest_daily_summary_date = previous_day
    persist_daily_activity_snapshot(previous_day, walking_snapshot, standing_snapshot, sitting_snapshot, sleeping_snapshot)
    return summary

def send_daily_telegram_summary(summary=None, force=False):
    global last_daily_summary_pinned_date, last_daily_summary_message_id, latest_daily_summary_text, latest_daily_summary_date
    if summary is None:
        summary = get_daily_summary()

    if activity_timing_paused and not force:
        return False, "Activity tracking paused because multiple people are in frame"

    summary_text = summary.get("telegram_summary_text") or build_daily_summary_text(summary)
    latest_daily_summary_text = summary_text
    latest_daily_summary_date = summary.get("date", str(date.today()))

    if not telegram_ready():
        return False, "Telegram is disabled or not configured"

    if not force and last_daily_summary_pinned_date == summary.get("date"):
        return False, "Daily summary already pinned"

    try:
        response = telegram_api_request(
            "sendMessage",
            {"chat_id": settings["chat_id"], "text": summary_text},
            timeout=5
        )
        message_id = response.get("result", {}).get("message_id")
        if message_id is None:
            return False, "Telegram did not return a message id"

        if last_daily_summary_message_id and last_daily_summary_message_id != message_id:
            unpin_telegram_message(last_daily_summary_message_id)

        pin_ok, pin_msg = pin_telegram_message(message_id)
        last_daily_summary_message_id = message_id
        last_daily_summary_pinned_date = summary.get("date")
        if pin_ok:
            add_system_event(f"Daily Telegram summary pinned for {summary.get('date')}")
            return True, "Daily summary pinned"

        add_system_event(f"Daily summary sent but pin failed: {pin_msg}", level="warning")
        return True, f"Daily summary sent, pin failed: {pin_msg}"
    except Exception as e:
        add_system_event(f"Daily summary send failed: {e}", level="error")
        return False, str(e)

@app.route("/api/daily-summary")
def api_daily_summary():
    return jsonify(get_daily_summary())

def build_activity_summary_text():
    snapshot = build_report_snapshot(include_remote=False)
    people = snapshot.get("people", [])
    falls = snapshot.get("falls", [])
    updated_at = time.strftime("%H:%M:%S", time.localtime())

    def latest_fall_for_person(person_name):
        for fall in falls:
            if fall.get("type") not in {"MINOR FALL", "MAJOR FALL"}:
                continue
            if str(fall.get("person", "")).strip() == str(person_name).strip():
                return fall
        return None

    lines = ["ElderlyCare Mini Dashboard", ""]
    if not people:
        lines.append("No resident activity yet.")
        lines.append(f"Updated: {updated_at}")
        return "\n".join(lines)

    person = resolve_telegram_primary_person(people)
    if not person:
        lines.append("No resident activity yet.")
        lines.append(f"Updated: {updated_at}")
        return "\n".join(lines)

    lines.extend([
        f"Resident: {person.get('person', 'Unknown')}",
        f"Current activity: {person.get('current_activity', 'AWAY')}",
        f"Walking: {person.get('walking_dur', '0m 0s')}",
        f"Standing: {person.get('standing_dur', '0m 0s')}",
        f"Sitting: {person.get('sitting_dur', '0m 0s')}",
        f"Sleeping: {person.get('sleeping_dur', '0m 0s')}",
        f"Monitored today: {person.get('monitored_dur', '0m 0s')}",
    ])
    fall = latest_fall_for_person(person.get("person", ""))
    if fall:
        lines.append(f"Last fall: {fall.get('type', 'FALL')} at {fall.get('time_str', '--:--:--')}")
    else:
        lines.append("Last fall: none")
    lines.append("")

    lines.append(f"Updated: {updated_at}")
    return "\n".join(lines).strip()

def send_or_update_activity_summary(force=False):
    global last_activity_summary_message_id, last_activity_summary_text, last_activity_summary_sent_at, activity_summary_dirty
    if not telegram_ready():
        return False, "Telegram is disabled or not configured"

    if activity_timing_paused and not force:
        return False, "Activity tracking paused because multiple people are in frame"

    message = build_activity_summary_text()
    if not force and message == last_activity_summary_text and not activity_summary_dirty:
        return False, "No change"

    now = time.time()
    payload_text = message
    ok = False
    result_msg = "Not sent"

    if last_activity_summary_message_id:
        ok, result_msg = edit_telegram_message(last_activity_summary_message_id, payload_text)
        if ok:
            last_activity_summary_text = payload_text
            last_activity_summary_sent_at = now
            activity_summary_dirty = False
            return True, "Edited"

    try:
        response = telegram_api_request(
            "sendMessage",
            {
                "chat_id": settings["chat_id"],
                "text": payload_text,
                "disable_notification": True
            },
            timeout=5
        )
        last_activity_summary_message_id = response.get("result", {}).get("message_id")
        last_activity_summary_text = payload_text
        last_activity_summary_sent_at = now
        activity_summary_dirty = False
        return True, "Sent"
    except Exception as e:
        add_system_event(f"Telegram activity summary failed: {e}", level="warning")
        return False, str(e)

def activity_summary_worker():
    interval_sec = 30.0
    while True:
        time.sleep(5.0)
        if not telegram_ready():
            continue
        if (time.time() - last_activity_summary_sent_at) < interval_sec and not activity_summary_dirty:
            continue
        send_or_update_activity_summary(force=False)

threading.Thread(target=activity_summary_worker, daemon=True).start()

video_job_lock = threading.Lock()
video_jobs = {}

def allowed_video_filename(filename):
    ext = os.path.splitext(filename or "")[1].lower()
    return ext in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

def resolve_video_fall_state(person_key, prev_state, proposed_state, current_velocity, current_v_velocity, aspect_ratio, now, confirm_window, fall_bank):
    active_fall_event = fall_bank["active_fall_event"]
    minor_fall_start_time = fall_bank["minor_fall_start_time"]
    recovery_mode = fall_bank["recovery_mode"]
    recovery_confirm_count = fall_bank["recovery_confirm_count"]
    lying_start_time = fall_bank["lying_start_time"]

    display_state = proposed_state if proposed_state != "UNKNOWN" else prev_state
    fast_fall_transition = prev_state in {"WALKING", "STANDING", "RECOVERED"} and (current_v_velocity > 2.0 or current_velocity > 4.0)
    major_fall_transition = prev_state in {"WALKING", "STANDING", "RECOVERED"} and current_v_velocity > 8.0

    if major_fall_transition and (display_state in ["LYING", "MINOR FALL"] or aspect_ratio > 1.2):
        display_state = "MAJOR FALL"

    if display_state == "MINOR FALL" and prev_state in ["WALKING", "STANDING", "SITTING", "RECOVERED"] and current_v_velocity < 2.5 and current_velocity < 3.2 and aspect_ratio < 1.7:
        display_state = "SITTING"

    if prev_state == "MAJOR FALL" and display_state in ["MINOR FALL", "LYING"]:
        display_state = "MAJOR FALL"

    if display_state == "LYING" and fast_fall_transition and prev_state in {"WALKING", "STANDING", "RECOVERED"}:
        display_state = "MINOR FALL"
    if display_state == "LYING" and prev_state in {"WALKING", "STANDING", "RECOVERED"} and (current_v_velocity > 0.4 or current_velocity > 1.0 or aspect_ratio > 1.45):
        display_state = "MINOR FALL"

    if current_v_velocity < -1.0 and prev_state in ["MAJOR FALL", "MINOR FALL", "LYING", "SLEEPING"]:
        if display_state in ["MINOR FALL", "LYING"]:
            display_state = "STANDING"

    if person_key in recovery_mode:
        if now > recovery_mode[person_key]:
            del recovery_mode[person_key]
        else:
            if display_state in ["MINOR FALL", "LYING"]:
                display_state = "STANDING"
            if now < (recovery_mode[person_key] - 5.0):
                display_state = "RECOVERED"

    is_currently_down = display_state in ["MAJOR FALL", "MINOR FALL", "LYING", "SLEEPING"]

    if is_currently_down and person_key not in active_fall_event:
        should_alert = False
        if prev_state in ["WALKING", "STANDING", "RECOVERED"]:
            if display_state in ["MINOR FALL", "MAJOR FALL"]:
                should_alert = True
            elif display_state == "LYING" and current_v_velocity > 4.0:
                should_alert = True
        if should_alert:
            active_fall_event[person_key] = "MINOR"
            if display_state == "LYING":
                display_state = "MINOR FALL"
            minor_fall_start_time[person_key] = now

    if is_currently_down:
        recovery_confirm_count[person_key] = 0
        if active_fall_event.get(person_key) == "MINOR" and person_key not in minor_fall_start_time:
            minor_fall_start_time[person_key] = now
        if active_fall_event.get(person_key) == "MINOR" and is_currently_down and (now - minor_fall_start_time.get(person_key, now) > confirm_window):
            active_fall_event[person_key] = "MAJOR"
            display_state = "MAJOR FALL"
    elif display_state in ["WALKING", "STANDING", "SITTING", "RECOVERED"]:
        recovery_confirm_count[person_key] = recovery_confirm_count.get(person_key, 0) + 1
        if recovery_confirm_count[person_key] > 30:
            if person_key in active_fall_event:
                recovery_mode[person_key] = now + 10.0
                del active_fall_event[person_key]
            if person_key in minor_fall_start_time:
                del minor_fall_start_time[person_key]
            recovery_confirm_count[person_key] = 0

    if display_state == "LYING":
        if person_key not in lying_start_time:
            lying_start_time[person_key] = now
        lying_duration = now - lying_start_time[person_key]
        sleep_threshold = SLEEPING_LYING_CONFIRM_SEC
        if prev_state in ["SITTING", "SLEEPING"]:
            sleep_threshold = SLEEPING_FROM_SITTING_SEC
        elif active_fall_event.get(person_key) == "MINOR":
            sleep_threshold = max(sleep_threshold, confirm_window + 2.0)
        if lying_duration >= sleep_threshold:
            display_state = "SLEEPING"
    elif display_state != "SLEEPING":
        if person_key in lying_start_time:
            del lying_start_time[person_key]

    return display_state

def create_video_job(input_path, output_path, original_name):
    job_id = uuid.uuid4().hex
    with video_job_lock:
        video_jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "original_name": original_name,
            "input_path": input_path,
            "output_path": output_path,
            "processed_frames": 0,
            "total_frames": 0,
            "progress": 0.0,
            "result_url": None,
            "error": None,
            "started_at": time.time(),
            "updated_at": time.time(),
        }
    threading.Thread(target=process_uploaded_video_job, args=(job_id,), daemon=True).start()
    return job_id

def update_video_job(job_id, **updates):
    with video_job_lock:
        job = video_jobs.get(job_id)
        if not job:
            return
        job.update(updates)
        job["updated_at"] = time.time()

def get_video_job(job_id):
    with video_job_lock:
        job = video_jobs.get(job_id)
        return dict(job) if job else None

def build_video_annotator_page(job=None, error_message=None, original_name=None):
    job = job or {}
    result_url = job.get("result_url")
    job_id = job.get("job_id")
    job_status = job.get("status", "")
    processed_frames = int(job.get("processed_frames") or 0)
    total_frames = int(job.get("total_frames") or 0)
    progress = float(job.get("progress") or 0.0)
    progress_text = f"{processed_frames} / {total_frames} frames" if total_frames else f"{processed_frames} frames"
    result_block = ""
    if result_url:
        result_block = f"""
        <div class="mt-4">
            <h4 class="mb-3">Annotated Video</h4>
            <div class="card border-0 shadow-sm p-3">
                <video id="annotated-video" class="w-100 rounded-3" controls playsinline preload="metadata" src="{result_url}"></video>
                <div class="d-flex flex-wrap gap-2 mt-3">
                    <button class="btn btn-outline-primary btn-sm" onclick="setRate(1)">1x</button>
                    <button class="btn btn-outline-primary btn-sm" onclick="setRate(2)">2x</button>
                    <button class="btn btn-outline-primary btn-sm" onclick="setRate(4)">4x</button>
                    <a class="btn btn-primary btn-sm ms-auto" href="{result_url}" download>Download Annotated Video</a>
                </div>
            </div>
        </div>
        """

    error_block = f'<div class="alert alert-danger">{error_message}</div>' if error_message else ""
    subtitle = f"Uploaded file: {original_name}" if original_name else "Upload a video and get an annotated activity replay."
    progress_block = ""
    if job_id and not result_url and job_status not in {"error", "completed"}:
        progress_block = f"""
        <div class="mt-4">
            <h4 class="mb-3">Processing Progress</h4>
            <div class="card border-0 shadow-sm p-4">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <div class="fw-semibold">Frames processed</div>
                    <div id="job-status" class="text-muted small">{job_status.title() if job_status else "Queued"}</div>
                </div>
                <div class="progress" style="height: 20px;">
                    <div id="progress-bar" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: {max(0.0, min(progress, 100.0)):.2f}%"></div>
                </div>
                <div id="progress-text" class="mt-2 small text-muted">{progress_text}</div>
            </div>
        </div>
        """

    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Video Annotator</title>
        <link rel="icon" href="/favicon.svg" type="image/svg+xml">
        <link rel="shortcut icon" href="/favicon.svg" type="image/svg+xml">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
        <style>
            body {{ background: linear-gradient(180deg, #0b1020 0%, #111827 100%); color: #e5e7eb; min-height: 100vh; }}
            .hero {{ max-width: 1100px; margin: 0 auto; padding: 32px 20px 48px; }}
            .panel {{ background: rgba(17,24,39,0.92); border: 1px solid #25324a; border-radius: 24px; box-shadow: 0 24px 80px rgba(0,0,0,0.35); }}
            .panel-inner {{ padding: 24px; }}
            .muted {{ color: #94a3b8; }}
            .dropzone {{ border: 2px dashed #334155; border-radius: 20px; padding: 28px; text-align: center; background: rgba(15,23,42,0.65); }}
            video {{ background: #020617; max-height: 70vh; }}
        </style>
    </head>
    <body>
        <div class="hero">
            <div class="d-flex justify-content-between align-items-center flex-wrap gap-3 mb-4">
                <div>
                    <h1 class="fw-bold mb-1">Video Annotator</h1>
                    <div class="muted">{subtitle}</div>
                </div>
                <a class="btn btn-outline-light" href="/">Back to Dashboard</a>
            </div>
            <div class="panel">
                <div class="panel-inner">
                    {error_block}
                    <form method="post" enctype="multipart/form-data" id="upload-form">
                        <input type="hidden" name="job_id" id="job_id" value="{job_id or ''}">
                        <div class="dropzone">
                            <div class="mb-3">
                                <label class="form-label fw-semibold">Choose a video file</label>
                                <input class="form-control form-control-lg" type="file" name="video_file" accept="video/*" required>
                            </div>
                            <button class="btn btn-primary btn-lg" type="submit">
                                <i class="fas fa-film me-2"></i>Upload and Annotate
                            </button>
                            <div class="mt-3 muted">Supported: MP4, MOV, AVI, MKV, WEBM, M4V</div>
                        </div>
                    </form>
                    {progress_block}
                    {result_block}
                </div>
            </div>
        </div>
        <script>
            function setRate(rate) {{
                const video = document.getElementById('annotated-video');
                if (video) video.playbackRate = rate;
            }}

            function renderJob(job) {{
                const progressWrap = document.getElementById('progress-wrap');
                if (!job) return;
                const statusEl = document.getElementById('job-status');
                const bar = document.getElementById('progress-bar');
                const text = document.getElementById('progress-text');
                if (statusEl) statusEl.textContent = job.status ? job.status.replace(/_/g, ' ') : 'Processing';
                if (bar) {{
                    const progress = Math.max(0, Math.min(100, job.progress || 0));
                    bar.style.width = progress.toFixed(2) + '%';
                    bar.textContent = progress >= 100 ? '100%' : progress.toFixed(0) + '%';
                }}
                if (text) {{
                    if (job.total_frames) {{
                        text.textContent = `${{job.processed_frames}} / ${{job.total_frames}} frames`;
                    }} else {{
                        text.textContent = `${{job.processed_frames}} frames processed`;
                    }}
                }}
            }}

            function injectResult(job) {{
                const container = document.querySelector('.panel-inner');
                if (!container || !job.result_url) return;
                const existing = document.getElementById('annotated-video');
                if (existing) return;
                const wrapper = document.createElement('div');
                wrapper.className = 'mt-4';
                wrapper.innerHTML = `
                    <h4 class="mb-3">Annotated Video</h4>
                    <div class="card border-0 shadow-sm p-3">
                        <video id="annotated-video" class="w-100 rounded-3" controls playsinline preload="metadata" src="${{job.result_url}}"></video>
                        <div class="d-flex flex-wrap gap-2 mt-3">
                            <button class="btn btn-outline-primary btn-sm" type="button" onclick="setRate(1)">1x</button>
                            <button class="btn btn-outline-primary btn-sm" type="button" onclick="setRate(2)">2x</button>
                            <button class="btn btn-outline-primary btn-sm" type="button" onclick="setRate(4)">4x</button>
                            <a class="btn btn-primary btn-sm ms-auto" href="${{job.result_url}}" download>Download Annotated Video</a>
                        </div>
                    </div>
                `;
                container.appendChild(wrapper);
            }}

            async function pollJob(jobId) {{
                if (!jobId) return;
                try {{
                    const res = await fetch(`/api/video-jobs/${{jobId}}`);
                    const job = await res.json();
                    renderJob(job);
                    if (job.status === 'completed') {{
                        injectResult(job);
                        return;
                    }}
                    if (job.status === 'error') {{
                        return;
                    }}
                    setTimeout(() => pollJob(jobId), 1000);
                }} catch (err) {{
                    setTimeout(() => pollJob(jobId), 1500);
                }}
            }}

            const existingJobId = document.getElementById('job_id')?.value;
            if (existingJobId) {{
                pollJob(existingJobId);
            }}
        </script>
    </body>
    </html>
    """

@app.route("/annotated-videos/<path:filename>")
def serve_annotated_video(filename):
    return send_from_directory(ANNOTATED_VIDEO_DIR, filename, as_attachment=False)

@app.route("/reports")
def reports_page():
    return build_reports_page()

@app.route("/reports/export")
def export_activity_report():
    period = str(request.args.get("period", "daily")).strip().lower()
    file_format = str(request.args.get("format", "csv")).strip().lower()
    if period not in {"daily", "weekly"}:
        return jsonify({"status": "error", "message": "Unsupported report period"}), 400
    if file_format not in {"csv", "pdf"}:
        return jsonify({"status": "error", "message": "Unsupported report format"}), 400

    anchor_key = "date" if period == "daily" else "end"
    anchor_date = parse_iso_date(request.args.get(anchor_key), date.today())
    report = build_activity_report_dataset(period=period, anchor_date=anchor_date)
    base_name = f"elderlycare_{period}_{report['start_date']}_to_{report['end_date']}"

    if file_format == "csv":
        data = build_activity_report_csv(report)
        return send_file(
            io.BytesIO(data),
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"{base_name}.csv"
        )

    pdf_bytes = build_activity_report_pdf(report)
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{base_name}.pdf"
    )

def process_uploaded_video_job(job_id):
    job = get_video_job(job_id)
    if not job:
        return
    input_path = job["input_path"]
    output_path = job["output_path"]
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        update_video_job(job_id, status="error", error="Could not open uploaded video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    ret, first_frame = cap.read()
    if not ret or first_frame is None:
        cap.release()
        update_video_job(job_id, status="error", error="Uploaded video has no readable frames", total_frames=total_frames)
        return

    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        update_video_job(job_id, status="error", error="Could not create annotated video output", total_frames=total_frames)
        return

    walking_total = defaultdict(float)
    standing_total = defaultdict(float)
    sitting_total = defaultdict(float)
    sleeping_total = defaultdict(float)
    person_state_local = {}
    person_last_time_local = {}
    person_last_pos_local = {}
    person_velocity_local = defaultdict(float)
    person_vertical_velocity_local = defaultdict(float)
    last_label_state = {}
    fall_bank = {
        "active_fall_event": {},
        "minor_fall_start_time": {},
        "recovery_mode": {},
        "recovery_confirm_count": defaultdict(int),
        "lying_start_time": {},
    }
    frame_count_local = 0
    frame_delta = 1.0 / fps
    video_time = 0.0

    try:
        update_video_job(job_id, status="processing", total_frames=total_frames, processed_frames=0, progress=0.0, error=None)
        while True:
            frame = first_frame if frame_count_local == 0 else None
            if frame is None:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

            raw_frame = frame
            proc_frame = resize_frame_for_processing(raw_frame)
            proc_h, proc_w = proc_frame.shape[:2]
            raw_h, raw_w = raw_frame.shape[:2]
            scale_x = raw_w / max(proc_w, 1)
            scale_y = raw_h / max(proc_h, 1)
            display_frame = raw_frame.copy()

            with model_lock, torch.inference_mode():
                results = model.track(
                    proc_frame,
                    persist=True,
                    conf=YOLO_CONFIDENCE,
                    imgsz=YOLO_IMGSZ,
                    max_det=YOLO_MAX_DET,
                    tracker=YOLO_TRACKER,
                    device=YOLO_DEVICE,
                    half=YOLO_HALF,
                    verbose=False
                )

            if results and results[0].keypoints is not None and results[0].boxes is not None and results[0].boxes.id is not None:
                max_people = min(int(settings.get("max_people_to_track", 4)), len(results[0].keypoints.xy))
                for i, kp in enumerate(results[0].keypoints.xy[:max_people]):
                    yolo_id = str(int(results[0].boxes.id[i]))
                    box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                    display_box = scale_box_to_frame(box, scale_x, scale_y, raw_w, raw_h)
                    keypoints = kp.cpu().numpy()
                    confidences = results[0].keypoints.conf[i].cpu().numpy()
                    display_keypoints = scale_keypoints_to_frame(keypoints, scale_x, scale_y)
                    draw_detailed_pose_overlay(display_frame, display_keypoints, confidences)

                    center_coords = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                    if yolo_id in person_last_pos_local:
                        last_pos = person_last_pos_local[yolo_id]
                        dist = float(np.sqrt((center_coords[0] - last_pos[0]) ** 2 + (center_coords[1] - last_pos[1]) ** 2))
                        v_dist = center_coords[1] - last_pos[1]
                        person_velocity_local[yolo_id] = person_velocity_local[yolo_id] * 0.8 + dist * 0.2
                        person_vertical_velocity_local[yolo_id] = person_vertical_velocity_local[yolo_id] * 0.8 + v_dist * 0.2
                    person_last_pos_local[yolo_id] = center_coords

                    prev_state = person_state_local.get(yolo_id, "UNKNOWN")
                    activity = classify_activity(
                        keypoints,
                        confidences,
                        velocity=person_velocity_local[yolo_id],
                        v_velocity=person_vertical_velocity_local[yolo_id],
                        aspect_ratio=((box[2] - box[0]) / max((box[3] - box[1]), 1)),
                        previous_state=prev_state
                    )
                    current_state = resolve_video_fall_state(
                        yolo_id,
                        prev_state,
                        activity,
                        person_velocity_local[yolo_id],
                        person_vertical_velocity_local[yolo_id],
                        ((box[2] - box[0]) / max((box[3] - box[1]), 1)),
                        video_time,
                        float(settings.get("fall_confirm_window_sec", 10.0)),
                        fall_bank
                    )
                    person_state_local[yolo_id] = current_state

                    now_local = video_time
                    if yolo_id not in person_last_time_local:
                        person_last_time_local[yolo_id] = now_local
                    delta = now_local - person_last_time_local[yolo_id]
                    if delta > 0:
                        if prev_state == "WALKING":
                            walking_total[yolo_id] += delta
                        elif prev_state == "STANDING":
                            standing_total[yolo_id] += delta
                        elif prev_state == "SITTING":
                            sitting_total[yolo_id] += delta
                        elif prev_state == "SLEEPING":
                            sleeping_total[yolo_id] += delta
                    person_last_time_local[yolo_id] = now_local

                    walk_str = format_duration(walking_total[yolo_id])
                    stand_str = format_duration(standing_total[yolo_id])
                    sit_str = format_duration(sitting_total[yolo_id])
                    sleep_str = format_duration(sleeping_total[yolo_id])
                    if "FALL" in current_state:
                        badge_color = (0, 0, 255)
                    elif current_state == "RECOVERED":
                        badge_color = (0, 255, 0)
                    elif current_state == "WALKING":
                        badge_color = (0, 255, 255)
                    elif current_state == "STANDING":
                        badge_color = (255, 200, 0)
                    else:
                        badge_color = (255, 0, 0)

                    if last_label_state.get(yolo_id) != current_state:
                        last_label_state[yolo_id] = current_state

                    draw_person_status_badge(
                        display_frame,
                        display_box,
                        yolo_id,
                        current_state,
                        badge_color,
                        walk_str,
                        stand_str,
                        sit_str,
                        sleep_str
                    )
                    if current_state in ["MINOR FALL", "MAJOR FALL"]:
                        draw_notice_banner(display_frame, f"{current_state} detected for ID {yolo_id}", (0, 0, 255) if current_state == "MAJOR FALL" else (0, 165, 255))

            writer.write(display_frame)
            frame_count_local += 1
            video_time += frame_delta
            if total_frames > 0:
                progress = (frame_count_local / total_frames) * 100.0
            else:
                progress = 0.0
            update_video_job(
                job_id,
                status="processing",
                processed_frames=frame_count_local,
                total_frames=total_frames,
                progress=progress
            )
    finally:
        cap.release()
        writer.release()

    result_url = f"/annotated-videos/{os.path.basename(output_path)}"
    update_video_job(
        job_id,
        status="completed",
        processed_frames=frame_count_local,
        total_frames=total_frames,
        progress=100.0,
        result_url=result_url,
        error=None
    )

@app.route("/annotate-video", methods=["GET", "POST"])
def annotate_video_page():
    if request.method == "GET":
        return build_video_annotator_page()

    upload = request.files.get("video_file")
    if not upload or not upload.filename:
        return build_video_annotator_page(error_message="Please choose a video file to upload.")

    if not allowed_video_filename(upload.filename):
        return build_video_annotator_page(error_message="Unsupported video format. Please upload MP4, MOV, AVI, MKV, WEBM, or M4V.")

    original_name = secure_filename(upload.filename)
    token = uuid.uuid4().hex[:10]
    input_name = f"{token}_{original_name}"
    output_name = f"{token}_annotated.mp4"
    input_path = os.path.join(VIDEO_UPLOAD_DIR, input_name)
    output_path = os.path.join(ANNOTATED_VIDEO_DIR, output_name)

    upload.save(input_path)
    add_system_event(f"Processing uploaded video: {original_name}")
    job_id = create_video_job(input_path, output_path, original_name)
    return build_video_annotator_page(job=get_video_job(job_id), original_name=original_name)

@app.route("/api/video-jobs/<job_id>")
def api_video_job_status(job_id):
    job = get_video_job(job_id)
    if not job:
        return jsonify({"status": "error", "error": "Unknown job"}), 404
    return jsonify(job)

def build_report_snapshot(include_remote=True):
    with data_lock:
        # Snapshot current state for reporting
        if SINGLE_PERSON_MODE:
            current_state_snapshot = {}
            active_states = [state for state in person_state.values() if state != "UNKNOWN"]
            if active_states:
                current_state_snapshot[SINGLE_PERSON_LABEL] = active_states[0]
        else:
            current_state_snapshot = person_state.copy()
        # Unnamed IDs are those in current state that don't have a manual mapping
        unnamed_ids = [] if SINGLE_PERSON_MODE else [pid for pid in current_state_snapshot if pid not in manual_id_map]
        
        # Sort people: currently active first, then by total monitored time
        all_pids = list(all_tracked_people)
        
        def get_activity_score(person_key):
            is_active = person_key in current_state_snapshot
            total_time = walking_time.get(person_key, 0) + standing_time.get(person_key, 0) + sitting_time.get(person_key, 0) + sleeping_time.get(person_key, 0)
            return (is_active, total_time)

        sorted_pids = sorted(all_pids, key=get_activity_score, reverse=True)
        
        report_data = []
        for person_key in sorted_pids[:10]:
            display_id = get_activity_label(person_key)
            monitored_total = walking_time.get(person_key, 0) + standing_time.get(person_key, 0) + sitting_time.get(person_key, 0) + sleeping_time.get(person_key, 0)
            report_data.append({
                "person": display_id,
                "walking_dur": format_duration(walking_time.get(person_key, 0)),
                "standing_dur": format_duration(standing_time.get(person_key, 0)),
                "sleeping_dur": format_duration(sleeping_time.get(person_key, 0)),
                "sitting_dur": format_duration(sitting_time.get(person_key, 0)),
                "monitored_dur": format_duration(monitored_total),
                "current_activity": current_state_snapshot.get(person_key, "AWAY"),
                "is_active": person_key in current_state_snapshot
            })
            
        # Ensure alerts and history are sorted by high-precision float timestamp
        alerts_copy = sorted(active_alerts, key=lambda x: x['timestamp'], reverse=True)
        falls_copy = sorted(fall_events, key=lambda x: x['timestamp'], reverse=True)[:10]

    snapshot = {
        "people": report_data,
        "falls": falls_copy, 
        "active_alerts": alerts_copy,
        "unnamed_ids": unnamed_ids
    }

    if include_remote:
        with data_lock:
            remote_reports = list(remote_edge_reports.values())
        remote_reports.sort(key=lambda item: item.get("updated_at", 0), reverse=True)

        for remote in remote_reports:
            node_id = remote.get("node_id", "remote")
            for person in remote.get("people", []):
                merged_person = dict(person)
                merged_person["person"] = f"{merged_person.get('person', 'Unknown')} [{node_id}]"
                snapshot["people"].append(merged_person)
            for alert in remote.get("active_alerts", []):
                merged_alert = dict(alert)
                merged_alert["person_id"] = f"{merged_alert.get('person_id', 'Unknown')} [{node_id}]"
                merged_alert["message"] = f"[{node_id}] {merged_alert.get('message', '')}"
                snapshot["active_alerts"].append(merged_alert)
            for fall_item in remote.get("falls", []):
                merged_fall = dict(fall_item)
                merged_fall["person"] = f"{merged_fall.get('person', 'Unknown')} [{node_id}]"
                snapshot["falls"].append(merged_fall)

        snapshot["active_alerts"] = sorted(snapshot["active_alerts"], key=lambda x: x['timestamp'], reverse=True)
        snapshot["falls"] = sorted(snapshot["falls"], key=lambda x: x['timestamp'], reverse=True)[:10]

    return snapshot

@app.route("/api/report")
def api_report():
    """API endpoint for JSON report with sorting and limiting"""
    return jsonify(build_report_snapshot(include_remote=True))

def run_server():
    # Runs Flask server in background thread
    try:
        print("Flask: Starting server...")
        app.run(
            host=settings["server_bind_host"],
            port=int(settings["server_port"]),
            debug=False,
            use_reloader=False,
            threaded=True
        )
    except Exception as e:
        print(f"Flask Error: {e}")
        import traceback
        traceback.print_exc()

print("Starting Flask server thread...")
threading.Thread(target=http_post_worker, daemon=True).start()
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

time.sleep(1)  # Give Flask time to start
print(f"✓ Flask server should be running at http://{settings['server_bind_host']}:{settings['server_port']}")
print(f"  Access the report at: http://127.0.0.1:{settings['server_port']}/")
print(f"Deployment mode: {settings['deployment_mode']} | Node ID: {settings['node_id']}")
if is_edge_mode():
    print(f"Central server URL: {get_central_server_url()}")
if IDENTITY_MODE == "tracker":
    print("Identity mode: tracker-only numbering (ReID disabled).")
if not FACE_RECOGNITION_AVAILABLE:
    print("Face registration mode: disabled.")
if use_dashboard_stream() and use_local_preview():
    video_output_label = "local preview window + dashboard preview"
elif use_dashboard_stream():
    video_output_label = "dashboard stream"
else:
    video_output_label = "local preview window"
print(f"Video output mode: {video_output_label}")

# ==================== YOLO11 Pose Model ====================
model = YOLO(os.path.join(BASE_DIR, "yolo11n-pose.pt"))  # latest & fast
try:
    model.to(YOLO_DEVICE)
except Exception as e:
    print(f"YOLO device move failed ({YOLO_DEVICE}): {e}")
    YOLO_DEVICE = "cpu"
    YOLO_DEVICE_NAME = "CPU fallback"
    CUDA_ACTIVE = False
    YOLO_HALF = False

try:
    model.fuse()
except Exception:
    pass

try:
    warmup_frame = np.zeros((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.uint8)
    with model_lock, torch.inference_mode():
        model.predict(
            warmup_frame,
            imgsz=YOLO_IMGSZ,
            conf=YOLO_CONFIDENCE,
            max_det=YOLO_MAX_DET,
            device=YOLO_DEVICE,
            half=YOLO_HALF,
            verbose=False
        )
except Exception as e:
    print(f"YOLO warmup skipped: {e}")

print(
    f"YOLO runtime: device={YOLO_DEVICE} ({YOLO_DEVICE_NAME}) | "
    f"precision={'FP16' if YOLO_HALF else 'FP32'} | "
    f"imgsz={YOLO_IMGSZ} | tracker={YOLO_TRACKER} | "
    f"max_det={YOLO_MAX_DET} | "
    f"camera={CAMERA_WIDTH}x{CAMERA_HEIGHT}@{CAMERA_FPS} | "
    f"processing={PROCESSING_WIDTH}x{PROCESSING_HEIGHT}"
)
fall_cooldown = {}  # Prevent spam alerts for same person

def classify_activity(keypoints, conf, velocity=0, v_velocity=0, aspect_ratio=1.0, previous_state=None):
    """
    Advanced skeleton-based activity classification with Impact Detection.
    """
    try:
        # Check confidence of critical joints
        critical_joints = [5, 6, 11, 12] # Shoulders and Hips
        
        # Calculate Midpoints and distances
        sho_y = (keypoints[5][1] + keypoints[6][1]) / 2
        sho_x = (keypoints[5][0] + keypoints[6][0]) / 2
        hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
        hip_x = (keypoints[11][0] + keypoints[12][0]) / 2
        
        dy = hip_y - sho_y
        dx = hip_x - sho_x
        angle = abs(np.degrees(np.arctan2(dx, dy)))
        torso_len = np.sqrt(dx**2 + dy**2)
        fast_fall_transition = previous_state in {"WALKING", "STANDING", "RECOVERED"} and (v_velocity > 2.0 or velocity > 4.0)
        major_fall_transition = previous_state in {"WALKING", "STANDING", "RECOVERED"} and v_velocity > 8.0

        if major_fall_transition and (angle > 30 or aspect_ratio > 1.2):
            return "MAJOR FALL"

        # Impact Detection: High vertical velocity (downward) + High angle or wide box
        # Threshold: > 8 pixels/frame downward is usually a fall
        if v_velocity > 8.0 and (angle > 30 or aspect_ratio > 1.2):
            return "MAJOR FALL"

        if any(conf[j] < 0.5 for j in critical_joints):
            if conf[0] > 0.5 and (conf[11] > 0.5 or conf[12] > 0.5):
                nose_y = keypoints[0][1]
                hip_y = (keypoints[11][1] + keypoints[12][1]) / 2 if (conf[11] > 0.5 and conf[12] > 0.5) else (keypoints[11][1] if conf[11] > 0.5 else keypoints[12][1])
                if nose_y > hip_y - 10:
                    return "LYING"
            
            # If joints are hidden but the box is very wide, likely lying down
            if aspect_ratio > 1.8:
                return "LYING"
            return "UNKNOWN"
        
        # 1. LYING (Horizontal and low)
        # If an upright person suddenly collapses into a horizontal pose, treat it as a fall first.
        posture_collapse = previous_state in {"WALKING", "STANDING", "RECOVERED"} and (angle > 42 or aspect_ratio > 1.35)
        if angle > 65 or aspect_ratio > 1.8:
            if fast_fall_transition or posture_collapse:
                return "MINOR FALL"
            return "LYING"

        # Backward falls can flatten without a huge downward spike. If the person
        # collapses from upright into a tilted/wide shape, keep it as a fall.
        if posture_collapse and (v_velocity > 0.4 or velocity > 1.0 or angle > 50 or aspect_ratio > 1.45):
            return "MINOR FALL"
        
        # Nose-to-ground ratio: If head is significantly lower than normal relative to torso
        if conf[0] > 0.5 and conf[15] > 0.5 and conf[16] > 0.5:
            ank_y = (keypoints[15][1] + keypoints[16][1]) / 2
            nose_y = keypoints[0][1]
            # Height of head from "floor" (ankles)
            head_height = abs(ank_y - nose_y)
            # Length of torso
            torso_len = np.sqrt(dx**2 + dy**2)
            # If head is very low (less than torso length away from floor), they are down
            if head_height < torso_len * 0.8:
                if fast_fall_transition:
                    return "MINOR FALL"
                return "LYING"

        if angle > 45 and conf[0] > 0.5 and keypoints[0][1] > hip_y:
            if fast_fall_transition:
                return "MINOR FALL"
            return "LYING"
        
        # 3. SITTING vs UPRIGHT (STANDING/WALKING)
        # Move sitting logic up to prevent "sitting on floor" from being "Minor Fall"
        is_sitting = False
        if conf[13] > 0.5 and conf[14] > 0.5:
            knee_y = (keypoints[13][1] + keypoints[14][1]) / 2
            if conf[15] > 0.5 and conf[16] > 0.5:
                ank_y = (keypoints[15][1] + keypoints[16][1]) / 2
                upper_leg = abs(knee_y - hip_y)
                lower_leg = abs(ank_y - knee_y)
                if upper_leg < lower_leg * 0.5: 
                    is_sitting = True
        
        if not is_sitting and conf[15] > 0.5 and conf[16] > 0.5:
            torso_len = np.sqrt(dx**2 + dy**2)
            ank_y = (keypoints[15][1] + keypoints[16][1]) / 2
            total_h = abs(ank_y - sho_y)
            # Sitting on floor with spread legs often results in torso_len / total_h > 0.6
            if torso_len / total_h > 0.6: 
                is_sitting = True
        
        # Additional floor sitting check: Hips close to floor (ankles) but torso not horizontal
        if not is_sitting and conf[11] > 0.5 and conf[12] > 0.5 and conf[15] > 0.5 and conf[16] > 0.5:
            ank_y = (keypoints[15][1] + keypoints[16][1]) / 2
            hip_to_floor = abs(ank_y - hip_y)
            torso_len = np.sqrt(dx**2 + dy**2)
            if hip_to_floor < torso_len * 0.4 and angle < 50:
                is_sitting = True

        if is_sitting:
            return "SITTING"

        # A slow sit-down should still show a seated lower-body shape.
        # Keep this conservative so a still standing person does not flicker into sitting.
        if previous_state in {"WALKING", "STANDING", "RECOVERED"}:
            if velocity < 2.5 and v_velocity < 1.8 and aspect_ratio > 0.45:
                if conf[13] > 0.5 and conf[14] > 0.5 and conf[15] > 0.5 and conf[16] > 0.5:
                    knee_y = (keypoints[13][1] + keypoints[14][1]) / 2
                    ank_y = (keypoints[15][1] + keypoints[16][1]) / 2
                    upper_leg = abs(knee_y - hip_y)
                    lower_leg = abs(ank_y - knee_y)
                    if upper_leg < lower_leg * 0.8:
                        return "SITTING"

        # 2. MINOR FALL (Significant tilt but not fully flat)
        # Require a stronger motion cue so controlled sitting is not treated as a fall.
        if (angle > 52 or aspect_ratio > 1.55) and (v_velocity > 2.0 or velocity > 4.0):
            return "MINOR FALL"
        
        # 4. STANDING vs WALKING (Using velocity + Pose)
        # Increased threshold to 3.0 for WALKING to reduce noise
        if velocity > 3.0:
            return "WALKING"
        
        # Fallback to pose-based walking detection if velocity is moderate
        if velocity > 0.8 and conf[15] > 0.5 and conf[16] > 0.5:
            feet_dist = abs(keypoints[15][0] - keypoints[16][0])
            shoulder_width = abs(keypoints[5][0] - keypoints[6][0])
            # Feet must be wider than shoulders to be 'walking' if velocity is low
            if feet_dist > shoulder_width * 1.1:
                return "WALKING"

        # Front-facing walking can have little center movement but clear leg-phase asymmetry.
        # Keep this conservative and only use it when the person is still upright.
        if angle < 30 and all(conf[j] > 0.5 for j in [11, 12, 13, 14, 15, 16]):
            left_upper = abs(keypoints[13][1] - keypoints[11][1])
            right_upper = abs(keypoints[14][1] - keypoints[12][1])
            left_lower = abs(keypoints[15][1] - keypoints[13][1])
            right_lower = abs(keypoints[16][1] - keypoints[14][1])
            knee_y_diff = abs(keypoints[13][1] - keypoints[14][1])
            ankle_y_diff = abs(keypoints[15][1] - keypoints[16][1])
            front_gait_score = abs(left_upper - right_upper) + abs(left_lower - right_lower) + 0.5 * (knee_y_diff + ankle_y_diff)
            if front_gait_score > max(14.0, torso_len * 0.16):
                return "WALKING"

        return "STANDING"

    except Exception:
        return "UNKNOWN"

def stabilize_activity_state(persistent_id, current_state, proposed_state):
    """Keep standing/sitting stable unless the new state persists for a few frames."""
    if persistent_id is None:
        return proposed_state

    if current_state == proposed_state:
        activity_transition_candidate.pop(persistent_id, None)
        activity_transition_count.pop(persistent_id, None)
        return proposed_state

    if current_state in {"STANDING", "WALKING"} and proposed_state in {"STANDING", "WALKING"}:
        cached_candidate = activity_transition_candidate.get(persistent_id)
        if cached_candidate != proposed_state:
            activity_transition_candidate[persistent_id] = proposed_state
            activity_transition_count[persistent_id] = 1
            return current_state

        activity_transition_count[persistent_id] += 1
        required_frames = WALKING_CONFIRM_FRAMES if proposed_state == "WALKING" else STANDING_CONFIRM_FRAMES
        if activity_transition_count[persistent_id] >= required_frames:
            activity_transition_candidate.pop(persistent_id, None)
            activity_transition_count.pop(persistent_id, None)
            return proposed_state
        return current_state

    if current_state in {"STANDING", "SITTING"} and proposed_state in {"STANDING", "SITTING"}:
        cached_candidate = activity_transition_candidate.get(persistent_id)
        if cached_candidate != proposed_state:
            activity_transition_candidate[persistent_id] = proposed_state
            activity_transition_count[persistent_id] = 1
            return current_state

        activity_transition_count[persistent_id] += 1
        required_frames = STANDING_SITTING_CONFIRM_FRAMES if proposed_state == "SITTING" else SITTING_STANDING_CONFIRM_FRAMES
        if activity_transition_count[persistent_id] >= required_frames:
            activity_transition_candidate.pop(persistent_id, None)
            activity_transition_count.pop(persistent_id, None)
            return proposed_state
        return current_state

    activity_transition_candidate.pop(persistent_id, None)
    activity_transition_count.pop(persistent_id, None)
    return proposed_state

def send_fall_alert(alert_msg, pid, fall_type, coords=None):
    global last_global_alert_time, last_alert_coords, last_alert_pid, status_message, status_expiry
    try:
        now = time.time()
        
        # Spatial-Temporal Squelch:
        # If we sent an alert of this type recently (< 5s) and it was in the same area (< 150px)
        # then it's likely a phantom ID/tracker drift for the same person.
        if coords and fall_type in last_alert_coords:
            prev_coords = last_alert_coords[fall_type]
            dist = np.sqrt((coords[0]-prev_coords[0])**2 + (coords[1]-prev_coords[1])**2)
            if dist < 150 and (now - last_global_alert_time) < 5:
                # If it's the SAME person (resolved name), definitely skip
                # If it's a DIFFERENT person but very close/recent, it's likely a ghost ID
                print(f"🤫 Squelching redundant {fall_type} for {pid} (likely ghost ID)")
                return
        
        # Update last alert state
        last_global_alert_time = now
        if coords: last_alert_coords[fall_type] = coords
        last_alert_pid[fall_type] = pid

        # Update status message for visual feedback on frame
        status_message = alert_msg
        status_expiry = now + 5

        print(f"⚠️  {alert_msg}! Sending alert...")
        # Log to DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO falls (timestamp, person_id, type, unix_timestamp) VALUES (?, ?, ?, ?)",
                  (datetime.now(), str(pid), fall_type, now))
        conn.commit()
        conn.close()

        trigger_payload = {"person_id": str(pid), "message": alert_msg, "type": fall_type}
        if is_edge_mode():
            queue_http_post(
                key=f"trigger:{fall_type}:{now}",
                url=f"{get_central_server_url()}/trigger",
                payload=trigger_payload,
                timeout=0.75,
                error_label="Central fall trigger"
            )
        else:
            queue_http_post(
                key=f"local-trigger:{fall_type}:{now}",
                url=f"http://127.0.0.1:{settings['server_port']}/trigger",
                payload=trigger_payload,
                timeout=0.5,
                error_label="Local fall trigger"
            )

        telegram_text = f"{pid}: {fall_type}\n{alert_msg}"
        telegram_category = f"fall:{fall_type.lower()}"
        if fall_type == "MAJOR FALL":
            send_telegram_burst_async(
                telegram_text,
                category=telegram_category,
                count=MAJOR_FALL_TELEGRAM_BURST_COUNT,
                delay_sec=MAJOR_FALL_TELEGRAM_BURST_DELAY_SEC,
                force=True
            )
            add_system_event(
                f"Major fall burst queued: {MAJOR_FALL_TELEGRAM_BURST_COUNT} Telegram notifications"
            )
        elif fall_type == "MINOR FALL":
            send_telegram_message_async(telegram_text, category=telegram_category)
        else:
            add_system_event(f"Recovery recorded silently for {pid}")
        print("✓ Fall alert sent and logged successfully")
    except Exception as e:
        print(f"✗ Failed to send/log fall alert: {e}")

def post_to_central(path, payload, timeout=2):
    if not is_edge_mode():
        return False
    try:
        response = requests.post(f"{get_central_server_url()}{path}", json=payload, timeout=timeout)
        response.raise_for_status()
        return True
    except Exception as e:
        add_system_event(f"Central sync failed for {path}: {e}", level="warning")
        return False

def send_node_heartbeat():
    payload = {
        "node_id": settings["node_id"],
        "deployment_mode": settings["deployment_mode"],
        "camera_available": camera_available,
        "has_frame": latest_stream_frame is not None
    }
    return queue_http_post(
        key="central-heartbeat",
        url=f"{get_central_server_url()}/api/node-heartbeat",
        payload=payload,
        timeout=0.5,
        error_label="Central heartbeat"
    )

def send_edge_report_snapshot():
    snapshot = build_report_snapshot(include_remote=False)
    snapshot.update({
        "node_id": settings["node_id"],
        "camera_available": camera_available,
        "has_frame": latest_stream_frame is not None
    })
    return queue_http_post(
        key="central-edge-report",
        url=f"{get_central_server_url()}/api/edge/report",
        payload=snapshot,
        timeout=0.75,
        error_label="Central edge report"
    )

# ==================== Camera Loop ====================
def configure_camera_capture(camera_handle, frame_width=None, frame_height=None):
    if camera_handle is None:
        return
    try:
        camera_handle.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    try:
        if len(CAMERA_FOURCC) == 4:
            camera_handle.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAMERA_FOURCC))
    except Exception:
        pass
    try:
        camera_handle.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    except Exception:
        pass

    width = frame_width if frame_width is not None else CAMERA_WIDTH
    height = frame_height if frame_height is not None else CAMERA_HEIGHT
    try:
        camera_handle.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    except Exception:
        pass
    try:
        camera_handle.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    except Exception:
        pass

def read_camera_frame(camera_handle, flush_old_frames=True):
    global last_camera_read_at
    if camera_handle is None or not camera_handle.isOpened():
        return False, None
    flush_count = 0
    if flush_old_frames:
        flush_count = CAMERA_FRAME_FLUSH_COUNT
        if last_camera_read_at > 0:
            elapsed = max(0.0, time.time() - last_camera_read_at)
            estimated_backlog = int(elapsed * max(1, CAMERA_FPS)) - 1
            flush_count = max(flush_count, min(24, estimated_backlog))
    for _ in range(max(0, flush_count)):
        if not camera_handle.grab():
            break
    ret, frame = camera_handle.read()
    last_camera_read_at = time.time()
    return ret, frame

def resize_frame_for_processing(frame):
    if frame is None or frame.size == 0:
        return frame
    h, w = frame.shape[:2]
    if w <= PROCESSING_WIDTH and h <= PROCESSING_HEIGHT:
        return frame
    scale = min(PROCESSING_WIDTH / max(w, 1), PROCESSING_HEIGHT / max(h, 1))
    if scale >= 1.0:
        return frame
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

def scale_box_to_frame(box, scale_x, scale_y, max_width, max_height):
    scaled_box = np.array([
        int(round(box[0] * scale_x)),
        int(round(box[1] * scale_y)),
        int(round(box[2] * scale_x)),
        int(round(box[3] * scale_y))
    ], dtype=int)
    scaled_box[0] = max(0, min(max_width, scaled_box[0]))
    scaled_box[1] = max(0, min(max_height, scaled_box[1]))
    scaled_box[2] = max(0, min(max_width, scaled_box[2]))
    scaled_box[3] = max(0, min(max_height, scaled_box[3]))
    return scaled_box

def scale_keypoints_to_frame(keypoints, scale_x, scale_y):
    scaled_keypoints = keypoints.copy()
    scaled_keypoints[:, 0] *= scale_x
    scaled_keypoints[:, 1] *= scale_y
    return scaled_keypoints

def resize_frame_for_stream(frame):
    if frame is None or frame.size == 0 or STREAM_OUTPUT_WIDTH <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= STREAM_OUTPUT_WIDTH:
        return frame
    scale = STREAM_OUTPUT_WIDTH / max(w, 1)
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (STREAM_OUTPUT_WIDTH, new_h), interpolation=cv2.INTER_AREA)

def prepare_motion_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (9, 9), 0)

def count_motion_pixels(prev_gray_frame, gray_frame, threshold_value=25):
    if prev_gray_frame is None or gray_frame is None or prev_gray_frame.shape != gray_frame.shape:
        return 0
    frame_delta = cv2.absdiff(prev_gray_frame, gray_frame)
    _, motion_mask = cv2.threshold(frame_delta, threshold_value, 255, cv2.THRESH_BINARY)
    motion_mask = cv2.dilate(motion_mask, None, iterations=2)
    return int(cv2.countNonZero(motion_mask))

def open_camera(frame_width=None, frame_height=None, max_retries=3, retry_delay=0.5):
    """Robust camera opener for Windows"""
    global last_camera_read_at
    try:
        camera_index = int(settings.get("preferred_camera", "0"))
    except Exception:
        camera_index = 0
    for _ in range(max(1, int(max_retries))): # Try up to N times
        # Using DSHOW (DirectShow) on Windows is much more stable for resolution changes
        c = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW) if sys.platform == "win32" else cv2.VideoCapture(camera_index)
        if c.isOpened():
            configure_camera_capture(c, frame_width=frame_width, frame_height=frame_height)
            last_camera_read_at = 0.0
            return c
        time.sleep(max(0.0, float(retry_delay)))
    return None

def update_stream_frame(frame):
    global pending_stream_frame, pending_stream_frame_seq
    if frame is None or not use_dashboard_stream():
        return
    with data_lock:
        pending_stream_frame = frame
        pending_stream_frame_seq += 1

def stream_encoder_worker():
    global latest_stream_frame, latest_stream_frame_id, last_stream_encode_at
    last_encoded_seq = -1
    while True:
        now = time.time()
        min_interval = 1.0 / get_effective_stream_max_fps()
        with data_lock:
            frame = pending_stream_frame
            frame_seq = pending_stream_frame_seq
        if frame is None or frame_seq == last_encoded_seq or (now - last_stream_encode_at) < min_interval:
            time.sleep(0.01)
            continue

        try:
            stream_frame = resize_frame_for_stream(frame)
            ok, encoded = cv2.imencode(".jpg", stream_frame, [int(cv2.IMWRITE_JPEG_QUALITY), STREAM_JPEG_QUALITY])
            if ok:
                with data_lock:
                    if frame_seq == pending_stream_frame_seq:
                        latest_stream_frame = encoded.tobytes()
                        latest_stream_frame_id += 1
                        last_stream_encode_at = now
                last_encoded_seq = frame_seq
            else:
                time.sleep(0.01)
        except Exception:
            time.sleep(0.01)

def clear_stream_frame():
    global latest_stream_frame, pending_stream_frame
    with data_lock:
        latest_stream_frame = None
        pending_stream_frame = None

def show_preview_window(frame):
    global preview_window_enabled, preview_window_initialized
    preview_window_enabled = use_local_preview()
    if not preview_window_enabled:
        if preview_window_initialized:
            try:
                cv2.destroyWindow(PREVIEW_WINDOW_NAME)
            except Exception:
                pass
            preview_window_initialized = False
        return
    if frame is None:
        return
    try:
        if not preview_window_initialized:
            cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(PREVIEW_WINDOW_NAME, PREVIEW_WINDOW_WIDTH, PREVIEW_WINDOW_HEIGHT)
            preview_window_initialized = True
        cv2.imshow(PREVIEW_WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc
            preview_window_enabled = False
            preview_window_initialized = False
            cv2.destroyWindow(PREVIEW_WINDOW_NAME)
    except Exception:
        preview_window_initialized = False

def draw_detailed_pose_overlay(frame, keypoints, confidences):
    if frame is None or keypoints is None or confidences is None:
        return

    # More complete COCO-style body layout with face, arms, torso, and legs.
    face_connections = [(0, 1), (0, 2), (1, 3), (2, 4)]
    upper_body_connections = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10)]
    torso_connections = [(5, 11), (6, 12), (11, 12)]
    lower_body_connections = [(11, 13), (13, 15), (12, 14), (14, 16)]
    all_connections = (
        face_connections
        + upper_body_connections
        + torso_connections
        + lower_body_connections
    )

    for start_idx, end_idx in all_connections:
        if confidences[start_idx] > 0.35 and confidences[end_idx] > 0.35:
            pt1 = (int(keypoints[start_idx][0]), int(keypoints[start_idx][1]))
            pt2 = (int(keypoints[end_idx][0]), int(keypoints[end_idx][1]))

            if start_idx <= 4 and end_idx <= 4:
                color = (255, 180, 0)
                thickness = 1
            elif start_idx in [5, 6, 7, 8, 9, 10] or end_idx in [5, 6, 7, 8, 9, 10]:
                color = (80, 220, 120)
                thickness = 2
            elif start_idx in [11, 12, 13, 14, 15, 16] or end_idx in [11, 12, 13, 14, 15, 16]:
                color = (80, 160, 255)
                thickness = 2
            else:
                color = (200, 200, 200)
                thickness = 2

            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

    # Draw joint markers with varied sizes for a more readable skeleton.
    for joint_idx in range(len(keypoints)):
        conf = confidences[joint_idx]
        if conf <= 0.35:
            continue

        point = (int(keypoints[joint_idx][0]), int(keypoints[joint_idx][1]))
        if joint_idx == 0:
            color = (0, 255, 255)
            radius = 5
        elif joint_idx in [5, 6, 11, 12]:
            color = (0, 255, 0)
            radius = 5
        elif joint_idx in [7, 8, 13, 14]:
            color = (255, 200, 0)
            radius = 4
        elif joint_idx in [9, 10, 15, 16]:
            color = (255, 120, 120)
            radius = 4
        else:
            color = (255, 255, 255)
            radius = 3

        cv2.circle(frame, point, radius + 2, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)

def draw_person_status_badge(frame, box, pid, current_state, color, walk_str, stand_str, sit_str, sleep_str):
    """Draw a stable person label next to the person's own bounding box."""
    if frame is None or box is None:
        return

    x1, y1, x2, y2 = [int(v) for v in box]
    label_lines = [
        f"ID {pid}",
        current_state,
        f"W:{walk_str} St:{stand_str} S:{sit_str} Sl:{sleep_str}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    line_scales = [0.62, 0.58, 0.48]
    line_thickness = [2, 2, 1]
    sizes = [cv2.getTextSize(text, font, scale, thick)[0] for text, scale, thick in zip(label_lines, line_scales, line_thickness)]
    text_w = max(width for width, _ in sizes)
    text_h = sum(height for _, height in sizes)
    padding_x = 10
    padding_y = 8
    gap = 4
    badge_w = text_w + padding_x * 2
    badge_h = text_h + padding_y * 2 + gap * (len(label_lines) - 1)

    left = max(0, min(x1, frame.shape[1] - badge_w - 2))
    above = y1 - badge_h - 10
    below = y2 + 10
    top = above if above >= 0 else min(max(0, below), max(0, frame.shape[0] - badge_h - 2))

    cv2.rectangle(frame, (left, top), (left + badge_w, top + badge_h), (14, 18, 24), -1)
    cv2.rectangle(frame, (left, top), (left + badge_w, top + badge_h), color, 2)

    baseline_y = top + padding_y + sizes[0][1]
    for idx, (text, scale, thick) in enumerate(zip(label_lines, line_scales, line_thickness)):
        if idx > 0:
            baseline_y += sizes[idx - 1][1] + gap
        cv2.putText(frame, text, (left + padding_x, baseline_y), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

def draw_notice_banner(frame, text, color=(0, 165, 255)):
    """Draw a compact status banner near the top of the frame."""
    if frame is None or not text:
        return
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thick = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thick)
    pad_x = 12
    pad_y = 10
    box_w = min(w - 20, text_w + pad_x * 2)
    box_h = text_h + pad_y * 2
    left = 10
    top = 10
    cv2.rectangle(frame, (left, top), (left + box_w, top + box_h), (16, 24, 34), -1)
    cv2.rectangle(frame, (left, top), (left + box_w, top + box_h), color, 2)
    cv2.putText(frame, text, (left + pad_x, top + pad_y + text_h), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

def enter_low_power_mode(frame, frame_width):
    global system_sleeping, low_power_overlay_suppressed, preview_window_initialized, cap, prev_sleep_gray, last_sleep_peek_at, camera_reconfigure_pending
    low_power_overlay_suppressed = True
    if preview_window_initialized:
        try:
            cv2.destroyWindow(PREVIEW_WINDOW_NAME)
        except Exception:
            pass
        preview_window_initialized = False
    clear_stream_frame()
    if cap is not None and cap.isOpened():
        configure_camera_capture(cap, frame_width=SLEEP_CAMERA_WIDTH, frame_height=SLEEP_CAMERA_HEIGHT)
    else:
        cap = open_camera(
            frame_width=SLEEP_CAMERA_WIDTH,
            frame_height=SLEEP_CAMERA_HEIGHT,
            max_retries=1,
            retry_delay=0.05
        )
    prev_sleep_gray = None
    last_sleep_peek_at = 0.0
    camera_reconfigure_pending = False
    add_system_event("WARD OUT OF FRAME. GOING INTO LOW POWER MODE.", level="warning")
    print(f"💤 Low Power Mode: No person detected for {LOW_POWER_IDLE_TIMEOUT_SEC:.1f}s. Turning off webcam...")
    system_sleeping = True

threading.Thread(target=stream_encoder_worker, daemon=True).start()

cap = open_camera()
camera_available = cap is not None
if not camera_available:
    print("Warning: Cannot open camera. Starting in dashboard-only mode.")
    add_system_event("Camera unavailable. Running in dashboard-only mode.", level="warning")

frame_count = 0
start_time = time.time()
last_detection = {}  # Track last frame when person was detected
last_motion_time = time.time()
out_of_frame_since = None
prev_gray = None
prev_sleep_gray = None
system_sleeping = False
low_power_overlay_suppressed = False
last_sleep_peek_at = 0.0
last_node_heartbeat_at = 0
last_edge_sync_at = 0

while True:
    frame_count += 1
    now = time.time()
    rollover_summary = rollover_daily_stats_if_needed()
    if rollover_summary is not None:
        threading.Thread(target=send_daily_telegram_summary, args=(rollover_summary,), daemon=True).start()
    frame = None

    if is_edge_mode() and (now - last_node_heartbeat_at) >= 5:
        send_node_heartbeat()
        last_node_heartbeat_at = now

    if not camera_active:
        time.sleep(1.0)
        continue
    
    # --- 1. Handle Sleep Mode Lifecycle ---
    if system_sleeping:
        sleep_wait_remaining = SLEEP_POLL_INTERVAL_SEC - max(0.0, now - last_sleep_peek_at)
        if sleep_wait_remaining > 0:
            time.sleep(min(0.1, sleep_wait_remaining))
            continue

        last_sleep_peek_at = now
        if cap is None or not cap.isOpened():
            cap = open_camera(
                frame_width=SLEEP_CAMERA_WIDTH,
                frame_height=SLEEP_CAMERA_HEIGHT,
                max_retries=1,
                retry_delay=0.05
            )
        if cap is None or not cap.isOpened():
            camera_available = False
            time.sleep(0.1)
            continue

        camera_available = True

        ret, peek_frame = read_camera_frame(cap, flush_old_frames=False)
        if not ret or peek_frame is None or peek_frame.size == 0:
            try:
                cap.release()
            except Exception:
                pass
            cap = None
            camera_available = False
            continue
            
        sleep_gray = prepare_motion_frame(peek_frame)
        motion_pixels = count_motion_pixels(prev_sleep_gray, sleep_gray)
        if prev_sleep_gray is None or prev_sleep_gray.shape != sleep_gray.shape:
            prev_sleep_gray = sleep_gray
            continue

        prev_sleep_gray = cv2.addWeighted(prev_sleep_gray, 0.85, sleep_gray, 0.15, 0)
        if motion_pixels < SLEEP_MOTION_MIN_PIXELS:
            clear_stream_frame()
            continue

        print(f"Motion detected in low power mode ({motion_pixels} px). Reopening live monitoring.")
        system_sleeping = False
        low_power_overlay_suppressed = False
        out_of_frame_since = None
        last_motion_time = now
        prev_sleep_gray = None
        prev_gray = None
        camera_reconfigure_pending = True
        frame = peek_frame
    # --- 2. Normal Camera Operation ---
    if not system_sleeping:
        if cap is None or not cap.isOpened():
            cap = open_camera()
            if cap is None:
                camera_available = False
                time.sleep(1)
                continue
            camera_available = True
            camera_reconfigure_pending = False

        if frame is None:
            if camera_reconfigure_pending:
                configure_camera_capture(cap)
                last_camera_read_at = 0.0
                camera_reconfigure_pending = False
            ret, frame = read_camera_frame(cap)
            if not ret or frame is None or frame.size == 0:
                print("Camera read failed. Retrying...")
                if cap:
                    cap.release()
                cap = None
                camera_available = False
                clear_stream_frame()
                continue

        raw_frame = frame
        frame = resize_frame_for_processing(raw_frame)

        # Motion detection to keep system awake
        gray = prepare_motion_frame(frame)
        motion_pixels = count_motion_pixels(prev_gray, gray)
        if motion_pixels > NORMAL_MOTION_MIN_PIXELS:
            last_motion_time = now
        if prev_gray is None or prev_gray.shape != gray.shape:
            prev_gray = gray
        else:
            prev_gray = cv2.addWeighted(prev_gray, 0.9, gray, 0.1, 0)

    last_frame = raw_frame.copy() # Store for registration
    try:
        h, w = raw_frame.shape[:2]
        process_h, process_w = frame.shape[:2]
        scale_x = w / max(process_w, 1)
        scale_y = h / max(process_h, 1)
        # Make a copy to display
        display_frame = raw_frame.copy()

        # Run YOLO tracking with higher confidence to reduce false positives
        if settings.get("enable_detection", True):
            with model_lock, torch.inference_mode():
                results = model.track(
                    frame,
                    persist=True,
                    conf=YOLO_CONFIDENCE,
                    imgsz=YOLO_IMGSZ,
                    max_det=YOLO_MAX_DET,
                    tracker=YOLO_TRACKER,
                    device=YOLO_DEVICE,
                    half=YOLO_HALF,
                    verbose=False
                )
        else:
            results = [type("EmptyResult", (), {"keypoints": None, "boxes": type("EmptyBoxes", (), {"id": None})()})()]

        detected_ids = set()  # Track which people are in this frame
        frame_assigned_pids = set()  # Prevent reusing one persistent id for multiple detections in the same frame
        normal_activity_monitoring_enabled = True
        person_visible_this_frame = bool(
            results[0].boxes is not None
            and getattr(results[0].boxes, "id", None) is not None
            and len(results[0].boxes.id) > 0
        )
        frame_people_count = int(len(results[0].boxes.id)) if person_visible_this_frame else 0
        if person_visible_this_frame and frame_people_count > 1:
            normal_activity_monitoring_enabled = False

        activity_timing_paused = not normal_activity_monitoring_enabled and person_visible_this_frame
        multi_person_count = frame_people_count if activity_timing_paused else 0
        activity_pause_notice = "MULTIPLE PEOPLE DETECTED - ACTIVITY TIMING PAUSED" if activity_timing_paused else ""
        
        if results[0].keypoints is not None and results[0].boxes.id is not None:
            # Keep system awake if people are detected
            last_motion_time = now
            max_people = int(settings.get("max_people_to_track", DEFAULT_SETTINGS["max_people_to_track"]))
            for i, kp in enumerate(results[0].keypoints.xy[:max_people]):
                # 1. Get YOLO track ID
                yolo_id = int(results[0].boxes.id[i])
                yolo_id_str = str(yolo_id)
                
                # Bounding box for movement check
                box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                display_box = scale_box_to_frame(box, scale_x, scale_y, w, h)
                center_coords = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

                current_embedding = None
                current_sig = None
                crop_x1 = crop_y1 = crop_x2 = crop_y2 = None
                if IDENTITY_MODE != "tracker":
                    crop_x1, crop_y1 = max(0, box[0]), max(0, box[1])
                    crop_x2, crop_y2 = min(process_w, box[2]), min(process_h, box[3])
                    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                        continue

                    person_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    if person_img.size == 0:
                        continue

                    current_embedding = reid_manager.get_embedding(person_img)
                    current_sig = get_color_signature(person_img)

                persistent_id = tracker_to_persistent.get(yolo_id_str)
                if persistent_id is not None and IDENTITY_MODE != "tracker":
                    validated_pid = validate_tracker_identity(
                        yolo_id_str,
                        persistent_id,
                        center_coords,
                        current_embedding=current_embedding,
                        current_sig=current_sig,
                        occupied_pids=frame_assigned_pids
                    )
                    if validated_pid is None:
                        tracker_to_persistent.pop(yolo_id_str, None)
                        persistent_id = None
                    else:
                        if validated_pid != persistent_id:
                            tracker_to_persistent[yolo_id_str] = validated_pid
                        persistent_id = validated_pid

                # 2. Map YOLO ID to Persistent ID (ReID)
                if persistent_id is None:
                    if IDENTITY_MODE == "tracker":
                        persistent_id = yolo_id_str
                        tracker_to_persistent[yolo_id_str] = persistent_id
                        person_start_pos[persistent_id] = center_coords
                        person_frames_seen[persistent_id] = 0
                        person_is_confirmed[persistent_id] = False
                    else:
                        persistent_id = find_recent_identity_match(
                            center_coords,
                            current_embedding=current_embedding,
                            current_sig=current_sig,
                            occupied_pids=frame_assigned_pids,
                            mapped_pid=None
                        )
                        if persistent_id is None:
                            with data_lock:
                                persistent_id = reid_manager.match_identity(
                                    current_embedding,
                                    current_sig=current_sig,
                                    blocked_ids=frame_assigned_pids
                                )
                        if persistent_id is None:
                            continue

                        tracker_to_persistent[yolo_id_str] = persistent_id

                        # Only initialize tracking state for genuinely new identities.
                        is_recent_identity = (
                            persistent_id in last_detection
                            and (frame_count - last_detection.get(persistent_id, 0)) <= TRACK_ASSOCIATION_MAX_AGE_FRAMES
                        )
                        if not is_recent_identity:
                            person_start_pos[persistent_id] = center_coords
                            person_frames_seen[persistent_id] = 0
                            person_is_confirmed[persistent_id] = False

                if persistent_id in frame_assigned_pids:
                    continue
                frame_assigned_pids.add(persistent_id)
                
                # IMPORTANT: Always mark as detected so they aren't pruned while being confirmed
                detected_ids.add(persistent_id)
                last_detection[persistent_id] = frame_count
                
                person_frames_seen[persistent_id] = person_frames_seen.get(persistent_id, 0) + 1

                # 3. Static Object Filtering (e.g., clothes on wall)
                # If a person hasn't moved at all in 60 frames (~2s), it's likely a static object
                is_static_object = False
                if not person_is_confirmed.get(persistent_id, False):
                    start_pos = person_start_pos.get(persistent_id, center_coords)
                    dist = np.sqrt((center_coords[0]-start_pos[0])**2 + (center_coords[1]-start_pos[1])**2)
                    
                    if dist > 30: # Moved 30 pixels? Confirmed human
                        person_is_confirmed[persistent_id] = True
                    elif person_frames_seen[persistent_id] > 60: 
                        is_static_object = True
                
                # Draw skeleton for ALL detections (including unconfirmed) so user sees tracking
                keypoints = kp.cpu().numpy()
                display_keypoints = scale_keypoints_to_frame(keypoints, scale_x, scale_y)
                confidences = results[0].keypoints.conf[i].cpu().numpy()
                
                # --- Draw Detailed Pose Overlay ---
                if settings.get("display_metrics_overlay", True) and not low_power_overlay_suppressed:
                    draw_detailed_pose_overlay(display_frame, display_keypoints, confidences)

                if is_static_object:
                    continue
                
                # Stabilization delay for logic processing (still show skeleton above)
                if not person_is_confirmed.get(persistent_id, False) and person_frames_seen[persistent_id] < 10:
                    continue

                # 4. Resolve a stable activity label for the overlay
                pid = get_activity_label(persistent_id)
                
                # 4. Periodically try to "Name" the Persistent ID using Face Recognition
                if FACE_RECOGNITION_AVAILABLE and not SINGLE_PERSON_MODE and frame_count % FACE_NAME_INTERVAL_FRAMES == 0 and persistent_id not in manual_id_map:
                    box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(process_w, box[2]), min(process_h, box[3])
                    person_img = frame[y1:y2, x1:x2]
                    
                    if person_img.size > 0:
                        with data_lock:
                            target_encodings = known_face_encodings[:]
                            target_names = known_face_names[:]
                        
                        if target_encodings:
                            rgb_person = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
                            face_locations = face_recognition.face_locations(rgb_person, number_of_times_to_upsample=1)
                            if face_locations:
                                face_encodings = face_recognition.face_encodings(rgb_person, face_locations)
                                for fe in face_encodings:
                                    matches = face_recognition.compare_faces(target_encodings, fe, tolerance=0.6)
                                    if True in matches:
                                        first_match_index = matches.index(True)
                                        real_name = str(target_names[first_match_index])
                                        rename_person(persistent_id, real_name)
                                        pid = real_name
                                        break
                
                # Update tracking metadata
                detected_ids.add(persistent_id)
                last_detection[persistent_id] = frame_count
                
                # 5. Classify Activity
                keypoints = kp.cpu().numpy()
                confidences = results[0].keypoints.conf[i].cpu().numpy()
                now = time.time()
                prev_s = person_state.get(persistent_id, "UNKNOWN")
                
                # Calculate velocity (rolling average displacement)
                if persistent_id in person_last_pos:
                    last_pos = person_last_pos[persistent_id]
                    dist = np.sqrt((center_coords[0]-last_pos[0])**2 + (center_coords[1]-last_pos[1])**2)
                    v_dist = center_coords[1] - last_pos[1] # Positive is downward
                    person_velocity[persistent_id] = person_velocity[persistent_id] * 0.8 + dist * 0.2
                    person_vertical_velocity[persistent_id] = person_vertical_velocity[persistent_id] * 0.8 + v_dist * 0.2
                
                person_last_pos[persistent_id] = center_coords
                current_velocity = person_velocity[persistent_id]
                current_v_velocity = person_vertical_velocity[persistent_id]
                
                # Aspect Ratio of bounding box (width/height)
                box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                bw = box[2] - box[0]
                bh = box[3] - box[1]
                aspect_ratio = bw / bh if bh > 0 else 0
                
                activity = classify_activity(
                    keypoints,
                    confidences,
                    velocity=current_velocity,
                    v_velocity=current_v_velocity,
                    aspect_ratio=aspect_ratio,
                    previous_state=prev_s
                )

                # --- 6. Body Scanning (Capture multi-angle signatures) ---
                # During the first 10 seconds of seeing a person, periodically capture different angles
                with data_lock:
                    id_data = reid_manager.identity_bank.get(persistent_id, {})
                    first_seen = id_data.get('first_seen', 0)
                    gallery_len = len(id_data.get('embeddings', [])) if 'embeddings' in id_data else 1
                
                if IDENTITY_MODE == "reid" and (now - first_seen) < 15 and frame_count % REID_SCAN_INTERVAL_FRAMES == 0:
                    box = results[0].boxes.xyxy[i].cpu().numpy().astype(int)
                    display_box = scale_box_to_frame(box, scale_x, scale_y, w, h)
                    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(process_w, box[2]), min(process_h, box[3])
                    if x2 > x1 and y2 > y1:
                        person_img = frame[y1:y2, x1:x2]
                        # Periodically update color signature to handle lighting changes
                        current_sig = get_color_signature(person_img)
                        emb = reid_manager.get_embedding(person_img)
                        
                        with data_lock:
                            if persistent_id in reid_manager.identity_bank:
                                # Update color signature
                                reid_manager.identity_bank[persistent_id]['color_sig'] = current_sig
                                # Update embedding via moving average if already matched
                                old_emb = reid_manager.identity_bank[persistent_id].get('embedding')
                                if old_emb is not None:
                                    updated_emb = 0.9 * old_emb + 0.1 * emb # Slower update during tracking
                                    norm = np.linalg.norm(updated_emb)
                                    if norm > 0:
                                        reid_manager.identity_bank[persistent_id]['embedding'] = updated_emb / norm
                                else:
                                    # Fallback for legacy gallery if needed
                                    reid_manager.add_to_gallery(persistent_id, emb)
                        
                        # Visual feedback for scanning
                        cv2.putText(display_frame, "Updating Body Signature...", (display_box[0], display_box[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                with data_lock:
                    if persistent_id not in person_state:
                        person_state[persistent_id] = "UNKNOWN"
                        person_last_time[persistent_id] = now
                        all_tracked_people.add(persistent_id)
                        print(f"✓ New person detected: ID {pid} (Internal: {persistent_id})")

                # State Machine logic for ESCALATION and RECOVERY
                new_state = activity
                if activity == "UNKNOWN":
                    new_state = prev_s
                
                # --- State Transition Refinements ---

                if new_state == "MINOR FALL" and prev_s in ["WALKING", "STANDING", "SITTING", "RECOVERED"] and current_v_velocity < 2.5 and current_velocity < 3.2 and aspect_ratio < 1.7:
                    new_state = "SITTING"

                # If already in a confirmed MAJOR FALL, stay there until recovery
                if prev_s == "MAJOR FALL" and new_state in ["MINOR FALL", "LYING"]:
                    new_state = "MAJOR FALL"

                # SUPPRESS "getting up" misclassification:
                # If they are moving UP (negative v_velocity) and were previously down, 
                # they are likely getting up. Force upright state to prevent false Minor Fall.
                if current_v_velocity < -1.0 and prev_s in ["MAJOR FALL", "MINOR FALL", "LYING", "SLEEPING"]:
                    if new_state in ["MINOR FALL", "LYING"]:
                        new_state = "STANDING" 

                confirm_window = float(settings.get("fall_confirm_window_sec", 10.0))

                # 4. Special case: If in recovery mode, show RECOVERED label briefly
                if persistent_id in recovery_mode:
                    if now > recovery_mode[persistent_id]: 
                        del recovery_mode[persistent_id]
                    else:
                        # If they briefly tilt while getting up/stabilizing, don't trigger a new fall
                        if new_state in ["MINOR FALL", "LYING"]:
                            new_state = "STANDING" # Keep them upright during stabilization
                        
                        # Show RECOVERED label for 5 seconds (of the 10s recovery window)
                        if now < (recovery_mode[persistent_id] - 5.0):
                            new_state = "RECOVERED"

                # 1. Handle Risk States (Lying or Minor Fall)
                is_currently_down = (new_state in ["MAJOR FALL", "MINOR FALL", "LYING", "SLEEPING"])
                
                # INITIAL FALL DETECTION
                if is_currently_down and persistent_id not in active_fall_event:
                    # Only treat controlled downward movement as a fall when it really looks like one.
                    should_alert = False
                    if prev_s in ["WALKING", "STANDING", "RECOVERED"]:
                        if new_state in ["MINOR FALL", "MAJOR FALL"]:
                            should_alert = True
                        elif new_state == "LYING" and current_v_velocity > 4.0:
                            should_alert = True
                    if should_alert:
                        active_fall_event[persistent_id] = "MINOR"
                        send_fall_alert(f"MINOR FALL (ID {pid})", pid, "MINOR FALL", coords=center_coords)
                        with data_lock:
                            fall_events.append({
                                "person": pid, "type": "MINOR FALL", "timestamp": now,
                                "time_str": time.strftime("%H:%M:%S", time.localtime(now))
                            })

                if is_currently_down:
                    recovery_confirm_count[persistent_id] = 0 # Reset recovery counter
                    if active_fall_event.get(persistent_id) == "MINOR" and persistent_id not in minor_fall_start_time:
                        # Track the duration only for confirmed fall events.
                        minor_fall_start_time[persistent_id] = now

                    # Escalation check: only for confirmed falls that remain down for the configured window.
                    if active_fall_event.get(persistent_id) == "MINOR" and is_currently_down and (now - minor_fall_start_time.get(persistent_id, now) > confirm_window):
                        send_fall_alert(f"MAJOR FALL (ID {pid}) - No recovery after {confirm_window:.1f}s", pid, "MAJOR FALL", coords=center_coords)
                        active_fall_event[persistent_id] = "MAJOR"
                        with data_lock:
                            fall_events.append({
                                "person": pid, "type": "MAJOR FALL", "timestamp": now,
                                "time_str": time.strftime("%H:%M:%S", time.localtime(now))
                            })
                
                # 2. Handle Potential Recovery (Upright: WALKING, STANDING, SITTING)
                elif new_state in ["WALKING", "STANDING", "SITTING", "RECOVERED"]:
                    # Require 30 frames (~1s at 30fps) of consistent upright pose before confirming recovery
                    recovery_confirm_count[persistent_id] = recovery_confirm_count.get(persistent_id, 0) + 1
                    
                    if recovery_confirm_count[persistent_id] > 30:
                        if persistent_id in active_fall_event:
                            recovery_mode[persistent_id] = now + 10.0 # 10s stabilization window
                            send_fall_alert(f"RECOVERED (ID {pid})", pid, "RECOVERED", coords=center_coords)
                            with data_lock:
                                fall_events.append({
                                    "person": pid, "type": "RECOVERED", "timestamp": now,
                                    "time_str": time.strftime("%H:%M:%S", time.localtime(now))
                                })
                            del active_fall_event[persistent_id]
                        
                        # Always clear "down" timers if confirmed upright
                        if persistent_id in minor_fall_start_time: del minor_fall_start_time[persistent_id]
                        if persistent_id in lying_start_time: del lying_start_time[persistent_id]
                        recovery_confirm_count[persistent_id] = 0

                # 3. SLEEPING logic (sustained lying or sitting-to-lying)
                if new_state == "LYING":
                    if persistent_id not in lying_start_time: 
                        lying_start_time[persistent_id] = now

                    lying_duration = now - lying_start_time[persistent_id]
                    sleep_threshold = SLEEPING_LYING_CONFIRM_SEC
                    if prev_s in ["SITTING", "SLEEPING"]:
                        sleep_threshold = SLEEPING_FROM_SITTING_SEC
                    elif active_fall_event.get(persistent_id) == "MINOR":
                        sleep_threshold = max(sleep_threshold, confirm_window + 2.0)

                    if lying_duration >= sleep_threshold:
                        new_state = "SLEEPING"
                elif new_state != "SLEEPING":
                    # Clear lying timer if they are not lying or already sleeping
                    if persistent_id in lying_start_time: del lying_start_time[persistent_id]

                # Stabilize the visible activity label so standing/sitting does not flicker.
                new_state = stabilize_activity_state(persistent_id, prev_s, new_state)
                
                # Accumulate time for CURRENT activity (every frame)
                duration = now - person_last_time[persistent_id]
                if duration > 0 and normal_activity_monitoring_enabled:
                    with data_lock:
                        prev_state = person_state.get(persistent_id, "UNKNOWN")
                        if prev_state == "WALKING": walking_time[persistent_id] += duration
                        elif prev_state == "STANDING": standing_time[persistent_id] += duration
                        elif prev_state == "SITTING": sitting_time[persistent_id] += duration
                        elif prev_state == "SLEEPING": sleeping_time[persistent_id] += duration
                    
                person_last_time[persistent_id] = now

                # Update state if changed
                if new_state != "UNKNOWN" and new_state != prev_s:
                    with data_lock:
                        person_state[persistent_id] = new_state
                    if normal_activity_monitoring_enabled:
                        notify_activity_change(persistent_id, new_state)

                # Overlay text
                if settings.get("display_metrics_overlay", True) and not low_power_overlay_suppressed:
                    with data_lock:
                        walk_str = format_duration(walking_time[persistent_id])
                        stand_str = format_duration(standing_time[persistent_id])
                        sit_str = format_duration(sitting_time[persistent_id])
                        sleep_str = format_duration(sleeping_time[persistent_id])
                        
                        # Color code based on state
                        current_s = person_state.get(persistent_id, "UNKNOWN")
                        if "FALL" in current_s:
                            color = (0, 0, 255) # Red for fall
                        elif current_s == "RECOVERED":
                            color = (0, 255, 0) # Green for recovery
                        elif current_s == "WALKING":
                            color = (0, 255, 255) # Yellow for walking
                        elif current_s == "STANDING":
                            color = (255, 200, 0) # Orange for standing
                        else:
                            color = (255, 0, 0) # Blue for sitting/sleeping/unknown
                        
                        draw_person_status_badge(
                            display_frame,
                            display_box,
                            pid,
                            current_s,
                            color,
                            walk_str,
                            stand_str,
                            sit_str,
                            sleep_str
                        )
        else:
            # No keypoints detected
            if settings.get("display_metrics_overlay", True) and not low_power_overlay_suppressed:
                cv2.putText(display_frame, "No pose detected", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        if activity_timing_paused and settings.get("display_metrics_overlay", True) and not low_power_overlay_suppressed:
            draw_notice_banner(display_frame, activity_pause_notice, (0, 165, 255))

        if person_visible_this_frame:
            out_of_frame_since = None
        else:
            if out_of_frame_since is None:
                out_of_frame_since = now
            elif (now - out_of_frame_since) >= LOW_POWER_IDLE_TIMEOUT_SEC:
                enter_low_power_mode(display_frame, w)
                prev_gray = None
                prev_sleep_gray = None
                continue

            if settings.get("display_metrics_overlay", True) and not low_power_overlay_suppressed:
                elapsed_out = now - out_of_frame_since
                remaining_out = max(0.0, LOW_POWER_IDLE_TIMEOUT_SEC - elapsed_out)
                cv2.putText(
                    display_frame,
                    f"OUT OF FRAME - SLEEP IN {remaining_out:.1f}s",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 165, 255),
                    2
                )

        # Clean up people not detected for 30 frames (about 1 second at 30fps)
        ids_to_remove = []
        for persistent_id in list(person_state.keys()):
            # Stability Patch: Don't remove named/important IDs from active tracking state easily
            if persistent_id not in detected_ids and (frame_count - last_detection.get(persistent_id, 0)) > 30:
                if persistent_id in manual_id_map:
                    # Named IDs get a much longer timeout (e.g. 5 minutes) before being cleared from memory
                    if (frame_count - last_detection.get(persistent_id, 0)) > 9000:
                         ids_to_remove.append(persistent_id)
                else:
                    ids_to_remove.append(persistent_id)
        
        with data_lock:
            for persistent_id in ids_to_remove:
                if persistent_id in person_state: del person_state[persistent_id]
                if persistent_id in person_last_time: del person_last_time[persistent_id]
                if persistent_id in last_detection: del last_detection[persistent_id]
                if persistent_id in lying_start_time: del lying_start_time[persistent_id]
                if persistent_id in minor_fall_start_time: del minor_fall_start_time[persistent_id]
                if persistent_id in recovery_mode: del recovery_mode[persistent_id]
                if persistent_id in active_fall_event: del active_fall_event[persistent_id]
                if persistent_id in recovery_confirm_count: del recovery_confirm_count[persistent_id]
                if persistent_id in person_start_pos: del person_start_pos[persistent_id]
                if persistent_id in person_frames_seen: del person_frames_seen[persistent_id]
                if persistent_id in person_is_confirmed: del person_is_confirmed[persistent_id]
                if persistent_id in person_velocity: del person_velocity[persistent_id]
                if persistent_id in person_last_pos: del person_last_pos[persistent_id]
                if persistent_id in activity_transition_candidate: del activity_transition_candidate[persistent_id]
                if persistent_id in activity_transition_count: del activity_transition_count[persistent_id]
                
                # Clean up tracker mapping to prevent stale entries
                yolo_keys = [k for k, v in tracker_to_persistent.items() if v == persistent_id]
                for k in yolo_keys: del tracker_to_persistent[k]
                
                print(f"Removed internal ID {persistent_id} from active tracking")

        # Log progress every 100 frames
        if frame_count % 100 == 0:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            print(f"Frame {frame_count} | FPS: {fps:.1f} | People tracked: {len(person_state)}")

        # Show status message if active
        if time.time() < status_expiry:
            cv2.rectangle(display_frame, (0, 0), (w, 40), (0, 255, 0), -1)
            cv2.putText(display_frame, status_message, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        show_preview_window(display_frame)
        update_stream_frame(display_frame)

        if is_edge_mode() and (time.time() - last_edge_sync_at) >= 2:
            send_edge_report_snapshot()
            last_edge_sync_at = time.time()

    except Exception as e:
        print(f"Error at frame {frame_count}: {e}")
        import traceback
        traceback.print_exc()
        break

cap.release()
print("Program exited.")
sys.exit(0)
