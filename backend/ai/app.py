"""
Flask Web Server for Smart Tourist Safety System
Real-time person detection with DUAL CAMERA support
"""
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2
import time
from loguru import logger
import sys
import threading
import json
import base64
import hashlib
from datetime import datetime, timedelta
import os
import uuid
import numpy as np
from collections import deque, Counter


# Local imports
sys.path.append('../')
from core.tracker import ByteTrack
from core.detector import PersonDetector
from core.reid import ReIDModel, cosine_similarity
from utils.video_reader import VideoReader
from database.utils import generate_did, generate_feature_id, compute_id_hash
from database.db_manager import DatabaseManager
from database.mongo_manager import MongoManager
from database.utils import generate_session_id
from services import fabric_client


# ----------------------------------------------------
# LOGGING CONFIG
# ----------------------------------------------------
logger.remove()
logger.add(sys.stdout, colorize=True,
           format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
logger.add("logs/app.log", rotation="10 MB")

# ----------------------------------------------------
# FLASK APP
# ----------------------------------------------------
app = Flask(__name__)
CORS(app)
# using thread async mode (eventlet/gevent optional)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ----------------------------------------------------
# PATHS
# ----------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# ----------------------------------------------------
# GLOBALS
# ----------------------------------------------------
detector = None
reid_model = None

video_reader_1 = None
video_reader_2 = None

is_running_1 = False
is_running_2 = False

# Keep trackers as globals but don't depend on them to always exist while generator runs.
tracker_1 = None
tracker_2 = None

# Phase 3.1 – Mongo tracking session state
active_sessions_cam1 = {}   # tracking_id -> session_id
active_sessions_cam2 = {}   # tracking_id -> session_id

# Re-ID config/state
REID_MODEL_NAME = "osnet_x0_25"
REID_EVERY_N_FRAMES = 8
REID_SIM_THRESHOLD = 0.80
REID_STABLE_WINDOW = 3
REID_STABLE_REQUIRED = 2
REID_MIN_CONF = 0.30
REID_MIN_BOX_WIDTH = 40
REID_MIN_BOX_HEIGHT = 80
REID_MIN_BOX_AREA_RATIO = 0.002
REID_LOG_EVERY_N_FRAMES = 30
REID_LOG_MIN_SCORE = 0.30
REID_GALLERY_REFRESH_SEC = 60
REID_MAX_TRACKS_PER_CYCLE = 4

reid_gallery = {}          # did -> embedding (np.ndarray)
reid_gallery_updated_at = 0.0
reid_matches_cam1 = {}     # tracking_id -> did
reid_matches_cam2 = {}     # tracking_id -> did
reid_match_scores_cam1 = {}  # tracking_id -> score
reid_match_scores_cam2 = {}  # tracking_id -> score
reid_votes_cam1 = {}       # tracking_id -> deque of dids
reid_votes_cam2 = {}       # tracking_id -> deque of dids
best_match_score_by_session = {}  # session_id -> best score

webcam_frame_count_1 = 0
webcam_frame_count_2 = 0
webcam_start_time_1 = None
webcam_start_time_2 = None


current_stats = {
    'camera1': {'fps': 0, 'persons': 0, 'detections': []},
    'camera2': {'fps': 0, 'persons': 0, 'detections': []},
    'total_persons': 0
}

# ----------------------------------------------------
# DATABASE INITIALIZATION
# try to initialize DBs but continue even if Mongo is down (we'll handle None checks)
# ----------------------------------------------------
try:
    db_manager = DatabaseManager()
    try:
        mongo_manager = MongoManager()
    except Exception as e:
        # Don't crash server if Mongo is not running; log and set to None.
        logger.error(f"MongoManager init failed: {e}")
        mongo_manager = None
    logger.success("Database managers initialized (sqlite ok, mongo may be None)")
except Exception as e:
    logger.error(f"Database initialization failed: {e}")
    db_manager = None
    mongo_manager = None

# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------
def safe_create_local_tracker(global_tracker, **kwargs):
    """
    Return a tracker instance to use in the generator loop.
    If a global tracker exists, use it. Otherwise create a temporary local one.
    This prevents race conditions where the global is set to None while generator is running.
    """
    if global_tracker is not None:
        return global_tracker, False  # (tracker_to_use, is_local_flag)
    else:
        return ByteTrack(**kwargs), True

def save_face_images(did, images):
    """
    Save base64 face images to disk and return public URLs.
    """
    saved_urls = []
    if not images:
        return saved_urls

    dest_dir = os.path.join(UPLOAD_DIR, did)
    os.makedirs(dest_dir, exist_ok=True)

    for idx, image_data in enumerate(images):
        if not image_data:
            continue
        try:
            header = ""
            b64data = image_data
            if isinstance(image_data, str) and image_data.startswith("data:"):
                header, b64data = image_data.split(",", 1)

            ext = "jpg"
            if "image/png" in header:
                ext = "png"
            elif "image/webp" in header:
                ext = "webp"
            elif "image/jpeg" in header or "image/jpg" in header:
                ext = "jpg"

            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{idx}_{uuid.uuid4().hex[:6]}.{ext}"
            file_path = os.path.join(dest_dir, filename)

            with open(file_path, "wb") as f:
                f.write(base64.b64decode(b64data))

            saved_urls.append(f"/uploads/{did}/{filename}")
        except Exception as e:
            logger.warning("Failed to save face image {}: {}", idx, e)

    return saved_urls


def compute_sha256_file(file_path):
    try:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return f"sha256:{hasher.hexdigest()}"
    except Exception as e:
        logger.warning("Failed to hash file {}: {}", file_path, e)
        return None


def resolve_upload_path(url):
    if not url or not isinstance(url, str):
        return None
    if not url.startswith("/uploads/"):
        return None
    rel = url[len("/uploads/"):]
    return os.path.join(UPLOAD_DIR, rel)


def compute_face_hash_from_urls(urls):
    if not urls:
        return None
    for url in urls:
        file_path = resolve_upload_path(url)
        if file_path and os.path.exists(file_path):
            return compute_sha256_file(file_path)
    return None


def utc_now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def extract_largest_person_crop(img_bgr, pad_ratio=0.08):
    """
    Try to crop the largest detected person from an image.
    Returns None if no person is detected or detector isn't ready.
    """
    if detector is None or img_bgr is None:
        return None
    try:
        detections = detector.detect(img_bgr)
    except Exception as e:
        logger.warning("Person crop detection failed: {}", e)
        return None

    if not detections:
        return None

    best = max(
        detections,
        key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1])
    )
    x1, y1, x2, y2 = map(int, best["bbox"])
    h, w = img_bgr.shape[:2]
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return img_bgr[y1:y2, x1:x2]

def refresh_reid_gallery(force=False):
    """
    Load Re-ID gallery from MongoDB (cached).
    """
    global reid_gallery, reid_gallery_updated_at
    if mongo_manager is None:
        return
    now = time.time()
    if not force and (now - reid_gallery_updated_at) < REID_GALLERY_REFRESH_SEC:
        return

    try:
        features = mongo_manager.get_all_tourist_features()
    except Exception as e:
        logger.warning("Failed to refresh Re-ID gallery: {}", e)
        return

    gallery = {}
    for feat in features:
        did = feat.get("did")
        emb = feat.get("reid_embedding")
        if not did or not emb:
            continue
        vec = np.array(emb, dtype=np.float32)
        if np.linalg.norm(vec) == 0:
            continue
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        gallery.setdefault(did, []).append(vec)

    merged = {}
    for did, vecs in gallery.items():
        avg = np.mean(np.stack(vecs), axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-8)
        merged[did] = avg.astype(np.float32)

    reid_gallery = merged
    reid_gallery_updated_at = now

def compute_reid_embedding_from_paths(paths, max_images=3, prefer_body=True):
    """
    Compute a single normalized embedding from saved image files.
    """
    if reid_model is None:
        return None

    embeddings = []
    for path in paths[:max_images]:
        try:
            abs_path = os.path.join(BASE_DIR, path.lstrip("/"))
            img = cv2.imread(abs_path)
            if img is None:
                continue
            crop = None
            if prefer_body:
                crop = extract_largest_person_crop(img)
                if crop is None:
                    logger.info("No full-body detected; using raw image for Re-ID embedding.")
            emb = reid_model.extract(crop if crop is not None else img)
            if emb is not None:
                embeddings.append(emb)
        except Exception as e:
            logger.warning("Re-ID embedding failed for {}: {}", path, e)

    if not embeddings:
        return None

    avg = np.mean(np.stack(embeddings), axis=0)
    avg = avg / (np.linalg.norm(avg) + 1e-8)
    return avg.astype(np.float32)

def match_reid(embedding):
    """
    Return (best_did, best_score) for a given embedding.
    """
    if embedding is None or not reid_gallery:
        return None, 0.0

    best_did = None
    best_score = 0.0
    for did, vec in reid_gallery.items():
        score = cosine_similarity(embedding, vec)
        if score > best_score:
            best_score = score
            best_did = did

    return best_did, best_score

def is_reid_candidate(track, frame_w, frame_h):
    """
    Apply quality gates before Re-ID to reduce false matches and CPU usage.
    """
    try:
        conf = float(track.get("confidence", 0.0))
        if conf < REID_MIN_CONF:
            return False

        x1, y1, x2, y2 = track.get("bbox", [0, 0, 0, 0])
        w = max(0, int(x2) - int(x1))
        h = max(0, int(y2) - int(y1))
        if w < REID_MIN_BOX_WIDTH or h < REID_MIN_BOX_HEIGHT:
            return False

        area = w * h
        if area < (frame_w * frame_h * REID_MIN_BOX_AREA_RATIO):
            return False

        return True
    except Exception:
        return False

def update_reid_votes(vote_map, tracking_id, did):
    """
    Stabilize matches by requiring the same DID to appear
    multiple times in a sliding window.
    """
    q = vote_map.get(tracking_id)
    if q is None:
        q = deque(maxlen=REID_STABLE_WINDOW)
        vote_map[tracking_id] = q
    q.append(did)

    counts = Counter([d for d in q if d])
    if not counts:
        return None

    best_did, count = counts.most_common(1)[0]
    if count >= REID_STABLE_REQUIRED:
        return best_did
    return None

def cleanup_reid_maps(current_ids, *maps):
    """
    Remove stale tracking IDs from Re-ID maps to avoid memory growth.
    """
    for m in maps:
        for tid in list(m.keys()):
            if tid not in current_ids:
                m.pop(tid, None)

def attach_reid_to_list(items, match_map, score_map):
    """
    Return a new list with DID/score attached per tracking_id.
    """
    out = []
    for obj in items:
        if not isinstance(obj, dict):
            continue
        new_obj = dict(obj)
        tid = new_obj.get("tracking_id")
        if tid in match_map:
            new_obj["did"] = match_map[tid]
            score = score_map.get(tid)
            if score is not None:
                new_obj["match_confidence"] = float(score)
        out.append(new_obj)
    return out

def log_reid_candidate(camera_id, tracking_id, did, score, frame_id):
    if did is None:
        return
    if frame_id % REID_LOG_EVERY_N_FRAMES != 0:
        return
    if score < REID_LOG_MIN_SCORE:
        return
    logger.info("Re-ID candidate {} TID {} -> {} (score {:.3f})", camera_id, tracking_id, did, score)

def save_reid_match_crop(session_id, crop_bgr):
    """
    Save best-match crop for audit/debugging and return its URL.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None

    dest_dir = os.path.join(UPLOAD_DIR, "reid_matches", session_id)
    os.makedirs(dest_dir, exist_ok=True)
    filename = "best.jpg"
    file_path = os.path.join(dest_dir, filename)

    try:
        cv2.imwrite(file_path, crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return f"/uploads/reid_matches/{session_id}/{filename}"
    except Exception as e:
        logger.warning("Failed to save Re-ID crop for {}: {}", session_id, e)
        return None

def maybe_store_best_match(session_id, did, score, crop_bgr, embedding):
    """
    Store best match crop/embedding per session if the score improves.
    """
    if not session_id or mongo_manager is None:
        return

    prev = best_match_score_by_session.get(session_id, 0.0)
    if score <= prev:
        return

    image_url = save_reid_match_crop(session_id, crop_bgr)
    payload = {
        "did": did,
        "score": float(score),
        "image_path": image_url,
        "updated_at": datetime.now()
    }
    if embedding is not None:
        payload["embedding"] = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

    try:
        mongo_manager.update_tracking_best_match(session_id, payload)
        best_match_score_by_session[session_id] = float(score)
    except Exception as e:
        logger.warning("Failed to store best match for {}: {}", session_id, e)

def update_last_seen_sqlite(did, camera_id):
    if not db_manager or not did:
        return
    try:
        db_manager.update_last_seen(did, camera_id)
    except Exception as e:
        logger.warning("Failed to update last_seen for {}: {}", did, e)
    
def handle_tracking_persistence(mongo, camera_id, frame_id, tracks, active_sessions, sample_every=3):
    """
    Phase 3.1: Persist tracking sessions into MongoDB.
    Safe: does nothing if mongo is None.
    """
    if mongo is None:
        return

    # Downsample tracklet updates to reduce DB growth.
    do_sample = (sample_every is None) or (sample_every <= 1) or (frame_id % sample_every == 0)
    current_ids = set()

    for track in tracks:
        tid = track["tracking_id"]
        current_ids.add(tid)

        if tid not in active_sessions:
            session_id = generate_session_id(camera_id, tid)
            mongo.start_tracking_session({
                "session_id": session_id,
                "camera_id": camera_id,
                "tracking_id": tid,
                "bbox": track["bbox"],
                "confidence": track["confidence"]
            })
            active_sessions[tid] = session_id
        else:
            if do_sample:
                mongo.update_tracking_session(
                    active_sessions[tid],
                    {
                        "frame_id": frame_id,
                        "bbox": track["bbox"],
                        "confidence": track["confidence"]
                    }
                )

    ended = set(active_sessions.keys()) - current_ids
    for tid in ended:
        mongo.end_tracking_session(active_sessions[tid])
        del active_sessions[tid]


# ----------------------------------------------------
# CAMERA STREAM FUNCTIONS (robust & defensive)
# ----------------------------------------------------
def generate_frames_camera1():
    global detector, video_reader_1, is_running_1, tracker_1, current_stats

    if not video_reader_1:
        logger.error("Camera 1 not initialized")
        # Return a small generator that yields nothing
        if False:
            yield b''
        return

    frame_count = 0
    start_time = time.time()
    process_every = 2
    last_tracked = []

    # Use a local tracker if global is None to avoid AttributeError when global cleared.
    # But prefer global so state (IDs) is maintained.
    while is_running_1:
        try:
            ret, frame = video_reader_1.read()
            if not ret or frame is None:
                logger.info("Camera 1 read returned no frame, stopping generator")
                # gracefully end stream
                is_running_1 = False
                break

            frame_count += 1
            resized = cv2.resize(frame, (640, 480))

            # get tracker safely
            t, is_local = safe_create_local_tracker(tracker_1, max_age=30, min_hits=3, iou_threshold=0.3)

            if frame_count % process_every == 0:
                # ensure detector exists
                if detector is None:
                    detections = []
                else:
                    detections = detector.detect(resized)

                # update tracking (guarded)
                try:
                    tracked = t.update(detections)
                except Exception as e:
                    logger.exception("Tracker update failed (camera1): {}", e)
                    tracked = last_tracked

                last_tracked = tracked
            else:
                tracked = last_tracked

            tracks = []

            for obj in tracked:
                # obj is already a dict
                if not isinstance(obj, dict):
                    continue

                bbox = obj.get("bbox")
                tracking_id = obj.get("tracking_id")
                confidence = obj.get("confidence")

                if bbox is None or tracking_id is None:
                    continue

                tracks.append({
                    "tracking_id": int(tracking_id),
                    "bbox": list(map(int, bbox)),
                    "confidence": float(confidence)
                })

            current_ids = {t["tracking_id"] for t in tracks}
            cleanup_reid_maps(current_ids, reid_votes_cam1, reid_matches_cam1, reid_match_scores_cam1)

            # Persist tracking sessions to MongoDB (Phase 3.1)
            try:
                handle_tracking_persistence(
                    mongo_manager,
                    "camera1",
                    frame_count,
                    tracks,
                    active_sessions_cam1
                )
            except Exception as e:
                logger.exception("Tracking persistence failed (camera1): {}", e)

            # Re-ID matching (Phase 5) - run periodically to avoid CPU spikes
            if reid_model and mongo_manager and (frame_count % REID_EVERY_N_FRAMES == 0):
                try:
                    refresh_reid_gallery()
                    processed = 0
                    h, w = resized.shape[:2]
                    for track in tracks:
                        if processed >= REID_MAX_TRACKS_PER_CYCLE:
                            break
                        tid = track["tracking_id"]
                        if tid in reid_matches_cam1:
                            continue
                        if not is_reid_candidate(track, w, h):
                            update_reid_votes(reid_votes_cam1, tid, None)
                            continue

                        x1, y1, x2, y2 = track["bbox"]
                        x1 = max(0, min(w - 1, x1))
                        y1 = max(0, min(h - 1, y1))
                        x2 = max(0, min(w, x2))
                        y2 = max(0, min(h, y2))
                        if x2 <= x1 or y2 <= y1:
                            update_reid_votes(reid_votes_cam1, tid, None)
                            continue

                        crop = resized[y1:y2, x1:x2]
                        emb = reid_model.extract(crop)
                        did, score = match_reid(emb)
                        log_reid_candidate("camera1", tid, did, score, frame_count)
                        stable_did = update_reid_votes(
                            reid_votes_cam1,
                            tid,
                            did if (did and score >= REID_SIM_THRESHOLD) else None
                        )
                        if did and score >= REID_SIM_THRESHOLD and stable_did == did:
                            session_id = active_sessions_cam1.get(tid)
                            if session_id:
                                mongo_manager.link_tracking_to_tourist(session_id, did, score)
                                fabric_client.link_tracking_session_async(session_id, did, score, utc_now_iso())
                                reid_matches_cam1[tid] = did
                                reid_match_scores_cam1[tid] = float(score)
                                maybe_store_best_match(session_id, did, score, crop, emb)
                                update_last_seen_sqlite(did, "camera1")
                        processed += 1
                except Exception as e:
                    logger.warning("Re-ID failed (camera1): {}", e)

            # If tracker used was local (temporary), we drop it (do not assign back)
            if not is_local:
                # global tracker_1 remains as-is
                pass

            annotated_tracked = attach_reid_to_list(tracked or [], reid_matches_cam1, reid_match_scores_cam1)

            # Draw detections (detector handles drawing)
            if detector:
                resized = detector.draw_tracked_detections(resized, annotated_tracked)

            # compute fps safely
            elapsed = max(1e-6, time.time() - start_time)
            fps = frame_count / elapsed

            # update stats
            current_stats["camera1"] = {
                "fps": round(fps, 1),
                "persons": len(tracked) if tracked is not None else 0,
                "detections": annotated_tracked
            }

            # emit combined stats periodically (every 5 frames)
            if frame_count % 5 == 0:
                update_combined_stats()

            ret2, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret2:
                logger.warning("Failed to encode frame for camera1")
                continue

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   buffer.tobytes() + b"\r\n")

        except GeneratorExit:
            # client disconnected, break
            logger.info("GeneratorExit in camera1 generator - client disconnected")
            break
        except Exception as e:
            logger.exception("Unhandled exception in generate_frames_camera1: {}", e)
            # avoid tight error loop — sleep slightly
            time.sleep(0.1)
            # continue loop or break — better to break to let client reconnect cleanly
            break

    # cleanup local resources if any
    try:
        if video_reader_1:
            # don't release video_reader here if other threads rely on it; caller stop_camera handles release
            pass
    except Exception:
        pass


def generate_frames_camera2():
    global detector, video_reader_2, is_running_2, tracker_2, current_stats

    if not video_reader_2:
        logger.error("Camera 2 not initialized")
        if False:
            yield b''
        return

    frame_count = 0
    start_time = time.time()
    process_every = 2
    last_tracked = []

    while is_running_2:
        try:
            ret, frame = video_reader_2.read()
            if not ret or frame is None:
                logger.info("Camera 2 read returned no frame, stopping generator")
                is_running_2 = False
                break

            frame_count += 1
            resized = cv2.resize(frame, (640, 480))

            t, is_local = safe_create_local_tracker(tracker_2, max_age=30, min_hits=3, iou_threshold=0.3)

            if frame_count % process_every == 0:
                if detector is None:
                    detections = []
                else:
                    detections = detector.detect(resized)

                try:
                    tracked = t.update(detections)
                except Exception as e:
                    logger.exception("Tracker update failed (camera2): {}", e)
                    tracked = last_tracked

                last_tracked = tracked
            else:
                tracked = last_tracked

            tracks = []

            for obj in tracked:
                # obj is already a dict
                if not isinstance(obj, dict):
                    continue

                bbox = obj.get("bbox")
                tracking_id = obj.get("tracking_id")
                confidence = obj.get("confidence")

                if bbox is None or tracking_id is None:
                    continue

                tracks.append({
                    "tracking_id": int(tracking_id),
                    "bbox": list(map(int, bbox)),
                    "confidence": float(confidence)
                })

            current_ids = {t["tracking_id"] for t in tracks}
            cleanup_reid_maps(current_ids, reid_votes_cam2, reid_matches_cam2, reid_match_scores_cam2)

            # Persist tracking sessions to MongoDB (Phase 3.1)
            try:
                handle_tracking_persistence(
                    mongo_manager,
                    "camera2",
                    frame_count,
                    tracks,
                    active_sessions_cam2
                )
            except Exception as e:
                logger.exception("Tracking persistence failed (camera2): {}", e)

            # Re-ID matching (Phase 5)
            if reid_model and mongo_manager and (frame_count % REID_EVERY_N_FRAMES == 0):
                try:
                    refresh_reid_gallery()
                    processed = 0
                    h, w = resized.shape[:2]
                    for track in tracks:
                        if processed >= REID_MAX_TRACKS_PER_CYCLE:
                            break
                        tid = track["tracking_id"]
                        if tid in reid_matches_cam2:
                            continue
                        if not is_reid_candidate(track, w, h):
                            update_reid_votes(reid_votes_cam2, tid, None)
                            continue

                        x1, y1, x2, y2 = track["bbox"]
                        x1 = max(0, min(w - 1, x1))
                        y1 = max(0, min(h - 1, y1))
                        x2 = max(0, min(w, x2))
                        y2 = max(0, min(h, y2))
                        if x2 <= x1 or y2 <= y1:
                            update_reid_votes(reid_votes_cam2, tid, None)
                            continue

                        crop = resized[y1:y2, x1:x2]
                        emb = reid_model.extract(crop)
                        did, score = match_reid(emb)
                        log_reid_candidate("camera2", tid, did, score, frame_count)
                        stable_did = update_reid_votes(
                            reid_votes_cam2,
                            tid,
                            did if (did and score >= REID_SIM_THRESHOLD) else None
                        )
                        if did and score >= REID_SIM_THRESHOLD and stable_did == did:
                            session_id = active_sessions_cam2.get(tid)
                            if session_id:
                                mongo_manager.link_tracking_to_tourist(session_id, did, score)
                                fabric_client.link_tracking_session_async(session_id, did, score, utc_now_iso())
                                reid_matches_cam2[tid] = did
                                reid_match_scores_cam2[tid] = float(score)
                                maybe_store_best_match(session_id, did, score, crop, emb)
                                update_last_seen_sqlite(did, "camera2")
                        processed += 1
                except Exception as e:
                    logger.warning("Re-ID failed (camera2): {}", e)


            annotated_tracked = attach_reid_to_list(tracked or [], reid_matches_cam2, reid_match_scores_cam2)

            if detector:
                resized = detector.draw_tracked_detections(resized, annotated_tracked)

            elapsed = max(1e-6, time.time() - start_time)
            fps = frame_count / elapsed

            current_stats["camera2"] = {
                "fps": round(fps, 1),
                "persons": len(tracked) if tracked is not None else 0,
                "detections": annotated_tracked
            }

            if frame_count % 5 == 0:
                update_combined_stats()

            ret2, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ret2:
                logger.warning("Failed to encode frame for camera2")
                continue

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   buffer.tobytes() + b"\r\n")

        except GeneratorExit:
            logger.info("GeneratorExit in camera2 generator - client disconnected")
            break
        except Exception as e:
            logger.exception("Unhandled exception in generate_frames_camera2: {}", e)
            time.sleep(0.1)
            break

# ----------------------------------------------------
# STATS EMISSION
# ----------------------------------------------------
def update_combined_stats():
    # sum persons safely
    try:
        total = int(current_stats.get("camera1", {}).get("persons", 0)) + int(current_stats.get("camera2", {}).get("persons", 0))
    except Exception:
        total = 0
    current_stats["total_persons"] = total

    all_dets = []

    for det in current_stats.get("camera1", {}).get("detections", []) or []:
        # ensure dict-like
        try:
            all_dets.append({**(det or {}), "camera": "Camera 1"})
        except Exception:
            all_dets.append({"camera": "Camera 1", "raw": str(det)})

    for det in current_stats.get("camera2", {}).get("detections", []) or []:
        try:
            all_dets.append({**(det or {}), "camera": "Camera 2"})
        except Exception:
            all_dets.append({"camera": "Camera 2", "raw": str(det)})

    # emit robust payload
    try:
        socketio.emit("stats", {
            "camera1": current_stats.get("camera1", {}),
            "camera2": current_stats.get("camera2", {}),
            "total_persons": total,
            "all_detections": all_dets
        })
    except Exception as e:
        logger.exception("Failed to emit stats via socketio: {}", e)

# ----------------------------------------------------
# ROUTES
# ----------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/register")
def register_page():
    """
    Render tourist registration page.
    """
    try:
        return render_template("register.html")
    except Exception as e:
        
        logger.exception("Failed to render register page: {}", e)
        return jsonify({"error": "Failed to load page"}), 500

@app.route("/uploads/<path:filename>")
def serve_uploads(filename):
    """
    Serve stored face images.
    """
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/video_feed_1")
def video_feed_1():
    return Response(generate_frames_camera1(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_2")
def video_feed_2():
    return Response(generate_frames_camera2(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/start_camera", methods=["POST"])
def start_camera():
    """
    Start camera stream. Accepts payload:
    { camera: 1|2, source_type: 'webcam'|'video'|'esp32cam', video_path: '...', esp32_url: '...' }
    """
    global video_reader_1, video_reader_2
    global is_running_1, is_running_2
    global tracker_1, tracker_2

    data = request.get_json() or {}
    cam = int(data.get("camera", 1))
    stype = data.get("source_type", "webcam")

    if stype == "webcam":
        source = cam - 1  # device index
    elif stype == "video":
        source = data.get("video_path")
    else:
        source = data.get("esp32_url")

    try:
        if cam == 1:
            # close previous if exists
            if video_reader_1:
                try:
                    video_reader_1.release()
                except Exception:
                    pass
            video_reader_1 = VideoReader(source)
            # create global tracker object so generator can reuse same instance
            tracker_1 = ByteTrack()
            is_running_1 = True
            info = video_reader_1.get_info()
        else:
            if video_reader_2:
                try:
                    video_reader_2.release()
                except Exception:
                    pass
            video_reader_2 = VideoReader(source)
            tracker_2 = ByteTrack()
            is_running_2 = True
            info = video_reader_2.get_info()

        logger.info(f"Started camera {cam} (source={source})")
        # return video metadata to client
        return jsonify({"status": "started", "camera": cam, "video_info": info})
    except Exception as e:
        logger.exception("Failed to start camera %s: %s", cam, e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/stop_camera", methods=["POST"])
def stop_camera():
    """
    Stop camera stream for camera 1 or 2.
    We set is_running flag to False and release VideoReader.
    We avoid setting tracker global to None immediately to prevent generator race;
    instead keep it but allow it to be reused or GC later.
    """
    global video_reader_1, video_reader_2
    global is_running_1, is_running_2
    global tracker_1, tracker_2
    global reid_matches_cam1, reid_matches_cam2
    global reid_match_scores_cam1, reid_match_scores_cam2
    global reid_votes_cam1, reid_votes_cam2

    cam = int(request.get_json().get("camera", 1))

    try:
        if cam == 1:
            is_running_1 = False
            if video_reader_1:
                try:
                    video_reader_1.release()
                except Exception:
                    logger.exception("Error releasing video_reader_1")
                video_reader_1 = None
            # do not immediately destroy tracker_1; let it be GC'd later
            tracker_1 = None
            reid_matches_cam1 = {}
            reid_match_scores_cam1 = {}
            reid_votes_cam1 = {}
        else:
            is_running_2 = False
            if video_reader_2:
                try:
                    video_reader_2.release()
                except Exception:
                    logger.exception("Error releasing video_reader_2")
                video_reader_2 = None
            tracker_2 = None
            reid_matches_cam2 = {}
            reid_match_scores_cam2 = {}
            reid_votes_cam2 = {}

        logger.info(f"Stopped camera {cam}")
        return jsonify({"status": "stopped", "camera": cam})
    except Exception as e:
        logger.exception("Failed to stop camera %s: %s", cam, e)
        return jsonify({"error": str(e)}), 500

@app.route("/api/stop_webcam", methods=["POST"])
def stop_webcam():
    """
    Stop webcam-based tracking and reset state.
    """
    global is_running_1, is_running_2
    global tracker_1, tracker_2
    global webcam_start_time_1, webcam_start_time_2
    global webcam_frame_count_1, webcam_frame_count_2
    global active_sessions_cam1, active_sessions_cam2
    global reid_matches_cam1, reid_matches_cam2
    global reid_match_scores_cam1, reid_match_scores_cam2
    global reid_votes_cam1, reid_votes_cam2

    cam = int(request.get_json().get("camera", 1))

    if cam == 1:
        is_running_1 = False
        tracker_1 = None
        webcam_start_time_1 = None
        webcam_frame_count_1 = 0
        active_sessions_cam1 = {}
        reid_matches_cam1 = {}
        reid_match_scores_cam1 = {}
        reid_votes_cam1 = {}
    else:
        is_running_2 = False
        tracker_2 = None
        webcam_start_time_2 = None
        webcam_frame_count_2 = 0
        active_sessions_cam2 = {}
        reid_matches_cam2 = {}
        reid_match_scores_cam2 = {}
        reid_votes_cam2 = {}

    return jsonify({"status": "stopped", "camera": cam})

@app.route("/api/webcam_frame", methods=["POST"])
def webcam_frame():
    """
    Receive a webcam frame from the browser and run detection/tracking.
    """
    global detector
    global tracker_1, tracker_2
    global is_running_1, is_running_2
    global webcam_frame_count_1, webcam_frame_count_2
    global webcam_start_time_1, webcam_start_time_2

    data = request.get_json() or {}
    cam = int(data.get("camera", 1))
    image_data = data.get("image")
    frame_id = int(data.get("frame_id", 0))

    if not image_data:
        return jsonify({"error": "No image data"}), 400

    try:
        if isinstance(image_data, str) and image_data.startswith("data:"):
            image_data = image_data.split(",", 1)[1]
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"Decode failed: {e}"}), 400

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    # Keep original size (no forced resize) to preserve aspect ratio
    resized = frame
    frame_h, frame_w = resized.shape[:2]

    if cam == 1:
        if tracker_1 is None:
            tracker_1 = ByteTrack()
        is_running_1 = True
    else:
        if tracker_2 is None:
            tracker_2 = ByteTrack()
        is_running_2 = True

    if detector is None:
        detections = []
    else:
        detections = detector.detect(resized)

    tracker = tracker_1 if cam == 1 else tracker_2
    try:
        tracked = tracker.update(detections)
    except Exception as e:
        logger.exception("Tracker update failed (webcam cam%s): {}", cam, e)
        tracked = []

    tracks = []
    for obj in tracked:
        if not isinstance(obj, dict):
            continue
        bbox = obj.get("bbox")
        tracking_id = obj.get("tracking_id")
        confidence = obj.get("confidence")
        if bbox is None or tracking_id is None:
            continue
        tracks.append({
            "tracking_id": int(tracking_id),
            "bbox": list(map(int, bbox)),
            "confidence": float(confidence)
        })

    current_ids = {t["tracking_id"] for t in tracks}
    if cam == 1:
        cleanup_reid_maps(current_ids, reid_votes_cam1, reid_matches_cam1, reid_match_scores_cam1)
    else:
        cleanup_reid_maps(current_ids, reid_votes_cam2, reid_matches_cam2, reid_match_scores_cam2)

    # Persist tracking sessions
    try:
        if cam == 1:
            handle_tracking_persistence(
                mongo_manager,
                "camera1",
                frame_id,
                tracks,
                active_sessions_cam1
            )
        else:
            handle_tracking_persistence(
                mongo_manager,
                "camera2",
                frame_id,
                tracks,
                active_sessions_cam2
            )
    except Exception as e:
        logger.warning("Tracking persistence failed (webcam cam%s): {}", cam, e)

    # Re-ID matching (periodic)
    if reid_model and mongo_manager and (frame_id % REID_EVERY_N_FRAMES == 0):
        try:
            refresh_reid_gallery()
            processed = 0
            h, w = resized.shape[:2]
            for track in tracks:
                if processed >= REID_MAX_TRACKS_PER_CYCLE:
                    break
                tid = track["tracking_id"]
                if cam == 1 and tid in reid_matches_cam1:
                    continue
                if cam == 2 and tid in reid_matches_cam2:
                    continue
                if not is_reid_candidate(track, w, h):
                    if cam == 1:
                        update_reid_votes(reid_votes_cam1, tid, None)
                    else:
                        update_reid_votes(reid_votes_cam2, tid, None)
                    continue

                x1, y1, x2, y2 = track["bbox"]
                x1 = max(0, min(w - 1, x1))
                y1 = max(0, min(h - 1, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))
                if x2 <= x1 or y2 <= y1:
                    if cam == 1:
                        update_reid_votes(reid_votes_cam1, tid, None)
                    else:
                        update_reid_votes(reid_votes_cam2, tid, None)
                    continue

                crop = resized[y1:y2, x1:x2]
                emb = reid_model.extract(crop)
                did, score = match_reid(emb)
                log_reid_candidate(f"webcam{cam}", tid, did, score, frame_id)
                stable_did = update_reid_votes(
                    reid_votes_cam1 if cam == 1 else reid_votes_cam2,
                    tid,
                    did if (did and score >= REID_SIM_THRESHOLD) else None
                )
                if did and score >= REID_SIM_THRESHOLD and stable_did == did:
                    session_id = (active_sessions_cam1 if cam == 1 else active_sessions_cam2).get(tid)
                    if session_id:
                        mongo_manager.link_tracking_to_tourist(session_id, did, score)
                        fabric_client.link_tracking_session_async(session_id, did, score, utc_now_iso())
                        if cam == 1:
                            reid_matches_cam1[tid] = did
                            reid_match_scores_cam1[tid] = float(score)
                        else:
                            reid_matches_cam2[tid] = did
                            reid_match_scores_cam2[tid] = float(score)
                        maybe_store_best_match(session_id, did, score, crop, emb)
                        update_last_seen_sqlite(did, "camera1" if cam == 1 else "camera2")
                processed += 1
        except Exception as e:
            logger.warning("Re-ID failed (webcam cam%s): {}", cam, e)

    if cam == 1:
        annotated_tracked = attach_reid_to_list(tracked or [], reid_matches_cam1, reid_match_scores_cam1)
        tracks_with_match = attach_reid_to_list(tracks, reid_matches_cam1, reid_match_scores_cam1)
    else:
        annotated_tracked = attach_reid_to_list(tracked or [], reid_matches_cam2, reid_match_scores_cam2)
        tracks_with_match = attach_reid_to_list(tracks, reid_matches_cam2, reid_match_scores_cam2)

    # Update stats and fps
    if cam == 1:
        if webcam_start_time_1 is None:
            webcam_start_time_1 = time.time()
        webcam_frame_count_1 += 1
        elapsed = max(1e-6, time.time() - webcam_start_time_1)
        fps = webcam_frame_count_1 / elapsed
        current_stats["camera1"] = {
            "fps": round(fps, 1),
            "persons": len(tracked),
            "detections": annotated_tracked
        }
    else:
        if webcam_start_time_2 is None:
            webcam_start_time_2 = time.time()
        webcam_frame_count_2 += 1
        elapsed = max(1e-6, time.time() - webcam_start_time_2)
        fps = webcam_frame_count_2 / elapsed
        current_stats["camera2"] = {
            "fps": round(fps, 1),
            "persons": len(tracked),
            "detections": annotated_tracked
        }

    update_combined_stats()

    return jsonify({
        "success": True,
        "detections": tracks_with_match,
        "frame_width": frame_w,
        "frame_height": frame_h
    })

@app.route("/api/register_tourist", methods=["POST"])
def register_tourist():
    global db_manager, mongo_manager

    if not db_manager:
        return jsonify({"error": "DB not initialized"}), 500

    data = request.get_json()

    did = generate_did("tourist")
    id_hash, salt = compute_id_hash(data["id_number"])
    visit_duration = data.get("visit_duration")
    expected_exit_timestamp = None
    try:
        if visit_duration:
            days = int(visit_duration)
            if days > 0:
                expected_exit_timestamp = (datetime.now() + timedelta(days=days)).isoformat()
    except (TypeError, ValueError):
        expected_exit_timestamp = None

    tourist_data = {
        "did": did,
        "id_hash": id_hash,
        "name": data["name"],
        "id_type": data["id_type"],
        "id_number": data["id_number"],
        "phone": data.get("phone", ""),
        "email": data.get("email", ""),
        "nationality": data.get("nationality"),
        "entry_point": data["entry_point"],
        "expected_exit_timestamp": expected_exit_timestamp,
        "itinerary": data.get("itinerary", [])
    }

    tourist_id, stored_hash = db_manager.register_tourist(tourist_data)

    face_images = data.get("face_images") or []
    face_image_urls = save_face_images(did, face_images)
    face_hash = compute_face_hash_from_urls(face_image_urls)

    # Optional: record DID on Fabric (non-blocking, safe to ignore failures)
    fabric_client.register_did(did, stored_hash, face_hash, utc_now_iso())

    feature_id = None
    if face_image_urls and mongo_manager:
        feature_id = generate_feature_id()
        reid_embedding = compute_reid_embedding_from_paths(face_image_urls)
        if reid_embedding is None:
            reid_embedding = np.zeros((512,), dtype=np.float32)

        mongo_manager.store_tourist_features({
            "feature_id": feature_id,
            "did": did,
            "reid_embedding": reid_embedding.tolist(),
            "camera_id": "registration_desk",
            "capture_angles": [{"angle": "front", "quality_score": 0.95}],
            "feature_quality": 0.95,
            "image_paths": face_image_urls
        })

        # refresh gallery so new tourist is matchable immediately
        refresh_reid_gallery(force=True)

    qr_data = {
        "did": did,
        "name": data["name"],
        "entry_point": data["entry_point"],
        "timestamp": datetime.now().isoformat()
    }

    return jsonify({
        "success": True,
        "tourist_id": tourist_id,
        "did": did,
        "id_hash": stored_hash,
        "feature_id": feature_id,
        "qr_data": qr_data
    })

@app.route("/api/search_tourist", methods=["GET"])
def search_tourist():
    if not db_manager:
        return jsonify({"error": "DB not initialized"}), 500

    did = request.args.get("did")
    if not did:
        return jsonify({"error": "DID required"}), 400

    tourist = db_manager.get_tourist_by_did(did)
    if not tourist:
        return jsonify({"error": "Not found"}), 404

    # decode itinerary if present, then remove encrypted fields to keep JSON safe
    try:
        itinerary_encrypted = tourist.get("itinerary_encrypted")
        if itinerary_encrypted:
            itinerary_json = db_manager.decrypt_data(itinerary_encrypted)
            tourist["itinerary"] = json.loads(itinerary_json)
    except Exception:
        tourist["itinerary"] = []

    for key in (
        "name_encrypted",
        "id_number_encrypted",
        "phone_encrypted",
        "email_encrypted",
        "itinerary_encrypted",
        "emergency_contact_encrypted",
    ):
        tourist.pop(key, None)

    # Attach face image URLs (if available) for UI display
    face_images = []
    if mongo_manager:
        try:
            features = mongo_manager.get_tourist_features(did)
            for feat in features:
                paths = feat.get("image_paths") or []
                if paths:
                    face_images.extend(paths)
        except Exception as e:
            logger.warning("Failed to load face images for {}: {}", did, e)

    if face_images:
        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for url in face_images:
            if url not in seen:
                seen.add(url)
                deduped.append(url)
        tourist["face_images"] = deduped
        tourist["face_image_count"] = len(deduped)

    return jsonify({"success": True, "tourist": tourist})

@app.route("/api/get_tourist_trajectory", methods=["GET"])
def get_tourist_trajectory():
    if not mongo_manager:
        return jsonify({"error": "DB not initialized"}), 500

    did = request.args.get("did")
    traj = mongo_manager.get_tourist_trajectory(did)
    return jsonify({"success": True, "did": did, "trajectory": traj})

@app.route("/api/status", methods=["GET"])
def api_status():
    # Unified status endpoint used by the dashboard
    return jsonify({
        "status": "running",
        "camera1_running": bool(is_running_1),
        "camera2_running": bool(is_running_2),
        "camera1": current_stats.get("camera1", {}),
        "camera2": current_stats.get("camera2", {}),
        "total_persons": current_stats.get("total_persons", 0),
        "device": getattr(detector, "device", "none"),
        "model_loaded": detector is not None
    })

# ----------------------------------------------------
# WEBSOCKET EVENTS
# ----------------------------------------------------
@socketio.on("connect")
def handle_connect():
    logger.info("Client connected")
    emit("connected", {"data": "Connected"})
    # when a client connects, push a current stats snapshot immediately
    try:
        socketio.emit("stats", {
            "camera1": current_stats.get("camera1", {}),
            "camera2": current_stats.get("camera2", {}),
            "total_persons": current_stats.get("total_persons", 0),
            "all_detections": []
        })
    except Exception:
        pass

@socketio.on("disconnect")
def handle_disconnect():
    logger.info("Client disconnected")

# ----------------------------------------------------
# SERVER START (BOTTOM ONLY)
# ----------------------------------------------------
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🚀 Smart Tourist Safety System - Dual Camera")
    logger.info("=" * 60)

    # Instantiate detector (this might be slow)
    try:
        detector = PersonDetector("models/yolov8n.pt", conf_threshold=0.4, img_size=320)
        logger.info("Detector initialized")
    except Exception as e:
        logger.exception("Failed to initialize detector: {}", e)
        detector = None

    # Instantiate Re-ID model (OSNet x0_25) for CPU
    try:
        reid_model = ReIDModel(model_name=REID_MODEL_NAME, device="cpu")
        logger.info("Re-ID model initialized: {}", REID_MODEL_NAME)
        refresh_reid_gallery(force=True)
    except Exception as e:
        logger.warning("Re-ID model not available: {}", e)
        reid_model = None

    # Run socketio/flask app
    socketio.run(app, host="0.0.0.0", port=5000,
                 debug=False, allow_unsafe_werkzeug=True)
