// Dual Camera Dashboard JavaScript
const socket = io();
let camera1Running = false;
let camera2Running = false;
let detectionUpdateInterval;

socket.on("stats", (data) => {
    console.log("Received stats:", data);
    updateStats(data);
});

// NEW: store webcam streams
let webcamStream1 = null;
let webcamStream2 = null;
let webcamInterval1 = null;
let webcamInterval2 = null;
let webcamFrameId1 = 0;
let webcamFrameId2 = 0;
let webcamSending1 = false;
let webcamSending2 = false;

const WEBCAM_SEND_INTERVAL_MS = 120; // ~8 FPS
const WEBCAM_MAX_WIDTH = 640;

function drawWebcamOverlay(cameraNum, payload) {
    const overlay = document.getElementById(`webcamOverlay${cameraNum}`);
    const video = document.getElementById(`webcamFeed${cameraNum}`);
    if (!overlay || !video) return;

    const detections = payload?.detections || [];
    const frameW = payload?.frame_width || WEBCAM_MAX_WIDTH;
    const frameH = payload?.frame_height || Math.round(frameW * 0.75);

    const container = video.parentElement;
    const vRect = video.getBoundingClientRect();
    const cRect = container ? container.getBoundingClientRect() : vRect;

    const displayW = vRect.width || frameW;
    const displayH = vRect.height || frameH;
    const dpr = window.devicePixelRatio || 1;

    overlay.width = displayW * dpr;
    overlay.height = displayH * dpr;
    overlay.style.width = `${displayW}px`;
    overlay.style.height = `${displayH}px`;
    overlay.style.left = `${vRect.left - cRect.left}px`;
    overlay.style.top = `${vRect.top - cRect.top}px`;

    const ctx = overlay.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, displayW, displayH);

    if (!detections || detections.length === 0) return;

    const scaleX = displayW / frameW;
    const scaleY = displayH / frameH;

    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 2;
    ctx.font = '14px sans-serif';
    ctx.fillStyle = '#10b981';

    detections.forEach(det => {
        const bbox = det.bbox || [];
        if (bbox.length !== 4) return;
        const [x1, y1, x2, y2] = bbox;
        const w = x2 - x1;
        const h = y2 - y1;
        if (w <= 0 || h <= 0) return;
        ctx.strokeRect(x1 * scaleX, y1 * scaleY, w * scaleX, h * scaleY);
        let label = `ID ${det.tracking_id || 'N/A'}`;
        if (det.did) {
            const score = det.match_confidence != null ? ` (${Number(det.match_confidence).toFixed(2)})` : '';
            label = `ID ${det.tracking_id || 'N/A'} | ${det.did}${score}`;
        }
        ctx.fillText(label, x1 * scaleX + 4, Math.max(14, y1 * scaleY - 4));
    });
}

function startWebcamStreaming(cameraNum) {
    const video = document.getElementById(`webcamFeed${cameraNum}`);
    const overlay = document.getElementById(`webcamOverlay${cameraNum}`);
    if (!video || !overlay) return;

    overlay.style.display = 'block';

    const captureCanvas = document.createElement('canvas');
    const captureCtx = captureCanvas.getContext('2d');

    const sendFrame = async () => {
        const isSending = cameraNum === 1 ? webcamSending1 : webcamSending2;
        if (isSending) return;
        if (video.readyState < 2 || !video.videoWidth || !video.videoHeight) return;

        if (cameraNum === 1) webcamSending1 = true;
        else webcamSending2 = true;

        try {
            const vw = video.videoWidth;
            const vh = video.videoHeight;
            const targetW = Math.min(WEBCAM_MAX_WIDTH, vw);
            const targetH = Math.round(vh * (targetW / vw));

            if (captureCanvas.width !== targetW || captureCanvas.height !== targetH) {
                captureCanvas.width = targetW;
                captureCanvas.height = targetH;
            }

            captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
            const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.7);
            const frameId = cameraNum === 1 ? ++webcamFrameId1 : ++webcamFrameId2;

            const response = await fetch('/api/webcam_frame', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    camera: cameraNum,
                    frame_id: frameId,
                    image: dataUrl
                })
            });

            if (response.ok) {
                const payload = await response.json();
                drawWebcamOverlay(cameraNum, payload);
            }
        } catch (e) {
            console.warn('Webcam frame send failed:', e);
        } finally {
            if (cameraNum === 1) webcamSending1 = false;
            else webcamSending2 = false;
        }
    };

    const intervalId = setInterval(sendFrame, WEBCAM_SEND_INTERVAL_MS);
    if (cameraNum === 1) webcamInterval1 = intervalId;
    else webcamInterval2 = intervalId;
}

function stopWebcamStreaming(cameraNum) {
    const overlay = document.getElementById(`webcamOverlay${cameraNum}`);
    if (overlay) {
        const ctx = overlay.getContext('2d');
        ctx.clearRect(0, 0, overlay.width, overlay.height);
        overlay.style.display = 'none';
    }

    if (cameraNum === 1 && webcamInterval1) {
        clearInterval(webcamInterval1);
        webcamInterval1 = null;
        webcamFrameId1 = 0;
        webcamSending1 = false;
    }
    if (cameraNum === 2 && webcamInterval2) {
        clearInterval(webcamInterval2);
        webcamInterval2 = null;
        webcamFrameId2 = 0;
        webcamSending2 = false;
    }
}

document.addEventListener('DOMContentLoaded', () => {
    console.log('Dual Camera Dashboard loaded');
    fetchSystemStatus();
    setupEventListeners();
    
    // Update detection list every 5 seconds to prevent scroll jump
    detectionUpdateInterval = setInterval(() => {
        if (camera1Running || camera2Running) {
            // Don't update - WebSocket handles it
        }
    }, 5000);
});

function setupEventListeners() {
    document.getElementById('videoSource1').addEventListener('change', () => handleSourceChange(1));
    document.getElementById('videoSource2').addEventListener('change', () => handleSourceChange(2));
}


function switchTab(cameraNum) {
    // Update tab buttons
    const tabs = document.querySelectorAll('.tab-btn');
    tabs.forEach((tab, index) => {
        if (index + 1 === cameraNum) {
            tab.classList.add('active');
        } else {
            tab.classList.remove('active');
        }
    });
    
    // Update settings panels
    const settings = document.querySelectorAll('.camera-settings');
    settings.forEach((setting, index) => {
        if (index + 1 === cameraNum) {
            setting.classList.add('active');
        } else {
            setting.classList.remove('active');
        }
    });
}


function handleSourceChange(camera) {
    const source = document.getElementById(`videoSource${camera}`).value;
    const videoPathGroup = document.getElementById(`videoPathGroup${camera}`);
    const esp32UrlGroup = document.getElementById(`esp32UrlGroup${camera}`);
    
    if (videoPathGroup && esp32UrlGroup) {
        videoPathGroup.style.display = source === 'video' ? 'block' : 'none';
        esp32UrlGroup.style.display = source === 'esp32cam' ? 'block' : 'none';
    }
}

// =============== NEW WEBCAM FUNCTIONS (REGISTER.JS STYLE) ================= //

async function startWebcam(cameraNum) {
    try {
        const constraints = {
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: "user"
            }
        };

        const stream = await navigator.mediaDevices.getUserMedia(constraints);

        if (cameraNum === 1) webcamStream1 = stream;
        else webcamStream2 = stream;

        const img = document.getElementById(`videoFeed${cameraNum}`);
        const video = document.getElementById(`webcamFeed${cameraNum}`);
        if (img) img.style.display = "none";
        if (video) {
            video.style.display = "block";
            video.srcObject = stream;
        }

        startWebcamStreaming(cameraNum);

        document.getElementById(`noVideoPlaceholder${cameraNum}`).style.display = "none";
        document.getElementById(`camera${cameraNum}Status`).textContent = 'Active';
        document.getElementById(`camera${cameraNum}Status`).classList.add('active');

        const badge = document.getElementById(`camera${cameraNum}Badge`);
        if (badge) {
            badge.textContent = "● Active";
            badge.classList.remove("inactive");
            badge.classList.add("active");
        }

        if (cameraNum === 1) camera1Running = true;
        else camera2Running = true;

        updateGlobalStatus();
        alert(`📷 Webcam for Camera ${cameraNum} started!`);

    } catch (error) {
        console.error('Webcam error:', error);
        alert('❌ Webcam access failed. Check permissions.');
    }
}

function stopWebcam(cameraNum) {
    stopWebcamStreaming(cameraNum);

    const stream = cameraNum === 1 ? webcamStream1 : webcamStream2;
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        if (cameraNum === 1) webcamStream1 = null;
        else webcamStream2 = null;
    }

    const img = document.getElementById(`videoFeed${cameraNum}`);
    const video = document.getElementById(`webcamFeed${cameraNum}`);
    if (video) {
        video.srcObject = null;
        video.style.display = "none";
    }
    if (img) img.style.display = "block";

    document.getElementById(`noVideoPlaceholder${cameraNum}`).style.display = "flex";
    document.getElementById(`camera${cameraNum}Status`).textContent = 'Inactive';
    document.getElementById(`camera${cameraNum}Status`).classList.remove('active');

    const badge = document.getElementById(`camera${cameraNum}Badge`);
    if (badge) {
        badge.textContent = "● Inactive";
        badge.classList.remove("active");
        badge.classList.add("inactive");
    }

    if (cameraNum === 1) camera1Running = false;
    else camera2Running = false;

    updateGlobalStatus();

    // notify backend to reset webcam state
    fetch('/api/stop_webcam', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera: cameraNum })
    }).catch(() => {});
}

// ========================== UPDATED startCamera ============================ //

async function startCamera(cameraNum) {
    const source = document.getElementById(`videoSource${cameraNum}`).value;

    // NEW: detect webcam mode
    if (source === "webcam") {
        return startWebcam(cameraNum);
    }

    // ORIGINAL CODE FOR VIDEO + ESP32
    const videoPath = document.getElementById(`videoPath${cameraNum}`)?.value || 'videos/sample.mp4';
    const esp32Url = document.getElementById(`esp32Url${cameraNum}`)?.value || '';

    const payload = {
        camera: cameraNum,
        source_type: source,
        video_path: videoPath,
        esp32_url: esp32Url
    };

    try {
        const response = await fetch('/api/start_camera', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            console.log(`Camera ${cameraNum} started:`, data);
            
            if (cameraNum === 1) camera1Running = true;
            else camera2Running = true;
            
            document.getElementById(`videoFeed${cameraNum}`).src = `/video_feed_${cameraNum}?t=` + Date.now();
            const webcamEl = document.getElementById(`webcamFeed${cameraNum}`);
            if (webcamEl) webcamEl.style.display = 'none';
            stopWebcamStreaming(cameraNum);
            const imgEl = document.getElementById(`videoFeed${cameraNum}`);
            if (imgEl) imgEl.style.display = 'block';
            document.getElementById(`noVideoPlaceholder${cameraNum}`).style.display = 'none';

            if (document.getElementById(`recordingIndicator${cameraNum}`)) {
                document.getElementById(`recordingIndicator${cameraNum}`).style.display = 'flex';
            }

            document.getElementById(`camera${cameraNum}Status`).textContent = 'Active';
            document.getElementById(`camera${cameraNum}Status`).classList.add('active');
            
            const badge = document.getElementById(`camera${cameraNum}Badge`);
            if (badge) {
                badge.textContent = '● Active';
                badge.classList.remove('inactive');
                badge.classList.add('active');
            }
            
            updateGlobalStatus();
            alert(`✅ Camera ${cameraNum} started!`);
        } else {
            throw new Error(data.error || 'Failed to start');
        }
    } catch (error) {
        console.error('Error:', error);
        alert(`❌ Failed to start Camera ${cameraNum}: ` + error.message);
    }
}

// ========================== UPDATED stopCamera ============================ //

async function stopCamera(cameraNum) {
    const source = document.getElementById(`videoSource${cameraNum}`).value;

    // NEW: stop webcam without backend call
    if (source === "webcam") {
        stopWebcam(cameraNum);
        alert(`⏹️ Webcam for Camera ${cameraNum} stopped`);
        return;
    }

    // ORIGINAL CODE for video/mp4 and ESP32
    try {
        const response = await fetch('/api/stop_camera', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ camera: cameraNum })
        });
        
        const data = await response.json();
        
        console.log(`Camera ${cameraNum} stopped:`, data);
        
        if (cameraNum === 1) camera1Running = false;
        else camera2Running = false;
        
        document.getElementById(`videoFeed${cameraNum}`).src = '';
        document.getElementById(`noVideoPlaceholder${cameraNum}`).style.display = 'flex';

        if (document.getElementById(`recordingIndicator${cameraNum}`)) {
            document.getElementById(`recordingIndicator${cameraNum}`).style.display = 'none';
        }

        document.getElementById(`camera${cameraNum}Status`).textContent = 'Inactive';
        document.getElementById(`camera${cameraNum}Status`).classList.remove('active');
        
        const badge = document.getElementById(`camera${cameraNum}Badge`);
        if (badge) {
            badge.textContent = '● Inactive';
            badge.classList.remove('active');
            badge.classList.add('inactive');
        }
        
        updateGlobalStatus();
        alert(`⏹️ Camera ${cameraNum} stopped`);
    } catch (error) {
        console.error('Error:', error);
        alert(`❌ Failed to stop Camera ${cameraNum}: ` + error.message);
    }
}


async function fetchSystemStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();
        
        document.getElementById('deviceInfo').textContent = 
            data.device ? data.device.toUpperCase() : 'N/A';
        
        camera1Running = data.camera1_running;
        camera2Running = data.camera2_running;
        
        updateGlobalStatus();
    } catch (error) {
        console.error('Error fetching status:', error);
    }
}

function updateStats(data) {
    // Update total person count
    document.getElementById('personsValue').textContent = data.total_persons || 0;
    
    // Update FPS (average of both cameras)
    const avgFps = ((data.camera1?.fps || 0) + (data.camera2?.fps || 0)) / 2;
    document.getElementById('fpsValue').textContent = Math.round(avgFps);
    
    // Update detection list (every 5 seconds to prevent scroll jump)
    const now = Date.now();
    if (!window.lastDetectionUpdate || now - window.lastDetectionUpdate > 5000) {
        if (data.all_detections && data.all_detections.length > 0) {
            updateDetectionsList(data.all_detections);
        } else {
            showEmptyDetections();
        }
        window.lastDetectionUpdate = now;
    }
}

function updateDetectionsList(detections) {
    const list = document.getElementById('detectionsList');
    
    list.innerHTML = detections.map((det, idx) => `
        <div class="detection-item">
            <strong>🎯 ${det.camera} - Tracking ID #${det.tracking_id || 'N/A'}</strong>
            <small>
                Confidence: ${(det.confidence * 100).toFixed(1)}%<br>
                ${det.did ? `Matched DID: ${det.did} ${det.match_confidence != null ? `(${Number(det.match_confidence).toFixed(2)})` : ''}<br>` : ''}
                BBox: [${det.bbox.join(', ')}]
            </small>
        </div>
    `).join('');
}

function showEmptyDetections() {
    const list = document.getElementById('detectionsList');
    list.innerHTML = `
        <div class="empty-state">
            <span class="empty-icon">📭</span>
            <p>No detections</p>
        </div>
    `;
}

function updateGlobalStatus() {
    const badge = document.getElementById('statusBadge');
    const systemStatus = document.getElementById('systemStatus');
    
    if (camera1Running || camera2Running) {
        badge.textContent = '● ACTIVE';
        badge.className = 'badge badge-active';
        systemStatus.textContent = 'Detecting';
    } else {
        badge.textContent = '● STOPPED';
        badge.className = 'badge badge-stopped';
        systemStatus.textContent = 'Ready';
    }
}

// Search tourist by DID
async function searchTourist() {
    const did = document.getElementById('searchDID').value.trim();
    
    if (!did) {
        alert('⚠️ Please enter a DID');
        return;
    }
    
    try {
        const response = await fetch(`/api/search_tourist?did=${encodeURIComponent(did)}`);
        const result = await response.json();
        
        if (response.ok && result.success) {
            displayTouristInfo(result.tourist);
            document.getElementById('searchResult').style.display = 'block';
        } else {
            alert(`❌ Tourist not found: ${result.error || 'Unknown error'}`);
            document.getElementById('searchResult').style.display = 'none';
        }
        
    } catch (error) {
        console.error('Search error:', error);
        alert(`❌ Search failed: ${error.message}`);
    }
}

// Display tourist information
function displayTouristInfo(tourist) {
    const infoDiv = document.getElementById('touristInfo');
    
    const statusColor = tourist.status === 'active' ? '#10b981' : '#6b7280';

    const images = tourist.face_images || [];
    let imageHtml = '';
    if (images.length > 0) {
        imageHtml = `
            <div class="face-image-panel">
                <img src="${images[0]}" alt="Tourist face" class="face-image-large">
            </div>
        `;
    }
    
    infoDiv.innerHTML = `
        <div class="tourist-info-layout">
            <div class="tourist-info-text">
                <p><strong>Name:</strong> ${tourist.name}</p>
                <p><strong>DID:</strong> ${tourist.did}</p>
                <p><strong>Nationality:</strong> ${tourist.nationality || 'N/A'}</p>
                <p><strong>Entry Point:</strong> ${tourist.entry_point}</p>
                <p><strong>Entry Time:</strong> ${new Date(tourist.entry_timestamp).toLocaleString()}</p>
                <p><strong>Status:</strong> <span style="color: ${statusColor}; font-weight: 600;">${tourist.status.toUpperCase()}</span></p>
                <p><strong>Last Seen:</strong> ${tourist.last_seen_camera || 'Not tracked yet'}</p>
            </div>
            ${imageHtml}
        </div>
    `;
    
    // Store DID for trajectory view
    window.currentTouristDID = tourist.did;
}

// View tourist trajectory
async function viewTrajectory() {
    const did = window.currentTouristDID;
    
    if (!did) {
        alert('⚠️ No tourist selected');
        return;
    }
    
    try {
        const response = await fetch(`/api/get_tourist_trajectory?did=${encodeURIComponent(did)}`);
        const result = await response.json();
        
        if (response.ok && result.success) {
            if (result.trajectory.length === 0) {
                alert('ℹ️ No tracking data available yet for this tourist');
            } else {
                displayTrajectory(result.trajectory);
            }
        } else {
            alert(`❌ Failed to fetch trajectory: ${result.error || 'Unknown error'}`);
        }
        
    } catch (error) {
        console.error('Trajectory fetch error:', error);
        alert(`❌ Failed to fetch trajectory: ${error.message}`);
    }
}

// Display trajectory (simple alert for now - will enhance later)
function displayTrajectory(trajectory) {
    let message = `📍 Tourist Trajectory (${trajectory.length} sessions):\n\n`;
    
    trajectory.forEach((session, idx) => {
        message += `${idx + 1}. Camera ${session.camera_id}\n`;
        message += `   Tracking ID: #${session.tracking_id}\n`;
        message += `   Start: ${new Date(session.start_time).toLocaleString()}\n`;
        message += `   Duration: ${session.duration || 'Active'}s\n`;
        message += `   Detections: ${session.num_detections}\n\n`;
    });
    
    alert(message);
}
