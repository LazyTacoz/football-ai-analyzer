"""
Football Analysis API Server
============================
FastAPI backend for the football analysis pipeline.

Endpoints:
- POST /upload: Upload video for processing
- GET /process/{video_id}: Start processing and stream results
- GET /stream/{video_id}: Stream processed video
- GET /status/{video_id}: Get processing status
- WebSocket /ws/{video_id}: Real-time frame streaming

Author: Uddeshya
"""
# Pre-load torch in main thread to avoid DLL init error on Windows


import os
import cv2
import json
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime
import uuid
import base64
import torch
_ = torch.zeros(1)
if torch.cuda.is_available():
    _ = torch.zeros(1).cuda()

from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from pydantic import BaseModel
import aiofiles
import numpy as np

from processor import FootballAnalysisProcessor, PitchConfig, DetectionConfig

# ============================================================
# CONFIGURATION
# ============================================================

# Directory setup
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
STATIC_DIR = BASE_DIR / "static"

# Create directories
for dir_path in [UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR]:
    dir_path.mkdir(exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("football-api")

# ============================================================
# STATE MANAGEMENT
# ============================================================

class VideoProcessingState:
    """Tracks the state of video processing jobs."""
    
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
    
    def create_job(self, video_id: str, filename: str) -> Dict[str, Any]:
        """Create a new processing job."""
        self.jobs[video_id] = {
            "id": video_id,
            "filename": filename,
            "status": "pending",  # pending, processing, completed, error
            "progress": 0.0,
            "total_frames": 0,
            "processed_frames": 0,
            "current_stats": {},
            "created_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            "error_message": None,
            "output_path": None,
        }
        return self.jobs[video_id]
    
    def update_job(self, video_id: str, **kwargs) -> None:
        """Update job state."""
        if video_id in self.jobs:
            self.jobs[video_id].update(kwargs)
    
    def get_job(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Get job state."""
        return self.jobs.get(video_id)


# Global state
processing_state = VideoProcessingState()
active_processors: Dict[str, FootballAnalysisProcessor] = {}
active_websockets: Dict[str, list] = {}

# ============================================================
# PYDANTIC MODELS
# ============================================================

class ProcessingOptions(BaseModel):
    """Options for video processing."""
    show_tracking: bool = True
    show_voronoi: bool = True
    show_radar: bool = True
    target_fps: int = 15  # Output FPS (lower = faster processing)

class ProcessingStatus(BaseModel):
    """Status response model."""
    id: str
    status: str
    progress: float
    total_frames: int
    processed_frames: int
    current_stats: Dict[str, Any]
    error_message: Optional[str] = None

# ============================================================
# LIFESPAN CONTEXT
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    logger.info("Starting Football Analysis API...")
    logger.info(f"Upload directory: {UPLOAD_DIR}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    yield
    # Cleanup on shutdown
    logger.info("Shutting down, cleaning up resources...")
    active_processors.clear()

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Football AI Analysis API",
    description="Computer vision pipeline for football match analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def encode_frame_to_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a numpy frame to JPEG bytes."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_params)
    return buffer.tobytes()

def encode_frame_to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode a numpy frame to base64 string."""
    jpeg_bytes = encode_frame_to_jpeg(frame, quality)
    return base64.b64encode(jpeg_bytes).decode('utf-8')

async def process_video_task(
    video_id: str,
    input_path: Path,
    options: ProcessingOptions
) -> None:
    """
    Background task for video processing.
    
    This processes the video frame-by-frame and:
    1. Saves annotated video to output directory
    2. Updates processing state with progress
    3. Broadcasts frames to connected WebSocket clients
    """
    try:
        processing_state.update_job(video_id, status="processing")
        
        # Initialize video capture
        cap = cv2.VideoCapture(str(input_path))
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        processing_state.update_job(video_id, total_frames=total_frames)
        logger.info(f"Processing video: {total_frames} frames at {input_fps} FPS, {frame_width}x{frame_height}")
        
        # Initialize processor
        processor = FootballAnalysisProcessor(
            model_path="C:\\Users\\nachi\\SY\\SEM3\\ediedi\\football_model_v3\\train\\weights\\best.pt",  
            pitch_config=PitchConfig(),
            detection_config=DetectionConfig()
        )
        active_processors[video_id] = processor
        processor.kinematics.video_fps = input_fps
        
        # Setup output video writer
        output_path = OUTPUT_DIR / f"{video_id}_processed.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_fps = min(options.target_fps, input_fps)
        
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            out_fps,
            (frame_width, frame_height)
        )
        
        # Calculate frame skip for target FPS
        frame_skip = max(1, int(input_fps / options.target_fps))
        
        # Processing loop
        frame_idx = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_idx += 1
            
            # Skip frames to achieve target FPS
            if frame_idx % frame_skip != 0:
                continue
            
            # Process frame
            result = processor.process_frame(
                frame,
                show_tracking=options.show_tracking,
                show_voronoi=options.show_voronoi,
                show_radar=options.show_radar
            )
            
            annotated_frame = result["annotated_frame"]
            radar_board = result["radar_board"]
            stats = result["stats"]
            
            # Write to output video
            writer.write(annotated_frame)
            processed_count += 1
            
            # Update progress
            progress = (frame_idx / total_frames) * 100
            processing_state.update_job(
                video_id,
                progress=progress,
                processed_frames=processed_count,
                current_stats=stats
            )
            
            # Broadcast to WebSocket clients
            if video_id in active_websockets:
                frame_data = {
                    "type": "frame",
                    "frame": encode_frame_to_base64(annotated_frame),
                    "radar": encode_frame_to_base64(radar_board) if radar_board is not None else None,
                    "stats": stats,
                    "progress": progress
                }
                
                # Send to all connected clients
                dead_sockets = []
                for ws in active_websockets[video_id]:
                    try:
                        await ws.send_json(frame_data)
                    except Exception:
                        dead_sockets.append(ws)
                
                # Clean up dead connections
                for ws in dead_sockets:
                    active_websockets[video_id].remove(ws)
            
            # Small delay to prevent overwhelming
            await asyncio.sleep(0.01)
        
        # Cleanup
        cap.release()
        writer.release()
        
        # Update completion state
        processing_state.update_job(
            video_id,
            status="completed",
            progress=100.0,
            completed_at=datetime.utcnow().isoformat(),
            output_path=str(output_path)
        )
        
        logger.info(f"Video processing completed: {video_id}")
        
        # Notify WebSocket clients of completion
        if video_id in active_websockets:
            completion_msg = {"type": "complete", "output_path": f"/output/{video_id}"}
            for ws in active_websockets[video_id]:
                try:
                    await ws.send_json(completion_msg)
                except Exception:
                    pass
        
    except Exception as e:
        logger.error(f"Processing error for {video_id}: {str(e)}")
        processing_state.update_job(
            video_id,
            status="error",
            error_message=str(e)
        )
        
        # Notify WebSocket clients of error
        if video_id in active_websockets:
            error_msg = {"type": "error", "message": str(e)}
            for ws in active_websockets[video_id]:
                try:
                    await ws.send_json(error_msg)
                except Exception:
                    pass
    
    finally:
        # Cleanup processor
        if video_id in active_processors:
            del active_processors[video_id]

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Football AI Analysis API",
        "version": "1.0.0"
    }

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload a video file for processing.
    
    Accepts: .mp4, .avi, .mov, .mkv files
    Returns: video_id for tracking the processing job
    """
    # Validate file type
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique ID
    video_id = str(uuid.uuid4())[:8]
    
    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{video_id}{file_ext}"
    
    async with aiofiles.open(upload_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    
    logger.info(f"Video uploaded: {video_id} ({file.filename})")
    
    # Create processing job
    job = processing_state.create_job(video_id, file.filename)
    
    return {
        "video_id": video_id,
        "filename": file.filename,
        "status": "uploaded",
        "message": "Video uploaded successfully. Call /process/{video_id} to start processing."
    }

@app.post("/process/{video_id}")
async def start_processing(
    video_id: str,
    options: ProcessingOptions = ProcessingOptions(),
    background_tasks: BackgroundTasks = None
):
    """
    Start processing an uploaded video.
    
    The processing runs as a background task. Use /status/{video_id}
    or WebSocket /ws/{video_id} to track progress.
    """
    job = processing_state.get_job(video_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if job["status"] == "processing":
        raise HTTPException(status_code=400, detail="Video is already being processed")
    
    # Find the uploaded file
    upload_files = list(UPLOAD_DIR.glob(f"{video_id}.*"))
    
    if not upload_files:
        raise HTTPException(status_code=404, detail="Video file not found")
    
    input_path = upload_files[0]
    
    # Initialize WebSocket client list
    active_websockets[video_id] = []
    
    # Start background processing
    background_tasks.add_task(process_video_task, video_id, input_path, options)
    
    return {
        "video_id": video_id,
        "status": "processing",
        "message": "Processing started. Connect to WebSocket /ws/{video_id} for real-time updates."
    }

@app.get("/status/{video_id}", response_model=ProcessingStatus)
async def get_status(video_id: str):
    """Get the current processing status of a video."""
    job = processing_state.get_job(video_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return ProcessingStatus(
        id=job["id"],
        status=job["status"],
        progress=job["progress"],
        total_frames=job["total_frames"],
        processed_frames=job["processed_frames"],
        current_stats=job["current_stats"],
        error_message=job.get("error_message")
    )

@app.get("/output/{video_id}")
async def get_output_video(video_id: str):
    """Download the processed video."""
    job = processing_state.get_job(video_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Video processing not completed")
    
    output_path = job.get("output_path")
    
    if output_path is None or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"analyzed_{job['filename']}"
    )

@app.delete("/video/{video_id}")
async def delete_video(video_id: str):
    """Delete video and associated files."""
    job = processing_state.get_job(video_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Stop processing if active
    if video_id in active_processors:
        del active_processors[video_id]
    
    # Delete files
    for dir_path in [UPLOAD_DIR, OUTPUT_DIR]:
        for file_path in dir_path.glob(f"{video_id}*"):
            file_path.unlink()
    
    # Remove from state
    if video_id in processing_state.jobs:
        del processing_state.jobs[video_id]
    
    return {"message": "Video deleted successfully"}

# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

@app.websocket("/ws/{video_id}")
async def websocket_endpoint(websocket: WebSocket, video_id: str):
    """
    WebSocket endpoint for real-time frame streaming.
    
    Sends JSON messages with structure:
    {
        "type": "frame" | "complete" | "error",
        "frame": base64_encoded_jpeg,  // for type=frame
        "radar": base64_encoded_jpeg,   // for type=frame
        "stats": {...},                 // for type=frame
        "progress": float               // for type=frame
    }
    """
    await websocket.accept()
    
    job = processing_state.get_job(video_id)
    
    if job is None:
        await websocket.send_json({"type": "error", "message": "Video not found"})
        await websocket.close()
        return
    
    # Add to active WebSocket list
    if video_id not in active_websockets:
        active_websockets[video_id] = []
    
    active_websockets[video_id].append(websocket)
    logger.info(f"WebSocket connected for video: {video_id}")
    
    try:
        # Send initial status
        await websocket.send_json({
            "type": "status",
            "status": job["status"],
            "progress": job["progress"]
        })
        
        # Keep connection alive until processing completes or client disconnects
        while True:
            try:
                # Wait for ping/messages from client
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=60.0  # 1 minute timeout
                )
                
                # Handle ping
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
                
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for video: {video_id}")
    finally:
        # Remove from active list
        if video_id in active_websockets and websocket in active_websockets[video_id]:
            active_websockets[video_id].remove(websocket)

# ============================================================
# STREAMING ENDPOINT (Alternative to WebSocket)
# ============================================================

@app.get("/stream/{video_id}")
async def stream_processing(video_id: str):
    """
    Server-Sent Events (SSE) endpoint for streaming processing updates.
    Alternative to WebSocket for simpler client implementations.
    """
    job = processing_state.get_job(video_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Video not found")
    
    async def event_generator():
        """Generate SSE events for processing updates."""
        last_progress = -1
        
        while True:
            job = processing_state.get_job(video_id)
            
            if job is None:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Job not found'})}\n\n"
                break
            
            current_progress = job["progress"]
            
            # Send update if progress changed
            if current_progress != last_progress:
                event_data = {
                    "type": "progress",
                    "status": job["status"],
                    "progress": job["progress"],
                    "processed_frames": job["processed_frames"],
                    "total_frames": job["total_frames"],
                    "stats": job["current_stats"]
                }
                yield f"data: {json.dumps(event_data)}\n\n"
                last_progress = current_progress
            
            # Check if completed or errored
            if job["status"] in ["completed", "error"]:
                final_data = {
                    "type": job["status"],
                    "output_path": f"/output/{video_id}" if job["status"] == "completed" else None,
                    "error": job.get("error_message")
                }
                yield f"data: {json.dumps(final_data)}\n\n"
                break
            
            await asyncio.sleep(0.5)
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# ============================================================
# SINGLE FRAME PROCESSING (For testing/demo)
# ============================================================

@app.post("/analyze-frame")
async def analyze_single_frame(file: UploadFile = File(...)):
    """
    Analyze a single frame/image (for testing purposes).
    
    Returns the annotated frame and radar board as base64.
    """
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    
    # Process
    processor = FootballAnalysisProcessor()
    result = processor.process_frame(frame)
    
    return {
        "annotated_frame": encode_frame_to_base64(result["annotated_frame"]),
        "radar_board": encode_frame_to_base64(result["radar_board"]) if result["radar_board"] is not None else None,
        "stats": result["stats"]
    }

# ============================================================
# CONFIGURATION ENDPOINTS
# ============================================================

@app.get("/config")
async def get_config():
    """Get current processor configuration."""
    pitch_config = PitchConfig()
    detection_config = DetectionConfig()
    
    return {
        "pitch": {
            "length_m": pitch_config.length,
            "width_m": pitch_config.width,
            "radar_width_px": pitch_config.radar_width,
            "radar_height_px": pitch_config.radar_height,
        },
        "detection": {
            "confidence_threshold": detection_config.confidence_threshold,
            "nms_threshold": detection_config.nms_threshold,
        }
    }


# ============================================================
# HEATMAP ENDPOINTS
# ============================================================

@app.get("/heatmap/{video_id}/{player_id}")
async def get_player_heatmap(video_id: str, player_id: int):
    """
    Get heatmap for a specific player.
    
    Returns base64-encoded heatmap image overlaid on the pitch.
    """
    if video_id not in active_processors:
        raise HTTPException(status_code=404, detail="Video processor not found. Process a video first.")
    
    processor = active_processors[video_id]
    heatmap = processor.get_player_heatmap(player_id)
    
    if heatmap is None:
        raise HTTPException(status_code=404, detail=f"No heatmap data for player {player_id}")
    
    return {
        "player_id": player_id,
        "heatmap": encode_frame_to_base64(heatmap),
        "stats": processor.get_player_stats(player_id)
    }


@app.get("/players/{video_id}")
async def get_tracked_players(video_id: str):
    """
    Get list of all tracked player IDs for a video.
    """
    if video_id not in active_processors:
        raise HTTPException(status_code=404, detail="Video processor not found")
    
    processor = active_processors[video_id]
    player_ids = processor.get_all_player_ids()
    
    # Get stats for each player
    players = []
    for pid in player_ids:
        stats = processor.get_player_stats(pid)
        players.append({
            "id": pid,
            "stats": stats
        })
    
    return {
        "video_id": video_id,
        "player_count": len(players),
        "players": players
    }


@app.get("/player-stats/{video_id}/{player_id}")
async def get_player_statistics(video_id: str, player_id: int):
    """
    Get detailed statistics for a specific player.
    """
    if video_id not in active_processors:
        raise HTTPException(status_code=404, detail="Video processor not found")
    
    processor = active_processors[video_id]
    stats = processor.get_player_stats(player_id)
    
    if not stats:
        raise HTTPException(status_code=404, detail=f"No data for player {player_id}")
    
    return {
        "player_id": player_id,
        "stats": stats
    }
@app.get("/kinematics/{video_id}")
async def get_kinematics(video_id: str):
    """Full kinematics report for all players."""
    if video_id not in active_processors:
        raise HTTPException(404, "Processor not found")
    proc = active_processors[video_id]
    all_ids = proc.get_all_player_ids()
    return {
        "players": [proc.kinematics.get_player_full_stats(pid) for pid in all_ids],
        "teams":   proc.kinematics.get_team_stats(),
        "top_speeds": proc.kinematics.get_top_speeds(5),
    }
@app.get("/debug-homography/{video_id}")
async def debug_homography(video_id: str):
    """Returns the homography debug frame as base64."""
    if video_id not in active_processors:
        raise HTTPException(404, "Processor not found")
    proc = active_processors[video_id]
    debug = proc.auto_homography.last_debug_frame
    if debug is None:
        raise HTTPException(404, "No debug frame available")
    return {"debug_frame": encode_frame_to_base64(debug)}

# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
