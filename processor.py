"""
Football Analysis Processor (v2.0)
==================================
Enhanced CV pipeline with:
1. Referee & out-of-field filtering
2. Persistent unique player IDs
3. Individual player heatmaps
4. Improved tracking accuracy (Kalman filtering)
5. Ball trajectory prediction

Author: Football AI Analyzer
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any, Set
from collections import defaultdict
from scipy.spatial import Voronoi
from scipy.ndimage import gaussian_filter
from sklearn.cluster import KMeans
import supervision as sv
from pitch_keypoint_detector import AutoHomographyEstimator
from player_kinematics import PlayerKinematicsTracker

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================

@dataclass
class PitchConfig:
    """Standard football pitch dimensions (FIFA regulations)."""
    length: float = 105.0  # meters
    width: float = 68.0    # meters
    penalty_area_length: float = 16.5
    penalty_area_width: float = 40.3
    goal_area_length: float = 5.5
    goal_area_width: float = 18.3
    center_circle_radius: float = 9.15
    radar_width: int = 700
    radar_height: int = 450

    def meters_to_radar(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert pitch meters to radar pixels."""
        px = int((x_m / self.length) * self.radar_width)
        py = int((y_m / self.width) * self.radar_height)
        return px, py


@dataclass
class DetectionConfig:
    """Detection and tracking configuration."""
    # YOLO confidence
    confidence_threshold: float = 0.35
    nms_threshold: float = 0.45
    
    # ByteTrack parameters (tuned for stability)
    track_thresh: float = 0.25
    track_buffer: int = 60        # Frames to keep lost tracks
    match_thresh: float = 0.85    # IoU matching threshold
    
    # Size filters (pixels)
    min_bbox_area: int = 1200
    max_bbox_area: int = 90000
    
    # Aspect ratio (height/width) - players are vertical
    min_aspect_ratio: float = 1.1
    max_aspect_ratio: float = 5.0


# ============================================================
# FEATURE 1: PITCH BOUNDARY DETECTOR
# ============================================================

class PitchBoundaryDetector:
    """
    Detects the playable pitch area using color segmentation.
    Filters out any detections outside the field boundaries.
    
    How it works:
    1. Convert frame to HSV color space
    2. Mask green pixels (grass)
    3. Find largest contour (the pitch)
    4. Create convex hull for boundary testing
    """
    
    def __init__(self):
        self.pitch_contour: Optional[np.ndarray] = None
        self.pitch_mask: Optional[np.ndarray] = None
        self.is_calibrated = False
        
        # HSV range for grass (works for most broadcasts)
        self.green_lower = np.array([35, 50, 50])
        self.green_upper = np.array([75, 255, 255])
    
    def detect_pitch(self, frame: np.ndarray) -> np.ndarray:
        """Detect pitch area and store boundary."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        
        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            self.pitch_contour = cv2.convexHull(largest)
            self.pitch_mask = np.zeros_like(mask)
            cv2.fillPoly(self.pitch_mask, [largest], 255)
            self.is_calibrated = True
        
        return mask
    
    def is_inside_pitch(self, x: float, y: float, margin: int = -40) -> bool:
        """Check if point is inside pitch boundary."""
        if not self.is_calibrated or self.pitch_contour is None:
            return True
        
        distance = cv2.pointPolygonTest(self.pitch_contour, (float(x), float(y)), True)
        return distance >= -margin
    
    def filter_detections(self, detections: sv.Detections) -> sv.Detections:
        """Remove detections outside the pitch."""
        if not self.is_calibrated or len(detections) == 0:
            return detections
        
        keep = []
        for i, bbox in enumerate(detections.xyxy):
            # Check feet position (bottom-center)
            feet_x = (bbox[0] + bbox[2]) / 2
            feet_y = bbox[3]
            if self.is_inside_pitch(feet_x, feet_y):
                keep.append(i)
        
        return detections[keep] if keep else sv.Detections.empty()


# ============================================================
# FEATURE 1: REFEREE CLASSIFIER
# ============================================================

class RefereeClassifier:
    """
    Classifies detections as referee vs player based on jersey color.
    
    Referee colors detected:
    - Black (most common)
    - Yellow/Fluorescent
    - Pink/Magenta
    
    Uses HSV color space for robust detection.
    """
    
    def __init__(self):
        self.referee_colors = {
            'black': {'lower': np.array([0, 0, 0]), 'upper': np.array([180, 100, 80])},
            'yellow': {'lower': np.array([20, 100, 100]), 'upper': np.array([35, 255, 255])},
            'pink': {'lower': np.array([140, 50, 100]), 'upper': np.array([170, 255, 255])},
        }
        self.threshold = 0.30  # 30% of torso must be referee color
    
    def _extract_torso(self, image: np.ndarray) -> np.ndarray:
        """Extract torso region from player crop."""
        h, w = image.shape[:2]
        y1, y2 = int(h * 0.15), int(h * 0.55)
        x1, x2 = int(w * 0.15), int(w * 0.85)
        torso = image[y1:y2, x1:x2]
        return torso if torso.size > 0 else image
    
    def is_referee(self, crop: np.ndarray) -> bool:
        """Determine if crop shows a referee."""
        if crop is None or crop.size == 0:
            return False
        
        torso = self._extract_torso(crop)
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
        total_pixels = torso.shape[0] * torso.shape[1]
        
        if total_pixels == 0:
            return False
        
        for color_range in self.referee_colors.values():
            mask = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
            ratio = np.sum(mask > 0) / total_pixels
            if ratio > self.threshold:
                return True
        
        return False
    
    def filter_referees(
        self, 
        detections: sv.Detections, 
        frame: np.ndarray
    ) -> Tuple[sv.Detections, sv.Detections]:
        """
        Separate players from referees.
        Returns: (players, referees)
        """
        if len(detections) == 0:
            return detections, sv.Detections.empty()
        
        player_idx, referee_idx = [], []
        
        for i, bbox in enumerate(detections.xyxy):
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            crop = frame[y1:y2, x1:x2]
            if self.is_referee(crop):
                referee_idx.append(i)
            else:
                player_idx.append(i)
        
        players = detections[player_idx] if player_idx else sv.Detections.empty()
        referees = detections[referee_idx] if referee_idx else sv.Detections.empty()
        
        return players, referees


# ============================================================
# FEATURE 2: TEAM COLOR CLASSIFIER (with ID persistence)
# ============================================================

class TeamColorClassifier:
    """
    Classifies players into teams using jersey brightness.
    Dark kit = Team 0, Light kit = Team 1.
    Uses temporal smoothing to prevent flickering.
    """
    
    def __init__(self):
        self.is_fitted = True  # Always ready
        self.team_history: Dict[int, List[int]] = defaultdict(list)
        self.confirmed_teams: Dict[int, int] = {}
        self.history_length = 15
        self.confirmation_threshold = 10
        self.brightness_threshold = 128  # Tunable 0-255
    
    def _extract_brightness(self, image: np.ndarray) -> float:
        h, w = image.shape[:2]
        y1, y2 = int(h * 0.18), int(h * 0.58)
        x1, x2 = int(w * 0.18), int(w * 0.82)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = image
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
    
    def fit(self, crops: List[np.ndarray]) -> None:
        # Compute median brightness to set threshold dynamically
        brightnesses = []
        for crop in crops:
            if crop is not None and crop.size > 0:
                brightnesses.append(self._extract_brightness(crop))
        if len(brightnesses) > 10:
            self.brightness_threshold = float(np.median(brightnesses))
            self.is_fitted = True
    
    def predict(self, crop: np.ndarray, player_id: int = -1) -> int:
        if player_id in self.confirmed_teams:
            return self.confirmed_teams[player_id]
        
        if crop is None or crop.size == 0:
            return 0
        
        brightness = self._extract_brightness(crop)
        raw_team = 0 if brightness < self.brightness_threshold else 1
        
        if player_id >= 0:
            self.team_history[player_id].append(raw_team)
            if len(self.team_history[player_id]) > self.history_length:
                self.team_history[player_id].pop(0)
            
            votes = np.bincount(self.team_history[player_id], minlength=2)
            team = int(np.argmax(votes))
            
            if len(self.team_history[player_id]) >= self.confirmation_threshold:
                if votes[team] / len(self.team_history[player_id]) > 0.7:
                    self.confirmed_teams[player_id] = team
            
            return team
        
        return raw_team


# ============================================================
# FEATURE 3: PLAYER HEATMAP GENERATOR
# ============================================================

class PlayerHeatmapGenerator:
    """
    Generates individual player heatmaps showing position frequency.
    
    Stores position history per player ID and creates
    Gaussian-smoothed density visualizations.
    """
    
    def __init__(self, pitch_config: PitchConfig):
        self.config = pitch_config
        self.positions: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.max_history = 5000
    
    def add_position(self, player_id: int, x: int, y: int) -> None:
        """Record a player position."""
        x = max(0, min(x, self.config.radar_width - 1))
        y = max(0, min(y, self.config.radar_height - 1))
        self.positions[player_id].append((x, y))
        
        if len(self.positions[player_id]) > self.max_history:
            self.positions[player_id].pop(0)
    
    def add_batch(self, player_ids: np.ndarray, positions: np.ndarray) -> None:
        """Add multiple positions at once."""
        for pid, pos in zip(player_ids, positions):
            if pid is not None and len(pos) == 2:
                self.add_position(int(pid), int(pos[0]), int(pos[1]))
    
    def generate_heatmap(self, player_id: int, sigma: float = 15.0) -> Optional[np.ndarray]:
        """Generate colored heatmap for a player."""
        if player_id not in self.positions or len(self.positions[player_id]) < 10:
            return None
        
        heatmap = np.zeros((self.config.radar_height, self.config.radar_width), dtype=np.float32)
        
        for x, y in self.positions[player_id]:
            heatmap[y, x] += 1
        
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        
        if heatmap.max() > 0:
            heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
        else:
            heatmap = heatmap.astype(np.uint8)
        
        return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    def generate_overlay(self, player_id: int, base_image: np.ndarray, alpha: float = 0.6) -> Optional[np.ndarray]:
        """Generate heatmap overlaid on pitch."""
        heatmap = self.generate_heatmap(player_id)
        if heatmap is None:
            return None
        
        if heatmap.shape[:2] != base_image.shape[:2]:
            heatmap = cv2.resize(heatmap, (base_image.shape[1], base_image.shape[0]))
        
        gray = cv2.cvtColor(heatmap, cv2.COLOR_BGR2GRAY)
        mask = gray > 10
        
        result = base_image.copy()
        result[mask] = cv2.addWeighted(base_image[mask], 1 - alpha, heatmap[mask], alpha, 0)
        return result
    
    def get_player_ids(self) -> List[int]:
        """Get all tracked player IDs."""
        return list(self.positions.keys())
    
    def get_player_stats(self, player_id: int) -> Dict[str, Any]:
        """Get movement statistics for a player."""
        if player_id not in self.positions or len(self.positions[player_id]) < 2:
            return {}
        
        pos = np.array(self.positions[player_id])
        diffs = np.diff(pos, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        
        return {
            "position_count": len(pos),
            "avg_x": float(np.mean(pos[:, 0])),
            "avg_y": float(np.mean(pos[:, 1])),
            "total_distance_px": float(np.sum(distances)),
        }
    
    def reset(self) -> None:
        """Clear all data."""
        self.positions.clear()


# ============================================================
# FEATURE 4: KALMAN FILTER TRACKER
# ============================================================

class KalmanTracker:
    """
    Kalman filter for smoothing trajectories.
    
    State: [x, y, vx, vy] (position + velocity)
    Helps with:
    - Smoothing noisy detections
    - Predicting during brief occlusions
    - Maintaining consistent trajectories
    """
    
    def __init__(self):
        self.filters: Dict[int, cv2.KalmanFilter] = {}
        self.lost_count: Dict[int, int] = {}
        self.max_lost = 15
    
    def _create_filter(self) -> cv2.KalmanFilter:
        """Create Kalman filter instance."""
        kf = cv2.KalmanFilter(4, 2)
        
        dt = 1.0
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        
        return kf
    
    def update(self, track_id: int, x: float, y: float) -> Tuple[float, float]:
        """Update tracker with measurement, return smoothed position."""
        if track_id not in self.filters:
            kf = self._create_filter()
            kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            self.filters[track_id] = kf
            self.lost_count[track_id] = 0
            return x, y
        
        kf = self.filters[track_id]
        self.lost_count[track_id] = 0
        
        kf.predict()
        measurement = np.array([[x], [y]], dtype=np.float32)
        kf.correct(measurement)
        
        state = kf.statePost
        return float(state[0]), float(state[1])
    
    def predict(self, track_id: int) -> Optional[Tuple[float, float]]:
        """Predict position without measurement."""
        if track_id not in self.filters:
            return None
        
        self.lost_count[track_id] += 1
        
        if self.lost_count[track_id] > self.max_lost:
            del self.filters[track_id]
            del self.lost_count[track_id]
            return None
        
        prediction = self.filters[track_id].predict()
        return float(prediction[0]), float(prediction[1])


# ============================================================
# FEATURE 4: BALL TRACKER (Specialized)
# ============================================================

class BallTracker:
    """
    Specialized tracker for the ball with interpolation.
    
    Ball tracking challenges:
    - Small object, often occluded
    - Fast movement, motion blur
    - Similar color to other objects
    
    Solution: Kalman + trajectory interpolation
    """
    
    def __init__(self):
        self.kalman = KalmanTracker()
        self.ball_id = -999
        self.history: List[Tuple[float, float, int]] = []
        self.max_history = 30
        self.last_frame = -1
        self.max_gap = 12  # Max frames to interpolate
    
    def update(self, x: float, y: float, frame: int) -> Tuple[float, float]:
        """Update with detected position."""
        sx, sy = self.kalman.update(self.ball_id, x, y)
        self.history.append((sx, sy, frame))
        
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        self.last_frame = frame
        return sx, sy
    
    def predict(self, frame: int) -> Optional[Tuple[float, float]]:
        """Predict position when not detected."""
        gap = frame - self.last_frame
        if gap > self.max_gap:
            return None
        
        return self.kalman.predict(self.ball_id)
    
    def get_trajectory(self, n: int = 10) -> List[Tuple[float, float]]:
        """Get recent trajectory for visualization."""
        return [(p[0], p[1]) for p in self.history[-n:]]


# ============================================================
# PERSPECTIVE TRANSFORMER
# ============================================================

class PerspectiveTransformer:
    """Maps camera coordinates to 2D tactical board."""
    
    def __init__(self, config: PitchConfig):
        self.config = config
        self.H: Optional[np.ndarray] = None
    
    def set_homography(self, src: List[Tuple], dst: List[Tuple]) -> bool:
        """Set homography from point correspondences."""
        if len(src) < 4 or len(dst) < 4:
            return False
        
        self.H, _ = cv2.findHomography(
            np.array(src, dtype=np.float32),
            np.array(dst, dtype=np.float32),
            cv2.RANSAC, 5.0
        )
        return self.H is not None
    
    def transform_point(self, x: float, y: float) -> Optional[Tuple[int, int]]:
        """Transform single point."""
        if self.H is None:
            return None
        
        pt = np.array([[[x, y]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H)
        tx, ty = transformed[0][0]
        
        tx = max(0, min(tx, self.config.radar_width - 1))
        ty = max(0, min(ty, self.config.radar_height - 1))
        
        return int(tx), int(ty)
    
    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform multiple points."""
        if self.H is None or len(points) == 0:
            return np.array([])
        
        pts = points.reshape(-1, 1, 2).astype(np.float32)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)


# ============================================================
# VORONOI ANALYZER
# ============================================================

class VoronoiAnalyzer:
    """Spatial control analysis using Voronoi diagrams."""
    
    def __init__(self, config: PitchConfig):
        self.config = config
    
    def compute(self, positions: np.ndarray) -> Optional[Voronoi]:
        """Compute Voronoi diagram with boundary points."""
        if len(positions) < 2:
            return None
        
        margin = 50
        w, h = self.config.radar_width, self.config.radar_height
        
        boundary = np.array([
            [-margin, -margin], [-margin, h + margin],
            [w + margin, -margin], [w + margin, h + margin],
            [w/2, -margin*2], [w/2, h + margin*2],
            [-margin*2, h/2], [w + margin*2, h/2]
        ])
        
        try:
            return Voronoi(np.vstack([positions, boundary]))
        except:
            return None
    
    def draw_overlay(
        self, 
        image: np.ndarray, 
        positions: np.ndarray, 
        teams: np.ndarray, 
        alpha: float = 0.25
    ) -> np.ndarray:
        """Draw Voronoi zones colored by team."""
        if len(positions) < 2:
            return image
        
        vor = self.compute(positions)
        if vor is None:
            return image
        
        overlay = image.copy()
        h, w = image.shape[:2]
        colors = [(0, 0, 200), (200, 0, 0)]  # Team A red, Team B blue
        
        for i in range(len(positions)):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 in region or len(region) == 0:
                continue
            
            polygon = np.array([vor.vertices[j] for j in region], dtype=np.int32)
            polygon = np.clip(polygon, [0, 0], [w-1, h-1])
            
            team = int(teams[i]) % 2 if i < len(teams) else 0
            cv2.fillPoly(overlay, [polygon], colors[team])
        
        return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    def calculate_possession(self, positions: np.ndarray, teams: np.ndarray) -> Dict[str, float]:
        """Calculate possession based on controlled area."""
        if len(positions) < 2:
            return {"team_a": 50.0, "team_b": 50.0}
        
        vor = self.compute(positions)
        if vor is None:
            return {"team_a": 50.0, "team_b": 50.0}
        
        w, h = self.config.radar_width, self.config.radar_height
        areas = {0: 0.0, 1: 0.0}
        
        for i in range(len(positions)):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 in region or len(region) == 0:
                continue
            
            polygon = np.array([vor.vertices[j] for j in region])
            polygon = np.clip(polygon, [0, 0], [w-1, h-1])
            
            # Shoelace formula
            n = len(polygon)
            area = 0.0
            for j in range(n):
                k = (j + 1) % n
                area += polygon[j][0] * polygon[k][1]
                area -= polygon[k][0] * polygon[j][1]
            area = abs(area) / 2.0
            
            team = int(teams[i]) % 2 if i < len(teams) else 0
            areas[team] += area
        
        total = areas[0] + areas[1]
        if total > 0:
            return {
                "team_a": round(areas[0] / total * 100, 1),
                "team_b": round(areas[1] / total * 100, 1)
            }
        
        return {"team_a": 50.0, "team_b": 50.0}


# ============================================================
# FEATURE 5: RADAR BOARD RENDERER
# ============================================================

class RadarBoardRenderer:
    """Renders 2D tactical board with player positions and IDs."""
    
    def __init__(self, config: PitchConfig):
        self.config = config
        self.base = self._create_pitch()
    
    def _create_pitch(self) -> np.ndarray:
        """Create pitch template."""
        w, h = self.config.radar_width, self.config.radar_height
        board = np.zeros((h, w, 3), dtype=np.uint8)
        board[:] = (34, 85, 34)  # Dark green
        
        line = (255, 255, 255)
        t = 2
        
        # Outer boundary
        cv2.rectangle(board, (0, 0), (w-1, h-1), line, t)
        
        # Center line
        cv2.line(board, (w//2, 0), (w//2, h), line, t)
        
        # Center circle
        radius = int((self.config.center_circle_radius / self.config.width) * h)
        cv2.circle(board, (w//2, h//2), radius, line, t)
        cv2.circle(board, (w//2, h//2), 3, line, -1)
        
        # Penalty areas
        pa_w = int((self.config.penalty_area_width / self.config.width) * h)
        pa_l = int((self.config.penalty_area_length / self.config.length) * w)
        pa_y1, pa_y2 = (h - pa_w) // 2, (h + pa_w) // 2
        
        cv2.rectangle(board, (0, pa_y1), (pa_l, pa_y2), line, t)
        cv2.rectangle(board, (w - pa_l, pa_y1), (w, pa_y2), line, t)
        
        # Goal areas
        ga_w = int((self.config.goal_area_width / self.config.width) * h)
        ga_l = int((self.config.goal_area_length / self.config.length) * w)
        ga_y1, ga_y2 = (h - ga_w) // 2, (h + ga_w) // 2
        
        cv2.rectangle(board, (0, ga_y1), (ga_l, ga_y2), line, t)
        cv2.rectangle(board, (w - ga_l, ga_y1), (w, ga_y2), line, t)
        
        return board
    
    def render(
        self,
        positions: np.ndarray,
        teams: np.ndarray,
        player_ids: Optional[np.ndarray] = None,
        ball_pos: Optional[Tuple[int, int]] = None,
        show_voronoi: bool = False,
        voronoi: Optional[VoronoiAnalyzer] = None,
        show_ids: bool = True
    ) -> np.ndarray:
        """Render tactical board."""
        board = self.base.copy()
        
        # Voronoi overlay
        if show_voronoi and voronoi and len(positions) >= 2:
            board = voronoi.draw_overlay(board, positions, teams, alpha=0.25)
        
        colors = [(0, 0, 255), (255, 100, 0)]  # Red, Blue
        
        # Draw players
        for i, (pos, team) in enumerate(zip(positions, teams)):
            x, y = int(pos[0]), int(pos[1])
            color = colors[int(team) % 2]
            
            # Dot with outline
            cv2.circle(board, (x, y), 10, (0, 0, 0), -1)
            cv2.circle(board, (x, y), 8, color, -1)
            
            # Player ID
            if show_ids and player_ids is not None and i < len(player_ids):
                pid = player_ids[i]
                if pid is not None:
                    text = str(int(pid))
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.putText(board, text, (x - tw//2, y + th//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw ball
        if ball_pos:
            bx, by = int(ball_pos[0]), int(ball_pos[1])
            cv2.circle(board, (bx, by), 6, (0, 0, 0), -1)
            cv2.circle(board, (bx, by), 5, (255, 255, 255), -1)
        
        return board


# ============================================================
# MAIN PROCESSOR CLASS
# ============================================================

class FootballAnalysisProcessor:
    """
    Main orchestrator for football video analysis.
    
    Pipeline:
    1. YOLO detection (persons + ball)
    2. Size/aspect filtering
    3. Pitch boundary filtering
    4. Referee classification
    5. ByteTrack tracking (unique IDs)
    6. Kalman smoothing
    7. Team classification
    8. Perspective transformation
    9. Heatmap recording
    10. Voronoi analysis
    11. Visualization
    """
    
    def __init__(
        self,
        model_path: str = "yolov8m.pt",
        pitch_config: Optional[PitchConfig] = None,
        detection_config: Optional[DetectionConfig] = None
    ):
        self.pitch_config = pitch_config or PitchConfig()
        self.detection_config = detection_config or DetectionConfig()
        
        # Components
        self.pitch_boundary = PitchBoundaryDetector()
        self.referee_classifier = RefereeClassifier()
        self.team_classifier = TeamColorClassifier()
        self.transformer = PerspectiveTransformer(self.pitch_config)
        self.voronoi = VoronoiAnalyzer(self.pitch_config)
        self.radar = RadarBoardRenderer(self.pitch_config)
        self.heatmaps = PlayerHeatmapGenerator(self.pitch_config)
        self.kalman = KalmanTracker()
        self.ball_tracker = BallTracker()
        self.auto_homography = AutoHomographyEstimator(self.pitch_config)
        self.kinematics = PlayerKinematicsTracker(self.pitch_config, video_fps=30)
        # Detection model (lazy loaded)
        self._model = None
        self._model_path = model_path
        
        # ByteTrack
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.detection_config.track_thresh,
            lost_track_buffer=self.detection_config.track_buffer,
            minimum_matching_threshold=self.detection_config.match_thresh,
            frame_rate=30
        )
        
        # Annotators
        self.trace = sv.TraceAnnotator(thickness=2, trace_length=50, position=sv.Position.BOTTOM_CENTER)
        
        # State
        self.frame_count = 0
        self.calibration_crops: List[np.ndarray] = []
        self.is_calibrated = False
        self.active_ids: Set[int] = set()
    
    @property
    def model(self):
        """Lazy load YOLO model."""
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
        return self._model
    
    def _detect(self, frame: np.ndarray) -> sv.Detections:
        """Run YOLO detection."""
        results = self.model(frame, conf=self.detection_config.confidence_threshold, verbose=False)[0]
        return sv.Detections.from_ultralytics(results)
    
    def _filter_size(self, detections: sv.Detections) -> sv.Detections:
        """Filter by size and aspect ratio."""
        if len(detections) == 0:
            return detections
        
        keep = []
        for i, bbox in enumerate(detections.xyxy):
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            area = w * h
            ar = h / max(w, 1)
            
            if (self.detection_config.min_bbox_area <= area <= self.detection_config.max_bbox_area and
                self.detection_config.min_aspect_ratio <= ar <= self.detection_config.max_aspect_ratio):
                keep.append(i)
        
        return detections[keep] if keep else sv.Detections.empty()
    
    def _split_detections(self, detections: sv.Detections) -> Tuple[sv.Detections, sv.Detections]:
        """Split into persons and balls using custom model class IDs.
        Custom model classes:
            0: ball
            1: goalkeeper  
            2: player
            3: referee
        """
        if len(detections) == 0:
            return sv.Detections.empty(), sv.Detections.empty()
        
        player_mask = (detections.class_id == 2) | (detections.class_id == 1)
        ball_mask = detections.class_id == 0
        
        persons = detections[player_mask]
        balls = detections[ball_mask]
        
        return persons, balls
    
    def _get_crops(self, frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
        """Extract player crops."""
        crops = []
        for bbox in detections.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crops.append(frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None)
        return crops
    
    def _get_feet(self, detections: sv.Detections) -> np.ndarray:
        """Get feet positions."""
        return np.array([[(b[0] + b[2]) / 2, b[3]] for b in detections.xyxy])
    
    def _setup_homography(self, w: int, h: int) -> None:
        """Set default homography."""
        src = [
            (int(w * 0.1), int(h * 0.3)),
            (int(w * 0.9), int(h * 0.3)),
            (int(w * 0.05), int(h * 0.95)),
            (int(w * 0.95), int(h * 0.95)),
        ]
        dst = [
            (0, 0),
            (self.pitch_config.radar_width, 0),
            (0, self.pitch_config.radar_height),
            (self.pitch_config.radar_width, self.pitch_config.radar_height),
        ]
        self.transformer.set_homography(src, dst)
    
    def process_frame(
        self,
        frame: np.ndarray,
        show_tracking: bool = True,
        show_voronoi: bool = True,
        show_radar: bool = True
    ) -> Dict[str, Any]:
        """Process single frame through full pipeline."""
        self.frame_count += 1
        h, w = frame.shape[:2]
        
        # Initialize on first frame
        if self.frame_count % 30 == 0:
            self.pitch_boundary.detect_pitch(frame)
            try:
                success = self.auto_homography.estimate(frame, self.transformer)
                if not success:
                    self._setup_homography(w, h)
            except Exception:
                self._setup_homography(w, h)

# Re-estimate every 5 seconds (handles zoom / camera pan)
        if self.frame_count % 150 == 0:
            self.auto_homography.estimate(frame, self.transformer)
        
        # === DETECTION ===
        detections = self._detect(frame)
        persons, balls = self._split_detections(detections)
        
        # === FILTERING ===
        persons = self._filter_size(persons)
        persons = self.pitch_boundary.filter_detections(persons)
        # Custom model detects referees directly via class_id == 3
        ref_mask = detections.class_id == 3
        referees = detections[ref_mask] if len(detections) > 0 else sv.Detections.empty()
        players = persons  # already filtered to players + goalkeepers only
        # Camera cut detection
        if self.frame_count % 30 == 0 and len(players) < 8:
            self.team_classifier.confirmed_teams.clear()
            self.calibration_crops = []
            self.is_calibrated = False
        
        # === TRACKING (Unique IDs) ===
        players = self.tracker.update_with_detections(players)
        tracker_ids = players.tracker_id if players.tracker_id is not None else np.array([])
        
        # === KALMAN SMOOTHING ===
        feet = self._get_feet(players)
        smoothed = []
        for i, (pos, tid) in enumerate(zip(feet, tracker_ids)):
            if tid is not None:
                sx, sy = self.kalman.update(int(tid), pos[0], pos[1])
                smoothed.append([sx, sy])
                self.active_ids.add(int(tid))
            else:
                smoothed.append(pos.tolist())
        smoothed = np.array(smoothed) if smoothed else np.array([])
        
        # === BALL TRACKING ===
        ball_pos = None
        ball_radar = None
        if len(balls) > 0:
            bbox = balls.xyxy[0]
            bx, by = (bbox[0] + bbox[2]) / 2, bbox[3]
            ball_pos = self.ball_tracker.update(bx, by, self.frame_count)
        else:
            ball_pos = self.ball_tracker.predict(self.frame_count)
        
        # === TEAM CLASSIFICATION ===
        crops = self._get_crops(frame, players)
        
        if not self.is_calibrated and self.frame_count <= 50:
            valid = [c for c in crops if c is not None and c.size > 0]
            self.calibration_crops.extend(valid)
            if len(self.calibration_crops) >= 150:
                self.team_classifier.fit(self.calibration_crops)
                self.is_calibrated = True
            teams = np.zeros(len(players), dtype=int)
        else:
            teams = np.array([
                self.team_classifier.predict(
                    crops[i],
                    int(tracker_ids[i]) if i < len(tracker_ids) and tracker_ids[i] is not None else -1
                )
                for i in range(len(crops))
            ])
        
        # === TRANSFORMATION ===
        radar_pos = np.array([])
        if self.transformer.H is not None and len(smoothed) > 0:
            radar_pos = self.transformer.transform_points(smoothed)
            if tracker_ids is not None and len(tracker_ids) > 0:
                self.heatmaps.add_batch(tracker_ids, radar_pos)
            
            if ball_pos:
                ball_radar = self.transformer.transform_point(*ball_pos)
            # === KINEMATICS (speed / distance) ===
            if len(radar_pos) > 0 and tracker_ids is not None and len(tracker_ids) > 0:
                self.kinematics.update(
                    tracker_ids=tracker_ids,
                    radar_positions=radar_pos,
                    frame_number=self.frame_count,
                    teams=teams
    )   
        # === POSSESSION ===
        possession = self.voronoi.calculate_possession(radar_pos, teams) if len(radar_pos) >= 2 else {"team_a": 50, "team_b": 50}
        
        # === ANNOTATION ===
        annotated = frame.copy()
        
        # Draw players
        for i, (bbox, team) in enumerate(zip(players.xyxy, teams)):
            x1, y1, x2, y2 = map(int, bbox)
            color = (50, 50, 255) if team == 0 else (255, 150, 50)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            if tracker_ids is not None and i < len(tracker_ids) and tracker_ids[i] is not None:
                label = f"#{int(tracker_ids[i])}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw referees
        for bbox in referees.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated, "REF", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            annotated = self.kinematics.annotate_frame(
                annotated,
                tracker_ids=tracker_ids if tracker_ids is not None else np.array([]),
                bboxes=players.xyxy,
                show_speed=True,
                show_distance=False
            )
            annotated = self.kinematics.draw_speed_legend(annotated)
        
        # Draw ball
        if ball_pos:
            bx, by = int(ball_pos[0]), int(ball_pos[1])
            cv2.circle(annotated, (bx, by), 12, (255, 255, 255), 2)
            cv2.circle(annotated, (bx, by), 4, (255, 255, 255), -1)
            
            trajectory = self.ball_tracker.get_trajectory(15)
            if len(trajectory) > 1:
                for j in range(len(trajectory) - 1):
                    p1 = (int(trajectory[j][0]), int(trajectory[j][1]))
                    p2 = (int(trajectory[j+1][0]), int(trajectory[j+1][1]))
                    alpha = (j + 1) / len(trajectory)
                    cv2.line(annotated, p1, p2, (int(255*alpha),)*3, 2)
        
        # Tracking trails
        if show_tracking and self.frame_count > 10:
            annotated = self.trace.annotate(scene=annotated, detections=players)
        
        # Voronoi overlay
        if show_voronoi and len(smoothed) >= 2:
            annotated = self.voronoi.draw_overlay(annotated, smoothed, teams, alpha=0.12)
        
        # === RADAR BOARD ===
        radar_board = None
        if show_radar:
            radar_board = self.radar.render(
                radar_pos if len(radar_pos) > 0 else np.array([]),
                teams, tracker_ids, ball_radar,
                show_voronoi=show_voronoi, voronoi=self.voronoi
            )
        
        # === STATS ===
        stats = {
            "frame_number": self.frame_count,
            "players_detected": len(players),
            "referees_detected": len(referees),
            "ball_detected": ball_pos is not None,
            "team_a_count": int(np.sum(teams == 0)) if len(teams) > 0 else 0,
            "team_b_count": int(np.sum(teams == 1)) if len(teams) > 0 else 0,
            "possession_team_a": possession["team_a"],
            "possession_team_b": possession["team_b"],
            "is_calibrated": self.is_calibrated,
            "active_player_ids": sorted(list(self.active_ids)),
            "player_speeds":    self.kinematics.get_all_speeds(),
            "player_distances": self.kinematics.get_all_distances(),
            "team_stats":       self.kinematics.get_team_stats(),
            "sprint_alerts":    self.kinematics.get_sprint_alerts(),
            "top_speeds":       self.kinematics.get_top_speeds(5),
            "homography_auto":  self.auto_homography.last_debug_frame is not None,
        }
        
        return {
            "annotated_frame": annotated,
            "radar_board": radar_board,
            "stats": stats,
        }
    
    def get_player_heatmap(self, player_id: int) -> Optional[np.ndarray]:
        """Get heatmap for specific player."""
        return self.heatmaps.generate_overlay(player_id, self.radar.base.copy())
    
    def get_all_player_ids(self) -> List[int]:
        """Get all tracked player IDs."""
        return self.heatmaps.get_player_ids()
    
    def get_player_stats(self, player_id: int) -> Dict[str, Any]:
        """Get player movement stats."""
        return self.heatmaps.get_player_stats(player_id)
    
    def reset(self) -> None:
        """Reset processor state."""
        self.frame_count = 0
        self.calibration_crops = []
        self.is_calibrated = False
        self.active_ids.clear()
        self.heatmaps.reset()
        self.team_classifier = TeamColorClassifier()
        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.detection_config.track_thresh,
            lost_track_buffer=self.detection_config.track_buffer,
            minimum_matching_threshold=self.detection_config.match_thresh,
            frame_rate=30
        )
