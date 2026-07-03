"""
Player Kinematics Module
=========================
Computes real-world speed (km/h) and total distance (metres)
for each tracked player using the existing homography matrix
and FIFA pitch dimensions.

HOW TO INTEGRATE:
-----------------
1. Copy this file into your backend/ folder
2. In processor.py, add this import:
       from player_kinematics import PlayerKinematicsTracker
3. In FootballAnalysisProcessor.__init__(), add:
       self.kinematics = PlayerKinematicsTracker(
           pitch_config=self.pitch_config,
           video_fps=30          # update this from actual video FPS
       )
4. In FootballAnalysisProcessor.process_frame(), AFTER the
   transformation step (after radar_pos is computed), add:

       # === KINEMATICS ===
       if len(radar_pos) > 0 and tracker_ids is not None:
           self.kinematics.update(
               tracker_ids=tracker_ids,
               radar_positions=radar_pos,   # already in pixel space
               frame_number=self.frame_count,
               teams=teams
           )

5. In the stats dict at the bottom of process_frame(), add:
       "player_speeds": self.kinematics.get_all_speeds(),
       "player_distances": self.kinematics.get_all_distances(),
       "sprint_alerts": self.kinematics.get_sprint_alerts(),

6. In main.py process_video_task(), pass the actual FPS:
       processor = FootballAnalysisProcessor(...)
       processor.kinematics.video_fps = input_fps   # set real FPS

HOW IT WORKS:
-------------
The radar board is already in a known coordinate space:
  radar_width  pixels = 105 metres (pitch length)
  radar_height pixels = 68  metres (pitch width)

So the pixel-to-metre scale factors are:
  x_scale = 105.0 / radar_width   (metres per pixel)
  y_scale = 68.0  / radar_height  (metres per pixel)

For each player between consecutive frames:
  dx_m = dx_px * x_scale
  dy_m = dy_px * y_scale
  distance_m = sqrt(dx_m² + dy_m²)
  speed_kmh = (distance_m / frame_interval_s) * 3.6

A rolling window (default 0.5s) smooths noisy per-frame speeds.

Sprint threshold: > 25 km/h (FIFA standard for high-intensity running)
Walking threshold: < 7 km/h
"""

import cv2
import numpy as np
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


# ============================================================
# CONSTANTS
# ============================================================

SPRINT_THRESHOLD_KMH    = 25.0   # High-intensity sprint
JOG_THRESHOLD_KMH       = 14.0   # Jogging
WALK_THRESHOLD_KMH      = 7.0    # Walking
MAX_PLAUSIBLE_SPEED_KMH = 38.0   # Usain Bolt ceiling — filter outliers

SMOOTHING_WINDOW_S  = 0.5        # seconds of rolling average
DISTANCE_PRECISION  = 2          # decimal places for distance output


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class PlayerKinematicsState:
    """Per-player running state."""
    player_id: int
    team: int = 0

    # Position history: (radar_x_px, radar_y_px, frame_number)
    position_history: deque = field(default_factory=lambda: deque(maxlen=300))

    # Speed history in km/h (for rolling average)
    speed_history: deque = field(default_factory=deque)

    # Cumulative distance in metres
    total_distance_m: float = 0.0

    # Current smoothed speed
    current_speed_kmh: float = 0.0

    # Peak speed observed
    peak_speed_kmh: float = 0.0

    # Sprint counters
    sprint_count: int = 0
    in_sprint: bool = False
    sprint_distance_m: float = 0.0

    # Walking / jogging / sprinting breakdown (metres)
    distance_by_intensity: Dict[str, float] = field(
        default_factory=lambda: {"walk": 0.0, "jog": 0.0, "run": 0.0, "sprint": 0.0}
    )


@dataclass
class SprintAlert:
    """Alert fired when a player enters a sprint."""
    player_id: int
    team: int
    speed_kmh: float
    position: Tuple[float, float]


# ============================================================
# MAIN TRACKER
# ============================================================

class PlayerKinematicsTracker:
    """
    Tracks real-world speed and distance for all players.

    Coordinate system assumption:
      radar_pos values are in pixels where:
        full width  (radar_width px)  = 105 m
        full height (radar_height px) = 68  m

    This matches the existing PerspectiveTransformer output.
    """

    def __init__(self, pitch_config, video_fps: float = 30.0):
        self.config     = pitch_config
        self.video_fps  = video_fps

        # Metres per radar pixel
        self.x_scale = 105.0 / pitch_config.radar_width   # m/px
        self.y_scale = 68.0  / pitch_config.radar_height  # m/px

        # Per-player state
        self.players: Dict[int, PlayerKinematicsState] = {}

        # Rolling window size (frames)
        self._smoothing_frames = max(1, int(video_fps * SMOOTHING_WINDOW_S))

        # Sprint alerts since last flush
        self._sprint_alerts: List[SprintAlert] = []

    # ----------------------------------------------------------
    # PUBLIC: UPDATE (call once per frame)
    # ----------------------------------------------------------

    def update(
        self,
        tracker_ids: np.ndarray,
        radar_positions: np.ndarray,
        frame_number: int,
        teams: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update kinematics for all currently visible players.

        Args:
            tracker_ids:    shape (N,) — ByteTrack IDs
            radar_positions: shape (N, 2) — [x_px, y_px] on radar board
            frame_number:   current frame index (used for dt calculation)
            teams:          shape (N,) — team assignments (0 or 1)
        """
        if len(tracker_ids) == 0 or len(radar_positions) == 0:
            return

        for i, tid in enumerate(tracker_ids):
            if tid is None:
                continue

            pid   = int(tid)
            pos   = radar_positions[i]
            team  = int(teams[i]) if teams is not None and i < len(teams) else 0

            # Initialise new player
            if pid not in self.players:
                self.players[pid] = PlayerKinematicsState(
                    player_id=pid, team=team
                )
                state = self.players[pid]
                state.position_history.append((pos[0], pos[1], frame_number))
                continue

            state = self.players[pid]
            state.team = team

            # Compute displacement from last known position
            last_x, last_y, last_frame = state.position_history[-1]
            dx_px = pos[0] - last_x
            dy_px = pos[1] - last_y
            d_frames = frame_number - last_frame

            if d_frames <= 0:
                state.position_history.append((pos[0], pos[1], frame_number))
                continue

            # Convert to metres
            dx_m = dx_px * self.x_scale
            dy_m = dy_px * self.y_scale
            dist_m = np.sqrt(dx_m**2 + dy_m**2)

            # Time interval
            dt_s = d_frames / self.video_fps

            # Instantaneous speed
            raw_speed_kmh = (dist_m / dt_s) * 3.6 if dt_s > 0 else 0.0

            # Filter implausible speeds (teleportation / ID swap artefacts)
            if raw_speed_kmh > MAX_PLAUSIBLE_SPEED_KMH:
                state.position_history.append((pos[0], pos[1], frame_number))
                continue

            # Rolling speed average
            state.speed_history.append(raw_speed_kmh)
            if len(state.speed_history) > self._smoothing_frames:
                state.speed_history.popleft()

            smoothed_speed = float(np.mean(list(state.speed_history)))
            state.current_speed_kmh = smoothed_speed

            # Peak
            if smoothed_speed > state.peak_speed_kmh:
                state.peak_speed_kmh = smoothed_speed

            # Accumulate distance
            state.total_distance_m += dist_m

            # Intensity breakdown
            intensity = self._classify_intensity(smoothed_speed)
            state.distance_by_intensity[intensity] += dist_m

            # Sprint detection
            self._handle_sprint(state, smoothed_speed, pos)

            # Store new position
            state.position_history.append((pos[0], pos[1], frame_number))

    # ----------------------------------------------------------
    # PUBLIC: GETTERS
    # ----------------------------------------------------------

    def get_speed(self, player_id: int) -> float:
        """Current smoothed speed for a player in km/h."""
        if player_id not in self.players:
            return 0.0
        return round(self.players[player_id].current_speed_kmh, 1)

    def get_distance(self, player_id: int) -> float:
        """Total distance covered by a player in metres."""
        if player_id not in self.players:
            return 0.0
        return round(self.players[player_id].total_distance_m, DISTANCE_PRECISION)

    def get_all_speeds(self) -> Dict[int, float]:
        """Speed for every tracked player {player_id: kmh}."""
        return {
            pid: round(s.current_speed_kmh, 1)
            for pid, s in self.players.items()
        }

    def get_all_distances(self) -> Dict[int, float]:
        """Total distance for every tracked player {player_id: metres}."""
        return {
            pid: round(s.total_distance_m, DISTANCE_PRECISION)
            for pid, s in self.players.items()
        }

    def get_player_full_stats(self, player_id: int) -> Dict[str, Any]:
        """Complete kinematics stats for one player."""
        if player_id not in self.players:
            return {}
        s = self.players[player_id]
        return {
            "player_id":         s.player_id,
            "team":              s.team,
            "speed_kmh":         round(s.current_speed_kmh, 1),
            "peak_speed_kmh":    round(s.peak_speed_kmh, 1),
            "total_distance_m":  round(s.total_distance_m, 2),
            "sprint_count":      s.sprint_count,
            "sprint_distance_m": round(s.sprint_distance_m, 2),
            "intensity_breakdown": {
                k: round(v, 2) for k, v in s.distance_by_intensity.items()
            },
            "intensity": self._classify_intensity(s.current_speed_kmh),
        }

    def get_team_stats(self) -> Dict[int, Dict[str, Any]]:
        """Aggregated stats per team."""
        teams: Dict[int, Dict[str, Any]] = {}
        for s in self.players.values():
            t = s.team
            if t not in teams:
                teams[t] = {
                    "total_distance_m": 0.0,
                    "avg_speed_kmh": [],
                    "sprint_count": 0,
                    "player_count": 0,
                }
            teams[t]["total_distance_m"] += s.total_distance_m
            teams[t]["avg_speed_kmh"].append(s.current_speed_kmh)
            teams[t]["sprint_count"] += s.sprint_count
            teams[t]["player_count"] += 1

        for t in teams:
            speeds = teams[t].pop("avg_speed_kmh")
            teams[t]["avg_speed_kmh"] = round(float(np.mean(speeds)), 1) if speeds else 0.0
            teams[t]["total_distance_m"] = round(teams[t]["total_distance_m"], 1)

        return teams

    def get_sprint_alerts(self) -> List[Dict]:
        """Flush and return sprint alerts since last call."""
        alerts = [
            {
                "player_id": a.player_id,
                "team": a.team,
                "speed_kmh": round(a.speed_kmh, 1),
                "position": a.position,
            }
            for a in self._sprint_alerts
        ]
        self._sprint_alerts.clear()
        return alerts

    def get_top_speeds(self, n: int = 5) -> List[Dict]:
        """Top N players by peak speed."""
        ranked = sorted(
            self.players.values(),
            key=lambda s: s.peak_speed_kmh,
            reverse=True
        )
        return [
            {
                "player_id": s.player_id,
                "team": s.team,
                "peak_speed_kmh": round(s.peak_speed_kmh, 1),
            }
            for s in ranked[:n]
        ]

    # ----------------------------------------------------------
    # VISUALISATION HELPER
    # ----------------------------------------------------------

    def annotate_frame(
        self,
        frame: np.ndarray,
        tracker_ids: np.ndarray,
        bboxes: np.ndarray,
        show_speed: bool = True,
        show_distance: bool = False,
    ) -> np.ndarray:
        """
        Draw speed (and optionally distance) labels above player bboxes.
        Call this AFTER the existing bbox drawing in process_frame().

        Args:
            frame:       annotated frame (already has bbox rectangles)
            tracker_ids: ByteTrack IDs for visible players
            bboxes:      xyxy bounding boxes, same order as tracker_ids
            show_speed:  overlay current speed in km/h
            show_distance: overlay cumulative distance in metres
        """
        if not show_speed and not show_distance:
            return frame

        annotated = frame  # mutate in place

        for i, tid in enumerate(tracker_ids):
            if tid is None or i >= len(bboxes):
                continue

            pid = int(tid)
            if pid not in self.players:
                continue

            s = self.players[pid]
            x1, y1 = int(bboxes[i][0]), int(bboxes[i][1])

            lines = []
            if show_speed:
                speed = round(s.current_speed_kmh, 1)
                intensity = self._classify_intensity(speed)
                colour = self._intensity_colour(intensity)
                lines.append((f"{speed} km/h", colour))
            if show_distance:
                dist = round(s.total_distance_m)
                lines.append((f"{dist}m", (200, 200, 200)))

            # Draw each line above the bbox
            for j, (text, colour) in enumerate(lines):
                offset_y = y1 - 8 - j * 18
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                # Background pill
                import cv2 as _cv2
                _cv2.rectangle(
                    annotated,
                    (x1, offset_y - th - 4),
                    (x1 + tw + 6, offset_y + 2),
                    (0, 0, 0), -1
                )
                _cv2.putText(
                    annotated, text,
                    (x1 + 3, offset_y),
                    _cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colour, 1, _cv2.LINE_AA
                )

        return annotated

    def draw_speed_legend(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw a small speed-intensity legend in the bottom-left corner.
        """
        import cv2 as _cv2
        legend = [
            ("Sprint  >25 km/h", (0, 0, 255)),
            ("Run     14-25",     (0, 165, 255)),
            ("Jog      7-14",     (0, 255, 0)),
            ("Walk    <7 km/h",  (200, 200, 200)),
        ]
        x0, y0 = frame.shape[1] - 200, frame.shape[0] - 10 - len(legend) * 22
        for i, (text, colour) in enumerate(legend):
            y = y0 + i * 22
            _cv2.rectangle(frame, (x0, y - 14), (x0 + 160, y + 4), (30, 30, 30), -1)
            _cv2.putText(frame, text, (x0 + 4, y),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, _cv2.LINE_AA)
        return frame

    # ----------------------------------------------------------
    # INTERNAL HELPERS
    # ----------------------------------------------------------

    @staticmethod
    def _classify_intensity(speed_kmh: float) -> str:
        if speed_kmh >= SPRINT_THRESHOLD_KMH:
            return "sprint"
        if speed_kmh >= JOG_THRESHOLD_KMH:
            return "run"
        if speed_kmh >= WALK_THRESHOLD_KMH:
            return "jog"
        return "walk"

    @staticmethod
    def _intensity_colour(intensity: str) -> Tuple[int, int, int]:
        return {
            "sprint": (0,   0,   255),   # red
            "run":    (0,   165, 255),   # orange
            "jog":    (0,   255, 0),     # green
            "walk":   (200, 200, 200),   # grey
        }.get(intensity, (255, 255, 255))

    def _handle_sprint(
        self,
        state: PlayerKinematicsState,
        speed_kmh: float,
        pos: np.ndarray,
    ) -> None:
        """Track sprint entry/exit and accumulate sprint distance."""
        if speed_kmh >= SPRINT_THRESHOLD_KMH:
            if not state.in_sprint:
                state.in_sprint   = True
                state.sprint_count += 1
                self._sprint_alerts.append(SprintAlert(
                    player_id=state.player_id,
                    team=state.team,
                    speed_kmh=speed_kmh,
                    position=(float(pos[0]), float(pos[1]))
                ))
        else:
            state.in_sprint = False