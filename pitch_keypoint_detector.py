"""
Automatic Pitch Keypoint Detector
==================================
Replaces the hardcoded _setup_homography() in processor.py with
automatic pitch line detection using Hough transforms.

HOW TO INTEGRATE:
-----------------
1. Copy this file into your backend/ folder
2. In processor.py, add this import at the top:
       from pitch_keypoint_detector import AutoHomographyEstimator
3. In FootballAnalysisProcessor.__init__(), add:
       self.auto_homography = AutoHomographyEstimator(self.pitch_config)
4. In process_frame(), replace the first-frame homography block:

    REMOVE:
        if self.frame_count == 1:
            self._setup_homography(w, h)

    REPLACE WITH:
        if self.frame_count == 1:
            success = self.auto_homography.estimate(frame, self.transformer)
            if not success:
                self._setup_homography(w, h)  # fallback if detection fails
            self.auto_homography_debug = self.auto_homography.last_debug_frame

5. Optional - re-run every N frames for adaptive tracking:
        if self.frame_count % 300 == 0:
            self.auto_homography.estimate(frame, self.transformer)

HOW IT WORKS:
-------------
Pipeline:
  1. Green grass segmentation → isolate pitch
  2. Edge detection (Canny) on masked region
  3. Probabilistic Hough Line Transform → pitch line segments
  4. Cluster lines by angle → horizontal (sidelines/goal lines)
    and vertical (center line / penalty lines)
  5. Find intersections of dominant line pairs
  6. Match intersections to known FIFA pitch keypoints
  7. Compute robust homography via RANSAC

Why this beats hardcoded points:
  - Works on different broadcast camera angles
  - Adapts to zoom changes
  - RANSAC filters outlier detections
  - Falls back gracefully if detection fails

Author: Football AI Analyzer
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
from sklearn.cluster import DBSCAN


# ============================================================
# DATA TYPES
# ============================================================

@dataclass
class PitchLine:
    """Represents a detected line in image coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def angle_deg(self) -> float:
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        return np.degrees(np.arctan2(dy, dx)) % 180

    @property
    def length(self) -> float:
        return np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)

    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class KeypointMatch:
    """A matched image <-> pitch coordinate pair."""
    image_pt: Tuple[float, float]   # pixels in camera frame
    pitch_pt:  Tuple[float, float]  # meters on FIFA pitch
    confidence: float = 1.0


# ============================================================
# PITCH KEYPOINT DEFINITIONS  (FIFA standard, in metres)
# Origin = top-left corner of pitch
# X = along length (0-105), Y = along width (0-68)
# ============================================================

FIFA_KEYPOINTS: Dict[str, Tuple[float, float]] = {
    # Corners
    "tl_corner":          (0.0,  0.0),
    "tr_corner":          (105.0, 0.0),
    "bl_corner":          (0.0,  68.0),
    "br_corner":          (105.0, 68.0),

    # Center line intersections
    "center_top":         (52.5,  0.0),
    "center_bottom":      (52.5,  68.0),
    "center_spot":        (52.5,  34.0),

    # Left penalty area
    "lpa_tl":             (0.0,   13.85),
    "lpa_tr":             (16.5,  13.85),
    "lpa_bl":             (0.0,   54.15),
    "lpa_br":             (16.5,  54.15),

    # Right penalty area
    "rpa_tl":             (88.5,  13.85),
    "rpa_tr":             (105.0, 13.85),
    "rpa_bl":             (88.5,  54.15),
    "rpa_br":             (105.0, 54.15),

    # Left goal area
    "lga_tl":             (0.0,   24.85),
    "lga_tr":             (5.5,   24.85),
    "lga_bl":             (0.0,   43.15),
    "lga_br":             (5.5,   43.15),

    # Right goal area
    "rga_tl":             (99.5,  24.85),
    "rga_tr":             (105.0, 24.85),
    "rga_bl":             (99.5,  43.15),
    "rga_br":             (105.0, 43.15),
}


# ============================================================
# LINE UTILITIES
# ============================================================

def line_intersection(
    l1: PitchLine, l2: PitchLine
) -> Optional[Tuple[float, float]]:
    """
    Compute intersection of two lines (infinite extension).
    Returns None if lines are parallel.
    """
    x1, y1, x2, y2 = l1.x1, l1.y1, l1.x2, l1.y2
    x3, y3, x4, y4 = l2.x1, l2.y1, l2.x2, l2.y2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    return (x, y)


def cluster_lines_by_angle(
    lines: List[PitchLine],
    eps: float = 12.0
) -> Dict[str, List[PitchLine]]:
    """
    Cluster lines into 'horizontal' and 'vertical' groups
    using DBSCAN on their angles.
    """
    if not lines:
        return {"horizontal": [], "vertical": []}

    angles = np.array([l.angle_deg for l in lines]).reshape(-1, 1)
    # Map angles so 170° and 10° cluster together
    angles_wrapped = np.where(angles > 90, angles - 180, angles)

    clustering = DBSCAN(eps=eps, min_samples=1).fit(angles_wrapped)
    labels = clustering.labels_

    # Find cluster centres
    unique_labels = set(labels)
    cluster_centres: Dict[int, float] = {}
    for lbl in unique_labels:
        if lbl == -1:
            continue
        members = angles_wrapped[labels == lbl]
        cluster_centres[lbl] = float(np.mean(members))

    # Assign to horizontal / vertical
    result: Dict[str, List[PitchLine]] = {"horizontal": [], "vertical": []}
    for i, line in enumerate(lines):
        lbl = labels[i]
        if lbl == -1:
            continue
        centre = cluster_centres[lbl]
        if abs(centre) < 35:          # near 0° → horizontal
            result["horizontal"].append(line)
        elif abs(centre) > 55:        # near 90° → vertical
            result["vertical"].append(line)
        # lines between 35–55° are ignored (diagonal noise)

    return result


def merge_collinear_lines(
    lines: List[PitchLine], angle_tol: float = 8.0, dist_tol: float = 30.0
) -> List[PitchLine]:
    """
    Merge nearby parallel line segments into single long lines.
    Reduces duplicate detections of the same pitch line.
    """
    if not lines:
        return []

    merged: List[PitchLine] = []
    used = [False] * len(lines)

    for i, a in enumerate(lines):
        if used[i]:
            continue
        group = [a]
        for j, b in enumerate(lines):
            if i == j or used[j]:
                continue
            angle_diff = abs(a.angle_deg - b.angle_deg) % 180
            if angle_diff > angle_tol and angle_diff < (180 - angle_tol):
                continue
            # Check distance between midpoints projected perpendicularly
            mx1, my1 = a.midpoint()
            mx2, my2 = b.midpoint()
            perp_dist = abs(
                (my2 - my1) * np.cos(np.radians(a.angle_deg)) -
                (mx2 - mx1) * np.sin(np.radians(a.angle_deg))
            )
            if perp_dist < dist_tol:
                group.append(b)
                used[j] = True

        # Build merged line spanning the full group
        all_pts = [(l.x1, l.y1, l.x2, l.y2) for l in group]
        xs = [p[0] for p in all_pts] + [p[2] for p in all_pts]
        ys = [p[1] for p in all_pts] + [p[3] for p in all_pts]
        merged.append(PitchLine(min(xs), min(ys), max(xs), max(ys)))
        used[i] = True

    return merged


# ============================================================
# AUTO HOMOGRAPHY ESTIMATOR
# ============================================================

class AutoHomographyEstimator:
    """
    Automatically estimates the camera-to-pitch homography by
    detecting pitch line intersections and matching them to
    known FIFA keypoint coordinates.

    Usage:
        estimator = AutoHomographyEstimator(pitch_config)
        success = estimator.estimate(frame, perspective_transformer)
    """

    def __init__(self, pitch_config):
        self.config = pitch_config
        self.last_debug_frame: Optional[np.ndarray] = None

        # Grass colour range in HSV (tuned for broadcast)
        self.green_lower = np.array([30, 35, 35])
        self.green_upper = np.array([85, 255, 255])

        # Minimum line length to consider (pixels)
        self.min_line_length = 80

        # Radar destination grid (pixels)
        self.radar_w = pitch_config.radar_width
        self.radar_h = pitch_config.radar_height

    # ----------------------------------------------------------
    # PUBLIC API
    # ----------------------------------------------------------

    def estimate(self, frame: np.ndarray, transformer) -> bool:
        """
        Run full detection pipeline.
        Updates transformer.H in-place.
        Returns True if homography was successfully estimated.
        """
        lines = self._detect_lines(frame)
        if len(lines) < 4:
            return False

        clustered = cluster_lines_by_angle(lines)
        h_lines = merge_collinear_lines(clustered["horizontal"])
        v_lines = merge_collinear_lines(clustered["vertical"])

        if len(h_lines) < 2 or len(v_lines) < 2:
            return False

        intersections = self._find_intersections(
            h_lines, v_lines, frame.shape
        )
        if len(intersections) < 4:
            return False

        matches = self._match_to_pitch(intersections, frame.shape)
        if len(matches) < 4:
            return False

        H = self._compute_homography(matches)
        if H is None:
            return False

        transformer.H = H
        self.last_debug_frame = self._draw_debug(
            frame, lines, intersections, matches
        )
        return True

    # ----------------------------------------------------------
    # STEP 1: LINE DETECTION
    # ----------------------------------------------------------

    def _detect_lines(self, frame: np.ndarray) -> List[PitchLine]:
        """
        Detect pitch lines using:
        1. Grass mask to isolate playable area
        2. Grayscale + edge detection
        3. Probabilistic Hough transform
        """
        h, w = frame.shape[:2]

        # --- Grass mask ---
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        grass_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_CLOSE, kernel)
        grass_mask = cv2.morphologyEx(grass_mask, cv2.MORPH_DILATE, kernel)

        # --- Edge detection on masked region ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        masked = cv2.bitwise_and(gray, gray, mask=grass_mask)

        blurred = cv2.GaussianBlur(masked, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        # --- Hough line transform ---
        raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=self.min_line_length,
            maxLineGap=25,
        )

        lines: List[PitchLine] = []
        if raw is not None:
            for seg in raw:
                x1, y1, x2, y2 = seg[0]
                pl = PitchLine(float(x1), float(y1), float(x2), float(y2))
                # Ignore very short and very diagonal lines
                if pl.length >= self.min_line_length:
                    ang = pl.angle_deg
                    if ang < 35 or ang > 55:   # skip ~45° diagonals
                        lines.append(pl)

        return lines

    # ----------------------------------------------------------
    # STEP 2: FIND INTERSECTIONS
    # ----------------------------------------------------------

    def _find_intersections(
        self,
        h_lines: List[PitchLine],
        v_lines: List[PitchLine],
        frame_shape: Tuple,
    ) -> List[Tuple[float, float]]:
        """
        Compute all H×V intersection points,
        keeping only those inside the frame.
        """
        fh, fw = frame_shape[:2]
        pts: List[Tuple[float, float]] = []

        for hl in h_lines:
            for vl in v_lines:
                pt = line_intersection(hl, vl)
                if pt is None:
                    continue
                x, y = pt
                # Keep points inside frame with small margin
                if -50 <= x <= fw + 50 and -50 <= y <= fh + 50:
                    pts.append((x, y))

        return pts

    # ----------------------------------------------------------
    # STEP 3: MATCH INTERSECTIONS TO FIFA KEYPOINTS
    # ----------------------------------------------------------

    def _match_to_pitch(
        self,
        intersections: List[Tuple[float, float]],
        frame_shape: Tuple,
    ) -> List[KeypointMatch]:
        """
        Match detected intersections to the most likely
        FIFA keypoints using spatial reasoning:

        Strategy:
        - Sort intersections by position (top-left to bottom-right)
        - Use known pitch geometry constraints to select candidates
        - Each intersection votes for the keypoint it best fits
        """
        fh, fw = frame_shape[:2]
        pts = np.array(intersections)

        # Normalise to [0,1] for position-based reasoning
        norm_pts = pts.copy()
        norm_pts[:, 0] /= fw
        norm_pts[:, 1] /= fh

        # Expected normalised positions of key intersections
        # (These are approximate and camera-dependent but give
        #  enough signal for RANSAC to refine)
        candidates = self._select_candidate_keypoints(norm_pts, pts)
        return candidates

    def _select_candidate_keypoints(
        self,
        norm_pts: np.ndarray,
        raw_pts: np.ndarray,
    ) -> List[KeypointMatch]:
        """
        Select the best keypoint candidates using
        a spatial constraint approach.

        We look for the structural pattern of:
        - Topmost points → near top sideline
        - Bottommost points → near bottom sideline
        - Leftmost vertical cluster → left penalty/goal area
        - Rightmost vertical cluster → right side
        - Centre cluster → centre line
        """
        matches: List[KeypointMatch] = []

        if len(norm_pts) == 0:
            return matches

        # Sort by y then x
        order = np.lexsort((norm_pts[:, 0], norm_pts[:, 1]))
        sorted_norm = norm_pts[order]
        sorted_raw  = raw_pts[order]

        n = len(sorted_raw)

        # --- Top row: topmost 20% of intersections ---
        top_mask = sorted_norm[:, 1] < 0.35
        top_pts  = sorted_raw[top_mask]

        # --- Bottom row: bottommost 20% ---
        bot_mask = sorted_norm[:, 1] > 0.65
        bot_pts  = sorted_raw[bot_mask]

        # --- Left cluster: leftmost 20% ---
        left_mask = sorted_norm[:, 0] < 0.25
        left_pts  = sorted_raw[left_mask]

        # --- Right cluster: rightmost 20% ---
        right_mask = sorted_norm[:, 0] > 0.75
        right_pts  = sorted_raw[right_mask]

        # --- Centre cluster: middle 20% in x ---
        cx_mask = (sorted_norm[:, 0] > 0.4) & (sorted_norm[:, 0] < 0.6)
        cx_pts  = sorted_raw[cx_mask]

        # Map clusters to FIFA meter coordinates
        # Top-left corner
        if len(top_pts) > 0 and len(left_pts) > 0:
            tl = self._closest_to(top_pts, (0, 0), sorted_raw.shape[0])
            matches.append(KeypointMatch(
                image_pt=(float(tl[0]), float(tl[1])),
                pitch_pt=self._to_radar(0.0, 0.0),
                confidence=0.8
            ))

        # Top-right corner
        if len(top_pts) > 0 and len(right_pts) > 0:
            tr = self._closest_to(top_pts, (1, 0), sorted_raw.shape[0])
            matches.append(KeypointMatch(
                image_pt=(float(tr[0]), float(tr[1])),
                pitch_pt=self._to_radar(105.0, 0.0),
                confidence=0.8
            ))

        # Bottom-left corner
        if len(bot_pts) > 0 and len(left_pts) > 0:
            bl = self._closest_to(bot_pts, (0, 1), sorted_raw.shape[0])
            matches.append(KeypointMatch(
                image_pt=(float(bl[0]), float(bl[1])),
                pitch_pt=self._to_radar(0.0, 68.0),
                confidence=0.8
            ))

        # Bottom-right corner
        if len(bot_pts) > 0 and len(right_pts) > 0:
            br = self._closest_to(bot_pts, (1, 1), sorted_raw.shape[0])
            matches.append(KeypointMatch(
                image_pt=(float(br[0]), float(br[1])),
                pitch_pt=self._to_radar(105.0, 68.0),
                confidence=0.8
            ))

        # Centre line top
        if len(cx_pts) > 0:
            ct = cx_pts[np.argmin(cx_pts[:, 1])]
            matches.append(KeypointMatch(
                image_pt=(float(ct[0]), float(ct[1])),
                pitch_pt=self._to_radar(52.5, 0.0),
                confidence=0.7
            ))

        # Centre line bottom
        if len(cx_pts) > 0:
            cb = cx_pts[np.argmax(cx_pts[:, 1])]
            matches.append(KeypointMatch(
                image_pt=(float(cb[0]), float(cb[1])),
                pitch_pt=self._to_radar(52.5, 68.0),
                confidence=0.7
            ))

        # Left penalty area corners (top)
        if len(left_pts) >= 2:
            left_sorted = left_pts[np.argsort(left_pts[:, 1])]
            lpa_tr = left_sorted[0]
            matches.append(KeypointMatch(
                image_pt=(float(lpa_tr[0]), float(lpa_tr[1])),
                pitch_pt=self._to_radar(16.5, 13.85),
                confidence=0.65
            ))

        # Right penalty area corners (top)
        if len(right_pts) >= 2:
            right_sorted = right_pts[np.argsort(right_pts[:, 1])]
            rpa_tl = right_sorted[0]
            matches.append(KeypointMatch(
                image_pt=(float(rpa_tl[0]), float(rpa_tl[1])),
                pitch_pt=self._to_radar(88.5, 13.85),
                confidence=0.65
            ))

        return matches

    def _closest_to(
        self,
        pts: np.ndarray,
        norm_target: Tuple[float, float],
        frame_size: int,
    ) -> np.ndarray:
        """Return the point from pts closest to a normalised target."""
        if len(pts) == 1:
            return pts[0]
        tx = norm_target[0] * frame_size
        ty = norm_target[1] * frame_size
        dists = np.sqrt((pts[:, 0] - tx)**2 + (pts[:, 1] - ty)**2)
        return pts[np.argmin(dists)]

    def _to_radar(self, pitch_x_m: float, pitch_y_m: float) -> Tuple[float, float]:
        """Convert FIFA metres to radar pixel coordinates."""
        rx = (pitch_x_m / 105.0) * self.radar_w
        ry = (pitch_y_m / 68.0)  * self.radar_h
        return (rx, ry)

    # ----------------------------------------------------------
    # STEP 4: COMPUTE HOMOGRAPHY VIA RANSAC
    # ----------------------------------------------------------

    def _compute_homography(
        self, matches: List[KeypointMatch]
    ) -> Optional[np.ndarray]:
        """
        Compute homography from matched point pairs using RANSAC
        to reject outlier matches.
        """
        if len(matches) < 4:
            return None

        src = np.array([m.image_pt for m in matches], dtype=np.float32)
        dst = np.array([m.pitch_pt  for m in matches], dtype=np.float32)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=15.0)

        if H is None:
            return None

        inliers = int(np.sum(mask)) if mask is not None else 0
        if inliers < 4:
            return None

        return H

    # ----------------------------------------------------------
    # DEBUG VISUALISATION
    # ----------------------------------------------------------

    def _draw_debug(
        self,
        frame: np.ndarray,
        lines: List[PitchLine],
        intersections: List[Tuple[float, float]],
        matches: List[KeypointMatch],
    ) -> np.ndarray:
        """
        Draw detected lines, intersections, and matched keypoints
        on a copy of the frame for debugging / demo purposes.
        Expose via /debug-homography/{video_id} endpoint if desired.
        """
        debug = frame.copy()

        # Draw all detected lines in cyan
        for l in lines:
            cv2.line(debug,
                     (int(l.x1), int(l.y1)),
                     (int(l.x2), int(l.y2)),
                     (255, 255, 0), 1)

        # Draw intersections as small yellow dots
        for (x, y) in intersections:
            cv2.circle(debug, (int(x), int(y)), 4, (0, 255, 255), -1)

        # Draw matched keypoints as larger green dots with labels
        for i, m in enumerate(matches):
            ix, iy = int(m.image_pt[0]), int(m.image_pt[1])
            cv2.circle(debug, (ix, iy), 8, (0, 255, 0), -1)
            cv2.circle(debug, (ix, iy), 10, (255, 255, 255), 2)
            label = f"K{i}({m.pitch_pt[0]:.0f},{m.pitch_pt[1]:.0f})"
            cv2.putText(debug, label, (ix + 12, iy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.putText(debug, f"Keypoints matched: {len(matches)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2)

        return debug