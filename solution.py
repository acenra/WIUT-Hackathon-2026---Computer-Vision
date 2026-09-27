from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


# ============================================================
# OFFICIAL CLASSES
# ============================================================

CLASSES: list[str] = [
    "accident",
    "near_miss",
    "red_light",
    "wrong_way",
    "illegal_u_turn",
    "stopped_vehicle",
    "jaywalking",
    "failure_to_yield",
    "illegal_turn",
    "solid_line_crossing",
    "stop_line",
    "congestion",
    "road_obstacle",
    "fire_smoke",
]

RISK_HORIZON_SEC = 5.0


# ============================================================
# CPU CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "weights" / "yolo11n.pt"

IMG_SIZE = 320

CONF = 0.30
IOU_NMS = 0.45

# Keep these values because the previous version was
# already close to the 3x runtime budget.
PART_A_DETECT_EVERY = 10
RISK_DETECT_EVERY = 15

MAX_MISSED = 5
MAX_HISTORY = 20

# Number of consecutive detector observations required
# before an event becomes active.
EVENT_CONFIRMATIONS = 2

# ============================================================
# COCO IDS
# ============================================================

PERSON = 0
BICYCLE = 1
CAR = 2
MOTORCYCLE = 3
BUS = 5
TRUCK = 7

VEHICLE_CLASSES = {
    CAR,
    MOTORCYCLE,
    BUS,
    TRUCK,
}

ROAD_USER_CLASSES = {
    PERSON,
    BICYCLE,
    MOTORCYCLE,
    CAR,
    BUS,
    TRUCK,
}


# ============================================================
# GLOBAL MODEL CACHE
# ============================================================

# Important for speed:
# detect_events() and RiskEstimator no longer load separate
# YOLO instances.
_GLOBAL_MODEL = None
_GLOBAL_MODEL_READY = False


def get_global_model():
    global _GLOBAL_MODEL
    global _GLOBAL_MODEL_READY

    if _GLOBAL_MODEL_READY:
        return _GLOBAL_MODEL

    _GLOBAL_MODEL_READY = True

    if YOLO is None:
        return None

    if not MODEL_PATH.exists():
        return None

    try:
        _GLOBAL_MODEL = YOLO(str(MODEL_PATH))
    except Exception:
        _GLOBAL_MODEL = None

    return _GLOBAL_MODEL


# ============================================================
# DATA
# ============================================================

@dataclass
class Detection:
    cls: int
    conf: float
    box: np.ndarray

    @property
    def cx(self) -> float:
        return float((self.box[0] + self.box[2]) * 0.5)

    @property
    def cy(self) -> float:
        return float((self.box[1] + self.box[3]) * 0.5)

    @property
    def w(self) -> float:
        return float(max(1.0, self.box[2] - self.box[0]))

    @property
    def h(self) -> float:
        return float(max(1.0, self.box[3] - self.box[1]))


@dataclass
class Track:
    track_id: int
    cls: int
    box: np.ndarray
    conf: float
    last_t: float

    history: list[
        tuple[float, float, float, float, float]
    ] = field(default_factory=list)

    missed: int = 0

    def update(
        self,
        det: Detection,
        t: float,
    ) -> None:

        self.cls = det.cls
        self.box = det.box.copy()
        self.conf = det.conf
        self.last_t = t
        self.missed = 0

        self.history.append(
            (
                float(t),
                det.cx,
                det.cy,
                det.w,
                det.h,
            )
        )

        if len(self.history) > MAX_HISTORY:
            self.history = self.history[-MAX_HISTORY:]

    @property
    def cx(self) -> float:
        return float((self.box[0] + self.box[2]) * 0.5)

    @property
    def cy(self) -> float:
        return float((self.box[1] + self.box[3]) * 0.5)


# ============================================================
# GEOMETRY
# ============================================================

def iou(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))

    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)

    inter = w * h

    aa = (
        max(0.0, float(a[2] - a[0]))
        * max(0.0, float(a[3] - a[1]))
    )

    ab = (
        max(0.0, float(b[2] - b[0]))
        * max(0.0, float(b[3] - b[1]))
    )

    union = aa + ab - inter

    if union <= 0:
        return 0.0

    return inter / union


def norm_distance(
    a: Track,
    b: Track,
    width: int,
    height: int,
) -> float:

    dx = (
        a.cx - b.cx
    ) / max(1, width)

    dy = (
        a.cy - b.cy
    ) / max(1, height)

    return math.sqrt(
        dx * dx + dy * dy
    )


# ============================================================
# DETECTOR
# ============================================================

class RoadDetector:

    def __init__(self) -> None:
        self.model = None

    def load(self) -> bool:
        self.model = get_global_model()
        return self.model is not None

    def detect(
        self,
        frame: np.ndarray,
    ) -> list[Detection]:

        if not self.load():
            return []

        try:
            results = self.model.predict(
                source=frame,
                imgsz=IMG_SIZE,
                conf=CONF,
                iou=IOU_NMS,
                device="cpu",
                verbose=False,
                max_det=40,
            )
        except Exception:
            return []

        if not results:
            return []

        boxes = results[0].boxes

        if boxes is None:
            return []

        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = (
                boxes.cls.cpu()
                .numpy()
                .astype(np.int32)
            )
        except Exception:
            return []

        detections: list[Detection] = []

        for box, conf, cls in zip(
            xyxy,
            confs,
            classes,
        ):

            cls = int(cls)

            if cls not in ROAD_USER_CLASSES:
                continue

            w = max(
                1.0,
                float(box[2] - box[0]),
            )

            h = max(
                1.0,
                float(box[3] - box[1]),
            )

            # Ignore tiny detections.
            if w * h < 100:
                continue

            detections.append(
                Detection(
                    cls=cls,
                    conf=float(conf),
                    box=np.asarray(
                        box,
                        dtype=np.float32,
                    ),
                )
            )

        return detections


# ============================================================
# TRACKER
# ============================================================

class SimpleTracker:

    def __init__(self) -> None:
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1

    def update(
        self,
        detections: list[Detection],
        t: float,
        width: int,
        height: int,
    ) -> list[Track]:

        tracks = list(
            self.tracks.values()
        )

        candidates: list[
            tuple[float, int, int]
        ] = []

        for ti, tr in enumerate(tracks):

            for di, det in enumerate(detections):

                if tr.cls != det.cls:
                    continue

                overlap = iou(
                    tr.box,
                    det.box,
                )

                dx = (
                    tr.cx - det.cx
                ) / max(1, width)

                dy = (
                    tr.cy - det.cy
                ) / max(1, height)

                distance = math.sqrt(
                    dx * dx + dy * dy
                )

                if (
                    overlap < 0.05
                    and distance > 0.10
                ):
                    continue

                score = (
                    overlap * 2.0
                    - distance
                )

                candidates.append(
                    (
                        score,
                        ti,
                        di,
                    )
                )

        candidates.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        used_tracks: set[int] = set()
        used_dets: set[int] = set()

        for _, ti, di in candidates:

            if ti in used_tracks:
                continue

            if di in used_dets:
                continue

            tr = tracks[ti]
            det = detections[di]

            tr.update(
                det,
                t,
            )

            used_tracks.add(ti)
            used_dets.add(di)

        # Unmatched tracks.
        for ti, tr in enumerate(tracks):

            if ti not in used_tracks:
                tr.missed += 1

        # New tracks.
        for di, det in enumerate(detections):

            if di in used_dets:
                continue

            tr = Track(
                track_id=self.next_id,
                cls=det.cls,
                box=det.box.copy(),
                conf=det.conf,
                last_t=t,
            )

            tr.update(
                det,
                t,
            )

            self.tracks[
                self.next_id
            ] = tr

            self.next_id += 1

        # Remove old tracks.
        dead = [
            tid
            for tid, tr in self.tracks.items()
            if tr.missed > MAX_MISSED
        ]

        for tid in dead:
            del self.tracks[tid]

        return list(
            self.tracks.values()
        )


# ============================================================
# MOTION
# ============================================================

def velocity(
    tr: Track,
    width: int,
    height: int,
) -> tuple[float, float, float]:

    if len(tr.history) < 2:
        return 0.0, 0.0, 0.0

    current = tr.history[-1]
    previous = tr.history[-2]

    dt = (
        current[0]
        - previous[0]
    )

    if dt <= 0:
        return 0.0, 0.0, 0.0

    vx = (
        current[1]
        - previous[1]
    ) / max(1, width) / dt

    vy = (
        current[2]
        - previous[2]
    ) / max(1, height) / dt

    speed = math.sqrt(
        vx * vx + vy * vy
    )

    return vx, vy, speed


def estimate_ttc(
    a: Track,
    b: Track,
    width: int,
    height: int,
) -> Optional[float]:

    if len(a.history) < 2:
        return None

    if len(b.history) < 2:
        return None

    avx, avy, _ = velocity(
        a,
        width,
        height,
    )

    bvx, bvy, _ = velocity(
        b,
        width,
        height,
    )

    rx = (
        b.cx - a.cx
    ) / max(1, width)

    ry = (
        b.cy - a.cy
    ) / max(1, height)

    rvx = bvx - avx
    rvy = bvy - avy

    distance = math.sqrt(
        rx * rx + ry * ry
    )

    if distance < 0.001:
        return 0.0

    closing = -(
        rx * rvx
        + ry * rvy
    ) / distance

    if closing <= 0:
        return None

    ttc = distance / closing

    if ttc < 0 or ttc > 8:
        return None

    return ttc


def relative_speed(
    a: Track,
    b: Track,
    width: int,
    height: int,
) -> float:

    avx, avy, _ = velocity(
        a,
        width,
        height,
    )

    bvx, bvy, _ = velocity(
        b,
        width,
        height,
    )

    return math.sqrt(
        (avx - bvx) ** 2
        + (avy - bvy) ** 2
    )


# ============================================================
# EVENT ANALYSIS
# ============================================================

def analyze(
    tracks: list[Track],
    width: int,
    height: int,
    t: float,
) -> dict[str, bool]:

    result = {
        label: False
        for label in CLASSES
    }

    vehicles = [
        tr
        for tr in tracks
        if (
            tr.cls in VEHICLE_CLASSES
            and tr.missed == 0
        )
    ]

    persons = [
        tr
        for tr in tracks
        if (
            tr.cls == PERSON
            and tr.missed == 0
        )
    ]

    # ========================================================
    # ACCIDENT / NEAR MISS
    # ========================================================

    for i in range(len(vehicles)):

        for j in range(i + 1, len(vehicles)):

            a = vehicles[i]
            b = vehicles[j]

            distance = norm_distance(
                a,
                b,
                width,
                height,
            )

            # Too far apart to be an immediate conflict.
            if distance > 0.10:
                continue

            ttc = estimate_ttc(
                a,
                b,
                width,
                height,
            )

            rel_speed = relative_speed(
                a,
                b,
                width,
                height,
            )

            overlap = iou(
                a.box,
                b.box,
            )

            # ------------------------------------------------
            # Strong collision evidence.
            #
            # Previous version used overlap > 0.15,
            # which produced too many short false accidents.
            # ------------------------------------------------

            direct_collision = (
                overlap >= 0.25
                and rel_speed >= 0.018
            )

            imminent_collision = (
                ttc is not None
                and ttc <= 0.60
                and distance <= 0.075
                and rel_speed >= 0.018
            )

            if (
                direct_collision
                or imminent_collision
            ):
                result[
                    "accident"
                ] = True

                continue

            # ------------------------------------------------
            # Near miss.
            # ------------------------------------------------

            strong_near_miss = (
                ttc is not None
                and ttc <= 1.10
                and distance <= 0.090
                and rel_speed >= 0.015
            )

            if strong_near_miss:
                result[
                    "near_miss"
                ] = True

    # ========================================================
    # STOPPED VEHICLE
    # ========================================================

    for tr in vehicles:

        if len(tr.history) < 3:
            continue

        if (
            t - tr.history[0][0]
            < 10.0
        ):
            continue

        xs = [
            item[1]
            for item in tr.history
        ]

        ys = [
            item[2]
            for item in tr.history
        ]

        movement = math.sqrt(
            (
                max(xs)
                - min(xs)
            ) ** 2
            +
            (
                max(ys)
                - min(ys)
            ) ** 2
        )

        if movement < max(
            width,
            height,
        ) * 0.025:

            if tr.conf >= 0.30:
                result[
                    "stopped_vehicle"
                ] = True

    # ========================================================
    # CONGESTION
    # ========================================================

    if len(vehicles) >= 6:

        slow = 0

        for tr in vehicles:

            _, _, speed = velocity(
                tr,
                width,
                height,
            )

            if speed < 0.012:
                slow += 1

        if (
            slow / len(vehicles)
            >= 0.65
        ):
            result[
                "congestion"
            ] = True

    # ========================================================
    # JAYWALKING
    # ========================================================

    for person in persons:

        # Person should be in the lower road area.
        if person.cy < height * 0.48:
            continue

        vx, vy, speed = velocity(
            person,
            width,
            height,
        )

        if speed < 0.012:
            continue

        # Horizontal motion across the road.
        if abs(vx) > abs(vy) * 0.8:

            result[
                "jaywalking"
            ] = True

            break

    # ========================================================
    # FAILURE TO YIELD
    # ========================================================

    for person in persons:

        for vehicle in vehicles:

            distance = norm_distance(
                person,
                vehicle,
                width,
                height,
            )

            if distance > 0.065:
                continue

            _, _, vehicle_speed = velocity(
                vehicle,
                width,
                height,
            )

            _, _, person_speed = velocity(
                person,
                width,
                height,
            )

            if (
                vehicle_speed > 0.018
                and person_speed > 0.008
            ):

                result[
                    "failure_to_yield"
                ] = True

                break

        if result[
            "failure_to_yield"
        ]:
            break

    return result


# ============================================================
# SEGMENT MERGING
# ============================================================

def merge_segments(
    segments: list[tuple[float, float]],
    min_duration: float,
    gap: float = 0.5,
) -> list[tuple[float, float]]:

    if not segments:
        return []

    segments.sort()

    result: list[list[float]] = []

    for start, end in segments:

        if end <= start:
            continue

        if not result:

            result.append(
                [start, end]
            )

            continue

        previous = result[-1]

        if (
            start
            <= previous[1] + gap
        ):

            previous[1] = max(
                previous[1],
                end,
            )

        else:

            result.append(
                [start, end]
            )

    return [
        (a, b)
        for a, b in result
        if b - a >= min_duration
    ]


# ============================================================
# TEMPORAL EVENT STABILIZATION
# ============================================================

def stabilize_flags(
    flags: dict[str, list[tuple[float, bool]]],
) -> dict[str, list[tuple[float, bool]]]:

    result = {}

    for label, values in flags.items():

        if label not in {
            "accident",
            "near_miss",
        }:

            result[label] = values
            continue

        output = []

        pending = 0
        pending_times: list[float] = []

        for t, active in values:

            if active:

                pending += 1
                pending_times.append(t)

                if pending >= EVENT_CONFIRMATIONS:

                    # Activate the current and previous
                    # confirmation point.
                    for pt in pending_times:
                        output.append(
                            (pt, True)
                        )

                    pending_times = []

            else:

                pending = 0
                pending_times = []

                output.append(
                    (t, False)
                )

        result[label] = output

    return result


# ============================================================
# PART A
# ============================================================

def detect_events(
    video_path: str,
) -> list[list]:

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        return []

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 30.0
    )

    n_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
        or 0
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
        or 0
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
        or 0
    )

    if fps <= 0:
        fps = 30.0

    duration = (
        n_frames / fps
        if n_frames > 0
        else 0.0
    )

    detector = RoadDetector()
    tracker = SimpleTracker()

    flags = {
        label: []
        for label in CLASSES
    }

    frame_idx = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        if (
            frame_idx
            % PART_A_DETECT_EVERY
            == 0
        ):

            t = frame_idx / fps

            detections = detector.detect(
                frame
            )

            tracks = tracker.update(
                detections,
                t,
                width,
                height,
            )

            current = analyze(
                tracks,
                width,
                height,
                t,
            )

            for label in CLASSES:

                flags[label].append(
                    (
                        t,
                        current[label],
                    )
                )

        frame_idx += 1

    cap.release()

    # Require repeated evidence for accident/near_miss.
    flags = stabilize_flags(flags)

    events: list[list] = []

    for label in CLASSES:

        active_start = None
        last_active = None

        segments = []

        for t, active in flags[label]:

            if active:

                if active_start is None:
                    active_start = t

                last_active = t

            else:

                if active_start is not None:

                    segments.append(
                        (
                            active_start,
                            last_active
                            if last_active is not None
                            else t,
                        )
                    )

                    active_start = None
                    last_active = None

        if active_start is not None:

            segments.append(
                (
                    active_start,
                    duration,
                )
            )

        # ----------------------------------------------------
        # Minimum event duration.
        # ----------------------------------------------------

        if label == "stopped_vehicle":
            minimum = 5.0

        elif label == "congestion":
            minimum = 1.0

        elif label == "accident":
            minimum = 0.60

        elif label == "near_miss":
            minimum = 0.60

        else:
            minimum = 0.50

        # Don't bridge long gaps between accident fragments.
        if label in {
            "accident",
            "near_miss",
        }:
            merge_gap = 0.50
        else:
            merge_gap = 0.75

        segments = merge_segments(
            segments,
            minimum,
            merge_gap,
        )

        for start, end in segments:

            start = max(
                0.0,
                float(start),
            )

            end = min(
                duration,
                float(end),
            )

            if end > start:

                events.append(
                    [
                        start,
                        end,
                        label,
                    ]
                )

    events.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    # Guarantee same-class non-overlap.
    last_end = {}

    cleaned = []

    for start, end, label in events:

        if label in last_end:

            start = max(
                start,
                last_end[label],
            )

        if end <= start:
            continue

        cleaned.append(
            [
                start,
                end,
                label,
            ]
        )

        last_end[label] = end

    return cleaned


# ============================================================
# PART B
# ============================================================

class RiskEstimator:

    def reset(
        self,
        meta: dict,
    ) -> None:

        self.meta = meta

        self.fps = float(
            meta.get(
                "fps",
                30.0,
            )
            or 30.0
        )

        self.width = int(
            meta.get(
                "width",
                0,
            )
            or 0
        )

        self.height = int(
            meta.get(
                "height",
                0,
            )
            or 0
        )

        self.frame_idx = 0

        self.last_detection_frame = (
            -999999
        )

        self.last_score = 0.0

        self.detector = RoadDetector()
        self.tracker = SimpleTracker()

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    def _risk(
        self,
        tracks: list[Track],
        t: float,
    ) -> float:

        vehicles = [
            tr
            for tr in tracks
            if (
                tr.cls in VEHICLE_CLASSES
                and tr.missed == 0
            )
        ]

        if len(vehicles) < 2:
            return 0.0

        best = 0.0

        for i in range(
            len(vehicles)
        ):

            for j in range(
                i + 1,
                len(vehicles),
            ):

                a = vehicles[i]
                b = vehicles[j]

                distance = norm_distance(
                    a,
                    b,
                    self.width,
                    self.height,
                )

                # For anticipation we allow a slightly larger
                # area than accident detection.
                if distance > 0.11:
                    continue

                ttc = estimate_ttc(
                    a,
                    b,
                    self.width,
                    self.height,
                )

                if ttc is None:
                    continue

                rel_speed = relative_speed(
                    a,
                    b,
                    self.width,
                    self.height,
                )

                if rel_speed < 0.008:
                    continue

                overlap = iou(
                    a.box,
                    b.box,
                )

                # ------------------------------------------------
                # TTC component.
                #
                # More conservative than the old version:
                # a 2-3 sec TTC should not automatically produce
                # a large alarm.
                # ------------------------------------------------

                if ttc <= 0.35:
                    risk_ttc = 1.00

                elif ttc <= 0.60:
                    risk_ttc = 0.88

                elif ttc <= 0.90:
                    risk_ttc = 0.70

                elif ttc <= 1.20:
                    risk_ttc = 0.52

                elif ttc <= 1.60:
                    risk_ttc = 0.34

                elif ttc <= 2.20:
                    risk_ttc = 0.20

                elif ttc <= 3.00:
                    risk_ttc = 0.09

                elif ttc <= 4.00:
                    risk_ttc = 0.035

                else:
                    risk_ttc = 0.0

                # ------------------------------------------------
                # Relative speed.
                # ------------------------------------------------

                speed_factor = np.clip(
                    (
                        rel_speed
                        - 0.008
                    ) / 0.032,
                    0.0,
                    1.0,
                )

                # ------------------------------------------------
                # Close-distance factor.
                # ------------------------------------------------

                distance_factor = np.clip(
                    (
                        0.11 - distance
                    ) / 0.08,
                    0.0,
                    1.0,
                )

                # ------------------------------------------------
                # Overlap.
                # ------------------------------------------------

                overlap_factor = np.clip(
                    overlap / 0.30,
                    0.0,
                    1.0,
                )

                risk = (
                    0.62 * risk_ttc
                    + 0.20 * speed_factor
                    + 0.12 * distance_factor
                    + 0.06 * overlap_factor
                )

                # Very low TTC is meaningful only when objects
                # are actually moving relative to each other.
                if (
                    ttc > 1.5
                    and rel_speed < 0.015
                ):
                    risk *= 0.65

                best = max(
                    best,
                    float(risk),
                )

        return float(
            np.clip(
                best,
                0.0,
                1.0,
            )
        )

    # --------------------------------------------------------
    # STEP
    # --------------------------------------------------------

    def step(
        self,
        frame: np.ndarray,
        t_sec: float,
    ) -> float:

        if (
            self.frame_idx
            - self.last_detection_frame
            >= RISK_DETECT_EVERY
        ):

            detections = (
                self.detector.detect(
                    frame
                )
            )

            tracks = (
                self.tracker.update(
                    detections,
                    t_sec,
                    self.width,
                    self.height,
                )
            )

            raw = self._risk(
                tracks,
                t_sec,
            )

            # Fast reaction to a new danger.
            #
            # Slower decay prevents the risk signal from
            # flickering between detector frames.
            if raw > self.last_score:

                alpha = 0.65

            else:

                alpha = 0.25

            self.last_score = (
                alpha * raw
                + (
                    1.0 - alpha
                )
                * self.last_score
            )

            self.last_score = float(
                np.clip(
                    self.last_score,
                    0.0,
                    1.0,
                )
            )

            self.last_detection_frame = (
                self.frame_idx
            )

        self.frame_idx += 1

        return float(
            self.last_score
        )
        
