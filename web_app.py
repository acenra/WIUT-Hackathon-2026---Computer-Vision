from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request

from solution import (
    CLASSES,
    RISK_DETECT_EVERY,
    RoadDetector,
    SimpleTracker,
    RiskEstimator,
    analyze,
    estimate_ttc,
    relative_speed,
    VEHICLE_CLASSES,
    PERSON,
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent

DEFAULT_VIDEO = ROOT / "samples" / "My Movie_0.mp4"


# ============================================================
# RISK PAIR ANALYSIS
# ============================================================

def calculate_risk_pair(
    vehicles,
    width,
    height,
):
    """
    Находит пару автомобилей с максимальным потенциальным
    collision risk.

    Возвращает:

        {
            "risk": float,
            "a_id": int,
            "b_id": int,
            "ttc": float | None,
            "rel_speed": float,
            "distance": float,
            "risk_ttc": float,
            "speed_factor": float,
            "distance_factor": float,
            "overlap_factor": float,
        }

    или None, если подходящей пары нет.
    """

    best = None

    for i in range(len(vehicles)):

        for j in range(i + 1, len(vehicles)):

            a = vehicles[i]
            b = vehicles[j]

            try:

                # ------------------------------------------------
                # Distance
                # ------------------------------------------------

                dx = (
                    a.cx - b.cx
                ) / max(
                    1,
                    width,
                )

                dy = (
                    a.cy - b.cy
                ) / max(
                    1,
                    height,
                )

                distance = (
                    dx * dx
                    +
                    dy * dy
                ) ** 0.5


                # ------------------------------------------------
                # TTC
                # ------------------------------------------------

                ttc = estimate_ttc(
                    a,
                    b,
                    width,
                    height,
                )


                # ------------------------------------------------
                # Relative speed
                # ------------------------------------------------

                rel = relative_speed(
                    a,
                    b,
                    width,
                    height,
                )


                # ------------------------------------------------
                # Bounding box overlap
                # ------------------------------------------------

                ax1, ay1, ax2, ay2 = map(
                    float,
                    a.box,
                )

                bx1, by1, bx2, by2 = map(
                    float,
                    b.box,
                )


                ix1 = max(
                    ax1,
                    bx1,
                )

                iy1 = max(
                    ay1,
                    by1,
                )

                ix2 = min(
                    ax2,
                    bx2,
                )

                iy2 = min(
                    ay2,
                    by2,
                )


                overlap = 0.0


                if (
                    ix2 > ix1
                    and iy2 > iy1
                ):

                    intersection = (
                        ix2 - ix1
                    ) * (
                        iy2 - iy1
                    )


                    area_a = max(
                        1.0,
                        (
                            ax2 - ax1
                        ) * (
                            ay2 - ay1
                        ),
                    )


                    area_b = max(
                        1.0,
                        (
                            bx2 - bx1
                        ) * (
                            by2 - by1
                        ),
                    )


                    overlap = (
                        intersection
                        /
                        min(
                            area_a,
                            area_b,
                        )
                    )


                # ------------------------------------------------
                # TTC risk
                #
                # Exactly the same thresholds as RiskEstimator.
                # ------------------------------------------------

                if ttc is None:

                    risk_ttc = 0.0

                elif ttc <= 0.35:

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
                # Other risk factors
                # ------------------------------------------------

                speed_factor = float(
                    np.clip(
                        (
                            rel
                            - 0.008
                        )
                        /
                        0.032,
                        0.0,
                        1.0,
                    )
                )


                distance_factor = float(
                    np.clip(
                        (
                            0.11
                            - distance
                        )
                        /
                        0.08,
                        0.0,
                        1.0,
                    )
                )


                overlap_factor = float(
                    np.clip(
                        overlap
                        /
                        0.30,
                        0.0,
                        1.0,
                    )
                )


                # ------------------------------------------------
                # Official pair risk formula
                # ------------------------------------------------

                pair_risk = (

                    0.62
                    * risk_ttc

                    +

                    0.20
                    * speed_factor

                    +

                    0.12
                    * distance_factor

                    +

                    0.06
                    * overlap_factor
                )


                # Same damping as RiskEstimator
                if (
                    ttc is not None
                    and ttc > 1.5
                    and rel < 0.015
                ):

                    pair_risk *= 0.65


                pair_risk = float(
                    np.clip(
                        pair_risk,
                        0.0,
                        1.0,
                    )
                )


                candidate = {

                    "risk": pair_risk,

                    "a_id": int(
                        a.track_id
                    ),

                    "b_id": int(
                        b.track_id
                    ),

                    "ttc": (
                        None
                        if ttc is None
                        else float(ttc)
                    ),

                    "rel_speed": float(
                        rel
                    ),

                    "distance": float(
                        distance
                    ),

                    "risk_ttc": float(
                        risk_ttc
                    ),

                    "speed_factor": float(
                        speed_factor
                    ),

                    "distance_factor": float(
                        distance_factor
                    ),

                    "overlap_factor": float(
                        overlap_factor
                    ),
                }


                if (
                    best is None
                    or candidate["risk"]
                    > best["risk"]
                ):

                    best = candidate


            except Exception:

                continue


    return best


# ============================================================
# LIVE SESSION
# ============================================================

class LiveSession:

    def __init__(self):

        self.lock = threading.Lock()

        self.thread = None
        self.stop_flag = False

        self.cap = None

        self.latest_jpeg = None

        self.state = {

            "running": False,
            "finished": False,

            "time": 0.0,
            "duration": 0.0,

            "risk": 0.0,

            "fps": 0.0,

            "objects": 0,
            "vehicles": 0,
            "persons": 0,

            "ttc": None,
            "rel_speed": 0.0,
            "distance": None,

            "conflict_a": None,
            "conflict_b": None,

            "pair_risk": 0.0,

            "risk_ttc": 0.0,
            "risk_speed": 0.0,
            "risk_distance": 0.0,
            "risk_overlap": 0.0,

            "events": [],
            "event_log": [],

            "message": "Ожидание",
        }


    # ========================================================
    # START
    # ========================================================

    def start(self, video_path: str):

        self.stop()

        path = Path(video_path)

        if not path.is_absolute():
            path = ROOT / path


        if not path.exists():

            with self.lock:

                self.state["message"] = (
                    f"Видео не найдено: {path}"
                )

            return False


        cap = cv2.VideoCapture(
            str(path)
        )


        if not cap.isOpened():

            with self.lock:

                self.state["message"] = (
                    "Не удалось открыть видео"
                )

            return False


        fps = float(
            cap.get(
                cv2.CAP_PROP_FPS
            )
            or 30.0
        )


        frame_count = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
            or 0
        )


        duration = (

            frame_count / fps

            if frame_count > 0

            else 0.0
        )


        self.cap = cap
        self.stop_flag = False


        with self.lock:

            self.latest_jpeg = None

            self.state = {

                "running": True,
                "finished": False,

                "time": 0.0,
                "duration": duration,

                "risk": 0.0,

                "fps": 0.0,

                "objects": 0,
                "vehicles": 0,
                "persons": 0,

                "ttc": None,
                "rel_speed": 0.0,
                "distance": None,

                "conflict_a": None,
                "conflict_b": None,

                "pair_risk": 0.0,

                "risk_ttc": 0.0,
                "risk_speed": 0.0,
                "risk_distance": 0.0,
                "risk_overlap": 0.0,

                "events": [],
                "event_log": [],

                "message": (
                    f"Запущено: {path.name}"
                ),
            }


        self.thread = threading.Thread(
            target=self._process,
            args=(fps,),
            daemon=True,
        )

        self.thread.start()

        return True


    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        self.stop_flag = True


        if self.cap is not None:

            try:
                self.cap.release()

            except Exception:
                pass


        if (
            self.thread is not None
            and self.thread.is_alive()
        ):

            self.thread.join(
                timeout=1.0
            )


        self.thread = None
        self.cap = None


        with self.lock:

            self.state["running"] = False


    # ========================================================
    # PROCESS VIDEO
    # ========================================================

    def _process(
        self,
        fps: float,
    ):

        cap = self.cap

        if cap is None:
            return


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


        # ----------------------------------------------------
        # Detector / tracker
        # ----------------------------------------------------

        detector = RoadDetector()

        tracker = SimpleTracker()


        # ----------------------------------------------------
        # RiskEstimator
        # ----------------------------------------------------

        risk = RiskEstimator()


        risk.reset({

            "fps": fps,

            "width": width,

            "height": height,
        })


        frame_idx = 0

        last_wall_time = time.perf_counter()

        processing_fps = 0.0

        event_log = []


        # ====================================================
        # FRAME LOOP
        # ====================================================

        while not self.stop_flag:

            ok, frame = cap.read()

            if not ok:
                break


            t_sec = (
                frame_idx / fps
            )


            # =================================================
            # YOLO DETECTION
            # =================================================

            should_detect = (

                frame_idx
                -
                risk.last_detection_frame

                >=

                RISK_DETECT_EVERY
            )


            if should_detect:

                detections = (
                    detector.detect(
                        frame
                    )
                )


                tracks = (
                    tracker.update(
                        detections,
                        t_sec,
                        width,
                        height,
                    )
                )


                # ---------------------------------------------
                # Official risk calculation
                # ---------------------------------------------

                raw_risk = risk._risk(
                    tracks,
                    t_sec,
                )


                if (
                    raw_risk
                    > risk.last_score
                ):

                    alpha = 0.65

                else:

                    alpha = 0.25


                risk.last_score = float(

                    max(

                        0.0,

                        min(

                            1.0,

                            alpha
                            * raw_risk

                            +

                            (
                                1.0
                                - alpha
                            )
                            * risk.last_score,
                        )
                    )
                )


                risk.last_detection_frame = (
                    frame_idx
                )


            else:

                tracks = list(
                    tracker.tracks.values()
                )


            # =================================================
            # EVENT ANALYSIS
            # =================================================

            current_events = analyze(

                tracks,

                width,

                height,

                t_sec,
            )


            active_events = [

                label

                for label in CLASSES

                if current_events.get(
                    label,
                    False,
                )
            ]


            # =================================================
            # EVENT LOG
            # =================================================

            previous_events = set()


            if event_log:

                previous_events = {

                    item["label"]

                    for item in event_log[-10:]

                    if (
                        t_sec
                        -
                        item["time"]
                        < 1.5
                    )
                }


            for label in active_events:

                if label not in previous_events:

                    event_log.append({

                        "time": round(
                            t_sec,
                            2,
                        ),

                        "label": label,
                    })


            event_log = event_log[-50:]


            # =================================================
            # OBJECT GROUPS
            # =================================================

            visible_tracks = [

                tr

                for tr in tracks

                if tr.missed == 0
            ]


            vehicles = [

                tr

                for tr in visible_tracks

                if tr.cls
                in VEHICLE_CLASSES
            ]


            persons = [

                tr

                for tr in visible_tracks

                if tr.cls == PERSON
            ]


            # =================================================
            # BEST RISK PAIR
            # =================================================

            risk_pair = calculate_risk_pair(

                vehicles,

                width,

                height,
            )


            # =================================================
            # TTC INFORMATION
            #
            # Keep the existing general TTC information.
            # =================================================

            best_ttc = None

            best_relative_speed = 0.0

            best_distance = None


            for i in range(
                len(vehicles)
            ):

                for j in range(
                    i + 1,
                    len(vehicles),
                ):

                    a = vehicles[i]

                    b = vehicles[j]


                    dx = (

                        a.cx
                        -
                        b.cx

                    ) / max(
                        1,
                        width,
                    )


                    dy = (

                        a.cy
                        -
                        b.cy

                    ) / max(
                        1,
                        height,
                    )


                    distance = (

                        dx * dx
                        +
                        dy * dy

                    ) ** 0.5


                    if (

                        best_distance
                        is None

                        or

                        distance
                        < best_distance
                    ):

                        best_distance = (
                            distance
                        )


                    ttc = estimate_ttc(

                        a,
                        b,
                        width,
                        height,
                    )


                    rel = relative_speed(

                        a,
                        b,
                        width,
                        height,
                    )


                    if ttc is not None:

                        if (

                            best_ttc
                            is None

                            or

                            ttc
                            < best_ttc
                        ):

                            best_ttc = ttc

                            best_relative_speed = (
                                rel
                            )


            # =================================================
            # DANGER IDS
            # =================================================

            danger_ids = set()


            if (
                risk_pair is not None
                and risk_pair["risk"] >= 0.60
            ):

                danger_ids = {

                    risk_pair["a_id"],

                    risk_pair["b_id"],
                }


            # =================================================
            # DRAW
            # =================================================

            output = frame.copy()


            # -------------------------------------------------
            # TRACKS
            # -------------------------------------------------

            for tr in visible_tracks:

                x1, y1, x2, y2 = map(
                    int,
                    tr.box,
                )


                names = {

                    0: "person",

                    1: "bicycle",

                    2: "car",

                    3: "motorcycle",

                    5: "bus",

                    7: "truck",
                }


                class_name = names.get(

                    tr.cls,

                    str(tr.cls),
                )


                # =================================================
                # BOX COLOR
                # =================================================

                if tr.track_id in danger_ids:

                    # 🔴 Potential collision
                    box_color = (
                        0,
                        0,
                        255,
                    )

                    box_thickness = 4

                else:

                    # 🟢 Normal
                    box_color = (
                        80,
                        220,
                        80,
                    )

                    box_thickness = 2


                # -------------------------------------------------
                # Box
                # -------------------------------------------------

                cv2.rectangle(

                    output,

                    (x1, y1),

                    (x2, y2),

                    box_color,

                    box_thickness,
                )


                # -------------------------------------------------
                # Label
                # -------------------------------------------------

                cv2.putText(

                    output,

                    (
                        f"#{tr.track_id} "
                        f"{class_name} "
                        f"{tr.conf:.2f}"
                    ),

                    (
                        x1,

                        max(
                            18,
                            y1 - 6,
                        ),
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.48,

                    box_color,

                    2 if tr.track_id in danger_ids else 1,

                    cv2.LINE_AA,
                )


                # -------------------------------------------------
                # Potential collision label
                # -------------------------------------------------

                if tr.track_id in danger_ids:

                    cv2.putText(

                        output,

                        "POTENTIAL COLLISION",

                        (
                            x1,

                            max(
                                38,
                                y1 - 25,
                            ),
                        ),

                        cv2.FONT_HERSHEY_SIMPLEX,

                        0.52,

                        (
                            0,
                            0,
                            255,
                        ),

                        2,

                        cv2.LINE_AA,
                    )


                # -------------------------------------------------
                # Trajectory
                # -------------------------------------------------

                points = [

                    (
                        int(item[1]),
                        int(item[2]),
                    )

                    for item
                    in tr.history[-15:]
                ]


                for k in range(
                    1,
                    len(points),
                ):

                    cv2.line(

                        output,

                        points[k - 1],

                        points[k],

                        (
                            255,
                            180,
                            40,
                        ),

                        2,
                    )


            # =================================================
            # HUD
            # =================================================

            risk_value = (
                risk.last_score
            )


            hud = [

                f"TIME      {t_sec:6.2f}s",

                f"RISK      {risk_value:.3f}",

                (
                    "TTC       "

                    +

                    (
                        "--"

                        if best_ttc is None

                        else f"{best_ttc:.2f}s"
                    )
                ),

                (
                    "REL SPEED "

                    f"{best_relative_speed:.4f}"
                ),

                (
                    "DIST      "

                    +

                    (
                        "--"

                        if best_distance is None

                        else f"{best_distance:.4f}"
                    )
                ),

                (
                    f"OBJECTS   "
                    f"{len(visible_tracks)}"
                ),

                (
                    f"VEHICLES  "
                    f"{len(vehicles)}"
                ),

                (
                    f"PERSONS   "
                    f"{len(persons)}"
                ),
            ]


            y = 30


            for line in hud:

                cv2.putText(

                    output,

                    line,

                    (12, y),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.58,

                    (
                        255,
                        255,
                        255,
                    ),

                    2,

                    cv2.LINE_AA,
                )

                y += 25


            # =================================================
            # POTENTIAL COLLISION HUD
            # =================================================

            if (
                risk_pair is not None
                and risk_pair["risk"] >= 0.60
            ):

                conflict_text = (

                    "CONFLICT  "

                    f"#{risk_pair['a_id']}"

                    " <-> "

                    f"#{risk_pair['b_id']}"

                    "  "

                    f"RISK {risk_pair['risk']:.3f}"
                )


                cv2.putText(

                    output,

                    conflict_text,

                    (
                        12,
                        y + 8,
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.60,

                    (
                        0,
                        0,
                        255,
                    ),

                    2,

                    cv2.LINE_AA,
                )


                y += 34


            # =================================================
            # RISK COMPONENTS
            # =================================================

            if (
                risk_pair is not None
                and risk_pair["risk"] >= 0.60
            ):

                component_text = (

                    f"TTC-RISK "
                    f"{risk_pair['risk_ttc']:.2f}   "

                    f"SPEED "
                    f"{risk_pair['speed_factor']:.2f}   "

                    f"DIST "
                    f"{risk_pair['distance_factor']:.2f}   "

                    f"OVERLAP "
                    f"{risk_pair['overlap_factor']:.2f}"
                )


                cv2.putText(

                    output,

                    component_text,

                    (
                        12,
                        y + 8,
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.46,

                    (
                        0,
                        180,
                        255,
                    ),

                    1,

                    cv2.LINE_AA,
                )


                y += 28


            # =================================================
            # EVENTS
            # =================================================

            if active_events:

                event_text = (

                    "EVENT: "

                    +

                    ", ".join(
                        active_events
                    )
                )


                cv2.putText(

                    output,

                    event_text,

                    (
                        12,
                        y + 8,
                    ),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.60,

                    (
                        60,
                        80,
                        255,
                    ),

                    2,

                    cv2.LINE_AA,
                )


            # =================================================
            # PROCESSING FPS
            # =================================================

            now = time.perf_counter()


            dt = (

                now
                -
                last_wall_time
            )


            last_wall_time = now


            if dt > 0:

                instant_fps = (
                    1.0 / dt
                )


                if processing_fps == 0:

                    processing_fps = (
                        instant_fps
                    )

                else:

                    processing_fps = (

                        0.9
                        * processing_fps

                        +

                        0.1
                        * instant_fps
                    )


            # =================================================
            # JPEG
            # =================================================

            ok_jpg, encoded = (

                cv2.imencode(

                    ".jpg",

                    output,

                    [

                        int(
                            cv2.IMWRITE_JPEG_QUALITY
                        ),

                        78,
                    ],
                )
            )


            if ok_jpg:

                with self.lock:

                    self.latest_jpeg = (
                        encoded.tobytes()
                    )


            # =================================================
            # STATE
            # =================================================

            with self.lock:

                self.state.update({

                    "running": True,

                    "finished": False,

                    "time": round(
                        t_sec,
                        3,
                    ),

                    "risk": round(
                        float(
                            risk.last_score
                        ),
                        4,
                    ),

                    "fps": round(
                        processing_fps,
                        1,
                    ),

                    "objects": len(
                        visible_tracks
                    ),

                    "vehicles": len(
                        vehicles
                    ),

                    "persons": len(
                        persons
                    ),

                    "ttc": (

                        None

                        if best_ttc is None

                        else round(
                            best_ttc,
                            3,
                        )
                    ),

                    "rel_speed": round(
                        best_relative_speed,
                        5,
                    ),

                    "distance": (

                        None

                        if best_distance is None

                        else round(
                            best_distance,
                            5,
                        )
                    ),

                    "conflict_a": (

                        None

                        if risk_pair is None

                        else risk_pair["a_id"]
                    ),

                    "conflict_b": (

                        None

                        if risk_pair is None

                        else risk_pair["b_id"]
                    ),

                    "pair_risk": (

                        0.0

                        if risk_pair is None

                        else round(
                            risk_pair["risk"],
                            4,
                        )
                    ),

                    "risk_ttc": (

                        0.0

                        if risk_pair is None

                        else round(
                            risk_pair["risk_ttc"],
                            4,
                        )
                    ),

                    "risk_speed": (

                        0.0

                        if risk_pair is None

                        else round(
                            risk_pair["speed_factor"],
                            4,
                        )
                    ),

                    "risk_distance": (

                        0.0

                        if risk_pair is None

                        else round(
                            risk_pair["distance_factor"],
                            4,
                        )
                    ),

                    "risk_overlap": (

                        0.0

                        if risk_pair is None

                        else round(
                            risk_pair["overlap_factor"],
                            4,
                        )
                    ),

                    "events": active_events,

                    "event_log": event_log,

                    "message": (
                        "Анализ видео"
                    ),
                })


            frame_idx += 1


            # Не sleep'им на 30 FPS.
            # CPU YOLO и так является bottleneck.

            time.sleep(
                0.001
            )


        # =====================================================
        # FINISHED
        # =====================================================

        cap.release()


        with self.lock:

            self.state["running"] = False

            self.state["finished"] = True

            self.state["message"] = (
                "Видео завершено"
            )


# ============================================================
# GLOBAL SESSION
# ============================================================

session = LiveSession()


# ============================================================
# ROUTES
# ============================================================

@app.get("/")
def index():

    default_video = str(

        DEFAULT_VIDEO.relative_to(
            ROOT
        )

    ).replace(
        "\\",
        "/",
    )


    return render_template(

        "index.html",

        default_video=default_video,
    )


# ============================================================
# START
# ============================================================

@app.post("/start")
def start():

    data = (

        request
        .get_json(
            silent=True
        )

        or {}
    )


    video = data.get(
        "video"
    )


    if not video:

        video = str(
            DEFAULT_VIDEO
        )


    success = session.start(
        video
    )


    with session.lock:

        state = dict(
            session.state
        )


    return jsonify({

        "ok": success,

        "state": state,
    })


# ============================================================
# STOP
# ============================================================

@app.post("/stop")
def stop():

    session.stop()


    return jsonify({

        "ok": True
    })


# ============================================================
# STATE
# ============================================================

@app.get("/state")
def state():

    with session.lock:

        return jsonify(

            dict(
                session.state
            )
        )


# ============================================================
# MJPEG VIDEO
# ============================================================

@app.get("/video_feed")
def video_feed():

    def generate():

        while True:

            with session.lock:

                frame = (
                    session.latest_jpeg
                )

                running = (
                    session.state[
                        "running"
                    ]
                )

                finished = (
                    session.state[
                        "finished"
                    ]
                )


            if frame:

                yield (

                    b"--frame\r\n"

                    b"Content-Type: "
                    b"image/jpeg\r\n\r\n"

                    +

                    frame

                    +

                    b"\r\n"
                )


            if (
                not running
                and finished
            ):

                break


            time.sleep(
                0.03
            )


    return Response(

        generate(),

        mimetype=(

            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "=========================================="
    )

    print(
        " Traffic Vision Live Monitor"
    )

    print(
        "=========================================="
    )

    print()

    print(
        "Open:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()


    app.run(

        host="127.0.0.1",

        port=5000,

        threaded=True,

        debug=False,
    )