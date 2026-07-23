import os
import sys
import time
import threading
import cv2
import yaml
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from realtime_foundation.camera.realsense_reader import RealSenseReader
from realtime_foundation.detection.yolo_segmenter import YoloSegmenter

CONFIG = os.path.join(PROJECT_ROOT, "realtime_foundation/config.yaml")


def resolve_project_path(path):
    if path is None or os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


class LatestFrameCamera:
    def __init__(self, camera):
        self.camera = camera
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._capture_loop, name="realsense-capture", daemon=True)
        self.latest_frame = None
        self.latest_frame_id = 0
        self.captured_frames = 0

    def start(self):
        self.camera.start()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.camera.stop()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def get_latest(self):
        with self.lock:
            return self.latest_frame_id, self.latest_frame

    def _capture_loop(self):
        while not self.stop_event.is_set():
            frame = self.camera.get_frame()
            if frame is None:
                continue

            with self.lock:
                self.latest_frame = frame
                self.latest_frame_id += 1
                self.captured_frames += 1

with open(CONFIG, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

camera_cfg = cfg["camera"]
yolo_cfg = cfg["yolo"]

camera = RealSenseReader(
    width=int(camera_cfg.get("width", 640)),
    height=int(camera_cfg.get("height", 480)),
    fps=int(camera_cfg.get("fps", 30)),
    serial=camera_cfg.get("serial"),
    depth_min=float(camera_cfg.get("depth_min", 0.001)),
    depth_max=float(camera_cfg.get("depth_max", 3.0)),
    align_to_color=bool(camera_cfg.get("align_to_color", True)),
    reset_before_start=bool(camera_cfg.get("reset_before_start", True)),
    verbose=True,
)

detector = YoloSegmenter(
    weights=resolve_project_path(yolo_cfg["weights"]),
    target_class=yolo_cfg.get("target_class"),
    target_class_id=yolo_cfg.get("target_class_id"),
    conf=float(yolo_cfg.get("conf", 0.35)),
    imgsz=yolo_cfg.get("imgsz", 640),
    device=yolo_cfg.get("device"),
    half=bool(yolo_cfg.get("half", True)),
    min_mask_area=int(yolo_cfg.get("min_mask_area", 100)),
    morph_kernel=int(yolo_cfg.get("morph_kernel", 5)),
)

frame_index = 0
processed_frames = 0
last_processed_frame_id = 0
last_infer_time = time.time()
last_stats_time = time.time()
last_stats_frame_id = 0

frame_buffer = LatestFrameCamera(camera)
frame_buffer.start()
try:
    while True:
        frame_id, frame = frame_buffer.get_latest()
        if frame is None or frame_id == last_processed_frame_id:
            time.sleep(0.001)
            continue

        last_processed_frame_id = frame_id

        color_rgb, depth, K = frame
        frame_index += 1
        processed_frames += 1

        result = detector.predict_mask(color_rgb)

        vis_rgb = color_rgb.copy()

        if result is not None:
            mask_bool = result.mask.astype(bool)

            overlay = vis_rgb.copy()
            overlay[mask_bool] = np.array([255, 0, 0], dtype=np.uint8)
            vis_rgb = cv2.addWeighted(vis_rgb, 0.65, overlay, 0.35, 0)

            x1, y1, x2, y2 = result.box_xyxy.astype(int)
            cv2.rectangle(vis_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = (
                f"{result.class_name} "
                f"conf={result.confidence:.2f} "
                f"area={result.area}"
            )
            cv2.putText(
                vis_rgb,
                text,
                (max(x1, 5), max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            print(
                f"[YOLO] frame={frame_index} "
                f"class={result.class_name} "
                f"conf={result.confidence:.3f} "
                f"area={result.area} "
                f"box={result.box_xyxy.astype(int).tolist()}"
            )
        else:
            cv2.putText(
                vis_rgb,
                "no detection",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )
            print(f"[YOLO] frame={frame_index} no detection")

        now = time.time()
        fps = 1.0 / max(now - last_infer_time, 1e-6)
        last_infer_time = now
        cv2.putText(
            vis_rgb,
            f"FPS {fps:.1f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("YOLO RealSense Mask Test", vis_rgb[..., ::-1])
        key = cv2.waitKey(1)
        if key in (27, ord("q")):
            break

        if now - last_stats_time >= 1.0:
            captured_delta = frame_id - last_stats_frame_id
            processed_delta = processed_frames
            dropped_delta = max(captured_delta - processed_delta, 0)
            print(
                f"[Stats] camera_fps={captured_delta / (now - last_stats_time):.1f} "
                f"yolo_fps={processed_delta / (now - last_stats_time):.1f} "
                f"dropped_latest_frames={dropped_delta}"
            )
            processed_frames = 0
            last_stats_frame_id = frame_id
            last_stats_time = now
finally:
    frame_buffer.stop()
    cv2.destroyAllWindows()