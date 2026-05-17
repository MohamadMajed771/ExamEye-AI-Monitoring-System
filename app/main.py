import cv2
import time
import sqlite3
import os
from datetime import datetime
from ultralytics import YOLO
from collections import deque


HEAD_TURN_RATIO_THRESHOLD = 0.2
HEAD_DOWN_RATIO_THRESHOLD = 0.22
SUSPICIOUS_TIME_THRESHOLD = 2
EVENT_COOLDOWN = 12

DB_PATH = "database/events.db"
SCREENSHOT_DIR = "screenshots"
CLIPS_DIR = "clips"

FRAME_BUFFER_SIZE = 300
FPS = 30
frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)

head_turn_timers = {}
head_down_timers = {}
last_logged_events = {}

tracker_to_student = {}
next_student_id = 1


def setup_database():
    os.makedirs("database", exist_ok=True)
    os.makedirs(SCREENSHOT_DIR, exist_ok=True)
    os.makedirs(CLIPS_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER,
            event_type TEXT,
            timestamp TEXT,
            confidence REAL,
            screenshot_path TEXT
        )
    """)

    conn.commit()
    conn.close()


def get_stable_student_id(tracker_id):
    global next_student_id

    if tracker_id in tracker_to_student:
        return tracker_to_student[tracker_id]

    student_id = next_student_id
    tracker_to_student[tracker_id] = student_id
    next_student_id += 1

    return student_id


def save_event(student_id, event_type, confidence, frame):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    screenshot_path = f"{SCREENSHOT_DIR}/student_{student_id}_{event_type}_{timestamp}.jpg"

    cv2.imwrite(screenshot_path, frame)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO events (
            student_id,
            event_type,
            timestamp,
            confidence,
            screenshot_path
        )
        VALUES (?, ?, ?, ?, ?)
    """, (
        int(student_id),
        event_type,
        timestamp,
        float(confidence),
        screenshot_path
    ))

    conn.commit()
    conn.close()

    print(f"Saved event: Student {student_id} - {event_type}")


def save_video_clip(student_id):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    clip_path = f"{CLIPS_DIR}/student_{student_id}_{timestamp}.mp4"

    if len(frame_buffer) == 0:
        return

    height, width, _ = frame_buffer[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(clip_path, fourcc, FPS, (width, height))

    for buffered_frame in frame_buffer:
        out.write(buffered_frame)

    out.release()
    print(f"Saved clip: {clip_path}")


def main():
    setup_database()

    model = YOLO("models/yolov8n-pose.pt")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening webcam")
        return

    print("ExamEye started. Press Q to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        annotated_frame = results[0].plot()
        frame_buffer.append(annotated_frame.copy())

        if results[0].keypoints is not None and results[0].boxes.id is not None:
            keypoints = results[0].keypoints.xy.cpu().numpy()
            tracker_ids = results[0].boxes.id.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()

            for person, tracker_id, box in zip(keypoints, tracker_ids, boxes):
                tracker_id = int(tracker_id)
                student_id = get_stable_student_id(tracker_id)

                nose = person[0]
                left_eye = person[1]
                right_eye = person[2]
                left_ear = person[3]
                right_ear = person[4]

                nose_x, nose_y = nose
                left_eye_x, left_eye_y = left_eye
                right_eye_x, right_eye_y = right_eye
                left_ear_x, left_ear_y = left_ear
                right_ear_x, right_ear_y = right_ear

                face_width = abs(left_ear_x - right_ear_x)

                if face_width > 20:
                    x1, y1, x2, y2 = map(int, box)

                    face_center_x = (left_ear_x + right_ear_x) / 2
                    head_turn_ratio = abs(nose_x - face_center_x) / face_width

                    eye_center_y = (left_eye_y + right_eye_y) / 2
                    head_down_ratio = (nose_y - eye_center_y) / face_width

                    # Head turn detection
                    if head_turn_ratio > HEAD_TURN_RATIO_THRESHOLD:
                        if student_id not in head_turn_timers:
                            head_turn_timers[student_id] = time.time()

                        elapsed_time = time.time() - head_turn_timers[student_id]

                        if elapsed_time > SUSPICIOUS_TIME_THRESHOLD:
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                            cv2.putText(
                                annotated_frame,
                                f"Student {student_id} HEAD TURN",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2
                            )

                            now = time.time()
                            last_time = last_logged_events.get(f"head_turn_{student_id}", 0)

                            if now - last_time > EVENT_COOLDOWN:
                                save_event(
                                    student_id=student_id,
                                    event_type="head_turn",
                                    confidence=round(float(head_turn_ratio), 2),
                                    frame=annotated_frame
                                )

                                save_video_clip(student_id)
                                last_logged_events[f"head_turn_{student_id}"] = now
                    else:
                        head_turn_timers.pop(student_id, None)

                    # Head down detection: possible hidden phone reading
                    if head_down_ratio > HEAD_DOWN_RATIO_THRESHOLD:
                        if student_id not in head_down_timers:
                            head_down_timers[student_id] = time.time()

                        elapsed_time = time.time() - head_down_timers[student_id]

                        if elapsed_time > SUSPICIOUS_TIME_THRESHOLD:
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                            cv2.putText(
                                annotated_frame,
                                f"Student {student_id} HEAD DOWN",
                                (x1, y2 + 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.8,
                                (0, 0, 255),
                                2
                            )

                            now = time.time()
                            last_time = last_logged_events.get(f"head_down_{student_id}", 0)

                            if now - last_time > EVENT_COOLDOWN:
                                save_event(
                                    student_id=student_id,
                                    event_type="head_down_possible_phone",
                                    confidence=round(float(head_down_ratio), 2),
                                    frame=annotated_frame
                                )

                                save_video_clip(student_id)
                                last_logged_events[f"head_down_{student_id}"] = now
                    else:
                        head_down_timers.pop(student_id, None)

        cv2.imshow("ExamEye - Event Logging", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()