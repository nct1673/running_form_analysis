"""
Side-view running pose analysis with MediaPipe Pose (legacy Solutions API).

Outputs:
  - <name>_annotated.mp4 : video with skeleton + joint angles drawn
  - <name>_angles.csv    : per-frame joint angles (knee, hip, ankle, elbow, trunk lean)
  - <name>_landmarks.csv : all 33 landmarks per frame (x, y, z, visibility) -> use as your reference pose

Usage:
  python running_pose.py path/to/running_side.mp4
  python running_pose.py path/to/running_side.mp4 --show --complexity 2
"""
import argparse
import csv
import math
import os

import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
PL = mp_pose.PoseLandmark


# ---------- geometry helpers ----------
def angle_3pt(a, b, c):
    """Angle ABC in degrees (b is the vertex). 180 = fully straight."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


def trunk_lean(shoulder, hip):
    """Trunk angle from vertical in degrees (0 = perfectly upright)."""
    dx, dy = shoulder - hip
    return math.degrees(math.atan2(abs(dx), -dy))  # image y points down


def pt(lms, idx, w, h):
    """Landmark -> pixel coords (using pixels avoids aspect-ratio distortion)."""
    lm = lms[idx]
    return np.array([lm.x * w, lm.y * h])


def near_side(lms):
    """Which side faces the camera, by summed visibility."""
    left = sum(lms[i].visibility for i in (PL.LEFT_SHOULDER, PL.LEFT_HIP, PL.LEFT_KNEE, PL.LEFT_ANKLE))
    right = sum(lms[i].visibility for i in (PL.RIGHT_SHOULDER, PL.RIGHT_HIP, PL.RIGHT_KNEE, PL.RIGHT_ANKLE))
    return "LEFT" if left >= right else "RIGHT"


def compute_angles(lms, w, h):
    P = lambda name: pt(lms, PL[name], w, h)
    side = near_side(lms)
    out = {"near_side": side}

    for s in ("LEFT", "RIGHT"):
        out[f"{s.lower()}_knee"] = angle_3pt(P(f"{s}_HIP"), P(f"{s}_KNEE"), P(f"{s}_ANKLE"))
        out[f"{s.lower()}_hip"] = angle_3pt(P(f"{s}_SHOULDER"), P(f"{s}_HIP"), P(f"{s}_KNEE"))
        out[f"{s.lower()}_ankle"] = angle_3pt(P(f"{s}_KNEE"), P(f"{s}_ANKLE"), P(f"{s}_FOOT_INDEX"))
        out[f"{s.lower()}_knee_vis"] = lms[PL[f"{s}_KNEE"]].visibility

    # Elbow + trunk from the near (camera-facing) side, which is more reliable
    out["elbow"] = angle_3pt(P(f"{side}_SHOULDER"), P(f"{side}_ELBOW"), P(f"{side}_WRIST"))
    mid_sh = (P("LEFT_SHOULDER") + P("RIGHT_SHOULDER")) / 2
    mid_hip = (P("LEFT_HIP") + P("RIGHT_HIP")) / 2
    out["trunk_lean"] = trunk_lean(mid_sh, mid_hip)
    return out


def put_label(img, text, xy, color=(255, 255, 255)):
    x, y = int(xy[0]), int(xy[1])
    cv2.putText(img, text, (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="side-view running video")
    ap.add_argument("--complexity", type=int, default=2, choices=[0, 1, 2],
                    help="0=lite, 1=full, 2=heavy (most accurate, slowest)")
    ap.add_argument("--show", action="store_true", help="show a live preview window")
    args = ap.parse_args()

    base = os.path.splitext(args.video)[0]
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(f"{base}_annotated.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    angle_rows, lm_rows = [], []
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=False,       # video mode: detect once, then track
        model_complexity=args.complexity,
        smooth_landmarks=True,         # reduces jitter between frames
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = pose.process(rgb)
            t = frame_idx / fps

            if results.pose_landmarks:
                lms = results.pose_landmarks.landmark

                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style())

                a = compute_angles(lms, w, h)
                angle_rows.append({"frame": frame_idx, "time_s": round(t, 4),
                                   **{k: (round(v, 2) if isinstance(v, float) else v) for k, v in a.items()}})

                # all 33 landmarks as the reference pose data
                row = {"frame": frame_idx, "time_s": round(t, 4)}
                for lm_enum in PL:
                    lm = lms[lm_enum]
                    n = lm_enum.name.lower()
                    row.update({f"{n}_x": lm.x, f"{n}_y": lm.y, f"{n}_z": lm.z, f"{n}_vis": lm.visibility})
                lm_rows.append(row)

                # draw angles next to joints
                s = a["near_side"]
                put_label(frame, f"L knee {a['left_knee']:.0f}", pt(lms, PL.LEFT_KNEE, w, h), (255, 200, 0))
                put_label(frame, f"R knee {a['right_knee']:.0f}", pt(lms, PL.RIGHT_KNEE, w, h), (0, 200, 255))
                put_label(frame, f"hip {a[s.lower() + '_hip']:.0f}", pt(lms, PL[f"{s}_HIP"], w, h))
                put_label(frame, f"elbow {a['elbow']:.0f}", pt(lms, PL[f"{s}_ELBOW"], w, h))
                put_label(frame, f"Trunk lean: {a['trunk_lean']:.1f} deg   near side: {s}", (10, 30))

            put_label(frame, f"frame {frame_idx}  t={t:.2f}s", (10, h - 15))
            writer.write(frame)

            if args.show:
                cv2.imshow("Running pose (q to quit, space to pause)", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord(" "):
                    cv2.waitKey(0)

            frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    for path, rows in ((f"{base}_angles.csv", angle_rows), (f"{base}_landmarks.csv", lm_rows)):
        if rows:
            with open(path, "w", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=rows[0].keys())
                wr.writeheader()
                wr.writerows(rows)

    print(f"Processed {frame_idx} frames, pose found in {len(angle_rows)}.")
    if angle_rows:
        for k in ("left_knee", "right_knee", "trunk_lean"):
            vals = [r[k] for r in angle_rows]
            print(f"  {k:<11} min {min(vals):6.1f}  max {max(vals):6.1f}  mean {np.mean(vals):6.1f}")
    print(f"Saved: {base}_annotated.mp4, {base}_angles.csv, {base}_landmarks.csv")


if __name__ == "__main__":
    main()