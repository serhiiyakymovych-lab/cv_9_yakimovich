# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 18:04:17 2025

@author: k4kuz
"""

import cv2
import numpy as np

def choose_tracker(tracker_type="KCF"):
    if tracker_type.upper() == "KCF":
        return cv2.TrackerKCF_create()
    elif tracker_type.upper() == "CSRT":
        return cv2.TrackerCSRT_create()
    elif tracker_type.upper() == "MEDIANFLOW":
        return cv2.legacy.TrackerMedianFlow_create()
    else:
        raise ValueError("Wrong tracker type")


def auto_select_roi(frame, max_corners=200, quality_level=0.01, min_distance=10, block_size=3):
    feature_params = dict(
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size
    )

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

    if p is None or len(p) <= 5:
        raise RuntimeError("To low points for tracking")

    points = np.int32(p).reshape(-1, 2)

    clusters = []
    used = np.zeros(len(points), dtype=bool)
    threshold_dist = 50

    for i, pt in enumerate(points):
        if used[i]:
            continue
        cluster = [pt]
        used[i] = True
        for j, other in enumerate(points):
            if not used[j] and np.linalg.norm(pt - other) < threshold_dist:
                cluster.append(other)
                used[j] = True
        clusters.append(cluster)

    largest_cluster = max(clusters, key=len)
    largest_cluster = np.array(largest_cluster)

    if len(largest_cluster) <= 5:
        raise RuntimeError("Low points on biggest cluster")

    x_min = np.min(largest_cluster[:, 0])
    y_min = np.min(largest_cluster[:, 1])
    x_max = np.max(largest_cluster[:, 0])
    y_max = np.max(largest_cluster[:, 1])
    w, h = x_max - x_min, y_max - y_min

    if w * h <= 800:
        raise RuntimeError("To small size")

    bbox = (x_min, y_min, w, h)
    cv2.rectangle(frame, (x_min, y_min), (x_min + w, y_min + h), (0, 255, 0), 2)
    cv2.imshow("Auto ROI", frame)
    cv2.waitKey(500)
    cv2.destroyWindow("Auto ROI")
    return bbox


def run_tracker(video_path, tracker_type="KCF", save_every_n_frames=15):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("No video read")

    bbox = auto_select_roi(frame)
    tracker = choose_tracker(tracker_type)
    tracker.init(frame, bbox)

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        success, bbox = tracker.update(frame)
        frame_count += 1

        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Tracking ({tracker_type})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Lost", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Tracker", frame)

        if frame_count % save_every_n_frames == 0:
            filename = f"frame_{tracker_type}_{frame_count}.png"
            cv2.imwrite(filename, frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "data/track.mp4" 
    run_tracker(video_path, tracker_type="csrt", save_every_n_frames=15)