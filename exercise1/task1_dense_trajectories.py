import cv2
import numpy as np
import os

trajectory_length = 15
sampling_step = 5 
min_flow_magnitude = 0.1

def dense_sample(frame, step=5):
    """ Sample points densely across the frame avoiding low-texture areas. """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=step)
    if corners is not None:
        return corners.reshape(-1, 2)
    
    return np.array([])

def compute_trajectory_shape(traj):
    """ Compute 30D trajectory shape descriptor as displacement vectors. """
    traj = np.array(traj)
    displacements = traj[1:] - traj[:-1]  # (T-1, 2)

    return displacements.flatten()

def track_points(cap, init_points, max_len=15):
    """ Track given points using dense optical flow (Farnebäck). """
    traj_list = []
    gray_prev = None

    trajectories = [[pt] for pt in init_points]

    for i in range(max_len):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray_prev is None:
            gray_prev = gray
            continue

        flow = cv2.calcOpticalFlowFarneback(
            gray_prev, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        new_points = []
        for t, pt in zip(trajectories, init_points):
            x, y = t[-1]
            if 0 <= int(x) < flow.shape[1] and 0 <= int(y) < flow.shape[0]:
                dx, dy = flow[int(y), int(x)]
                new_pt = [x + dx, y + dy]
                t.append(new_pt)
                new_points.append(new_pt)

        init_points = new_points
        gray_prev = gray

    for t in trajectories:
        if len(t) == max_len:
            traj_list.append(t)

    return traj_list


def main(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return

    init_points = dense_sample(frame, step=sampling_step)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind
    trajectories = track_points(cap, init_points, max_len=trajectory_length)

    descriptors = []
    for traj in trajectories:
        descriptor = compute_trajectory_shape(traj)
        if np.linalg.norm(descriptor) > min_flow_magnitude:
            descriptors.append(descriptor)

    descriptors = np.array(descriptors)
    print(f"Extracted {len(descriptors)} trajectory descriptors.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, descriptors)
    print(f"Saved descriptors to {output_path}")

