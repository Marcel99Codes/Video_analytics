import cv2
import numpy as np
import os

volume_size = 32
volume_len = 15
tube_div = (2, 2, 3)

def extract_volume(frames, center):
    """ Extract a 32x32x15 volume around trajectory center. """
    x, y = int(center[0]), int(center[1])
    half = volume_size // 2
    vol = []

    for frame in frames:
        h, w = frame.shape
        xmin, xmax = max(0, x - half), min(w, x + half)
        ymin, ymax = max(0, y - half), min(h, y + half)
        patch = np.zeros((volume_size, volume_size), dtype=np.uint8)
        cropped = frame[ymin:ymax, xmin:xmax]
        patch[0:cropped.shape[0], 0:cropped.shape[1]] = cropped
        vol.append(patch)

    # shape: (H, W, T)
    return np.stack(vol, axis=-1) 


def compute_hog(volume, bins=8):
    """ Compute histogram of gradients for each tube, return 96D vector. """
    h, w, t = volume.shape
    gx = cv2.Sobel(volume, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(volume, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle = np.mod(angle, 360.0)

    descriptor = []
    dx, dy, dt = h // 2, w // 2, t // 3
    for i in range(2):
        for j in range(2):
            for k in range(3):
                sub_mag = mag[i*dx:(i+1)*dx, j*dy:(j+1)*dy, k*dt:(k+1)*dt]
                sub_angle = angle[i*dx:(i+1)*dx, j*dy:(j+1)*dy, k*dt:(k+1)*dt]
                hist = np.histogram(sub_angle, bins=bins, range=(0, 360), weights=sub_mag)[0]
                descriptor.extend(hist)
    # 2*2*3*8 = 96D
    return np.array(descriptor)


def compute_hof(prev_frames, next_frames, bins=9):
    """ Compute histogram of optical flow for each tube, return 108D vector. """
    h, w = prev_frames[0].shape
    t = len(prev_frames)

    hof_vol = np.zeros((h, w, t))

    flows = []
    for i in range(t):
        flow = cv2.calcOpticalFlowFarneback(
            prev_frames[i], next_frames[i], None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        flows.append(flow)

    descriptor = []
    dx, dy, dt = h // 2, w // 2, t // 3
    for i in range(2):
        for j in range(2):
            for k in range(3):
                sub_flows = [f[i*dx:(i+1)*dx, j*dy:(j+1)*dy] for f in flows[k*dt:(k+1)*dt]]
                mags = []
                angles = []
                for f in sub_flows:
                    mag, ang = cv2.cartToPolar(f[..., 0], f[..., 1], angleInDegrees=True)
                    mags.append(mag)
                    angles.append(ang)
                mags = np.stack(mags).flatten()
                angles = np.stack(angles).flatten()
                hist = np.histogram(angles, bins=bins, range=(0, 360), weights=mags)[0]
                descriptor.extend(hist)

    # 2×2×3×9 = 108D
    return np.array(descriptor)

def compute_mbh(flow_vol, bins=8):
    """ Compute MBHx and MBHy as HoG of x and y flow channels. """
    h, w = flow_vol[0].shape[:2]
    t = len(flow_vol)

    mbhx = np.zeros((h, w, t))
    mbhy = np.zeros((h, w, t))

    for i in range(t):
        fx = cv2.Sobel(flow_vol[i][..., 0], cv2.CV_32F, 1, 0, ksize=1)
        fy = cv2.Sobel(flow_vol[i][..., 1], cv2.CV_32F, 0, 1, ksize=1)
        mbhx[..., i] = fx
        mbhy[..., i] = fy

    def compute_sub(mb):
        gx = mb
        gy = np.zeros_like(mb)
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        angle = np.mod(angle, 360.0)
        descriptor = []
        dx, dy, dt = h // 2, w // 2, t // 3
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    mag_block = mag[i*dx:(i+1)*dx, j*dy:(j+1)*dy, k*dt:(k+1)*dt]
                    angle_block = angle[i*dx:(i+1)*dx, j*dy:(j+1)*dy, k*dt:(k+1)*dt]
                    hist = np.histogram(angle_block, bins=bins, range=(0, 360), weights=mag_block)[0]
                    descriptor.extend(hist)
        return np.array(descriptor)

    # 96D each
    return compute_sub(mbhx), compute_sub(mbhy)


def main(video_path, trajectory_file, output_file):
    descriptors = np.load(trajectory_file)
    cap = cv2.VideoCapture(video_path)

    all_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        all_frames.append(gray)

    all_descriptors = []

    for i in range(0, len(descriptors)):
        center = np.array([volume_size // 2, volume_size // 2])

        start = i
        end = i + volume_len
        if end >= len(all_frames):
            break

        volume = extract_volume(all_frames[start:end], center)

        prevs = all_frames[start:end-1]
        nexts = all_frames[start+1:end]

        #Parameters from lecture/paper
        flows = [cv2.calcOpticalFlowFarneback(p, n, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                 for p, n in zip(prevs, nexts)]

        hog = compute_hog(volume)
        hof = compute_hof(prevs, nexts)
        mbhx, mbhy = compute_mbh(flows)

        combined = np.concatenate([descriptors[i], hog, hof, mbhx, mbhy])
        all_descriptors.append(combined)

    all_descriptors = np.array(all_descriptors)
    np.save(output_file, all_descriptors)
