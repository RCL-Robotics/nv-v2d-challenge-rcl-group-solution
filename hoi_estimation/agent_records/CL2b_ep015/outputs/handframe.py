import numpy as np
def hand_frames(kp):
    """Right-hand frame (MHR70: wrist 41, index base 28, middle base 32, pinky base 40). Returns (F,4,4) world."""
    o = kp[:, 41]; a = kp[:, 32] - o; a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = kp[:, 28] - kp[:, 40]; b -= (b * a).sum(1, keepdims=True) * a; b /= np.linalg.norm(b, axis=1, keepdims=True)
    c = np.cross(a, b)
    T = np.tile(np.eye(4), (len(kp), 1, 1)); T[:, :3, 0] = a; T[:, :3, 1] = b; T[:, :3, 2] = c; T[:, :3, 3] = o
    return T
