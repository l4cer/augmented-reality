import cv2

import numpy as np


def compute_pose(points_2D: np.ndarray, K: np.ndarray) -> np.ndarray:
    points_3D = np.array([[-0.5,  0.5, 0.0],
                          [ 0.5,  0.5, 0.0],
                          [ 0.5, -0.5, 0.0],
                          [-0.5, -0.5, 0.0]])

    _, rvec, tvec = cv2.solvePnP(
        points_3D, points_2D, K, np.zeros((5, 1)), flags=cv2.SOLVEPNP_IPPE_SQUARE)

    pose = np.eye(4)

    pose[:3, :3] = cv2.Rodrigues(rvec)[0]
    pose[:3, 3:] = tvec

    u = np.copy(pose[:3, 0])
    v = np.copy(pose[:3, 1])

    pose[:3, 0] = -v
    pose[:3, 1] =  u

    return pose
