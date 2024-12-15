import cv2

import trimesh

import numpy as np

from typing import Tuple


class Model:
    def __init__(self, filename: str, scale: float) -> None:
        model = trimesh.load(filename)
        model = model.apply_scale(scale)

        self.vertices = model.vertices

        self.faces = model.faces
        self.normals = model.face_normals

    def set_intrinsic_matrix(self, f: float) -> None:
        self.K = np.linalg.inv(np.array([[ f,  0, -1],
                                         [ 0,  f, -1],
                                         [ 0,  0,  1]]))

    def project_vertices(self, pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vertices = self.vertices @ pose[:3, :3].T + pose[:3, 3]

        light_rays  = np.mean(vertices[self.faces], axis=1)
        light_rays /= np.linalg.norm(light_rays, axis=1, keepdims=True) + 1e-8

        vertices = vertices @ self.K.T
        vertices[:, 0] = vertices[:, 0] / vertices[:, 2]
        vertices[:, 1] = vertices[:, 1] / vertices[:, 2]
        vertices = np.round(vertices[:, :2]).astype(int)

        # Calculate diffuse intensities
        intensities = np.einsum("ij,ij->i", self.normals, light_rays)
        intensities = np.nan_to_num(intensities, nan=0.0)
        intensities = np.maximum(64, 255 * intensities).astype(int)

        return vertices, intensities


def draw_axes(frame: np.ndarray, model: Model, pose: np.ndarray) -> None:
    for i in range(3):
        color = [(0, 0, 180), (0, 180, 0), (180, 0, 0)][i]

        pi = model.K @ pose[:3, 3]
        pf = model.K @ pose[:3, i] * 0.5 + pi

        pi = np.round(pi / pi[2]).astype(int)[:2]
        pf = np.round(pf / pf[2]).astype(int)[:2]

        cv2.arrowedLine(frame, pi, pf, color, 2)


def draw_box(frame: np.ndarray, model: Model, pose: np.ndarray) -> None:
    p1 = model.K @ (pose[:3, 3] - 0.5 * pose[:3, 1] - 0.5 * pose[:3, 0])
    p2 = model.K @ (pose[:3, 3] + 0.5 * pose[:3, 1] - 0.5 * pose[:3, 0])
    p3 = model.K @ (pose[:3, 3] + 0.5 * pose[:3, 1] + 0.5 * pose[:3, 0])
    p4 = model.K @ (pose[:3, 3] - 0.5 * pose[:3, 1] + 0.5 * pose[:3, 0])

    p5 = model.K @ pose[:3, 2] + p1
    p6 = model.K @ pose[:3, 2] + p2
    p7 = model.K @ pose[:3, 2] + p3
    p8 = model.K @ pose[:3, 2] + p4

    p = [p1, p2, p3, p4, p5, p6, p7, p8]
    p = [np.round(pi / pi[2]).astype(int)[:2] for pi in p]
    p = np.array(p)

    cv2.drawContours(frame, [p[[0, 1, 2, 3]]], -1, (255, 0, 255), 1)
    cv2.drawContours(frame, [p[[4, 5, 6, 7]]], -1, (255, 0, 255), 1)
    cv2.drawContours(frame, [p[[0, 1, 5, 4]]], -1, (255, 0, 255), 1)
    cv2.drawContours(frame, [p[[2, 3, 7, 6]]], -1, (255, 0, 255), 1)


def render_model(frame: np.ndarray, model: Model, pose: np.ndarray) -> None:
    vertices, intensities = model.project_vertices(pose)

    for intensity, faces in zip(intensities, model.faces):
        color = (int(intensity), int(intensity), int(intensity))
        cv2.fillPoly(frame, [vertices[faces]], color)
