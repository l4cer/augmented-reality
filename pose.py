import numpy as np

from typing import List


def quadratic_solver(a: float, b: float, c: float) -> List[float]:
    if np.isclose(a, 0):
        if np.isclose(b, 0):
            return []

        return [-c / b]

    b, c = b / a, c / a
    delta = b**2 - 4 * c

    if np.isclose(delta, 0):
        return [-b / 2]

    if delta < 0:
        return []

    return [(-b + np.sqrt(delta)) / 2, (-b - np.sqrt(delta)) / 2]


def compute_score(points: np.ndarray) -> float:
    sides = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)

    score = np.sum(np.abs(sides - 1))
    score += np.abs(np.linalg.norm(points[0] - points[2]) - np.sqrt(2))
    score += np.abs(np.linalg.norm(points[1] - points[3]) - np.sqrt(2))

    vectors = points[1:] - points[0]
    normal = np.cross(vectors[0], vectors[1])

    score += np.abs(np.dot(vectors[2], normal / np.linalg.norm(normal)))

    return score


def pnp_solver(contour: np.ndarray, k: float) -> np.ndarray:
    proj = np.array([[k * x - 1, k * y - 1, 1] for x, y in contour])
    proj = proj / np.linalg.norm(proj, axis=1, keepdims=True)

    b = np.eye(4)
    for i in range(3):
        for j in range(i + 1, 4):
            dot = np.dot(proj[i], proj[j])
            b[i, j] = 2 * dot
            b[j, i] = 2 * dot

    best_score = 0
    solution = None

    A  = b[1,3] * b[0,2]**2 - 4 * b[1,3]
    A += b[0,1] * (2 * b[0,3] - b[2,3] * b[0,2])
    A += b[2,1] * (2 * b[2,3] - b[0,3] * b[0,2])

    B  = 4 * b[1,3] * b[0,2]
    B -= 2 * b[0,1] * b[2,3]
    B -= 2 * b[0,3] * b[1,2]

    C = 4 * b[1,3]

    for prod in quadratic_solver(A, B, C):
        if prod < 0 or np.isclose(prod, 0):
            continue

        for squared_s1 in quadratic_solver(1, -(2 + prod * b[0,2]), prod**2):
            if squared_s1 > 0:
                s1 = np.sqrt(squared_s1)
                s3 = prod / s1

                B2 = -(b[0,1] * s1 + b[2,1] * s3)
                B4 = -(b[0,3] * s1 + b[2,3] * s3)

                all_s2 = quadratic_solver(2, B2, b[0,2] * s1 * s3)
                all_s4 = quadratic_solver(2, B4, b[0,2] * s1 * s3)

                for s2 in all_s2:
                    for s4 in all_s4:
                        if s2 > 0 and s4 > 0:
                            points = np.array([proj[0] * s1,
                                               proj[1] * s2,
                                               proj[2] * s3,
                                               proj[3] * s4])

                            score = compute_score(points)

                            if solution is None or score < best_score:
                                best_score = score
                                solution = points[:]

    return solution


def compute_pose(contour: np.ndarray, k: float) -> np.ndarray:
    points = pnp_solver(contour, k)

    if points is None:
        return None

    u = points[3] - points[0]
    v = points[1] - points[0]

    # Gram–Schmidt process
    v -= u * np.dot(v, u) / np.dot(u, u)

    pose = np.eye(4)
    pose[:3, 0] = u / np.linalg.norm(u)
    pose[:3, 1] = v / np.linalg.norm(v)

    pose[:3, 2] = np.cross(pose[:3, 0], pose[:3, 1])
    pose[:3, 2] = pose[:3, 2] /np.linalg.norm(pose[:3, 2])

    pose[:3, 3] = np.mean(points, axis=0)

    return pose
