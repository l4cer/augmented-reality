import cv2

import numpy as np

from typing import Tuple, List


def extract_contours(image: np.ndarray, min_length: float = 40.0) -> np.ndarray:
    blured = cv2.GaussianBlur(image, (3, 3), 0)

    _, thresh = cv2.threshold(blured, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filtered = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4 and cv2.arcLength(approx, True) > min_length:
            filtered.append([item[0] for item in approx])

    return np.asarray(filtered)


def improve_contour(contour: np.ndarray, corners: np.ndarray, max_dist: float = 10.0) -> None:
    for index, corner in enumerate(contour):
        distances = np.linalg.norm(corners - corner, axis=1)
        argmin = np.argmin(distances)

        if distances[argmin] < max_dist:
            contour[index] = corners[argmin]


def decode_marker(image: np.ndarray, contour: np.ndarray) -> Tuple[int, int]:
    minimum = np.min(contour, axis=0)
    maximum = np.max(contour, axis=0)

    size = max(maximum - minimum) + 1

    cropped = np.zeros((size, size), dtype=image.dtype)

    xi, xf = minimum[1], min(minimum[1] + size, image.shape[0])
    yi, yf = minimum[0], min(minimum[0] + size, image.shape[1])

    cropped = image[xi:xf, yi:yf]

    points_A = np.array(contour - minimum, dtype=np.float32)
    points_B = np.array([[   0,    0],
                         [size,    0],
                         [size, size],
                         [   0, size]], dtype=np.float32)

    H, _ = cv2.findHomography(points_A, points_B, cv2.RANSAC, 4.0)

    kwargs = {
        "borderMode": cv2.BORDER_TRANSPARENT,
        "flags": cv2.INTER_LINEAR
    }

    warpped = cv2.warpPerspective(cropped, H, (size, size), image, **kwargs)

    _, thresh = cv2.threshold(warpped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    for msize in range(6, 10):
        bits = np.zeros((msize, msize), dtype=np.uint8)
        border = int(0.2 * (size // msize))

        for i in range(msize):
            for j in range(msize):
                xi, xf = i * size // msize + border, (i+1) * size // msize - border
                yi, yf = j * size // msize + border, (j+1) * size // msize - border

                bits[i, j] = 1 if np.mean(thresh[xi:xf, yi:yf]) > 127 else 0

        parser = {
            6: cv2.aruco.DICT_4X4_250,
            7: cv2.aruco.DICT_5X5_250,
            8: cv2.aruco.DICT_6X6_250,
            9: cv2.aruco.DICT_7X7_250
        }

        dictionary = cv2.aruco.getPredefinedDictionary(parser[msize])
        valid, number, rotation = dictionary.identify(bits[1:-1, 1:-1], 0.6)

        if valid:
            return number, rotation

    return None, None


def extract_markers(image: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    theta = cv2.cornerHarris(image, 2, 3, 0.04)

    j, i = np.where(theta > 0.01 * np.max(theta))
    corners = np.concatenate((i[:, np.newaxis], j[:, np.newaxis]), axis=1)

    markers = []
    for contour in extract_contours(image):
        improve_contour(contour, corners)

        number, rotation = decode_marker(image, contour)
        if number is None:
            continue

        markers.append((number, np.roll(contour, rotation, axis=0)))

    return markers