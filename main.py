import cv2

import numpy as np

from marker import extract_markers
from pose import compute_pose


def draw_guizmo(image: np.ndarray, pose: np.ndarray, f: float) -> np.ndarray:
    K = np.array([[ f,  0, -1],
                  [ 0,  f, -1],
                  [ 0,  0,  1]])

    Ki = np.linalg.inv(K)

    for i in range(3):
        color = [(0,0,180), (0,180,0), (180,0,0)][i]

        pi = Ki @ pose[:3, 3]
        pf = Ki @ (pose[:3, 3] + 0.5 * pose[:3, i])

        pi = np.round(pi / pi[2]).astype(int)[:2]
        pf = np.round(pf / pf[2]).astype(int)[:2]

        image = cv2.arrowedLine(image, pi, pf, color, 2)

    return image


def main(filename: str) -> None:
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    k = 2 / max(*image.shape)

    for number, contour in extract_markers(gray):
        pose = compute_pose(contour, k)

        #cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)
        #cv2.circle(image, contour[0], 2, (0, 255, 0), -1)
        image = cv2.putText(
            image, str(number), contour[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        if pose is not None:
            image = draw_guizmo(image, pose, k)

    cv2.imshow("Markers", image)
    cv2.waitKey(0)


if __name__ == "__main__":
    main("images/aruco2.jpg")
    main("images/aruco3.png")
    main("images/aruco4.jpg")
