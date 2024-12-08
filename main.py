import cv2

import numpy as np

from marker import extract_markers
from pose import compute_pose


def draw_axes(frame: np.ndarray, pose: np.ndarray) -> np.ndarray:
    f = 2 / max(*frame.shape)

    K = np.linalg.inv(np.array([[ f,  0, -1],
                                [ 0,  f, -1],
                                [ 0,  0,  1]]))

    for i in range(3):
        color = [(0, 0, 180), (0, 180, 0), (180, 0, 0)][i]

        pi = K @ pose[:3, 3]
        pf = K @ pose[:3, i] * 0.5 + pi

        pi = np.round(pi / pi[2]).astype(int)[:2]
        pf = np.round(pf / pf[2]).astype(int)[:2]

        frame = cv2.arrowedLine(frame, pi, pf, color, 2)

    return frame


def draw_box(frame: np.ndarray, pose: np.ndarray) -> np.ndarray:
    f = 2 / max(*frame.shape)

    K = np.linalg.inv(np.array([[ f,  0, -1],
                                [ 0,  f, -1],
                                [ 0,  0,  1]]))

    p1 = K @ (pose[:3, 3] - 0.5 * pose[:3, 1] - 0.5 * pose[:3, 0])
    p2 = K @ (pose[:3, 3] + 0.5 * pose[:3, 1] - 0.5 * pose[:3, 0])
    p3 = K @ (pose[:3, 3] + 0.5 * pose[:3, 1] + 0.5 * pose[:3, 0])
    p4 = K @ (pose[:3, 3] - 0.5 * pose[:3, 1] + 0.5 * pose[:3, 0])

    p5 = K @ pose[:3, 2] + p1
    p6 = K @ pose[:3, 2] + p2
    p7 = K @ pose[:3, 2] + p3
    p8 = K @ pose[:3, 2] + p4

    p = [p1, p2, p3, p4, p5, p6, p7, p8]
    p = [np.round(pi / pi[2]).astype(int)[:2] for pi in p]
    p = np.array(p)

    cv2.drawContours(frame, [p[[0, 1, 2, 3]]], -1, (255, 0, 255), 1)
    cv2.drawContours(frame, [p[[4, 5, 6, 7]]], -1, (255, 0, 255), 1)
    cv2.drawContours(frame, [p[[0, 1, 5, 4]]], -1, (255, 0, 255), 1)
    cv2.drawContours(frame, [p[[2, 3, 7, 6]]], -1, (255, 0, 255), 1)

    return frame


def process_frame(frame: np.ndarray, debug: bool = False) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    f = 2 / max(*frame.shape)

    for number, contour in extract_markers(gray):
        pose = compute_pose(contour, f)
        contour = np.round(contour).astype(int)

        if debug:
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), 1)
            cv2.circle(frame, contour[0], 2, (0, 255, 0), -1)
            frame = cv2.putText(
                frame, str(number), contour[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            if pose is not None:
                frame = draw_box(frame, pose)
                frame = draw_axes(frame, pose)

    return frame


def main(device: int, debug: bool = False) -> None:
    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        print("Cannot open camera")
        exit

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame. Exiting...")
            break

        frame = process_frame(frame, debug=debug)
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(0, debug=True)
