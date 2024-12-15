import sys

import cv2

import numpy as np

from marker import extract_markers
from  pose  import compute_pose
from render import Model, draw_axes, draw_box, render_model


help_docstring = """
Usage: python main.py [options]

Options:
  -h                 Show this help message and exit
  --debug            Display contours and axes
  --device=<int>     Specify video capture device

Examples:
  python main.py
  python main.py --debug
  python main.py --device=0
"""

models = {
    0: Model("models/chess_bishop.stl", 0.03),
    1: Model("models/chess_rook.stl",   0.03),
    2: Model("models/chess_queen.stl",  0.03),
    7: Model("models/chess_pawn.stl",   0.03),
    9: Model("models/chess_king.stl",   0.03)
}


def process_frame(frame: np.ndarray, debug: bool = False) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    for number, contour in extract_markers(gray):
        if number not in models:
            continue

        pose = compute_pose(contour, models[number].K)
        contour = np.round(contour).astype(int)

        if pose is None:
            continue

        if debug:
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), 1)
            cv2.circle(frame, contour[0], 2, (0, 255, 0), -1)
            cv2.putText(
                frame, str(number), contour[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            draw_axes(frame, models[number], pose)
            draw_box(frame, models[number], pose)

        else:
            render_model(frame, models[number], pose)

    return frame


def main(device: int, debug: bool = False) -> None:
    cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        print("Cannot open camera")
        exit

    ret, frame = cap.read()
    for model in models.values():
        model.set_intrinsic_matrix(2 / max(*frame.shape))

    print("Press \33[34mQ\33[0m to quit")

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
    device = 0
    debug = False

    for arg in sys.argv[1:]:
        if arg == "-h":
            print(help_docstring)
            exit(0)

        elif arg.startswith("--device="):
            device = int(arg.split("=", 1)[1])

        elif arg == "--debug":
            debug = True

    main(device, debug=debug)
