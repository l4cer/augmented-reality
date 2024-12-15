import cv2
import sys
import numpy as np

from marker import extract_markers
from pose import compute_pose
from render import *

model_paths = [
    "3d_models/scad_chess_bishop.stl",
    "3d_models/scad_chess_king.stl",
    "3d_models/scad_chess_knight.stl",
    "3d_models/scad_chess_pawn.stl",
    "3d_models/scad_chess_queen.stl",
    "3d_models/scad_chess_rook.stl",
    "3d_models/scad_chess_rook.stl",
    "3d_models/scad_chess_rook.stl",
    "3d_models/scad_chess_rook.stl",
    "3d_models/scad_chess_rook.stl",
    "3d_models/Jolteon.stl",
]


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


def process_frame(frame: np.ndarray, objects : np.ndarray , debug: bool = False, displayToggle = [True, True]) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    f = 2 / max(*frame.shape)

    for number, contour in extract_markers(gray):
        pose, rvec, tvec = compute_pose(contour, f)
        cameraMatrix = np.linalg.inv(np.array([[ f,  0, -1],
                                                [ 0,  f, -1],
                                                [ 0,  0,  1]]))

        contour = np.round(contour).astype(int)
        distortion_coeffs = np.zeros(4) 
        light_pos = np.array([0, 0, 0]) 

        if debug:
            cv2.drawContours(frame, [contour], -1, (0, 0, 255), 1)
            cv2.circle(frame, contour[0], 2, (0, 255, 0), -1)
            frame = cv2.putText(
                frame, str(number), contour[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

            if pose is not None:
                frame = draw_box(frame, pose)
                frame = draw_axes(frame, pose)
        
        if number > 8 :
            continue
        else :
            points, faces, normals, colors = objects[number]
            image_points, _ = cv2.projectPoints(points , rvec, tvec, cameraMatrix, distortion_coeffs)

            # display vertices
            if displayToggle[0]:
                for k in range(len(image_points)):
                    color = tuple([int(colors[k][i]) for i in range(4)])
                    center = tuple(np.round(image_points[k].ravel()).astype(int)) 
                    cv2.circle(frame, center, 3, color, -1)

            # display filled polygons
            if displayToggle[1]:
                # Compute light direction for each face
                rotation_matrix, _ = cv2.Rodrigues(rvec)  
                points = points.reshape(-1, 3) 
                points_camera_space = (rotation_matrix @ points.T).T + tvec.T  

                face_centers = np.mean(points_camera_space[faces], axis=1) 
                light_directions = light_pos - face_centers 
                light_directions /= np.linalg.norm(light_directions, axis=1, keepdims=True) + 1e-8
                light_directions = light_directions.reshape(-1, 3)

                # Calculate diffuse intensity
                intensities = np.maximum(0, np.einsum('ij,ij->i', normals, light_directions))
                intensities = np.nan_to_num(intensities, nan=0.0)  
                intensities = 255 * intensities



                for k in range(len(image_points)):
                    vcolor = int(intensities[k])
                    color = (vcolor, vcolor, vcolor)
                    points = image_points[faces[k]].astype(int)
                    #cv2.fillConvexPoly(frame, points, color)
                    cv2.fillPoly(frame, [points], color)
                   
    return frame


def main(device: int, debug: bool = False, displayToggle = [True, True]) -> None:
    cap = cv2.VideoCapture(device)
    objects = [load_model(path) for path in model_paths]

    if not cap.isOpened():
        print("Cannot open camera")
        exit

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame. Exiting...")
            break

        frame = process_frame(frame, objects, debug=debug, displayToggle=displayToggle)
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":


    # Default
    testModel = False
    debug = False
    show_vertex = True
    show_polygon = False
    path = ""

    for arg in sys.argv[1:]:
        if arg == "-h":
            print("""
Usage: python main.py [options]

Options:
  -h                Show this help message and exit
  --debug           Display contours and axes
  --noVertex        Disable vertex rendering
  --polygon         Enable polygon rendering
  --testModel       Display the 1st model before launching the camera
  --path=<path>     Specify a custom path to a 3D model file. 
                    Replaces the first default model in the list.

Examples:
  python main.py --debug --polygon --path="3d_models/my_model.stl"
  python main.py --testModel --noVertex
    """)
            exit(0)
        elif arg == "--debug":
            debug = True
        elif arg == "--noVertex":
            show_vertex = False
        elif arg == "--polygon":
            show_polygon = True
        elif arg == "--testModel":
            testModel = True
        elif arg.startswith("--path="):
            path = arg.split("=", 1)[1] 
            model_paths[0] = path # replace the first model by the one provided

    displayToggle = [show_vertex, show_polygon]

    if testModel:
        test_model("3d_models/Jolteon.stl")
    main(0, debug, displayToggle)
