import trimesh


def test_model(path):

    # Load the file
    model = trimesh.load(path)

    # Check if the model is loaded successfully
    if model.is_empty:
        print("Failed to load the file.")
    else:
        print("file loaded successfully!")
        print(f"Number of vertices: {len(model.vertices)}")
        print(f"Number of faces: {len(model.faces)}")

    # Visualize the model
    model.show()

def load_model(path):

    model = trimesh.load(path)
    # Check if the model is loaded successfully
    if model.is_empty:
        print("Failed to load the model")
    else:
        print("model loaded successfully!")

    model = model.apply_scale(0.1) # Ideally, save the model as the small version

    # Check if vertex colors exist
    if model.visual.vertex_colors.any():
        colors = model.visual.vertex_colors
    else : 
        colors = None

    # Extract vertices (3D points)
    vertices = model.vertices  # Shape: (N, 3)

    # Convert vertices into the required format for cv2.projectPoints
    points_3d = vertices.reshape(-1, 1, 3)  # Shape: (N, 1, 3)

    # Extract faces and normals
    faces = model.faces
    normals = model.face_normals
    
    return points_3d, faces, normals , colors