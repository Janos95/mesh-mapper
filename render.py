import os
import trimesh
import polyscope as ps
import numpy as np

def load_and_render_mesh(mesh_path):
    # Load mesh using trimesh
    mesh = trimesh.load(mesh_path)
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)

    ps.init()
    ps.set_ground_plane_mode("none")

    ps_mesh = ps.register_surface_mesh("my_mesh", verts, faces)
    ps_mesh.set_color([0.5, 0.5, 0.5])  # Set color to grey

    bbox = mesh.bounding_box
    bbox_center = bbox.centroid
    bbox_extent = bbox.extents


    # Define camera positions and names
    directions = ['x_plus', 'x_minus', 'y_plus', 'y_minus', 'z_plus', 'z_minus']
    camera_positions = []
    for dir in directions:
        if 'x' in dir:
            dist = np.linalg.norm(bbox_extent[[1, 2]]) + bbox_extent[0] / 2
            sign = 1 if 'plus' in dir else -1
            camera_positions.append(bbox_center + np.array([sign * dist, 0, 0]))
        elif 'y' in dir:
            dist = np.linalg.norm(bbox_extent[[0, 2]]) + bbox_extent[1] / 2
            sign = 1 if 'plus' in dir else -1
            camera_positions.append(bbox_center + np.array([0, sign * dist, 0]))
        else:
            dist = np.linalg.norm(bbox_extent[[0, 1]]) + bbox_extent[2] / 2
            sign = 1 if 'plus' in dir else -1
            camera_positions.append(bbox_center + np.array([0, 0, sign * dist]))

    look_at_points = [bbox_center] * 6  # Look at the center of the bounding box

    ps.set_screenshot_extension(".jpg")

    base_filename = os.path.splitext(os.path.basename(mesh_path))[0]

    for i, (cam_pos, look_at, dir) in enumerate(zip(camera_positions, look_at_points, directions)):
        ps.look_at(cam_pos, look_at)
        screenshot_filename = os.path.join('images', base_filename, f"{dir}.jpg")
        dir = os.path.join('images', base_filename)
        if not os.path.exists(dir):
            os.makedirs(dir)

        ps.screenshot(screenshot_filename)

# Specify the directory containing the 3D files
directory = 'meshes'

paths = []
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    if filepath.lower().endswith(('.stl', '.obj', '.ply', '.off')):
        paths.append(filepath)


paths = paths[:10]

for mesh_path in paths:
    load_and_render_mesh(mesh_path)