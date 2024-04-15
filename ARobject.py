import cv2
import numpy as np
import open3d as o3d

# Load camera matrix and distortion coefficients from previous calibration (Homework Assignment #3)
camera_matrix = np.load('camera_matrix.npy')
dist_coeffs = np.load('dist_coeffs.npy')

# Load a simple 3D model for AR visualization
mesh = o3d.geometry.TriangleMesh.create_box(width=0.1, height=0.1, depth=0.1)

# Initialize Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window()

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Undistort the frame using camera matrix and distortion coefficients
    h, w = frame.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Estimate camera pose
    # This part should be implemented based on your specific camera pose estimation method

    # Render the AR object using Open3D
    vis.clear_geometries()
    vis.add_geometry(mesh)
    vis.update_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()

    # Combine undistorted frame with AR visualization
    vis_img = np.asarray(vis.capture_screen_float_buffer(True))
    combined_frame = cv2.addWeighted(undistorted_frame, 0.5, vis_img, 0.5, 0)

    # Display the combined frame
    cv2.imshow('AR Visualization', combined_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
