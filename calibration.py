import numpy as np
import cv2

# Number of chessboard corners
num_corners_x = 7
num_corners_y = 6

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((num_corners_x * num_corners_y, 3), np.float32)
objp[:, :2] = np.mgrid[0:num_corners_x, 0:num_corners_y].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
obj_points = []  # 3d points in real world space
img_points = []  # 2d points in image plane.

# Number of calibration images to capture
num_images = 10

# Initialize video capture
cap = cv2.VideoCapture(0)

# Capture calibration images
for i in range(num_images):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (num_corners_x, num_corners_y), None)

    # If found, add object points, image points (after refining them)
    if ret:
        obj_points.append(objp)
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners_subpix)

        # Draw and display the corners
        cv2.drawChessboardCorners(frame, (num_corners_x, num_corners_y), corners_subpix, ret)
        cv2.imshow('Chessboard Corners', frame)
        cv2.waitKey(500)  # Show the image for 500 milliseconds

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calibrate the camera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Save camera matrix and distortion coefficients to files
np.save('camera_matrix.npy', camera_matrix)
np.save('dist_coeffs.npy', dist_coeffs)

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
