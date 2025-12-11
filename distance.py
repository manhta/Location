import cv2 as cv
import numpy as np
from ultralytics import YOLO

# Load YOLO model
yolo = YOLO('Location/model/yolov8x.pt')

# Camera intrinsic parameters (lap cam)
fx, fy = 659.68, 648.23     # Focal lengths in pixels
m, n = 640, 480             # Image resolution
P = (320, 240)              # Principal point (u0, v0)
u0, v0 = P


# Camera extrinsic parameters
phi, omega, kappa = -15.47, 0, 0 # Rotation angles
alpha = np.radians(90 - np.absolute(phi))

phi_rad, omega_rad, kappa_rad = np.radians(phi), np.radians(omega), np.radians(kappa)  # Convert rotation angles into radians
R = np.array([
    [1, 0, 0],
    [0, np.cos(phi_rad), -np.sin(phi_rad)],
    [0, np.sin(phi_rad), np.cos(phi_rad)]
])
R = np.around(R, decimals=3)

R_inv = np.around(np.linalg.inv(R), decimals=3)  # Inverse rotation matrix
h = 0.925  # Camera height (m)
C = np.array([0, h, 0]).reshape(-1, 1)  # Camera position in world coordinates
t = -np.dot(R, C).reshape(-1, 1)  # Translation matrix
t = np.around(t, decimals=3)

def InverseProjection(point_pixel: tuple) -> np.ndarray:
    """
    Converts a 2D pixel coordinate to a 3D world coordinate.
    """
    u, v = point_pixel
    v = 2 * v0 - v  # Flip vertical coordinate

    r21, r11 = R[2][1], R[1][1]
    ty, tz = t[1][0], t[2][0]

    z_c = round((ty * r11 + tz * r21) / (((v - v0) * r11) / fy + r21), 3)
    x_c = round(((u - u0) / fx) * z_c, 3)
    y_c = round(((v - v0) / fy) * z_c, 3)
    
    Xc = np.array([x_c, y_c, z_c]).reshape(-1, 1)
    Xw = np.around(np.dot(R_inv, (Xc - t)), 3)
    return Xw

def DistanceEstimation(point_pixel: tuple) -> float:
    """
    Estimates the distance from the camera to a given pixel location.
    """
    point = InverseProjection(point_pixel)
    x, z = point[0][0], point[2][0]
    return round(np.sqrt(x**2 + z**2), 3)

def main():
    cam = cv.VideoCapture(1)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        results = yolo.predict(frame, conf=0.7)
        print(frame.shape[:2])
        if results:
            boxes = results[0].boxes.xyxy.tolist()
            for box in boxes:
                pt1, pt2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                
                cv.rectangle(frame, pt1, pt2, (0, 255, 0), 1)
                
                point = (int((pt1[0] + pt2[0]) / 2), pt2[1])
                pred_dis = DistanceEstimation(point_pixel=point)
                

                cv.putText(frame, f'Dist: {pred_dis} m', (30,30), 
                           cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
        
        cv.imshow('frame', frame)

        if cv.waitKey(1) == ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()