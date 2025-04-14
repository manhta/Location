import cv2 as cv
import numpy as np
from ultralytics import YOLO

# Load YOLO model
yolo = YOLO('model/yolov8x.pt')

# Camera intrinsic parameters
fx, fy = 642.76, 653.31   # Focal lengths in pixels
m, n = 640, 480             # Image resolution
P = (320, 240)              # Principal point (u0, v0)
u0, v0 = P
B = 51.14

# Camera extrinsic parameters
phi, omega, kappa = 0, 0, 0 # Rotation angles
# alpha = np.radians(90 - np.absolute(phi))
alpha = 90

phi_rad, omega_rad, kappa_rad = np.radians(phi), np.radians(omega), np.radians(kappa)  # Convert rotation angles into radians
R = np.array([
    [1, 0, 0],
    [0, np.cos(phi_rad), -np.sin(phi_rad)],
    [0, np.sin(phi_rad), np.cos(phi_rad)]
])
R = np.around(R, decimals=3)

R_inv = np.around(np.linalg.inv(R), decimals=3)     # Inverse rotation matrix
h = 0.9                                          # Camera height (m)
C = np.array([0, h, 0]).reshape(-1, 1)              # Camera position in world coordinates
t = -np.dot(R, C).reshape(-1, 1)                    # Translation matrix
t = np.around(t, decimals=3)

# Camera latitude and longitude
cam_lat = 21.003973
cam_lon = 105.842590

# Camera azimuth angle 
psi = np.radians(275)


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
    return Xw, Xc


def distanceEstimation(Xw) -> float:
    """
    Estimates the distance from the camera to a given pixel location.
    """
    x, z = Xw[0][0], Xw[2][0]
    return round(np.sqrt(x**2 + z**2), 3)


def gpsEstimation(Xc, psi, cam_lat, cam_lon):
    """
    Triển khai công thức (18) đến (31) để tính tọa độ GPS của vật thể
    """
    x_c, y_c, z_c = Xc[0][0], Xc[1][0], Xc[2][0]
    # --- (18) đến (20): Chuyển Camera frame → Body frame ---
    x_b = x_c
    y_b = y_c * np.cos(alpha) + z_c * np.sin(alpha)
    z_b = - y_c * np.sin(alpha) + z_c * np.cos(alpha)

    # --- (21) đến (23): Body frame → ENU ---
    E = z_b * np.sin(psi) + x_b * np.cos(psi)
    N = z_b * np.cos(psi) - x_b * np.sin(psi)
    U = -y_b  # không dùng trong bước tính GPS

    # --- (24): Góc phương vị b ---
    b = np.arctan(abs(E / N))  # theo công thức gốc, lấy trị tuyệt đối

    # --- (25): Khoảng cách s ---
    s = np.sqrt(E**2 + N**2)

    # --- (26) và (27): Dịch chuyển theo X và Y ---
    dX = s * np.sin(b)
    dY = s * np.cos(b)

    # --- (28) và (29): Tính độ lệch kinh độ và vĩ độ ---
    # 2 số kia đang xem Hanoi là bao nhiêu nhé !!!
    delta_lon = dX / (103967.713 * np.cos(np.deg2rad(cam_lat)))
    delta_lat = dY / 110717.067
    
    # --- (30) và (31): Tính tọa độ GPS cuối cùng ---
    lat_object = cam_lat + delta_lat
    lon_object = cam_lon + delta_lon

    return lat_object, lon_object

def gpsTest(Xw, psi, cam_lat, cam_lon):
    x_w, z_w = Xw[0][0], Xw[2][0]
    E = z_w * np.sin(psi) + x_w * np.cos(psi)
    N = z_w * np.cos(psi) - x_w * np.sin(psi)

    # --- (24): Góc phương vị b ---
    b = np.arctan(abs(E / N))  # theo công thức gốc, lấy trị tuyệt đối

    # --- (25): Khoảng cách s ---
    s = np.sqrt(E**2 + N**2)

    # --- (26) và (27): Dịch chuyển theo X và Y ---
    dX = s * np.sin(b)
    dY = s * np.cos(b)

    # --- (28) và (29): Tính độ lệch kinh độ và vĩ độ ---
    # 2 số kia đang xem Hanoi là bao nhiêu nhé !!!
    delta_lon = dX / (103967.713 * np.cos(np.deg2rad(cam_lat)))
    delta_lat = dY / 110717.067
    
    # --- (30) và (31): Tính tọa độ GPS cuối cùng ---
    lat_object = cam_lat + delta_lat
    lon_object = cam_lon + delta_lon

    return lat_object, lon_object

def locationEstimation(point_pixel, psi, cam_lat, cam_lon):
    Xw, Xc = InverseProjection(point_pixel)
    estimatedDistance = distanceEstimation(Xw)
    estimatedGPS = gpsEstimation(Xc, psi, cam_lat, cam_lon)
    testGPS = gpsTest(Xw, psi, cam_lat, cam_lon)

    return estimatedDistance, estimatedGPS, testGPS

def main():
    cam = cv.VideoCapture(1)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        results = yolo.predict(frame, conf=0.7)

        if results:
            boxes = results[0].boxes.xyxy.tolist()
            for box in boxes:
                pt1, pt2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
                
                cv.rectangle(frame, pt1, pt2, (0, 255, 0), 1)
                
                point = (int((pt1[0] + pt2[0]) / 2), pt2[1])
                dis, gps, gps_test = locationEstimation(point, psi, cam_lat, cam_lon)
                
                cv.circle(frame, point, 1, (0, 0, 255), 1)
                cv.putText(frame, f'Dist: {dis} m', (10,30), 
                           cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                cv.putText(frame, f'GPS: ({gps[0]:.6f}, {gps[1]:.6f})', (10,100), 
                           cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                cv.putText(frame, f'GPS test: ({gps_test[0]:.6f}, {gps_test[1]:.6f})', (10,100), 
                           cv.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        
        cv.imshow('frame', frame)

        if cv.waitKey(1) == ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()