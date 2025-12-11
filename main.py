import cv2 as cv
import numpy as np
from ultralytics import YOLO

# Load YOLO model
yolo = YOLO()
class_dict = yolo.names

# Camera intrinsic parameters
fx, fy = 642.76, 653.31   # Focal lengths in pixels
fx_2, fy_2 = 665.08, 677.78 # Focal lengths in pixels, calibrate using zhang's calibration method
m, n = 640, 480             # Image resolution
P = (320, 240)              # Principal point (u0, v0)
u0, v0 = P
B = 44.76

halfCamHorizontalFOV = 45 #in degree 

# Camera extrinsic parameters
phi, omega, kappa = 0, 0, 0 # Rotation angles
alpha = np.radians(90 - np.absolute(phi))

phi_rad, omega_rad, kappa_rad = np.radians(phi), np.radians(omega), np.radians(kappa)  # Convert rotation angles into radians
R = np.array([
    [1, 0, 0],
    [0, np.cos(phi_rad), -np.sin(phi_rad)],
    [0, np.sin(phi_rad), np.cos(phi_rad)]
])
R = np.around(R, decimals=3)

R_inv = np.around(np.linalg.inv(R), decimals=3)     # Inverse rotation matrix
h = 0.93                                      # Camera height (m)
C = np.array([0, h, 0]).reshape(-1, 1)              # Camera position in world coordinates
t = -np.dot(R, C).reshape(-1, 1)                    # Translation matrix
t = np.around(t, decimals=3)

# Camera latitude and longitude
cam_lat = 21.003947
cam_lon = 105.842602

# Camera azimuth angle 
psi = np.radians(90)


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

    "Using Zhang's method"
    z_c_2 = round((ty * r11 + tz * r21) / (((v - v0) * r11) / fy_2 + r21), 3)
    x_c_2 = round(((u - u0) / fx_2) * z_c_2, 3)
    y_c_2 = round(((v - v0) / fy_2) * z_c_2, 3)
    
    Xc_2 = np.array([x_c_2, y_c_2, z_c_2]).reshape(-1, 1)
    Xw_2 = np.around(np.dot(R_inv, (Xc_2 - t)), 3)

    return Xw, Xc, Xw_2, Xc_2


def distanceEstimation(Xw) -> float:
    """
    Estimates the distance from the camera to a given pixel location.
    """
    x, z = Xw[0][0], Xw[2][0]
    return round(np.sqrt(x**2 + z**2), 3)

def distanceEstimation2(Xw) -> float: 
    """
    Estimates the distance from the camera to a given pixel location.
    Using Zhang's method
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

def korGPS(point_pixel, distance):
    x = point_pixel[0]
    deltaX = np.abs(x-320)
    az = np.radians((deltaX/320)*halfCamHorizontalFOV)

    if x<320:
        az = (0-az)

    deltaLong = distance*np.sin(psi+az)
    deltaLat = distance*np.cos(psi+az)

    lat_obj = cam_lat + (deltaLat/110717.067)
    long_obj = cam_lon + (deltaLong/103967.713)

    return lat_obj, long_obj


def locationEstimation(point_pixel, psi, cam_lat, cam_lon):
    Xw, Xc, Xw_2, Xc_2 = InverseProjection(point_pixel)
    estimatedDistance = distanceEstimation(Xw)
    estimatedDistance_2 = distanceEstimation(Xw_2)
    estimatedGPS = gpsEstimation(Xc, psi, cam_lat, cam_lon)
    estimatedGPS_2 = gpsEstimation(Xc_2, psi, cam_lat, cam_lon)
    testGPS = gpsTest(Xw, psi, cam_lat, cam_lon)
    testGPS_2 = gpsTest(Xw_2, psi, cam_lat, cam_lon)
    koreaGPS = korGPS(point_pixel, estimatedDistance) 
    koreaGPS_2 = korGPS(point_pixel, estimatedDistance_2) 

    return estimatedDistance, estimatedGPS, testGPS, koreaGPS, estimatedDistance_2, estimatedGPS_2, testGPS_2, koreaGPS_2

def main():
    cam = cv.VideoCapture(0)
    
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        # frame = cv.flip(frame, 1)
        results = yolo.predict(frame, conf=0.2)

        if results:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].tolist()
                class_id = int(boxes.cls[i].item())

                if class_dict[class_id] == "person":
                    pt1, pt2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            
                    cv.rectangle(frame, pt1, pt2, (0, 255, 0), 1)

                    point = (int((pt1[0] + pt2[0]) / 2), pt2[1])
                    dis, gps, gps_test, koreaGPS, dis2, gps2, gps_test2, koreaGPS2 = locationEstimation(point, psi, cam_lat, cam_lon)
            
                    cv.circle(frame, point, 2, (0, 0, 255), -1)
                    cv.putText(frame, f'Point: {point}', (10,35), 
                               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
                    cv.putText(frame, f'Dis: {dis} m', (10,55), 
                               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
                    cv.putText(frame, f'Dis (Z): {dis2} m', (10,75), 
                               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
                    cv.putText(frame, f'GPS (1): ({gps2[0]:.6f}, {gps2[1]:.6f})', (10,95), 
                               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
                    cv.putText(frame, f'GPS (2): ({gps_test2[0]:.6f}, {gps_test2[1]:.6f})', (10,115), 
                               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
                    cv.putText(frame, f'GPS (3): ({koreaGPS2[0]:.6f}, {koreaGPS2[1]:.6f})', (10,135), 
                               cv.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
        
        cv.imshow('frame', frame)

        if cv.waitKey(1) == ord('q'):
            break

    cam.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()