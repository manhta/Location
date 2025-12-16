import cv2 as cv
import numpy as np
from ultralytics import YOLO
import socket
import sys
import time
from datetime import datetime, timezone

# Image resolution
RESOLUTION = (640,480)

## Set socket host and port
HOST = "localhost"
PORT = 5000

## Set camera parameters
#  Camera intrinsic parameters
F_X, F_Y = 665.08, 677.78     # Focal lengths in pixels, calibrate using zhang's calibration method
M, N = RESOLUTION             # Image resolution
P = (320, 240)                # Principal point (u0, v0)
U_0, V_0 = P
B = 44.76
HALF_CAM_HORIZONTAL_FOV = 45  # in degrees 

# Camera extrinsic parameters
PHI, OMEGA, KAPPA = 0, 0, 0 # Rotation angles
ALPHA = np.radians(90 - np.absolute(PHI))

PHI_RAD, OMEGA_RAD, KAPPA_RAD = np.radians(PHI), np.radians(OMEGA), np.radians(KAPPA)  # Convert rotation angles into radians
R = np.array([
    [1, 0, 0],
    [0, np.cos(PHI_RAD), -np.sin(PHI_RAD)],
    [0, np.sin(PHI_RAD), np.cos(PHI_RAD)]
])
R = np.around(R, decimals=3)

R_INV = np.around(np.linalg.inv(R), decimals=3)     # Inverse rotation matrix
H = 0.93                                            # Camera height (m)
C = np.array([0, H, 0]).reshape(-1, 1)              # Camera position in world coordinates
T = -np.dot(R, C).reshape(-1, 1)                    # Translation matrix
T = np.around(T, decimals=3)

# Camera azimuth angle 
PSI = np.radians(90)

# Camera latitude and longitude
CAM_LAT = None
CAM_LON = None

## Load YOLO model
yolo_path = "model/yolo11n_ncnn_model"
yolo = YOLO(yolo_path, task="detect")
yolo_class_list = yolo.names

def inverse_projection(point_pixel: tuple) -> np.ndarray:
    """Converts a 2D pixel coordinate to a 3D world coordinate"""
    u, v = point_pixel
    v = 2 * V_0 - v  # Flip vertical coordinate

    r21, r11 = R[2][1], R[1][1]
    ty, tz = T[1][0], T[2][0]

    z_c = round((ty * r11 + tz * r21) / (((v - V_0) * r11) / F_Y + r21), 3)
    x_c = round(((u - U_0) / F_X) * z_c, 3)
    y_c = round(((v - V_0) / F_Y) * z_c, 3)
    
    X_c = np.array([x_c, y_c, z_c]).reshape(-1, 1)  # Object position in camera coordinates
    X_w = np.around(np.dot(R_INV, (X_c - T)), 3)    # Object position in world coordinates

    return X_w

def distance_estimation(X_w) -> float:
    """Estimates the distance from the camera to a given pixel location"""
    x, z = X_w[0][0], X_w[2][0]
    return round(np.sqrt(x**2 + z**2), 3)

def gps_estimation(point_pixel, distance):
    x = point_pixel[0]
    delta_x = np.abs(x - 320)
    az = np.radians((delta_x/320)*HALF_CAM_HORIZONTAL_FOV)

    if x < 320:
        az = (0 - az)

    delta_lon = distance * np.sin(PSI+az)
    delta_lat = distance * np.cos(PSI+az)

    obj_lat = round(CAM_LAT + (delta_lat/110717.067), 5) if CAM_LAT is not None else None
    obj_lon = round(CAM_LON + (delta_lon/103967.713), 5) if CAM_LON is not None else None

    return (obj_lat, obj_lon)

def location_estimation(point_in_pixels):
    X_w = inverse_projection(point_in_pixels)
    estimated_distance = distance_estimation(X_w)
    estimated_gps = gps_estimation(point_in_pixels, estimated_distance)
    
    return estimated_distance, estimated_gps

def open_socket_server() -> socket.socket:
    """
    Initialize and bind a TCP server socket.
    Does not accept connections.
    """

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        s.bind((HOST, PORT))
        s.listen(1)
        s.settimeout(1.0)
        print("Server opened, waiting for connection...")
        return s
    except Exception as e:
        print("Cannot open socket server, try again!")
        sys.exit(1)

def encoding_data(data: str):
    try:
        comma = data.find(',')
        lat = float(data[1:comma])
        lon = float(data[comma+2:-2])
        return lat, lon
    except Exception:
        return None, None

def main():
    global CAM_LAT, CAM_LON
    
    # FPS 
    prev_frame_time = 0
    new_frame_time = 0
    fps = 0

    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print(f"Cannot open camera")
        sys.exit(1)

    socket_server = open_socket_server()
    while True:
        try:
            conn, addr = socket_server.accept()
            break
        except socket.timeout:
            continue

    conn.settimeout(1.0)
    print("Client connected:", addr)
    print("Operation started!")

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                print("Lost connection to camera!")
                break
            
            try:
                data = conn.recv(1024).decode(encoding="utf-8", errors="ignore")
                if not data:
                    print("Client disconnected")
                    break
                data = data.strip()
                print(f"Receive data from client! Location: {data}")

                if data != "NO FIX":
                    CAM_LAT, CAM_LON = encoding_data(data)
            except socket.timeout:
                pass
            
            try:
                if ret:
                    if frame.shape[:2] != RESOLUTION:
                        frame = cv.resize(frame, RESOLUTION, interpolation=cv.INTER_LANCZOS4)
                    frame = cv.flip(frame, 1)

                    results = yolo(frame, verbose = False)

                    if results:
                        boxes = results[0].boxes

                        for i in range(len(boxes)):
                            class_id = int(boxes[i].cls.item())
                            class_name = yolo_class_list.get(class_id)

                            if class_name == "person":
                                box = boxes[i].xyxy.cpu().numpy()[0]
                                pt1 = (int(box[0]), int(box[1]))
                                pt2 = (int(box[2]), int(box[3]))

                                point = (int((pt1[0] + pt2[0]) / 2), pt2[1])
                                estimated_distance, estimated_gps = location_estimation(point)

                                cv.rectangle(frame, pt1, pt2, (0,255,0), 1)
                                cv.putText(frame, f"D: {str(round(estimated_distance, 4))} m", (pt1[0], (pt1[1]-20)), cv.FONT_HERSHEY_COMPLEX, 0.6, (0,255,0))
                                cv.putText(frame, f"Loc: {str(estimated_gps[0])}, {str(estimated_gps[1])}", (pt1[0], (pt1[1]-5)), cv.FONT_HERSHEY_COMPLEX, 0.6, (0,255,0))
                                
                    # Display FPS
                    new_frame_time = time.time()
                    dt = new_frame_time - prev_frame_time
                    fps = str(int(1 / (dt))) if dt > 0 else 0
                    cv.putText(frame, f"FPS: {fps}", (5,20), cv.FONT_HERSHEY_COMPLEX, 0.75, (0,255,0), 1)
                    prev_frame_time = new_frame_time
                    
                    # Display frame
                    cv.imshow("frame", frame)

                    # Wait for keyboard command
                    key = cv.waitKey(1)
                    if key == ord('q'):
                        print("\n\033[93mFinish, exit command!\033[0m")
                        break
                else:
                    print("Lost connection to camera!")            
                    break

            except Exception as e:
                print("Error during processing:", e)
                break

    except KeyboardInterrupt:
        print("\n\033[93mFinish, keyboard interrupt!\033[0m")
    
    finally:
        try:
            cam.release()
            cv.destroyAllWindows()
        except Exception:
            pass

if __name__ == "__main__":
    main()