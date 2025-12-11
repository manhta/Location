import cv2 as cv
import numpy as np
from ultralytics import YOLO
from nmea import parse_fields, parse_gga, parse_gll, parse_gsa, parse_rmc, parse_vtg, dm_to_deg, deg_to_dms, nmea_checksum_ok
import serial
import sys
import re
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# Set local timezone
LOCAL_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")

# Image resolution
RESOLUTION = (640,480)

## Set the port and baud rate for the Arduino; switch the port to /dev/serial0 when using a Raspberry Pi
PORT = "COM5"
BAUD = 115200
TIMEOUT = 0.5
REFRESH_HZ = 1.0

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
yolo_path = "Your_model_path"
yolo = YOLO(yolo_path)
yolo_class_list = yolo.names


# Box width (including borders). Keep inner content <= (BOX_WIDTH - 2).
BOX_WIDTH = 78
INNER = BOX_WIDTH - 2

# --------- ANSI helpers ---------
RESET = "\033[0m"
BOLD = "\033[1m"
DIM  = "\033[2m"
FG = {
    "gray": "\033[90m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
}
CLEAR = "\033[2J\033[H"  # clear screen + move cursor home
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")

def color(txt, c="white", bold=False, dim=False):
    s = FG.get(c, "")
    if bold:
        s += BOLD
    if dim:
        s += DIM
    return f"{s}{txt}{RESET}"

def vis_len(s: str) -> int:
    """Visible length ignoring ANSI escape codes."""
    return len(ANSI_RE.sub("", s))

def pad_to(s: str, width: int) -> str:
    """Right-pad with spaces to visible width; never slice through ANSI."""
    v = vis_len(s)
    if v < width:
        return s + (" " * (width - v))
    return s  # assume caller keeps strings within width

def bar_line(left: str, right: str = "") -> str:
    """Build a single content line between vertical borders with left/right parts."""
    l = left
    r = right
    gap = INNER - vis_len(l) - vis_len(r)
    if gap < 0:
        # If overflow, drop right first, then trim left by appending ellipsis
        r = ""
        gap = INNER - vis_len(l)
        if gap < 0:
            l = l[:max(0, INNER-3)] + "..."  # safe fallback (no ANSI expected here)
            gap = INNER - vis_len(l)
    return f"{color('│','blue')}{l}{' '*gap}{r}{color('│','blue')}"

def draw(state):
    # Prepare strings
    lat = state["lat"]
    lon = state["lon"]
    alt = state["alt"]
    spd = state["speed_kmh"]
    cog = state["cog"]
    hdop = state["hdop"]
    sats = state["sats_used"] if state["sats_used"] is not None else state["numsats"]
    fix_dim = state["fix_dim"]
    valid = state["valid"]
    utc_ts = state["utc_ts"]

    fix_label = "VALID" if valid else "NO FIX"
    fix_colored = color(fix_label, "green" if valid else "red", bold=True)

    lat_s = f"{lat:.6f}" if lat is not None else "-"
    lon_s = f"{lon:.6f}" if lon is not None else "-"
    lat_d = deg_to_dms(lat, True) if lat is not None else "-"
    lon_d = deg_to_dms(lon, False) if lon is not None else "-"

    alt_s = f"{alt:.2f} m" if alt is not None else "-"
    spd_s = f"{spd:.2f} km/h" if spd is not None else "-"
    cog_s = f"{cog:.2f}°" if cog is not None else "-"
    hdop_s = f"{hdop:.2f}" if hdop is not None else "-"
    sats_s = f"{sats}" if sats is not None else "-"
    dim_map = {"1":"NO FIX", "2":"2D", "3":"3D", None:"?"}
    dim_s = dim_map.get(fix_dim, str(fix_dim))

    utc_s = (utc_ts.isoformat() if utc_ts else datetime.now(timezone.utc).isoformat())
    loc_s = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Draw box
    top = color("┌" + "─"*INNER + "┐", "blue")
    sep = color("├" + "─"*INNER + "┤", "blue")
    bot = color("└" + "─"*INNER + "┘", "blue")

    print(CLEAR, end="")
    # Title
    left = " GPS Monitor  /dev/serial0 @ {baud}".format(baud=BAUD)
    right = f"Local: {loc_s} "
    print(top)
    print(bar_line(left, right))
    print(sep)

    # Status line
    status = f" Status: {fix_colored}   Dim: {dim_s:>3}   HDOP: {hdop_s:>5}   Sats: {sats_s:>2} "
    print(bar_line(status))

    # UTC line
    utc_line = f" UTC:    {utc_s} "
    print(bar_line(utc_line))

    print(sep)

    # Position lines
    pos1 = f" Lat: {lat_s:>11}   Lon: {lon_s:>11}   Alt: {alt_s:>10} "
    print(bar_line(pos1))

    pos2 = f" DMS: {lat_d:<22}  {lon_d:<22} "
    print(bar_line(pos2))

    print(sep)

    # Motion line
    motion = f" Speed: {spd_s:<14}  Course: {cog_s:<12} "
    print(bar_line(motion))
    print(bot)

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

def main():
    global CAM_LAT, CAM_LON
    
    # FPS 
    prev_frame_time = 0
    new_frame_time = 0
    fps = 0
    
    try:
        ser = serial.Serial(port=PORT, baudrate=BAUD, timeout=TIMEOUT)
    except Exception as e:
        print(f"Cannot open {PORT} @ {BAUD}: {e}")
        sys.exit(1)

    cam = cv.VideoCapture(0)
    if not cam.isOpened():
        print(f"Cannot open camera")
        sys.exit(1)
    
    last = {
        "lat": None, "lon": None,
        "alt": None,
        "speed_kmh": None, "cog": None,
        "fix_dim": None, "hdop": None, "sats_used": None, "numsats": None,
        "valid": False, "utc_ts": None
    }

    last_draw = 0.0
    print(CLEAR, end="")

    try:
        while True:
            ret, frame = cam.read()
            if frame.shape[:2] != RESOLUTION:
                frame = cv.resize(frame, RESOLUTION, interpolation=cv.INTER_LANCZOS4)
            frame = cv.flip(frame, 1)

            raw = ser.readline()
            
            try:
                line = raw.decode("ascii", errors="ignore").strip()
            except:
                line = ""
            
            if line.startswith("$") and nmea_checksum_ok(line):
                    fields = parse_fields(line)
                    typ = fields[0][-3:]

                    if typ == "GLL":
                        p = parse_gll(fields)
                        if p:
                            last["lat"] = p["lat"] if p["lat"] is not None else last["lat"]
                            last["lon"] = p["lon"] if p["lon"] is not None else last["lon"]
                            last["valid"] = (p.get("status","") == "A")
                            utc = p.get("utc","")
                            if len(utc) >= 6:
                                try:
                                    today = datetime.now(timezone.utc).date()
                                    hh, mm, ss = int(utc[0:2]), int(utc[2:4]), float(utc[4:])
                                    sec = int(ss)
                                    micro = int((ss-sec)*1e6)
                                    last["utc_ts"] = datetime(today.year, today.month, today.day, hh, mm, sec, micro, tzinfo=timezone.utc)
                                except Exception:
                                    pass

                    elif typ == "VTG":
                        v = parse_vtg(fields)
                        if v:
                            last["speed_kmh"] = v["s_kmh"] if v["s_kmh"] is not None else last["speed_kmh"]
                            last["cog"] = v["cog"] if v["cog"] is not None else last["cog"]

                    elif typ == "RMC":
                        r = parse_rmc(fields)
                        if r:
                            last["lat"] = r["lat"] if r["lat"] is not None else last["lat"]
                            last["lon"] = r["lon"] if r["lon"] is not None else last["lon"]
                            last["speed_kmh"] = (r["sog_kn"] * 1.852) if r["sog_kn"] is not None else last["speed_kmh"]
                            last["cog"] = r["cog"] if r["cog"] is not None else last["cog"]
                            last["utc_ts"] = r["ts"] if r["ts"] is not None else last.get("utc_ts")
                            last["valid"] = (r.get("status","") == "A")

                    elif typ == "GGA":
                        g = parse_gga(fields)
                        if g:
                            last["lat"] = g["lat"] if g["lat"] is not None else last["lat"]
                            last["lon"] = g["lon"] if g["lon"] is not None else last["lon"]
                            last["alt"] = g["alt"] if g["alt"] is not None else last["alt"]
                            last["hdop"] = g["hdop"] if g["hdop"] is not None else last["hdop"]
                            last["numsats"] = g["numsats"] if g["numsats"] is not None else last["numsats"]
                            fixq = g.get("fix")
                            if fixq is not None and fixq > 0:
                                last["valid"] = True

                    elif typ == "GSA":
                        s = parse_gsa(fields)
                        if s:
                            last["fix_dim"] = s.get("mode2")
                            last["hdop"] = s.get("hdop") if s.get("hdop") is not None else last["hdop"]
                            last["sats_used"] = s.get("used")
            
            if last["lat"] is not None:
                CAM_LAT = last["lat"]
            if last["lon"] is not None:
                CAM_LON = last["lon"]

            try:
                if ret:
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

                    # Draw at fixed rate
                    now = time.time()
                    if now - last_draw >= 1.0/REFRESH_HZ:
                        draw(last)
                        last_draw = now
                    
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
            ser.close()
            cam.release()
            cv.destroyAllWindows()
        except Exception:
            pass

if __name__ == "__main__":
    main()