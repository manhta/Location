import serial
import socket
import re
import sys
import time
import platform
from datetime import datetime, timezone
from nmea import parse_fields, parse_gga, parse_gll, parse_gsa, parse_rmc, parse_vtg, deg_to_dms, nmea_checksum_ok


## Set the port and baud rate for the Arduino; switch the port to /dev/serial0 when using a Raspberry Pi
PORT = "COM5" if platform.system() == "Windows" else "/dev/serial0"
BAUD = 115200
TIMEOUT = 0.5
REFRESH_HZ = 1.0

# Set socket port 
HOST_ADDR = "localhost"
SOCKET_PORT = 5000

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

def connect_serial() -> serial.Serial:
    try:
        ser = serial.Serial(port=PORT, baudrate=BAUD, timeout=TIMEOUT)
        print(f"Serial port opened successfully!")
        return ser
    except Exception as e:
        print(f"Cannot open {PORT} @ {BAUD}: {e}, check connection!")
        print("\n\033[93mExit!\033[0m")
        sys.exit(1)

def connect_socket() -> socket.socket:
    while True:
        try: 
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)
            s.connect((HOST_ADDR, SOCKET_PORT))
            print("Connected to socket server!")
            s.settimeout(None)
            return s
        except Exception as e:
            print("Waiting for socket server open...", e)
            time.sleep(1)

def main():
    # Camera latitude and longitude
    CAM_LAT = None
    CAM_LON = None

    client = connect_socket()
    ser = connect_serial()
    time.sleep(3)
    ser.reset_input_buffer()
    print("Start reading gps signal!")

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
                        gsa = parse_gsa(fields)
                        if gsa:
                            last["fix_dim"] = gsa.get("mode2")
                            last["hdop"] = gsa.get("hdop") if gsa.get("hdop") is not None else last["hdop"]
                            last["sats_used"] = gsa.get("used")

            CAM_LAT = last["lat"] if last["lat"] is not None else CAM_LAT
            CAM_LON = last["lon"] if last["lon"] is not None else CAM_LON
            if CAM_LAT is not None and CAM_LON is not None:
                location = f"({CAM_LAT:.5f}, {CAM_LON:.5f})"
            else:
                location = "NO FIX"
            
            # Send location to server and display location on terminal
            now = time.time()
            if now - last_draw >= 1.0 / REFRESH_HZ:
                draw(state=last)
                try:
                    client.sendall(location.encode(encoding="utf-8"))
                except (BrokenPipeError, OSError):
                    print("Server closed connection. Client exiting.")
                    break
                last_draw = now

    except KeyboardInterrupt:
        print("\n\033[93mFinish, keyboard interrupt!\033[0m")
    
    finally:
        try:
            ser.close()
            client.close()
            print("Serial port close and disconnect to socket server")
        except Exception:
            pass


if __name__ == "__main__":
    main()