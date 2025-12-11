from datetime import datetime, timezone
from zoneinfo import ZoneInfo

LOCAL_TIMEZONE = ZoneInfo("Asia/Ho_Chi_Minh")

# --------- NMEA helpers ---------
def nmea_checksum_ok(line: str) -> bool:
    if not line.startswith("$"):
        return False
    star = line.find("*")
    if star == -1:
        return False
    data = line[1:star]
    cs = line[star+1:star+3]
    calc = 0
    for ch in data:
        calc ^= ord(ch)
    try:
        return calc == int(cs, 16)
    except Exception:
        return False

def dm_to_deg(dm: str, hemi: str):
    if not dm or not hemi or "." not in dm:
        return None
    try:
        dot = dm.index(".")
        deg_len = dot - 2
        deg = int(dm[:deg_len])
        minutes = float(dm[deg_len:])
        val = deg + minutes/60.0
        if hemi in ("S","W"):
            val = -val
        return val
    except Exception:
        return None

def deg_to_dms(deg: float, is_lat=True):
    if deg is None:
        return None
    hemi = ("N","S") if is_lat else ("E","W")
    h = hemi[0] if deg >= 0 else hemi[1]
    d = abs(deg)
    D = int(d)
    M_full = (d - D) * 60.0
    M = int(M_full)
    S = (M_full - M) * 60.0
    return f"{D:02d}°{M:02d}'{S:05.2f}\" {h}"

# --------- Parsers ---------
def parse_fields(line):
    star = line.find("*")
    core = line[1:star] if star != -1 else line[1:]
    return core.split(",")

def parse_gll(fields):
    try:
        lat = dm_to_deg(fields[1], fields[2])
        lon = dm_to_deg(fields[3], fields[4])
        utc = fields[5] if len(fields) > 5 else ""
        status = fields[6] if len(fields) > 6 else ""
        mode = fields[7] if len(fields) > 7 else ""
        return {"lat": lat, "lon": lon, "utc": utc, "status": status, "mode": mode}
    except Exception:
        return None

def parse_vtg(fields):
    def tof(x):
        try: return float(x)
        except: return None
    try:
        cog_t = tof(fields[1])
        s_kn = tof(fields[5]) if len(fields) > 5 else None
        s_kmh = tof(fields[7]) if len(fields) > 7 else None
        mode = fields[9] if len(fields) > 9 else ""
        return {"cog": cog_t, "s_kn": s_kn, "s_kmh": s_kmh, "mode": mode}
    except Exception:
        return None

def parse_rmc(fields):
    def tof(x):
        try: return float(x)
        except: return None
    try:
        utc = fields[1]
        status = fields[2]
        lat = dm_to_deg(fields[3], fields[4])
        lon = dm_to_deg(fields[5], fields[6])
        sog_kn = tof(fields[7])
        cog = tof(fields[8])
        date = fields[9]
        ts = None
        if len(utc) >= 6 and len(date) == 6:
            try:
                ts = datetime.strptime(date + utc[:6], "%d%m%y%H%M%S").replace(tzinfo=timezone.utc)
            except Exception:
                ts = None
        return {"lat": lat, "lon": lon, "utc": utc, "status": status, "sog_kn": sog_kn, "cog": cog, "ts": ts}
    except Exception:
        return None

def parse_gga(fields):
    def tof(x):
        try: return float(x)
        except: return None
    def toi(x):
        try: return int(x)
        except: return None
    try:
        utc = fields[1]
        lat = dm_to_deg(fields[2], fields[3])
        lon = dm_to_deg(fields[4], fields[5])
        fix = toi(fields[6])
        numsats = toi(fields[7])
        hdop = tof(fields[8])
        alt = tof(fields[9])
        return {"lat": lat, "lon": lon, "utc": utc, "fix": fix, "numsats": numsats, "hdop": hdop, "alt": alt}
    except Exception:
        return None

def parse_gsa(fields):
    try:
        mode1 = fields[1]
        mode2 = fields[2]
        svs = fields[3:15]
        used = sum(1 for s in svs if s.strip())
        pdop = float(fields[15]) if len(fields) > 15 and fields[15] else None
        hdop = float(fields[16]) if len(fields) > 16 and fields[16] else None
        vdop = float(fields[17]) if len(fields) > 17 and fields[17] else None
        return {"mode1": mode1, "mode2": mode2, "used": used, "pdop": pdop, "hdop": hdop, "vdop": vdop}
    except Exception:
        return None

