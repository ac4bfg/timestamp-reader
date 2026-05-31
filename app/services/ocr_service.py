import os
import re
import io
import time
from PIL import Image
import pytesseract
import numpy as np

# ── Konfigurasi ──────────────────────────────────────────────────────────────

# Sesuaikan path ini jika Tesseract tidak terinstall di lokasi default (bisa diatur via ENV nanti)
TESSERACT_PATH = os.environ.get("TESSERACT_PATH", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Area crop: ambil X% kanan dan Y% bawah dari gambar
CROP_RIGHT_FRACTION  = 0.65
CROP_RIGHT_STEP      = 0.35
CROP_RIGHT_MAX       = 1.00
CROP_BOTTOM_FRACTION = 0.35

# Dynamic threshold untuk RGB pure white detection
THRESHOLD_START = 252
THRESHOLD_MIN   = 200
THRESHOLD_STEP  = 8

# ── Regex pola koordinat ──────────────────────────────────────────────────────

COORD_PATTERNS = [
    re.compile(r'(\d{1,3}[,\.]\d{1,20}\s*[SsNn])\s+(\d{1,3}[,\.]\d{1,20}\s*[EeBbWw])', re.IGNORECASE),
    re.compile(r'(\d{1,3}[,\.]\d{2,20})\s*([SsNn])\s+(\d{1,3}[,\.]\d{2,20})\s*([EeBbWw])', re.IGNORECASE),
    re.compile(r'(\d{1,2}[,\.]\d{2,20})\s+(\d{2,3}[,\.]\d{2,20}\s*[EeBbWw])', re.IGNORECASE),
    re.compile(r'\b(\d{1,2}[,\.]\d{2,20})\s+(\d{3}[,\.]\d{2,20})\b'),
    re.compile(r'\b(69\d{2,8})\s*[SsNn]?\s+(110)[,\.]?(\d{2,8})\s*[EeBbWw]?'),
    re.compile(r'6[,\.]9\d{1,20}\s*[SsNn]?\s+110[,\.]\d{2,20}\s*[EeBbWw]?'),
]

LAT_INDONESIA_MIN = 6.0
LAT_INDONESIA_MAX = 11.0
LON_INDONESIA_MIN = 95.0
LON_INDONESIA_MAX = 141.0

TIMESTAMP_PATTERN = re.compile(r'(\d{1,2}\s+[A-Za-z]+\s+\d{4})(?:\s+\d{1,2}[.:]\d{2}(?:[.:]\d{2})?)?', re.IGNORECASE)

LAT_MIN, LAT_MAX = -7.20, -6.80
LON_MIN, LON_MAX = 109.90, 110.25
COORD_DECIMALS = 4

# ─────────────────────────────────────────────────────────────────────────────

def validate_coord(lat: str, lon: str) -> tuple[str, str, str]:
    try:
        lat_f = float(lat)
        lon_f = float(lon)
    except (ValueError, TypeError):
        return lat, lon, ""

    lat_r = round(lat_f, COORD_DECIMALS)
    lon_r = round(lon_f, COORD_DECIMALS)
    lat_out = f"{lat_r:.{COORD_DECIMALS}f}"
    lon_out = f"{lon_r:.{COORD_DECIMALS}f}"

    lat_valid = LAT_MIN <= lat_r <= LAT_MAX
    lon_valid = LON_MIN <= lon_r <= LON_MAX

    if not (lat_valid and lon_valid):
        if lon_valid:
            lat_str = str(abs(lat_r))
            for skip in range(1, 3):
                candidate = lat_str[skip:]
                if candidate.startswith("."):
                    candidate = "6" + candidate
                try:
                    candidate_f = -abs(float(candidate))
                    if LAT_MIN <= candidate_f <= LAT_MAX:
                        corrected = f"{candidate_f:.{COORD_DECIMALS}f}"
                        return corrected, lon_out, f"Lat dikoreksi otomatis dari {lat_r} → {corrected} (noise OCR)"
                except ValueError:
                    continue
            warning = f"Lat di luar area proyek (lat={lat_r}, lon={lon_r}) — periksa manual"
            return lat_out, lon_out, warning
        warning = f"Koordinat keduanya di luar area proyek (lat={lat_r}, lon={lon_r}) — kemungkinan OCR error, dikosongkan"
        return "", "", warning
    return lat_out, lon_out, ""

def normalize_coord(value: str, direction: str) -> str:
    value = value.replace(",", ".")
    direction = direction.upper()
    if direction in ("S", "W"):
        return f"-{value}"
    return value

def extract_from_text(text: str) -> dict:
    result = {"latitude": "", "longitude": "", "timestamp": "", "raw_text": text.strip(), "warning": ""}

    cleaned = text.replace("°", "")
    cleaned = re.sub(r'[oO](?=\d)', "0", cleaned)
    cleaned = re.sub(r'(?<=\d)[oO]', "0", cleaned)
    cleaned = re.sub(r'[Qq](?=\d)', "0", cleaned)
    cleaned = re.sub(r'(?<=\d)\)', ",", cleaned)
    cleaned = re.sub(r"['’ʼ]", "", cleaned)
    cleaned = re.sub(r'\bI[Oo0]C\b', "110", cleaned)
    cleaned = re.sub(r'\bI[Oo0]0\b', "110", cleaned)
    cleaned = re.sub(r'\b1[Oo]C\b', "110", cleaned)
    cleaned = re.sub(r'\b1[Oo]0\b', "110", cleaned)
    cleaned = re.sub(r'(\d+[,\.])\s+(\d+)', r'\1\2', cleaned)
    cleaned = re.sub(r'(\d+\.\d+)\s+\.(\d+)', r'\1\2', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)

    for pattern in COORD_PATTERNS:
        m = pattern.search(cleaned)
        if m:
            if len(m.groups()) == 2:
                raw_lat = m.group(1).strip()
                raw_lon = m.group(2).strip()
                lat_num = re.sub(r'[A-Za-z\s]', '', raw_lat)
                lat_dir = re.search(r'[NSns]', raw_lat)
                lon_num = re.sub(r'[A-Za-z\s]', '', raw_lon)
                lon_dir = re.search(r'[EWBbew]', raw_lon)

                if lat_dir:
                    result["latitude"] = normalize_coord(lat_num, lat_dir.group())
                else:
                    try:
                        lat_f = float(lat_num.replace(",", "."))
                        if LAT_INDONESIA_MIN <= lat_f <= LAT_INDONESIA_MAX:
                            result["latitude"] = f"-{lat_num.replace(',', '.')}"
                        else:
                            result["latitude"] = lat_num.replace(",", ".")
                    except ValueError:
                        result["latitude"] = lat_num.replace(",", ".")

                if lon_dir:
                    result["longitude"] = normalize_coord(lon_num, lon_dir.group())
                else:
                    result["longitude"] = lon_num.replace(",", ".")
            elif len(m.groups()) == 3:
                lat_raw = m.group(1)
                lon_int = m.group(2)
                lon_dec = m.group(3)
                lat_str = f"{lat_raw[0]}.{lat_raw[1:5]}" if len(lat_raw) >= 5 else f"{lat_raw[0]}.{lat_raw[1:]}"
                result["latitude"]  = f"-{lat_str}"
                result["longitude"] = f"{lon_int}.{lon_dec}"
            elif len(m.groups()) == 4:
                result["latitude"]  = normalize_coord(m.group(1), m.group(2))
                result["longitude"] = normalize_coord(m.group(3), m.group(4))
            else:
                raw = m.group(0)
                parts = raw.strip().split()
                if len(parts) >= 2:
                    raw_lat = parts[0]
                    raw_lon = parts[-1]
                    lat_num = re.sub(r'[A-Za-z]', '', raw_lat).replace(",", ".")
                    lat_dir = re.search(r'[NSns]', raw_lat)
                    lon_num = re.sub(r'[A-Za-z]', '', raw_lon).replace(",", ".")
                    try:
                        lat_f2 = float(lat_num)
                        result["latitude"] = f"-{lat_num}" if (lat_dir and lat_dir.group().upper() == 'S') or (not lat_dir and LAT_INDONESIA_MIN <= lat_f2 <= LAT_INDONESIA_MAX) else lat_num
                    except ValueError:
                        result["latitude"] = lat_num
                    result["longitude"] = lon_num
            break

    m_ts = TIMESTAMP_PATTERN.search(cleaned)
    if m_ts:
        result["timestamp"] = m_ts.group(1).strip()

    if result["latitude"] and result["longitude"]:
        lat, lon, warning = validate_coord(result["latitude"], result["longitude"])
        result["latitude"]  = lat
        result["longitude"] = lon
        result["warning"]   = warning

    return result

def preprocess(img: Image.Image, threshold: int = THRESHOLD_START) -> Image.Image:
    img = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)
    arr = np.array(img.convert("RGB"))

    try:
        import cv2
        gray = np.array(img.convert("L"))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        tophat_mask = tophat > 15
    except ImportError:
        tophat_mask = np.ones(arr.shape[:2], dtype=bool)
        cv2 = None

    rgb_mask = (arr[..., 0] >= threshold) & (arr[..., 1] >= threshold) & (arr[..., 2] >= threshold)
    combined_mask = tophat_mask & rgb_mask
    fg = combined_mask.astype(np.uint8) * 255

    if cv2 is not None:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    try:
        from scipy import ndimage
        labeled, num = ndimage.label(fg > 128)
        if num > 0:
            sizes = ndimage.sum(fg > 128, labeled, range(1, num + 1))
            bboxes = ndimage.find_objects(labeled)
            total = fg.shape[0] * fg.shape[1]
            MIN_BLOB = 30
            MAX_BLOB = total * 0.05
            MAX_ASPECT = 8.0

            keep = np.zeros(num + 1, dtype=bool)
            for i, (s, bbox) in enumerate(zip(sizes, bboxes), start=1):
                if s < MIN_BLOB or s > MAX_BLOB:
                    continue
                h_b = bbox[0].stop - bbox[0].start
                w_b = bbox[1].stop - bbox[1].start
                if h_b > 0 and (w_b / h_b > MAX_ASPECT or h_b / w_b > MAX_ASPECT):
                    continue
                keep[i] = True
            fg = np.where(keep[labeled], 255, 0).astype(np.uint8)
    except ImportError:
        pass

    combined = 255 - fg
    from PIL import Image as PILImage
    return PILImage.fromarray(combined, mode='L')

def crop_bottom_right(img: Image.Image, right_fraction: float = CROP_RIGHT_FRACTION) -> Image.Image:
    w, h = img.size
    left = int(w * (1 - right_fraction))
    top  = int(h * (1 - CROP_BOTTOM_FRACTION))
    return img.crop((left, top, w, h))

def crop_coord_line(img: Image.Image, right_fraction: float = CROP_RIGHT_FRACTION) -> Image.Image:
    w, h = img.size
    left = int(w * (1 - right_fraction))
    top  = int(h * 0.78)
    bot  = int(h * 0.88)
    return img.crop((left, top, w, bot))

def build_right_fractions() -> list[float]:
    fracs = []
    rf = CROP_RIGHT_FRACTION
    while rf <= CROP_RIGHT_MAX + 1e-9:
        fracs.append(round(rf, 2))
        rf += CROP_RIGHT_STEP
    if not fracs or fracs[-1] < CROP_RIGHT_MAX:
        fracs.append(CROP_RIGHT_MAX)
    return fracs

def detect_watermark_bbox(img: Image.Image, threshold: int = THRESHOLD_START) -> tuple | None:
    from PIL import ImageChops, ImageFilter
    w, h = img.size
    sx, sy = int(w * 0.50), int(h * 0.50)
    search = img.crop((sx, sy, w, h))
    sw, sh = search.size

    scaled = search.resize((sw * 3, sh * 3), Image.LANCZOS)
    r, g, b = scaled.split()
    min_rg  = ImageChops.darker(r, g)
    min_rgb = ImageChops.darker(min_rg, b)
    binary  = min_rgb.point(lambda px: 0 if px > threshold else 255)
    binary  = binary.filter(ImageFilter.MinFilter(3))

    arr = np.array(binary)
    black = (arr == 0)
    row_counts = np.sum(black, axis=1)
    total_rows = arr.shape[0]

    min_pixels = arr.shape[1] * 0.05
    text_rows  = np.where(row_counts > min_pixels)[0]

    if len(text_rows) < 5:
        return None

    gaps = np.diff(text_rows)
    big_gap = np.where(gaps > total_rows * 0.08)[0]

    if len(big_gap) > 0:
        last_cluster_start = text_rows[big_gap[-1] + 1]
        cluster_rows = text_rows[text_rows >= last_cluster_start]
    else:
        cluster_rows = text_rows

    if len(cluster_rows) < 3:
        return None

    row_top = int(cluster_rows[0])
    row_bot = int(cluster_rows[-1])

    cluster_mask = black[row_top:row_bot + 1, :]
    cols_with_text = np.any(cluster_mask, axis=0)
    col_indices = np.where(cols_with_text)[0]

    if len(col_indices) == 0:
        return None
    col_left = int(col_indices[0])

    pad_row = max(10, int(total_rows * 0.02))
    pad_col = max(10, int(arr.shape[1] * 0.01))

    row_top  = max(0, row_top  - pad_row)
    row_bot  = min(total_rows - 1, row_bot + pad_row)
    col_left = max(0, col_left - pad_col)

    scale = 3
    orig_left = sx + col_left // scale
    orig_top  = sy + row_top  // scale
    orig_bot  = sy + row_bot  // scale
    orig_left = min(orig_left, int(w * 0.40))

    return (orig_left, orig_top, w, orig_bot)

def smart_crop(img: Image.Image, threshold: int = THRESHOLD_START) -> Image.Image:
    bbox = detect_watermark_bbox(img, threshold)
    if bbox:
        return img.crop(bbox)
    return crop_bottom_right(img)

def ocr_image(img: Image.Image, psm: int = 4) -> str:
    config = f"--psm {psm} --oem 3 -l eng"
    return pytesseract.image_to_string(img, config=config)

def ocr_with_dynamic_threshold(img: Image.Image) -> tuple[str, int]:
    last_text = ""
    t = THRESHOLD_START
    while t >= THRESHOLD_MIN:
        processed = preprocess(img, threshold=t)
        for psm in (4, 6, 11):
            text = ocr_image(processed, psm=psm)
            parsed = extract_from_text(text)
            if parsed["latitude"]:
                return text, t
        last_text = text
        t -= THRESHOLD_STEP
    return last_text, THRESHOLD_MIN

def process_image_from_bytes(image_bytes: bytes) -> dict:
    result = {
        "timestamp": "",
        "latitude":  "",
        "longitude": "",
        "warning":   "",
        "error":     "",
    }
    try:
        img = Image.open(io.BytesIO(image_bytes))

        # Handle image rotation via EXIF if any
        try:
            from PIL import ImageOps
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass

        right_fractions = build_right_fractions()

        # Tahap 1: fixed crop
        for rf in right_fractions:
            crop1 = crop_bottom_right(img, right_fraction=rf)
            text1, t1 = ocr_with_dynamic_threshold(crop1)
            parsed1 = extract_from_text(text1)
            if parsed1["latitude"]:
                result.update(parsed1)
                return result

        # Tahap 2: smart crop
        crop2 = smart_crop(img)
        text2, t2 = ocr_with_dynamic_threshold(crop2)
        parsed2 = extract_from_text(text2)
        if parsed2["latitude"]:
            result.update(parsed2)
            return result

        # Tahap 3: strip koordinat
        for rf in right_fractions:
            strip = crop_coord_line(img, right_fraction=rf)
            text3, t3 = ocr_with_dynamic_threshold(strip)
            parsed3 = extract_from_text(text3)
            if parsed3["latitude"]:
                result.update(parsed3)
                return result

    except Exception as e:
        result["error"] = str(e)
    
    return result
