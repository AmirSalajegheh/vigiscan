import cv2
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
import json

# مسیرها
MODEL_PATH = "Data/yolo9e.pt"
FONT_PATH = "Data/Titr.ttf"
VIDEO_PATH = "Data/3.mp4"
CONFIG_PATH = "config.json"

# خواندن تنظیمات از config.json
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    options = json.load(f)

# نمایش متن فارسی
def draw_farsi_text(frame, text, position, color, font, align_right=False):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    color_rgb = (color[2], color[1], color[0])

    if align_right:
        bbox = draw.textbbox((0, 0), bidi_text, font=font)
        text_width = bbox[2] - bbox[0]
        x, y = position
        x = x - text_width
        position = (x, y)

    draw.text(position, bidi_text, font=font, fill=color_rgb)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# بارگذاری مدل و آماده‌سازی
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO(MODEL_PATH).to(device)
font = ImageFont.truetype(FONT_PATH, 14)
CONFIDENCE_THRESHOLD = 0.05
panel_width = 200
colors = {
    "safe": (0, 150, 0),
    "partial": (0, 180, 255),
    "unsafe": (0, 0, 255)
}

# بارگذاری ویدیو
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter("output_gui.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width + panel_width, height))
results = model.track(source=VIDEO_PATH, device=device, stream=True, conf=CONFIDENCE_THRESHOLD, persist=True)

id_map = {}
next_id = 1

for r in results:
    frame = r.orig_img.copy()
    boxes = r.boxes
    names = model.names

    persons, helmets, gloves, glasses, vests = [], [], [], [], []

    for box in boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < CONFIDENCE_THRESHOLD:
            continue

        class_name = names[class_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        tid = int(box.id[0]) if box.id is not None else None
        info = {"coords": (x1, y1, x2, y2), "id": tid}

        if class_name == "person":
            persons.append(info)
        elif class_name == "helmet":
            helmets.append(info)
        elif class_name == "gloves":
            gloves.append(info)
        elif class_name == "glasses":
            glasses.append(info)
        elif class_name in ["safety-vest", "safety-suit"]:
            vests.append(info)

    status_list = []

    for person in persons:
        x1, y1, x2, y2 = person["coords"]
        tid = person["id"]
        if tid not in id_map:
            id_map[tid] = next_id
            next_id += 1
        pid = id_map[tid]

        top_area = (x1, y1, x2, y1 + (y2 - y1) // 4)

        def intersects(item_list, area, threshold):
            for item in item_list:
                ix1 = max(area[0], item["coords"][0])
                iy1 = max(area[1], item["coords"][1])
                ix2 = min(area[2], item["coords"][2])
                iy2 = min(area[3], item["coords"][3])
                iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
                inter_area = iw * ih
                item_area = (item["coords"][2] - item["coords"][0]) * (item["coords"][3] - item["coords"][1])
                if item_area > 0 and inter_area / item_area > threshold:
                    return True
            return False

        has_helmet = intersects(helmets, top_area, 0.3) if options.get("helmet", True) else True
        has_gloves = intersects(gloves, person["coords"], 0.8) if options.get("gloves", True) else True
        has_glasses = intersects(glasses, person["coords"], 0.4) if options.get("glasses", True) else True
        has_vest = intersects(vests, person["coords"], 0.4) if options.get("vest", True) or options.get("suit", True) else True

        checks = [has_helmet, has_gloves, has_glasses, has_vest]
        color = colors["safe"] if all(checks) else colors["unsafe"] if not any(checks) else colors["partial"]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        frame = draw_farsi_text(frame, f"ﻩﺭﺎﻤﺷ {pid}", (x1, y1 - 30), color, font)

        status_lines = [f"ﻩﺭﺎﻤﺷ {pid}"]
        if options.get("vest", True) or options.get("suit", True):
            status_lines.append("ﺩﺭﺍﺩ ﻪﻘﯿﻠﺟ" if has_vest else "ﺩﺭﺍﺪﻧ ﻪﻘﯿﻠﺟ")
        if options.get("glasses", True):
            status_lines.append("ﺩﺭﺍﺩ ﮏﻨﯿﻋ" if has_glasses else "ﺩﺭﺍﺪﻧ ﮏﻨﯿﻋ")
        if options.get("gloves", True):
            status_lines.append("ﺩﺭﺍﺩ ﺶﮐ‌ﺖﺳﺩ" if has_gloves else "ﺩﺭﺍﺪﻧ ﺶﮐ‌ﺖﺳﺩ")
        if options.get("helmet", True):
            status_lines.append("ﺩﺭﺍﺩ ﻩﻼﮐ" if has_helmet else "ﺩﺭﺍﺪﻧ ﻩﻼﮐ")

        status_list.append({"lines": status_lines, "color": color})

    display = np.full((height, width + panel_width, 3), 255, dtype=np.uint8)
    display[:, :width] = frame

    y_offset = 20
    for entry in status_list:
        for line in entry["lines"]:
            if "ﻩﺭﺎﻤﺷ" in line:  # برای عنوان (مثلاً شماره فرد)، از رنگ اصلی entry استفاده کن
                text_color = entry["color"]
            elif "ﺩﺭﺍﺩ" in line:
                text_color = (0, 150, 0)  # سبز
            elif "ﺩﺭﺍﺪﻧ" in line:
                text_color = (0, 0, 255)  # قرمز
            else:
                text_color = entry["color"]

            display = draw_farsi_text(display, line, (width + 10, y_offset), text_color, font)
            y_offset += 25
        y_offset += 10

    out.write(display)
    cv2.imshow("PPE Detection", display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("خروجی در فایل output_gui.mp4 ذخیره شد.")