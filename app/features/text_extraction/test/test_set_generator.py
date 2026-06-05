import os
import re
import urllib.request
from PIL import Image, ImageDraw, ImageFont

GUTENBERG_URL = "https://www.gutenberg.org/files/1342/1342-0.txt"  # Pride and Prejudice
OUTPUT_DIR    = "dataset"
IMAGES_DIR    = os.path.join(OUTPUT_DIR, "images")
GT_DIR        = os.path.join(OUTPUT_DIR, "ground_truth")

PAGE_W        = 1240
PAGE_H        = 1754
MARGIN        = 100
FONT_SIZE     = 26
LINE_SPACING  = 12
NUM_PAGES     = 30

FONT_CANDIDATES = [
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/Library/Fonts/Times New Roman.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
]


def load_font():
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            print(f"  Font: {path}")
            return ImageFont.truetype(path, FONT_SIZE)
    print("  Font: Pillow default (no system font found)")
    return ImageFont.load_default(size=FONT_SIZE)


def fetch_text(url):
    with urllib.request.urlopen(url) as response:
        raw = response.read().decode("utf-8", errors="ignore")

    start = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG", raw, re.IGNORECASE)
    end   = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG",   raw, re.IGNORECASE)
    if start:
        raw = raw[start.end():]
    if end:
        raw = raw[:end.start()]

    raw = re.sub(r"\r\n|\r", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def wrap_paragraph(para, font, max_width, draw):
    words = para.split()
    lines, current = [], []
    for word in words:
        candidate = " ".join(current + [word])
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def build_line_stream(text, font, max_width, draw):
    for para in text.split("\n\n"):
        para = para.strip().replace("\n", " ")
        if not para:
            continue
        yield from wrap_paragraph(para, font, max_width, draw)
        yield ""


def paginate(text, font, max_width, max_lines, draw):
    stream   = list(build_line_stream(text, font, max_width, draw))
    pages    = []
    i        = 0
    while i < len(stream) and len(pages) < NUM_PAGES:
        chunk = stream[i:i + max_lines]
        while chunk and chunk[-1] == "":
            chunk.pop()
        if chunk:
            pages.append(chunk)
        i += max_lines
    return pages


def render_page(lines, font):
    img  = Image.new("RGB", (PAGE_W, PAGE_H), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    line_h = FONT_SIZE + LINE_SPACING
    y = MARGIN
    for line in lines:
        if line:
            draw.text((MARGIN, y), line, font=font, fill=(0, 0, 0))
        y += line_h
    return img


def save_ground_truth(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(GT_DIR, exist_ok=True)

    print("Loading font...")
    font = load_font()

    print("Fetching text from Project Gutenberg...")
    text = fetch_text(GUTENBERG_URL)

    max_width  = PAGE_W - 2 * MARGIN
    line_h     = FONT_SIZE + LINE_SPACING
    max_lines  = (PAGE_H - 2 * MARGIN) // line_h

    tmp_img  = Image.new("RGB", (PAGE_W, PAGE_H))
    tmp_draw = ImageDraw.Draw(tmp_img)

    print("Paginating...")
    pages = paginate(text, font, max_width, max_lines, tmp_draw)

    print(f"Rendering {len(pages)} pages...\n")
    for i, lines in enumerate(pages):
        page_id    = f"page_{i + 1:03d}"
        image_path = os.path.join(IMAGES_DIR, f"{page_id}.png")
        gt_path    = os.path.join(GT_DIR,     f"{page_id}.txt")

        render_page(lines, font).save(image_path)
        save_ground_truth(lines, gt_path)
        print(f"  {page_id}  ({len(lines)} lines)")

    print(f"\nDone. Dataset saved to '{OUTPUT_DIR}/'")
    print(f"  {IMAGES_DIR}/  — {len(pages)} PNG images")
    print(f"  {GT_DIR}/  — {len(pages)} ground truth .txt files")


if __name__ == "__main__":
    main()