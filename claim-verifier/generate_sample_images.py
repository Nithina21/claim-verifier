#!/usr/bin/env python3
"""
generate_sample_images.py

Generates synthetic (but visually plausible) demo images for a handful of
sample claims, so the claim-verifier CLI can be run end-to-end without
needing real customer photos.

These are simple procedurally-drawn illustrations (not photos) — good
enough for Claude's vision model to reason about shapes, marks, and
described damage, and good enough to demo SUPPORTED / CONTRADICTED /
INSUFFICIENT outcomes.

Run:
    python generate_sample_images.py
"""

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

random.seed(42)

OUT_ROOT = Path(__file__).parent / "images"


def base_canvas(w=800, h=600, color=(235, 235, 238)):
    img = Image.new("RGB", (w, h), color)
    return img, ImageDraw.Draw(img)


def add_noise_texture(img, intensity=6):
    """Light grain so images don't look perfectly flat / synthetic to the model."""
    px = img.load()
    w, h = img.size
    for _ in range(w * h // 25):
        x, y = random.randint(0, w - 1), random.randint(0, h - 1)
        r, g, b = px[x, y]
        n = random.randint(-intensity, intensity)
        px[x, y] = (max(0, min(255, r + n)), max(0, min(255, g + n)), max(0, min(255, b + n)))
    return img


def label(draw, text, xy=(20, 20), color=(90, 90, 95)):
    draw.text(xy, text, fill=color)


# ---------------------------------------------------------------------------
# CAR scenes
# ---------------------------------------------------------------------------

def car_silhouette(draw, body_color=(60, 90, 150), ox=80, oy=200):
    """Draw a simple side-profile car body as a base scene."""
    # body
    draw.rounded_rectangle([ox, oy, ox + 600, oy + 140], radius=30, fill=body_color)
    # cabin
    draw.polygon([(ox + 130, oy), (ox + 220, oy - 90), (ox + 420, oy - 90), (ox + 480, oy)], fill=body_color)
    # windows
    draw.polygon([(ox + 150, oy - 5), (ox + 230, oy - 80), (ox + 290, oy - 80), (ox + 280, oy - 5)], fill=(200, 220, 230))
    draw.polygon([(ox + 300, oy - 5), (ox + 300, oy - 80), (ox + 405, oy - 80), (ox + 440, oy - 5)], fill=(200, 220, 230))
    # wheels
    for wx in (ox + 130, ox + 470):
        draw.ellipse([wx - 45, oy + 110, wx + 45, oy + 200], fill=(25, 25, 28))
        draw.ellipse([wx - 22, oy + 133, wx + 22, oy + 177], fill=(170, 170, 175))
    # front bumper block
    draw.rounded_rectangle([ox - 30, oy + 90, ox + 20, oy + 150], radius=10, fill=(40, 40, 45))
    # rear bumper block
    draw.rounded_rectangle([ox + 580, oy + 90, ox + 630, oy + 150], radius=10, fill=(40, 40, 45))
    # headlight
    draw.ellipse([ox - 10, oy + 30, ox + 30, oy + 65], fill=(255, 248, 220))
    # taillight
    draw.ellipse([ox + 590, oy + 30, ox + 625, oy + 65], fill=(200, 40, 40))
    # side mirror
    draw.rectangle([ox + 200, oy - 15, ox + 215, oy + 5], fill=(30, 30, 32))


def img_case001_front_bumper_headlight_damage():
    """SUPPORTED case: front bumper dent + left headlight damage clearly visible (close-up)."""
    img, d = base_canvas(800, 600, (210, 213, 218))
    car_silhouette(d, ox=60, oy=230)
    # zoom emphasis box around front area
    d.rectangle([10, 250, 220, 420], outline=(200, 30, 30), width=3)
    # dent: irregular dark crescent on bumper
    d.ellipse([10, 320, 70, 360], fill=(35, 35, 38))
    d.ellipse([20, 325, 55, 350], fill=(20, 20, 22))
    # crack lines radiating from dent
    for ang in (20, 55, 100, 150):
        x2 = 40 + 35 * math.cos(math.radians(ang))
        y2 = 340 + 35 * math.sin(math.radians(ang))
        d.line([40, 340, x2, y2], fill=(15, 15, 18), width=2)
    # damaged headlight: cracked lens, darker fill, jagged lines
    d.ellipse([45, 255, 90, 295], fill=(180, 180, 150))
    for _ in range(6):
        x1, y1 = random.randint(48, 88), random.randint(258, 292)
        x2, y2 = random.randint(48, 88), random.randint(258, 292)
        d.line([x1, y1, x2, y2], fill=(90, 90, 70), width=1)
    label(d, "Front-left: dent + cracked headlight (close-up)", (10, 20))
    add_noise_texture(img)
    return img


def img_case001_full_view():
    img, d = base_canvas(800, 600, (220, 222, 226))
    car_silhouette(d, ox=80, oy=200)
    label(d, "Full vehicle context shot", (10, 20))
    add_noise_texture(img)
    return img


def img_case003_door_dent():
    """SUPPORTED: deep dent on door panel."""
    img, d = base_canvas(800, 600, (215, 218, 222))
    # door panel close-up (flat panel, not full car)
    d.rounded_rectangle([100, 100, 700, 500], radius=20, fill=(60, 90, 150))
    d.line([100, 300, 700, 300], fill=(40, 65, 110), width=3)  # body crease line
    # deep dent: dark concentric shading
    cx, cy = 420, 320
    for i, r in enumerate([90, 70, 50, 30]):
        shade = 50 - i * 8
        d.ellipse([cx - r, cy - r * 0.6, cx + r, cy + r * 0.6], fill=(shade, shade + 15, shade + 50))
    label(d, "Door panel close-up: deep dent visible", (10, 20))
    add_noise_texture(img)
    return img


def img_case999_pristine_panel():
    """CONTRADICTED demo: claimed dent, but panel is actually undamaged/clean."""
    img, d = base_canvas(800, 600, (215, 218, 222))
    d.rounded_rectangle([100, 100, 700, 500], radius=20, fill=(60, 90, 150))
    d.line([100, 300, 700, 300], fill=(40, 65, 110), width=3)
    # subtle clean highlight/reflection, no dent shading at all
    d.polygon([(200, 150), (260, 150), (220, 450), (160, 450)], fill=(110, 140, 195))
    label(d, "Door panel close-up: smooth, no deformation visible", (10, 20))
    add_noise_texture(img)
    return img


def img_case_insufficient_dark():
    """INSUFFICIENT demo: very dark / blurry image where part can't be assessed."""
    img, d = base_canvas(800, 600, (25, 25, 28))
    car_silhouette(d, ox=80, oy=200, body_color=(15, 20, 35))
    img = img.filter(ImageFilter.GaussianBlur(radius=6))
    d2 = ImageDraw.Draw(img)
    label(d2, "Low-light parking photo", (10, 20), color=(120, 120, 125))
    return img


# ---------------------------------------------------------------------------
# LAPTOP scenes
# ---------------------------------------------------------------------------

def laptop_base(draw, ox=150, oy=150, w=500, h=320, screen_fill=(40, 45, 60)):
    # base/keyboard deck
    draw.rounded_rectangle([ox, oy + h, ox + w, oy + h + 40], radius=8, fill=(170, 170, 175))
    # screen
    draw.rounded_rectangle([ox, oy, ox + w, oy + h], radius=8, fill=(20, 20, 22))
    draw.rounded_rectangle([ox + 15, oy + 15, ox + w - 15, oy + h - 15], radius=4, fill=screen_fill)


def img_case017_cracked_screen():
    """SUPPORTED: laptop screen with a clear crack pattern."""
    img, d = base_canvas(800, 600, (230, 230, 232))
    laptop_base(d, screen_fill=(70, 110, 160))
    # crack: spiderweb pattern from an impact point
    cx, cy = 420, 260
    d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(10, 10, 12))
    for ang in range(0, 360, 30):
        length = random.randint(60, 140)
        x2 = cx + length * math.cos(math.radians(ang))
        y2 = cy + length * math.sin(math.radians(ang))
        d.line([cx, cy, x2, y2], fill=(15, 15, 18), width=2)
        # secondary branch
        bx = cx + (length * 0.6) * math.cos(math.radians(ang))
        by = cy + (length * 0.6) * math.sin(math.radians(ang))
        bx2 = bx + 30 * math.cos(math.radians(ang + 25))
        by2 = by + 30 * math.sin(math.radians(ang + 25))
        d.line([bx, by, bx2, by2], fill=(15, 15, 18), width=1)
    label(d, "Laptop screen close-up: crack visible", (10, 20))
    add_noise_texture(img)
    return img


def img_case017_full_laptop():
    img, d = base_canvas(800, 600, (235, 235, 238))
    laptop_base(d, screen_fill=(70, 110, 160))
    label(d, "Full laptop context shot", (10, 20))
    add_noise_texture(img)
    return img


def img_case028_hinge_damage():
    """SUPPORTED: hinge area visibly broken / gapped."""
    img, d = base_canvas(800, 600, (228, 228, 230))
    laptop_base(d, oy=180, h=280, screen_fill=(50, 55, 70))
    # zoom into hinge area: screen sitting at a wrong/skewed angle, gap visible
    hinge_x = 150 + 250
    d.line([hinge_x - 40, 180 + 280, hinge_x + 40, 180 + 280], fill=(90, 90, 95), width=6)
    # crack/gap at hinge
    d.polygon([(hinge_x - 20, 460), (hinge_x + 20, 460), (hinge_x + 10, 478), (hinge_x - 10, 478)], fill=(15, 15, 16))
    for i in range(5):
        x = hinge_x - 20 + i * 10
        d.line([x, 460, x + 4, 478], fill=(5, 5, 5), width=1)
    label(d, "Hinge close-up: visible gap / misalignment", (10, 20))
    add_noise_texture(img)
    return img


def img_case_laptop_keyboard_normal():
    """CONTRADICTED demo: claim is screen crack, but image shows an intact keyboard, no screen visible damage context given."""
    img, d = base_canvas(800, 600, (232, 232, 235))
    laptop_base(d, screen_fill=(60, 100, 150))
    # draw keyboard grid (intact, no damage)
    kx, ky = 180, 480
    for row in range(4):
        for col in range(12):
            x = kx + col * 35
            y = ky + row * 22
            d.rounded_rectangle([x, y, x + 28, y + 16], radius=3, fill=(60, 60, 65))
    label(d, "Keyboard close-up: keys intact, no screen visible", (10, 20))
    add_noise_texture(img)
    return img


# ---------------------------------------------------------------------------
# PACKAGE scenes
# ---------------------------------------------------------------------------

def box_base(draw, ox=150, oy=150, w=500, h=350, fill=(180, 140, 90)):
    draw.rectangle([ox, oy, ox + w, oy + h], fill=fill)
    # tape line
    draw.rectangle([ox + w // 2 - 15, oy, ox + w // 2 + 15, oy + h], fill=(200, 195, 170))
    # shipping label
    draw.rectangle([ox + 40, oy + 40, ox + 220, oy + 140], fill=(245, 245, 245))
    for i in range(5):
        d_y = oy + 55 + i * 16
        draw.line([ox + 55, d_y, ox + 200, d_y], fill=(80, 80, 80), width=3)


def img_case029_corner_crushed():
    """SUPPORTED: package corner visibly crushed/caved in."""
    img, d = base_canvas(800, 600, (225, 225, 228))
    box_base(d)
    # crushed corner: jagged inward dent at bottom-right corner
    ox, oy, w, h = 150, 150, 500, 350
    crushed_pts = [
        (ox + w - 10, oy + h),
        (ox + w - 90, oy + h - 15),
        (ox + w - 60, oy + h - 70),
        (ox + w - 110, oy + h - 110),
        (ox + w - 20, oy + h - 130),
        (ox + w, oy + h - 40),
    ]
    d.polygon(crushed_pts, fill=(140, 105, 65))
    for i in range(4):
        x1, y1 = crushed_pts[i]
        x2, y2 = crushed_pts[i + 1]
        d.line([x1, y1, x2, y2], fill=(90, 65, 35), width=3)
    label(d, "Package corner close-up: crushed / caved in", (10, 20))
    add_noise_texture(img)
    return img


def img_case030_seal_torn():
    """SUPPORTED: seal/tape torn open."""
    img, d = base_canvas(800, 600, (225, 225, 228))
    box_base(d)
    ox, oy, w, h = 150, 150, 500, 350
    # torn tape: jagged gap down the middle seam
    midx = ox + w // 2
    pts_left = [(midx - 15, oy)]
    pts_right = [(midx + 15, oy)]
    y = oy
    while y < oy + h:
        y += 25
        pts_left.append((midx - 15 + random.randint(-12, 12), y))
        pts_right.append((midx + 15 + random.randint(-12, 12), y))
    d.polygon(pts_left + pts_right[::-1], fill=(70, 50, 30))
    label(d, "Package seal close-up: torn open along seam", (10, 20))
    add_noise_texture(img)
    return img


def img_case_package_intact():
    """INSUFFICIENT / CONTRADICTED demo: claim is water damage but box looks dry & intact, label legible."""
    img, d = base_canvas(800, 600, (225, 225, 228))
    box_base(d, fill=(190, 150, 100))
    label(d, "Package full view: dry, label legible, no visible staining", (10, 20))
    add_noise_texture(img)
    return img


# ---------------------------------------------------------------------------
# Manifest: maps case folder -> list of (filename, generator function)
# ---------------------------------------------------------------------------

MANIFEST = {
    # CAR — SUPPORTED: front bumper dent + left headlight damage (matches claims.csv case_001)
    "test/case_001": [
        ("img_1.jpg", img_case001_front_bumper_headlight_damage),
        ("img_2.jpg", img_case001_full_view),
    ],
    # CAR — SUPPORTED: deep door dent (matches claims.csv case_003)
    "test/case_003": [
        ("img_1.jpg", img_case003_door_dent),
    ],
    # CAR — CONTRADICTED demo: claim says "dent" but panel is undamaged
    "test/case_003_contradicted_demo": [
        ("img_1.jpg", img_case999_pristine_panel),
    ],
    # CAR — INSUFFICIENT demo: too dark to assess
    "test/case_insufficient_demo": [
        ("img_1.jpg", img_case_insufficient_dark),
    ],
    # LAPTOP — SUPPORTED: cracked screen (matches claims.csv case_017)
    "test/case_017": [
        ("img_1.jpg", img_case017_cracked_screen),
        ("img_2.jpg", img_case017_full_laptop),
    ],
    # LAPTOP — SUPPORTED: hinge damage (matches claims.csv case_028)
    "test/case_028": [
        ("img_1.jpg", img_case028_hinge_damage),
    ],
    # LAPTOP — CONTRADICTED demo: claim says screen crack, image only shows intact keyboard
    "test/case_017_contradicted_demo": [
        ("img_1.jpg", img_case_laptop_keyboard_normal),
    ],
    # PACKAGE — SUPPORTED: crushed corner (matches claims.csv case_029)
    "test/case_029": [
        ("img_1.jpg", img_case029_corner_crushed),
    ],
    # PACKAGE — SUPPORTED: torn seal (matches claims.csv case_030)
    "test/case_030": [
        ("img_1.jpg", img_case030_seal_torn),
    ],
    # PACKAGE — CONTRADICTED demo: claim says water damage, box looks dry/intact
    "test/case_water_contradicted_demo": [
        ("img_1.jpg", img_case_package_intact),
    ],
}


def main():
    OUT_ROOT.mkdir(exist_ok=True)
    total = 0
    for case_dir, files in MANIFEST.items():
        full_dir = OUT_ROOT / case_dir
        full_dir.mkdir(parents=True, exist_ok=True)
        for filename, gen_fn in files:
            img = gen_fn()
            out_path = full_dir / filename
            img.save(out_path, quality=88)
            total += 1
            print(f"  wrote {out_path.relative_to(OUT_ROOT.parent)}")
    print(f"\nGenerated {total} sample image(s) across {len(MANIFEST)} case folder(s).")


if __name__ == "__main__":
    main()
