"""Compose episode covers with an exact title and a checked center 3:4 crop."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

WIDTH, HEIGHT = 1080, 1920
CROP = (0, 240, 1080, 1680)


def face(font, size, index=0):
    return ImageFont.truetype(str(font), size, index=index)


def fit_font(value, path, size, width, index=0):
    while True:
        font = face(path, size, index)
        if font.getlength(value) <= width:
            return font
        size -= 2


def photograph(path, width, height, focus):
    with Image.open(path) as source:
        picture = source.convert("RGB")
    factor = max(width / picture.width, height / picture.height)
    picture = picture.resize((round(picture.width * factor), round(picture.height * factor)),
                             Image.Resampling.LANCZOS)
    left = min(picture.width - width, max(0, round(focus * picture.width - width / 2)))
    top = (picture.height - height) // 2
    return picture.crop((left, top, left + width, top + height))


def background(row, root):
    paths = [root / name for name in row["images"]]
    focus = row.get("focus", [.5] * len(paths))
    if len(paths) == 2:
        picture = Image.new("RGB", (WIDTH, HEIGHT))
        picture.paste(photograph(paths[0], 585, HEIGHT, focus[0]), (0, 0))
        second = photograph(paths[1], 592, HEIGHT, focus[1])
        mask = Image.new("L", second.size, 255)
        blend = Image.linear_gradient("L").transpose(Image.Transpose.ROTATE_90)
        mask.paste(blend.resize((97, HEIGHT)), (0, 0))
        picture.paste(second, (488, 0), mask)
    else:
        picture = photograph(paths[0], WIDTH, HEIGHT, focus[0])
    tone = row.get("tone", "warm")
    tint = {"warm": (42, 24, 12), "cool": (12, 20, 31), "wine": (47, 16, 25)}[tone]
    ambient = Image.blend(picture.filter(ImageFilter.GaussianBlur(35)), Image.new("RGB", picture.size, tint), .82)
    if row.get("portrait"):
        # Keep the original faces in the lower part of the poster, clear of the title.
        shift = row.get("shift", 520)
        original = picture.crop((0, 260, WIDTH, HEIGHT))
    else:
        shift = row.get("shift", 340)
        original = picture
    blend = Image.new("L", original.size, 255)
    blend.paste(Image.linear_gradient("L").resize((WIDTH, 190)), (0, 0))
    ambient.paste(ImageEnhance.Contrast(original).enhance(1.03), (0, shift), blend)
    shading = Image.new("L", (1, HEIGHT))
    shading.putdata([round(190 * max(0, min(1, (690 - y) / 300))) for y in range(HEIGHT)])
    ambient.paste(Image.new("RGB", ambient.size, (13, 11, 12)), (0, 0), shading.resize(ambient.size))
    return ambient


def letterpress(image, text, font, top, high, low, outline=3):
    matte = Image.new("L", image.size)
    draw = ImageDraw.Draw(matte)
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    point = ((WIDTH - (r - l)) // 2 - l, top - t)
    draw.text(point, text, font=font, fill=255)
    edge = Image.new("L", image.size)
    ImageDraw.Draw(edge).text(point, text, font=font, fill=255, stroke_width=outline, stroke_fill=255)
    shade = Image.new("L", image.size)
    shade.paste(edge, (0, 7))
    image.paste((8, 6, 8), (0, 0), shade.filter(ImageFilter.GaussianBlur(8)))
    image.paste((44, 30, 23), (0, 0), edge)
    ramp = Image.new("RGB", (1, HEIGHT))
    for y in range(HEIGHT):
        progress = min(1, max(0, (y - top) / max(1, b - t)))
        ramp.putpixel((0, y), tuple(round(a + (z - a) * progress) for a, z in zip(high, low)))
    image.paste(ramp.resize(image.size), (0, 0), matte)


def compose(row, root, title, episode, brush, serif):
    canvas = background(row, root)
    first, punctuation, last = title.partition("？")
    first += punctuation
    if not punctuation and len(first) > 9:
        middle = (len(first) + 1) // 2
        first, last = first[:middle], first[middle:]
    letterpress(canvas, first, fit_font(first, brush, 120, 904), 292 if last else 355,
                (255, 239, 189), (210, 158, 90), 4)
    if last:
        letterpress(canvas, last, fit_font(last, serif, 82, 920, index=2), 433,
                    (255, 246, 228), (225, 205, 174), 3)
    draw = ImageDraw.Draw(canvas)
    badge = f'第{episode["number"]}集'
    badge_font = face(serif, 54, 2)
    detail = episode["title"]
    detail_font = fit_font(detail, serif, 35, 610, index=2)
    bw = round(badge_font.getlength(badge))
    tw = round(detail_font.getlength(detail))
    left = (WIDTH - bw - tw - 69) // 2
    draw.line((left, 573, left + 28, 573), fill=(200, 158, 101), width=2)
    draw.text((left + 44, 574), badge, font=badge_font, fill=(255, 218, 141),
              stroke_width=2, stroke_fill=(23, 15, 12), anchor="lm")
    draw.text((left + bw + 69, 574), detail, font=detail_font,
              fill=(243, 225, 195), stroke_width=1, stroke_fill=(23, 15, 12), anchor="lm")
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("script", "plan", "brush-font", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--serif-font", type=Path,
                        default=Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"))
    args = parser.parse_args()
    manifest = json.loads(args.plan.read_text(encoding="utf-8"))
    episodes = json.loads(args.script.read_text(encoding="utf-8"))["episodes"]
    root = args.plan.parent
    out = args.output
    (out / "3x4").mkdir(parents=True, exist_ok=True)
    (out / "9x16").mkdir(parents=True, exist_ok=True)
    receipts = []
    for episode in episodes:
        number = episode["number"]
        if args.episode and number != args.episode:
            continue
        row = manifest["episodes"][str(number)]
        cover = compose(row, root, manifest["title"], episode, args.brush_font, args.serif_font)
        image_name = f'第{number:02d}集_{episode["title"]}.jpg'
        def save(picture, folder):
            path = out / folder / image_name
            picture.save(path, quality=96, subsampling=0, optimize=True)
            return {"path": str(path.relative_to(out)), "size": list(picture.size),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        receipts.append({"episode": number, "title": manifest["title"], "label": f"第{number}集",
                         "source_kind": row["source_kind"], "sources": row["images"],
                         "cover_3x4": save(cover.crop(CROP), "3x4"), "cover_9x16": save(cover, "9x16")})
    if not args.episode:
        (out / "封面清单.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(receipts)} episode covers exported to {out}")


if __name__ == "__main__":
    main()
