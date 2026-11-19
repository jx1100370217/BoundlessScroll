"""Compose a cover and accompanying publication caption for each episode."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

WIDTH, HEIGHT = 1080, 1920
CROP = (0, 240, 1080, 1680)
DEFAULT_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")


def face(path, size):
    return ImageFont.truetype(str(path), size, index=2 if path.suffix.lower() == ".ttc" else 0)


def fit_font(value, path, size, width):
    while size >= 24:
        font = face(path, size)
        if font.getlength(value) <= width:
            return font
        size -= 2
    raise ValueError("Split the cover title into shorter lines")


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
    tint = {"warm": (21, 12, 17), "cool": (11, 15, 26), "wine": (29, 10, 19)}[row.get("tone", "warm")]
    ambient = Image.blend(picture.filter(ImageFilter.GaussianBlur(40)), Image.new("RGB", picture.size, tint), .43)
    top_blend = Image.new("L", picture.size, 255)
    top_blend.paste(Image.linear_gradient("L").resize((WIDTH, 112)), (0, 0))
    ambient.paste(picture, (0, row.get("picture_shift", 118)), top_blend)
    shade = Image.new("L", (1, HEIGHT))
    shade.putdata([round(241 * min(1, max(0, (y - 728) / 850)) ** .87) for y in range(HEIGHT)])
    ambient.paste(tint, (0, 0), shade.resize(ambient.size))
    return ambient


def headline(image, text, font, x, top, accent):
    # Offset the upper strokes slightly to give the stacked headline a forward lean.
    left, upper, right, bottom = font.getbbox(text)
    h = bottom - upper + 24
    slant = .12 if accent else .085
    w = right - left + 24 + round(slant * h)

    def lettering(stroke):
        mask = Image.new("L", (w, h))
        ImageDraw.Draw(mask).text((12 - left, 12 - upper), text, font=font, fill=255, stroke_width=stroke)
        return mask.transform(mask.size, Image.Transform.AFFINE, (1, slant, -slant * h, 0, 1, 0),
                              resample=Image.Resampling.BICUBIC)

    face_mask = lettering(0)
    edge = lettering(7)
    point = (x - 12, top - 12)
    image.paste((7, 5, 11), (point[0] + 4, point[1] + 9), edge.filter(ImageFilter.GaussianBlur(4)))
    image.paste((30, 15, 19), point, edge)
    if not accent:
        image.paste((255, 247, 234), point, face_mask)
        return (x, top, x + w - 12, top + h - 12)
    shades = [(255, round(240 - 59 * (y / h)), round(170 - 100 * (y / h))) for y in range(h)]
    gradient = Image.new("RGB", (1, h))
    gradient.putdata(shades)
    image.paste(gradient.resize((w, h)), point, face_mask)
    return (x, top, x + w - 12, top + h - 12)


def compose(row, root, lines, episode, title_font, label_font):
    canvas = background(row, root)
    # Title and number sit at the lower edge of the common 3:4 framing.
    slots = ((84, 1054, 123, 876), (108, 1180, 132, 864), (77, 1311, 184, 925))
    rectangles = []
    for idx, (value, (x, top, size, max_width)) in enumerate(zip(lines, slots[-len(lines):])):
        chosen = fit_font(value, title_font, size, max_width)
        rectangles.append(headline(canvas, value, chosen, x, top, idx == len(lines) - 1))
    draw = ImageDraw.Draw(canvas)
    badge = f'第{episode["number"]}集'
    font = face(label_font, 62)
    left, top = 83, 1536
    width = round(font.getlength(badge)) + 57
    draw.rounded_rectangle((left, top, left + width, top + 88), radius=15,
                           fill=(134, 26, 42), outline=(224, 145, 90), width=2)
    draw.text((left + width / 2, top + 42), badge, font=font, fill=(255, 247, 231),
              anchor="mm", stroke_width=1, stroke_fill=(134, 26, 42))
    rectangles.append((left, top, left + width, top + 88))
    if any(not (CROP[0] < a < c < CROP[2] and CROP[1] < b < d < CROP[3])
           for a, b, c, d in rectangles):
        raise ValueError(f"{badge} cover lettering fell outside the common crop")
    return canvas


def title_lines(manifest):
    title = manifest["title"]
    first, punctuation, last = title.partition("？")
    default = [first + punctuation, last] if last else [title]
    if len(default) == 1 and len(title) > 10:
        middle = (len(title) + 1) // 2
        default = [title[:middle], title[middle:]]
    lines = manifest.get("title_lines", default)
    if not 1 <= len(lines) <= 3 or not all(lines) or "".join(lines) != title:
        raise ValueError("Cover line breaks must preserve the full title")
    return lines


def save_release_copy(out, filename, title, label, copy, *, ai_generated):
    folder = out / "发布文案"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / Path(filename).with_suffix(".txt").name
    topics = " ".join("#" + topic for topic in copy["topics"])
    parts = [f"《{title}》{label}", copy["hook"].strip()]
    disclosure = "内容由AI生成\n" if ai_generated else ""
    parts.append(disclosure + topics)
    path.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    return str(path.relative_to(out))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("script", "plan", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--episode", type=int)
    parser.add_argument("--title-font", "--brush-font", type=Path)
    parser.add_argument("--label-font", "--serif-font", type=Path, default=DEFAULT_FONT)
    args = parser.parse_args()
    manifest = json.loads(args.plan.read_text(encoding="utf-8"))
    episodes = json.loads(args.script.read_text(encoding="utf-8"))["episodes"]
    root = args.plan.parent
    out = args.output
    (out / "3x4").mkdir(parents=True, exist_ok=True)
    (out / "9x16").mkdir(parents=True, exist_ok=True)
    lines = title_lines(manifest)
    title_font = args.title_font or root / manifest.get("title_font", DEFAULT_FONT)
    receipts = []
    for episode in episodes:
        number = episode["number"]
        if args.episode and number != args.episode:
            continue
        row = manifest["episodes"][str(number)]
        cover = compose(row, root, lines, episode, title_font, args.label_font)
        image_name = f'第{number:02d}集_{episode["title"]}.jpg'

        def save(picture, folder):
            path = out / folder / image_name
            picture.save(path, quality=96, subsampling=0, optimize=True)
            return {"path": str(path.relative_to(out)), "size": list(picture.size),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

        label = f"第{number}集"
        receipt = {"episode": number, "title": manifest["title"], "title_lines": lines, "label": label,
                   "source_kind": row["source_kind"], "sources": row["images"],
                   "cover_3x4": save(cover.crop(CROP), "3x4"), "cover_9x16": save(cover, "9x16")}
        if "release_copy" in row:
            receipt["release_copy"] = save_release_copy(out, image_name, manifest["title"], label, row["release_copy"],
                                                        ai_generated=manifest.get("ai_generated", False))
        receipts.append(receipt)
    if not args.episode:
        (out / "封面清单.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{len(receipts)} episode covers exported to {out}")


if __name__ == "__main__":
    main()
