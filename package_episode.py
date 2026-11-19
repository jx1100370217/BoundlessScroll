"""Add character IDs, time jumps and a vertical end card to a finished 1080p episode."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory


PACKAGE_LAYOUT_VERSION = 2


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def probe(path, *args):
    output = subprocess.check_output(["ffprobe", "-v", "error", *args, "-of", "json", str(path)])
    return json.loads(output)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audio_packets(path):
    return probe(path, "-select_streams", "a:0", "-show_packets", "-show_data_hash", "sha256",
                 "-show_entries", "packet=pts,dts,duration,data_hash")["packets"]


def timestamp(frame, fps):
    hundredths = frame * 100 // fps
    whole, fraction = divmod(hundredths, 100)
    minutes, seconds = divmod(whole, 60)
    return f"{minutes // 60}:{minutes % 60:02}:{seconds:02}.{fraction:02}"


def ass_text(value):
    if any(char in value for char in "{}\\"):
        raise ValueError(f"Unsupported ASS control character in {value!r}")
    return value.replace("\n", r"\N")


class CaptionPlacementError(ValueError):
    pass


def continuation_metrics(width, height):
    size = round(min(width, height) * .127)
    return size, size + 6


def card_bounds(card, width, height):
    kind = card["variant"]
    if kind == "name":
        margin = round(width * .066)
        x = width - margin - 137 if card.get("side") == "right" else margin
        return x, round(height * card.get("top", .18)), 137, 276
    if kind == "end":
        size, step = continuation_metrics(width, height)
        w, h = size + 32, step * len(card["text"]) + 16
        center = round(width * (.175 if card.get("side") == "left" else .825))
        top = min(round(height * card.get("top", .34)), int(height * .77) - h)
        return center - w // 2, top, w, h
    align = {"left": .275, "right": .725}.get(card.get("side"), .5)
    return round(width * align) - 207, round(height * card.get("top", .077)) - 73, 414, 164


def position_cards(source, cards, geometry, detector_path):
    import cv2

    width, height = geometry
    model = cv2.FaceDetectorYN.create(str(detector_path), "", (width // 2, height // 2),
                                     score_threshold=.70, nms_threshold=.3)
    capture = cv2.VideoCapture(str(source))
    result = []
    try:
        for card in cards:
            faces = []
            first, last = card["start_frame"], card["end_frame_exclusive"] - 1
            for frame in {round(first + (last - first) * point) for point in (0, .25, .5, .75, 1)}:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, picture = capture.read()
                if not ok:
                    raise CaptionPlacementError(f'{card["id"]}: could not check frame {frame}')
                _, found = model.detect(cv2.resize(picture, (width // 2, height // 2),
                                                  interpolation=cv2.INTER_AREA))
                if found is not None:
                    faces.extend(tuple(float(value) * 2 for value in face[:4])
                                 for face in found if min(face[2], face[3]) >= 22)
            choices = [card]
            if card["variant"] == "name":
                choices += [dict(card, side=side, top=top) for top in (.41, .49, .57)
                            for side in (card.get("side", "left"), "right", "left")]
            elif card["variant"] == "end":
                choices += [dict(card, side=side, top=top) for top in (.34, .14, .46)
                            for side in ("left", "right")]
            else:
                choices += [dict(card, side=side, top=.077)
                            for side in ("center", "right", "left")]
            for candidate in choices:
                x, y, w, h = card_bounds(candidate, width, height)
                if all(x + w + 10 <= a or a + b + 10 <= x or y + h + 10 <= c or c + d + 10 <= y
                       for a, c, b, d in faces):
                    result.append(candidate)
                    break
            else:
                raise CaptionPlacementError(f'{card["id"]}: confirm a position clear of faces')
    finally:
        capture.release()
    return result

def cards_ass(cards, width, height, fps):
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Brush,Ma Shan Zheng,76,&H00DAE7EE,&H000000FF,&H00120E0B,&H78120E0B,0,0,0,0,100,100,0,0,1,1,2,5,0,0,0,1
Style: Role,Noto Serif CJK SC,29,&H00B8CCDB,&H000000FF,&H00120E0B,&H78120E0B,0,0,0,0,100,100,0,0,1,0,1,8,0,0,0,1
Style: Shape,Ma Shan Zheng,1,&H00000000,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    events = []
    for card in cards:
        kind = card["variant"]
        start, end = (timestamp(card[key], fps) for key in ("start_frame", "end_frame_exclusive"))
        fade = r"\fad(330,0)" if kind == "end" else r"\fad(333,300)"

        def event(layer, style, tags, value):
            events.append(f"Dialogue: {layer},{start},{end},{style},,0,0,0,,{{{tags}{fade}}}{value}")

        def rectangle(x, y, w, h, color, alpha):
            event(0, "Shape", fr"\an7\pos({x},{y})\1c&H{color}&\1a&H{alpha}&\p1",
                  f"m 0 0 l {w} 0 {w} {h} 0 {h}")

        if kind == "name":
            right = card.get("side") == "right"
            x, y, w, h = card_bounds(card, width, height)
            rectangle(x, y, w, h, "120B08", "34")
            rectangle(x + (130 if right else 7), y + 13, 3, 250, "343C92", "16")
            name = ass_text("\n".join(card["text"]))
            role = ass_text("\n".join(card["subtitle"]))
            event(1, "Brush", fr"\an8\pos({x + (84 if right else 51)},{y + 18})\fs63", name)
            event(1, "Role", fr"\an8\pos({x + (29 if right else 108)},{y + 24})", role)
            continue
        if kind == "end":
            left, top, w, h = card_bounds(card, width, height)
            size, step = continuation_metrics(width, height)
            outer = 1 if card.get("side") != "left" else -1
            x = left + w // 2
            rectangle(x + outer * (w // 2 - 6), top + 29, 2, h - 58, "394C9C", "28")
            for row, char in enumerate(card["text"]):
                y = top + 8 + step // 2 + row * step
                event(1, "Brush", fr"\an5\pos({x - outer * 5},{y})\fs{size}\1c&HBBE6FA&\bord2.6\shad3",
                      ass_text(char))
            continue
        left, top, w, _ = card_bounds(card, width, height)
        x, y = left + w // 2, top + 73
        rectangle(left, top, w, 164, "100C08", "42")
        rectangle(left, y + 90, w, 2, "343C92", "1B")
        event(1, "Brush", fr"\an5\pos({x},{y})\fs58\fsp4", ass_text(card["text"]))
    return header + "\n".join(events) + "\n"


def compile_cards(timeline, script, episode, content):
    fps = timeline["fps"]
    shots = timeline.get("shots") or timeline["segments"]
    frame_count = timeline.get("frames") or timeline["total_frames"]
    episode_numbers = {entry["number"] for entry in script["episodes"]}
    if episode not in episode_numbers:
        raise ValueError("Episode does not exist in the source script")
    by_id = {shot["id"] if "id" in shot else f'{shot["group"]:03}-{shot["segment"]:03}': shot
             for shot in shots}
    events = []
    time_cards = content.get("timeCards", content.get("titleCards", []))
    for card in content["nameCards"] + [card for card in time_cards if card.get("variant") == "time"]:
        shot = by_id[card["anchor"]]
        start = shot["start_frame"] + round(card.get("offset", 0) * fps)
        end = start + round(card["duration"] * fps)
        if not shot["start_frame"] <= start < end <= shot["end_frame_exclusive"]:
            raise ValueError(f'{card["id"]}: card timing falls outside the anchor shot')
        events.append(dict(card, start_frame=start, end_frame_exclusive=end))
    if episode != max(episode_numbers):
        last = shots[-1]
        events.append({"id": "continuation", "variant": "end", "text": "未完待续", "side": "right",
                       "anchor": last.get("id") or f'{last["group"]:03}-{last["segment"]:03}',
                       "start_frame": max(last["start_frame"], frame_count - round(2.8 * fps)),
                       "end_frame_exclusive": frame_count})
    return fps, frame_count, events


def package(source, timeline, script, episode, content, brush, output, detector=None):
    fps, frame_count, events = compile_cards(timeline, script, episode, content)
    video = probe(source, "-select_streams", "v:0", "-show_streams")["streams"][0]
    geometry = (video["width"], video["height"])
    if (geometry not in {(1080, 1920), (1920, 1080)} or int(video["nb_frames"]) != frame_count
            or video["r_frame_rate"] != f"{fps}/1"):
        raise ValueError("Picture geometry, frame count, or fps does not match the 1080p delivery timeline")
    if timeline.get("file_sha256") and timeline["file_sha256"] != digest(source):
        raise ValueError("Input is not the picture identified by the release record")
    if source.resolve() == output.resolve():
        raise ValueError("Choose a separate path for the packaged review edition")
    if detector:
        events = position_cards(source, events, geometry, detector)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".package-", dir=output.parent) as folder:
        tmp = Path(folder)
        (tmp / "cards.ass").write_text(cards_ass(events, *geometry, fps), encoding="utf-8")
        (tmp / "fonts").mkdir()
        (tmp / "fonts" / "MaShanZheng.ttf").symlink_to(brush.resolve(strict=True))
        target = tmp / "packaged.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                        "-map", "0:v:0", "-map", "0:a:0", "-vf", "ass=cards.ass:fontsdir=fonts",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "13", "-threads", "8",
                        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(target)],
                       cwd=tmp, check=True)
        encoded = probe(target, "-select_streams", "v:0", "-show_streams")["streams"][0]
        for key in ("width", "height", "r_frame_rate", "nb_frames", "start_time", "duration"):
            if encoded[key] != video[key]:
                raise ValueError(f"Packaged picture changed {key}")
        if audio_packets(source) != audio_packets(target):
            raise ValueError("Encoded audio or its packet timestamps have changed")
        receipt = {"layout_version": PACKAGE_LAYOUT_VERSION, "episode": episode,
                   "picture": str(source), "source_sha256": digest(source),
                   "sha256": digest(target), "geometry": list(geometry), "fps": fps, "frames": frame_count,
                   "audio_packet_identity_verified": True, "face_placement_sampled": bool(detector),
                   "cards": events, "status": "needs_review"}
        data = tmp / "packaging.json"
        data.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        data.replace(output.with_suffix(".packaging.json"))
        (tmp / "cards.ass").replace(output.with_suffix(".cards.ass"))
        if source.with_suffix(".srt").is_file():
            shutil.copyfile(source.with_suffix(".srt"), output.with_suffix(".srt"))
        target.replace(output)
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "timeline", "script", "cards", "brush-font", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--face-detector", type=Path)
    parser.add_argument("--episode", required=True, type=int)
    args = parser.parse_args()
    package(args.input.resolve(), load(args.timeline), load(args.script), args.episode,
            load(args.cards), args.brush_font, args.output.resolve(), args.face_detector)
