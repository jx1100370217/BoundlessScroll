"""Check native H3 takes before they become an episode assembly input."""
import csv
from functools import lru_cache
import io
import re
import subprocess

from native_words import canonical, distance


class ShotRejected(ValueError):
    pass


def vertical_gold(frame):
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    crop = frame[round(height * .035):round(height * .61), round(width * .12):round(width * .90)]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (14, 72, 159), (37, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 4), np.uint8))
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    glyphs = [(x, y) for (_, _, w, h, area), (x, y) in zip(stats[1:], centers[1:])
              if 16 <= w <= 100 and 43 <= h <= 102 and area >= 54 and .08 < area / (w * h) < .85]
    for x, _ in glyphs:
        column = sorted(y for px, y in glyphs if abs(px - x) <= 20)
        count, previous = 1, column[0]
        for y in column[1:]:
            if y - previous > 117:
                count = 1
            elif y - previous >= 45:
                count += 1
            else:
                continue
            if count >= 4:
                return True
            previous = y
    return False


def visible_words(video):
    """Find repeated superimposed lower-screen text in the raw takes."""
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    hits = []
    vertical_hits = 0
    try:
        for part in (.20, .41, .62, .82):
            capture.set(cv2.CAP_PROP_POS_FRAMES, round(part * (count - 1)))
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not decode native take {video}")
            vertical_hits += vertical_gold(frame)
            height, width = frame.shape[:2]
            crop = frame[round(height * .58):round(height * .92), round(width * .06):round(width * .94)]
            grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            saturation = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
            mask = np.uint8(255 * (~((saturation < 90) & (grey >= 211))))
            ok, encoded = cv2.imencode(".png", mask)
            if not ok:
                raise ValueError("Could not prepare frame for typography check")
            report = subprocess.run(["tesseract", "stdin", "stdout", "-l", "chi_sim",
                                     "--psm", "6", "tsv"], input=encoded.tobytes(),
                                    capture_output=True, check=True).stdout.decode("utf-8")
            value = "".join(re.sub(r"[^\u3400-\u9fff]", "", row["text"])
                            for row in csv.DictReader(io.StringIO(report), delimiter="\t")
                            if row["conf"] and float(row["conf"]) >= 45 and 16 <= int(row["height"]) <= 225)
            if len(value) >= 2:
                hits.append(value)
    finally:
        capture.release()
    if vertical_hits >= 2:
        hits += ["疑似金色竖排字"] * vertical_hits
    return hits


@lru_cache(maxsize=2)
def recognizer(name):
    from faster_whisper import WhisperModel

    return WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=4, local_files_only=True)


def transcribe(video, model):
    import numpy as np

    raw = subprocess.check_output(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video),
                                   "-vn", "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"])
    speech = recognizer(model).transcribe(np.frombuffer(raw, dtype=np.float32), language="zh", beam_size=5,
                                          temperature=0, word_timestamps=True,
                                          condition_on_previous_text=False)[0]
    parts = list(speech)
    return ("".join(row.text for row in parts),
            [dict(text=word.word, start=word.start, end=word.end) for row in parts for word in row.words])


def inspect_take(path, expected):
    from pypinyin import lazy_pinyin

    overlays = visible_words(path)
    if len(overlays) >= 2:
        return {"status": "rejected", "reason": "发现模型附加的疑似文字", "visible_words": overlays}
    if not expected:
        return {"status": "checked", "expected": "", "words": []}
    target = lazy_pinyin(canonical(expected))
    comparisons = []
    for model in ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", "Systran/faster-whisper-small"):
        observed, words = transcribe(path, model)
        mismatch = distance(target, lazy_pinyin(canonical(observed)))
        comparisons.append({"model": model, "recognized": observed, "phonetic_distance": mismatch})
        if not mismatch:
            return {"status": "checked", "expected": expected, "recognized": observed,
                    "words": words, "dialogue": comparisons}
    return {"status": "rejected", "reason": "H3 原声与原定对白不一致", "expected": expected,
            "dialogue": comparisons}
