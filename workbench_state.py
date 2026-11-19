"""Read Boundless Scroll production evidence for the workbench."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlparse

import requests

from production_progress import DIRECTORY, read_json


PHASES = {
    "preparing": "正在准备素材", "waiting_assets": "等待制作素材",
    "sampling": "正在生成镜头", "assembling": "正在拼接原声音画",
    "exporting": "正在封装成片", "validating": "正在核对交付文件",
}
EVENT_NAMES = {
    **PHASES, "planned": "分镜计划已保存", "sampled": "镜头采样已结束", "assembled": "已拼接镜头",
    "exported": "已完成导出", "failed": "制作出错", "interrupted": "制作已中断",
}
MEDIA_TYPES = {
    ".mp4": "video/mp4", ".webm": "video/webm", ".png": "image/png",
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
    ".flac": "audio/flac", ".wav": "audio/wav", ".srt": "application/x-subrip",
}


@dataclass
class Production:
    key: str
    title: str
    manifest: Path
    output: Path
    scenes: list
    progress: dict


def modified(path):
    try:
        return path.stat().st_mtime
    except OSError:
        return 0


def file_in(directory, name):
    directory = directory.resolve()
    path = (directory / name).resolve() if isinstance(name, str) and name else None
    return path if path and path.is_relative_to(directory) and path.is_file() else None


def media_url(media, path):
    if path and path.suffix.lower() in MEDIA_TYPES and path.is_file():
        path = path.resolve()
        key = hashlib.sha256(os.fsencode(path)).hexdigest()[:24]
        media[key] = path
        return "/media/" + key
    return None


def list_productions(root):
    projects = {}
    for episode in read_json(root / "production/script.json", {}).get("episodes", []):
        key = "ep%02d" % episode["number"]
        projects[key] = Production(key, episode["title"], root / "examples" / key / "project.json",
                                   root / "renders" / key, episode["scenes"], {})
    for manifest in sorted((root / "examples").glob("*/project.json")):
        key = manifest.parent.name
        title = "首尾帧入门样例" if key == "starter" else key
        projects.setdefault(key, Production(key, title, manifest, root / "renders" / key, [], {}))
    snapshots = [(p.stem, read_json(p)) for p in (root / DIRECTORY).glob("*.json")]
    for key, run in sorted(((k, r) for k, r in snapshots if r and r.get("manifest") and r.get("output")),
                           key=lambda item: item[1].get("updated", 0)):
        manifest = Path(run["manifest"])
        output = Path(run["output"])
        match = next((p for p in projects.values() if p.manifest == manifest and p.output == output), None)
        if match:
            match.progress = run
        else:
            projects[key] = Production(key, output.name, manifest, output, [], run)
    return projects


def worker_active(progress):
    pid = progress.get("pid")
    if progress.get("phase") not in PHASES or not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def get_queue(engine, progress):
    recorded = progress.get("engine")
    if recorded and urlparse(recorded).hostname in {"127.0.0.1", "localhost", "::1"}:
        engine = recorded
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(engine.rstrip("/") + "/queue", timeout=(0.6, 0.8))
            response.raise_for_status()
            data = response.json()
        running = {row[1] for row in data["queue_running"]}
        pending = {row[1] for row in data["queue_pending"]}
        return True, running, pending
    except (requests.RequestException, ValueError, IndexError, KeyError, TypeError):
        return False, set(), set()


@lru_cache(maxsize=24)
def output_digest(path, stamp, size):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def matches_digest(path, expected):
    if not path or not expected:
        return False
    try:
        stat = path.stat()
        return output_digest(path, stat.st_mtime_ns, stat.st_size) == expected
    except OSError:
        return False


@lru_cache(maxsize=24)
def picture_geometry(path, stamp, size):
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                 "-show_entries", "stream=width,height", "-of", "json", str(path)],
                                capture_output=True, text=True, check=True, timeout=5)
        stream = json.loads(result.stdout)["streams"][0]
        return [stream["width"], stream["height"]]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError):
        return None


def slot_state(receipt, video, queue, reviewed):
    online, running, pending = queue
    job = receipt.get("pending_job") or receipt.get("failed_job") or receipt.get("job")
    if job in running:
        return "running"
    if job in pending:
        return "queued"
    if receipt.get("quality", {}).get("status") == "rejected":
        return "failed"
    if receipt.get("error") or receipt.get("failed_job"):
        return "failed"
    if video:
        return "reviewed" if reviewed else "rendered"
    if job:
        return "receipt" if online else "offline"
    return "planned"


def native_shots(project, native, visual, queue, media):
    titles = {item["id"]: item["title"] for item in
              read_json(project.manifest.parent / "storyboard.json", {}).get("segments", [])}
    checked = visual.get("checked", {})
    shots = []
    for spec in read_json(native / "episode.json", []):
        name = spec["id"]
        accepted = read_json(native / "accepted" / (name + ".json"), {})
        receipt = read_json(native / "render/receipts" / (name + ".json"), accepted.get("receipt", {}))
        video = file_in(native / "render", receipt.get("video")) if receipt.get("status") == "complete" else None
        first = file_in(native / "assets", spec.get("first"))
        last = file_in(native / "assets", spec.get("last"))
        verified = (name in checked and receipt.get("job")
                    and accepted.get("receipt", {}).get("job") == receipt["job"])
        shots.append({
            "id": name, "title": titles.get(spec.get("source_shot"), name), "text": spec.get("text", ""),
            "frames": spec["frames"] - spec.get("trim", 0), "anchors": bool(first and last),
            "image": media_url(media, first), "video": media_url(media, video),
            "state": slot_state(receipt, video, queue, verified),
            "updated": modified(video) if video else modified(native / "render/receipts" / (name + ".json")),
        })
    return shots


def sequence_shots(project, config, timeline, queue, media):
    clips = project.output / "clips"
    candidates = {}
    selected = project.progress.get("receipts", {})
    finished = {}
    if project.progress.get("phase") not in {"sampling", "assembling"}:
        for receipt in sorted(clips.glob("*/*.json"), key=modified):
            candidates[receipt.stem] = receipt
        finished = {(item["group"], item["segment"]): item for item in timeline.get("segments", [])}
    shots = []
    for group_number, group in enumerate(config.get("groups", []), 1):
        images = group.get("frames", [])
        previous_frames = 0
        for segment_number, segment in enumerate(group.get("segments", []), 1):
            label = f"g{group_number:03}-s{segment_number:03}"
            segment_frames = segment["num_frames"]
            trim = 0 if segment.get("cut_in") else min(config.get("context_length", 22), previous_frames)
            count = segment_frames - trim
            previous_frames = segment_frames
            item = finished.get((group_number, segment_number), {})
            receipt_path = candidates.get(label)
            if label in selected:
                receipt_path = Path(selected[label])
            elif item.get("path"):
                receipt_path = (project.output / item["path"]).with_suffix(".json")
            if receipt_path:
                receipt_path = receipt_path.resolve()
                if not receipt_path.is_relative_to(clips.resolve()):
                    receipt_path = None
            receipt = read_json(receipt_path, {}) if receipt_path else {}
            video = receipt_path.with_suffix(".mp4") if receipt.get("sha256") else None
            if video and (not video.is_file() or (config.get("check_native_shots")
                          and receipt.get("quality", {}).get("status") != "checked")):
                video = None
            keyframes = segment.get("keyframes", images[segment_number - 1:segment_number + 1])
            first, last = [
                (project.manifest.parent / name).resolve() if isinstance(name, str) else None
                for name in keyframes
            ] if len(keyframes) == 2 else (None, None)
            shots.append({
                "id": segment.get("id", label), "label": label, "title": segment.get("title", f"机位 {group_number}"),
                "frames": item.get("frames", count), "text": segment.get("speech_text") or segment.get("prompt", ""),
                "anchors": bool(first and first.is_file() and last and last.is_file()),
                "image": media_url(media, first), "video": media_url(media, video),
                "state": slot_state(receipt, video, queue, False),
                "updated": modified(video) if video else modified(receipt_path) if receipt_path else 0,
            })
    return shots


def export_file(project):
    registered = project.progress.get("delivery")
    candidates = [Path(registered)] if registered else []
    candidates.extend(sorted(project.output.glob("*_H3原生1080P.mp4")))
    candidates.append(project.output / "delivery_1080p.mp4")
    if project.scenes and re.fullmatch(r"ep\d+", project.key):
        candidates.append(project.output / f"第{project.key[2:]}集_{project.title}.mp4")
    return next((p for p in candidates if p.is_file()), None)


def snapshot(root, selected, engine, media):
    projects = list_productions(root)
    if not projects:
        return {"projects": []}
    project = projects.get(selected) or next(iter(projects.values()))
    config = read_json(project.manifest, {})
    native = (project.manifest.parent / config["native_episode"]) if config.get("native_episode") else None
    report = read_json(native / "native-release.json", {}) if native else {}
    evaluation_path = native / "qa/final-evaluation.json" if native else None
    visual_path = native / "qa/visual-review.json" if native else None
    visual_review = read_json(visual_path, {}) if visual_path else {}
    timeline = read_json(project.output / "timeline.json", {})
    queue = get_queue(engine, project.progress)
    shots = (native_shots(project, native, visual_review, queue, media) if native else
             sequence_shots(project, config, timeline, queue, media))
    fps = report.get("fps", config.get("fps", 24))
    total = len(shots)
    anchors = sum(shot["anchors"] for shot in shots)
    rendered = sum(shot["state"] in {"rendered", "reviewed"} for shot in shots)
    reviewed = sum(shot["state"] == "reviewed" for shot in shots)
    running = sum(shot["state"] == "running" for shot in shots)
    queued = sum(shot["state"] == "queued" for shot in shots)
    movie = export_file(project)
    movie_time = modified(movie) if movie else 0
    evaluation = (read_json(evaluation_path, {}) if evaluation_path and movie_time
                  and modified(evaluation_path) >= movie_time else {})
    preview = movie or file_in(project.output, "visual.mp4")
    geometry = None
    if preview:
        try:
            stat = preview.stat()
            geometry = picture_geometry(preview, stat.st_mtime_ns, stat.st_size)
        except OSError:
            pass
    source_size = report.get("source_size") or [config.get("width"), config.get("height")]
    target_size = report.get("delivery_size") or ([1080, 1920] if
                  (source_size[1] or 0) > (source_size[0] or 0) else [1920, 1080])
    measurements = evaluation.get("measurements", [])
    verified = bool(total and reviewed == total and measurements and visual_review.get("unresolved") == []
                    and modified(visual_path) >= movie_time
                    and evaluation.get("native_audio_verified") and evaluation.get("complete_subtitles")
                    and evaluation.get("geometry") == target_size == geometry
                    and evaluation.get("source_geometry") == source_size
                    and all(a and a >= b for a, b in zip(source_size, target_size))
                    and matches_digest(movie, report.get("file_sha256")))
    active = worker_active(project.progress)
    phase = project.progress.get("phase")
    if running or queued:
        status, tone = ("正在生成镜头" if running else "镜头已排队"), "running"
    elif active:
        status, tone = PHASES[phase], "running"
    elif phase == "failed":
        status, tone = "制作出错", "failed"
    elif phase in PHASES or phase == "interrupted":
        status, tone = "制作已中断", "failed"
    elif verified:
        status, tone = "成片已验收", "done"
    elif movie:
        status, tone = "成片待验收", "waiting"
    elif total and rendered == total:
        status, tone = "镜头已生成", "waiting"
    elif rendered:
        status, tone = "镜头已部分生成", "waiting"
    else:
        status, tone = "待制作", "idle"
    scenes = project.scenes or ([{
        "heading": "镜头计划",
        "beats": [{"text": shot["text"] or shot["title"]} for shot in shots],
    }] if shots else [])
    phase_step = lambda done, live, label, detail, section: {
        "label": label, "state": "running" if live else "done" if done else "idle",
        "detail": detail, "section": section,
    }
    stages = [
        phase_step(bool(scenes), False, "剧本", f"{len(scenes)} 个场景" if scenes else "待准备", "screenplay"),
        phase_step(bool(total), active and phase == "preparing", "分镜计划",
                   f"{total} 个镜头" if total else "待编排", "storyboard"),
        phase_step(bool(total and anchors == total), active and phase == "waiting_assets", "关键帧",
                   f"{anchors}/{total} 组" if total else "待制作", "storyboard"),
        phase_step(bool(total and rendered == total), bool(running or queued or (active and phase == "sampling")),
                   "镜头生成", f"{rendered}/{total} 镜头" if total else "待生成", "storyboard"),
        phase_step(verified or bool(total and reviewed == total), active and phase == "validating",
                   "音画复核", f"{reviewed}/{total} 镜头" if total else "待复核", "records"),
        phase_step(bool(report or timeline), active and phase == "assembling", "剪辑决定",
                   "时间线就绪" if report or timeline else "待拼接", "records"),
        phase_step(bool(movie), active and phase == "exporting", "合成导出",
                   "MP4 已保存" if movie else "待导出", "renders"),
        phase_step(verified, False, "交付", "原生 1080p" if verified else "待验收", "renders"),
    ]
    frames = report.get("frames") or timeline.get("total_frames") or sum(shot["frames"] for shot in shots)
    render = {"video": media_url(media, preview), "name": preview.name if preview else None,
              "subtitle": media_url(media, preview.with_suffix(".srt")) if preview else None,
              "geometry": geometry,
              "preview": bool(preview and not movie), "verified": verified}
    records = []
    if all(source_size):
        records.append({"label": "采样设置", "title": f"{source_size[0]}×{source_size[1]} · {fps} fps",
                        "text": "画面与声音同时采样" if config.get("native_audio") else "读取现有镜头配置"})
    if total:
        records.append({"label": "镜头进度", "title": f"{rendered} / {total} 个镜头",
                        "text": f"生成中 {running} · 排队 {queued} · 已复核 {reviewed}"})
    if evaluation.get("native_audio_verified"):
        records.append({"label": "原声音轨", "title": "H3 原声已核对", "text": "音轨来自同一次 H3 采样"})
    offsets = [abs(m["offset25"]) for m in measurements if isinstance(m.get("offset25"), (int, float))]
    if offsets:
        records.append({"label": "口型复测", "title": f"{len(offsets)} 段对白 · 最大 {max(offsets)} 帧",
                        "text": f"核对了 {sum(m.get('native_face_samples', 0) for m in measurements)} 个人脸区域"})
    if verified:
        records.append({"label": "画面复核", "title": f"{reviewed} 个镜头已逐镜检查",
                        "text": "已核对画面清晰度、可见人脸和五官稳定性"})
    if project.progress.get("error"):
        records.insert(0, {"label": "任务出错", "title": "最近一次运行", "text": project.progress["error"]})
    events = [{"time": event["time"], "title": EVENT_NAMES[event["phase"]]}
              for event in project.progress.get("events", []) if event["phase"] in EVENT_NAMES]
    events += [{"time": s["updated"], "title": s["title"], "shot": s["id"], "state": s["state"]}
               for s in shots if s["updated"]]
    if movie:
        events.append({"time": movie_time, "title": "成片已导出"})
    if verified:
        events.append({"time": max(modified(visual_path), modified(evaluation_path)), "title": "成片已验收"})
    events.sort(key=lambda event: event["time"])
    current = project.progress.get("current_shot") if active else None
    return {
        "projects": [{"id": p.key, "title": p.title, "episode": bool(p.scenes)} for p in projects.values()],
        "id": project.key, "title": project.title, "status": status, "tone": tone, "online": queue[0],
        "shots": shots, "stages": stages, "scenes": scenes, "fps": fps, "seconds": frames / fps,
        "render": render, "records": records, "events": events, "current": current,
        "updated": max([project.progress.get("updated", 0), modified(project.manifest)]
                       + [event["time"] for event in events]),
    }
