#!/usr/bin/env python3
"""Render ordered FL2VA keyframes with ComfyUI and join the trimmed shots at 24 fps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import requests


BASE = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE = "minimax_h3_video_vae_fp16.safetensors"
TURBO = "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"


def node(kind, **inputs):
    return {"class_type": kind, "inputs": inputs}


def build_graph(cfg, segment, first, last, folder, index, seed, previous_folder):
    anchor = node("MiniMaxH3ImageToVideo", clip=["2", 0], vae=["3", 0],
                  prompt="\n".join(filter(None, [cfg.get("master_prompt"), segment["prompt"]])),
                  width=cfg["width"], height=cfg["height"], length=segment["num_frames"],
                  first_frame=["6", 0], last_frame=["7", 0])
    graph = {
        "1": node("UNETLoader", unet_name=UNET, weight_dtype="default"),
        "2": node("CLIPLoader", clip_name=CLIP, type="minimax", device="default"),
        "3": node("VAELoader", vae_name=VAE),
        "4": node("LoraLoaderModelOnly", model=["1", 0], lora_name=cfg.get("lora", TURBO),
                  strength_model=0.8),
        "5": node("MiniMaxH3SigmaShift", model=["4", 0], shift_video=12.0, shift_audio=3.0),
        "6": node("LoadImage", image=first),
        "7": node("LoadImage", image=last),
        "8": anchor,
        "9": node("MiniMaxH3MotionContext", conditioning=["8", 0], vae=["3", 0],
                  latent=["8", 1], context_length=str(cfg.get("context_length", 22)),
                  audio_context_length=0),
        "10": node("RandomNoise", noise_seed=seed),
        "11": node("KSamplerSelect", sampler_name="euler"),
        "12": node("BasicScheduler", model=["5", 0], scheduler="simple",
                   steps=cfg.get("steps", 8), denoise=1.0),
        "13": node("BasicGuider", model=["5", 0], conditioning=["9", 0]),
        "14": node("SamplerCustomAdvanced", noise=["10", 0], guider=["13", 0],
                   sampler=["11", 0], sigmas=["12", 0], latent_image=["8", 1]),
        "15": node("VAEDecode", vae=["3", 0], samples=["14", 0]),
        "16": node("MiniMaxH3MotionContextTrim", images=["15", 0], trim_frames=["9", 1]),
        "17": node("CreateVideo", images=["16", 0], fps=24.0),
        "18": node("SaveVideo", video=["17", 0], filename_prefix=f"{folder}/shot{index:03}",
                   format="mp4", **{"format.codec": "auto"}),
    }
    if cfg.get("native_audio"):
        graph["21"] = node("VAELoader", vae_name="minimax_h3_audio_vae_fp32.safetensors")
        graph["22"] = node("VAEDecodeAudio", samples=["14", 0], vae=["21", 0])
        graph["16"]["inputs"].update(audio=["22", 0], fps=24., match_tail=True)
        graph["17"]["inputs"]["audio"] = ["16", 1]
        if segment.get("guide_audio"):
            graph["23"] = node("LoadAudio", audio=segment["guide_audio"])
            start = cfg.get("context_length", 22) if index > 1 else 0
            audio = ["23", 0]
            if start:
                graph["25"] = node("TrimAudioDuration", audio=audio, start_index=start / 24,
                                   duration=(segment["num_frames"] - start) / 24)
                audio = ["25", 0]
            graph["24"] = node("MiniMaxH3AddGuide", positive=["8", 0], latent=["8", 1],
                               frame_idx=start, audio_vae=["21", 0], audio=audio)
            graph["9"]["inputs"]["conditioning"] = ["24", 0]
    if previous_folder:
        graph["19"] = node("MiniMaxH3MotionContextLoadLatent", latent_path=previous_folder,
                           clip_index=index - 1)
        graph["9"]["inputs"]["context_latent"] = ["19", 0]
    graph["20"] = node("MiniMaxH3MotionContextSaveLatent", latent=["14", 0],
                       filename_prefix=f"{folder}/clip", clip_index=index)
    return graph


def save_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def response_json(response):
    if not response.ok:
        raise RuntimeError(f"ComfyUI HTTP {response.status_code}: {response.text[:3500]}")
    return response.json()


def upload(session, path):
    name = hashlib.sha256(path.read_bytes()).hexdigest() + path.suffix.lower()
    with path.open("rb") as stream:
        info = response_json(session.post(BASE + "/upload/image", files={"image": (name, stream)},
                             data={"subfolder": "h3_multi_fl2va", "overwrite": "true"}, timeout=120))
    return "/".join(filter(None, (info.get("subfolder"), info["name"])))


def get_result(session, job):
    while True:
        history = response_json(session.get(f"{BASE}/history/{job}", timeout=30)).get(job)
        if history:
            status = history["status"]
            if status["status_str"] != "success":
                errors = [v for event, v in status["messages"] if event == "execution_error"]
                message = "\n".join(str(e.get("exception_message", e)) for e in errors)
                raise RuntimeError(f"ComfyUI job {job} failed: {message or status['status_str']}")
            return history["outputs"]["18"]["images"][0]
        queue = response_json(session.get(BASE + "/queue", timeout=30))
        active = {entry[1] for rows in queue.values() for entry in rows}
        if job not in active:
            # History and queue are separate endpoints. Recheck after a completion race.
            if job not in response_json(session.get(f"{BASE}/history/{job}", timeout=30)):
                raise RuntimeError(f"ComfyUI lost queued job {job}")
        time.sleep(4)


def frame_count(path):
    result = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path)],
        text=True)
    return int(result.strip())


def run(manifest, output):
    cfg = json.loads(manifest.read_text(encoding="utf-8"))
    if cfg.get("native_episode"):
        episode = manifest.parent / cfg["native_episode"]
        subprocess.run([sys.executable, str(episode / "native_render.py"), str(episode / "shots.json")], check=True)
        return
    context = cfg.get("context_length", 22)
    if cfg.get("fps", 24) != 24 or context not in (5, 22, 39, 56):
        raise ValueError("H3 requires 24 fps and a motion context of 5, 22, 39 or 56 frames")
    if any(cfg[k] <= 0 or cfg[k] % 32 for k in ("width", "height")):
        raise ValueError("Width and height must be positive multiples of 32")
    if cfg.get("native_audio"):
        from finish_episode import delivery_scale
        delivery_scale(cfg["width"], cfg["height"])
    session = requests.Session()
    session.trust_env = False
    frames = {}
    if not cfg["groups"]:
        raise ValueError("At least one group is required")
    for group in cfg["groups"]:
        if len(group["frames"]) != len(group["segments"]) + 1 or not group["segments"]:
            raise ValueError("Each group needs one more image than segments")
        for idx, seg in enumerate(group["segments"]):
            count = seg["num_frames"]
            if not isinstance(count, int) or not 5 <= count <= 3600 or count % 17 != 5:
                raise ValueError(f"Invalid 17n+5 sampling frame count: {count}")
            if idx and count <= min(context, group["segments"][idx - 1]["num_frames"]):
                raise ValueError("A continuation must include frames after its pinned head")
        for name in group["frames"]:
            if name not in frames:
                frames[name] = upload(session, (manifest.parent / name).resolve())

    output.mkdir(parents=True, exist_ok=True)
    elapsed = 0
    timeline = []
    for gid, group in enumerate(cfg["groups"], start=1):
        previous_folder = None
        for idx, segment in enumerate(group["segments"], start=1):
            label = f"g{gid:03}-s{idx:03}"
            guide = segment.get("guide_audio")
            if guide:
                if guide not in frames:
                    frames[guide] = upload(session, (manifest.parent / guide).resolve())
                segment = dict(segment, guide_audio=frames[guide])
            graph = build_graph(cfg, segment, frames[group["frames"][idx - 1]],
                                frames[group["frames"][idx]], "pending", idx,
                                cfg.get("seed", 61025) + len(timeline), previous_folder)
            # Image names carry their content hash. The previous latent's folder carries
            # the chain hash. Appending scenes therefore reuses exactly the matching prefix.
            signature = hashlib.sha256(json.dumps(graph, sort_keys=True).encode() + os.fsencode(output))
            key = signature.hexdigest()[:20]
            folder = f"h3_multi_fl2va/{key}"
            graph["18"]["inputs"]["filename_prefix"] = f"{folder}/shot{idx:03}"
            graph["20"]["inputs"]["filename_prefix"] = f"{folder}/clip"
            previous_folder = folder
            clips = output / "clips" / key
            clips.mkdir(parents=True, exist_ok=True)
            clip = clips / (label + ".mp4")
            receipt = clips / (label + ".json")
            saved = json.loads(receipt.read_text()) if receipt.exists() else {}
            ready = (clip.is_file() and saved.get("graph") == graph and
                     hashlib.sha256(clip.read_bytes()).hexdigest() == saved.get("sha256"))
            if ready and idx < len(group["segments"]):
                ready = response_json(session.post(BASE + "/h3_motion_context/slot_exists",
                                      json={"latent_path": folder, "clip_index": idx},
                                      headers={"Origin": BASE}, timeout=30))["exists"]
            if not ready:
                job = None
                if receipt.exists():
                    job = json.loads(receipt.read_text()).get("pending_job")
                if not job:
                    job = response_json(session.post(BASE + "/prompt",
                                        json={"prompt": graph}, timeout=60))["prompt_id"]
                    save_json(receipt, {"pending_job": job, "graph": graph})
                print(f"{label}: {job}", flush=True)
                try:
                    info = get_result(session, job)
                except RuntimeError as exc:
                    save_json(receipt, {"failed_job": job, "error": str(exc), "graph": graph})
                    raise
                partial = clip.with_suffix(".part")
                with session.get(BASE + "/view", params={k: info[k] for k in
                                 ("filename", "subfolder", "type")}, stream=True, timeout=120) as res:
                    res.raise_for_status()
                    with partial.open("wb") as stream:
                        for chunk in res.iter_content(1024 * 1024):
                            stream.write(chunk)
                count = frame_count(partial)
                if count < 1:
                    raise RuntimeError(f"{label}: empty render")
                partial.replace(clip)
                save_json(receipt, {"job": job, "output": info, "frames": count, "graph": graph,
                                    "sha256": hashlib.sha256(clip.read_bytes()).hexdigest()})
            count = frame_count(clip)
            timeline.append({"group": gid, "segment": idx, "path": str(clip.relative_to(output)),
                             "prompt": segment["prompt"], "sampled_frames": segment["num_frames"],
                             "frames": count, "start_frame": elapsed, "end_frame_exclusive": elapsed + count})
            elapsed += count
            print(f"{label}: {count} rendered frames after trim", flush=True)

    playlist = output / "concat.txt"
    playlist.write_text("".join(f"file '{row['path']}'\n" for row in timeline))
    tmp = output / "visual.partial.mp4"
    if cfg.get("native_audio"):
        from native_media import join_native
        join_native([(output / row["path"],row["frames"]) for row in timeline], tmp)
    else:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "1",
                        "-i", str(playlist), "-map", "0:v:0", "-c:v", "copy", "-movflags",
                        "+faststart", str(tmp)], check=True)
    if frame_count(tmp) != elapsed:
        raise RuntimeError("Joined picture frame count differs from trimmed segment totals")
    tmp.replace(output / "visual.mp4")
    save_json(output / "timeline.json", {"fps": 24, "width": cfg["width"], "height": cfg["height"],
              "total_frames": elapsed, "duration_seconds": elapsed / 24, "segments": timeline})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    project, output = args.project.resolve(), args.output.resolve()
    run(project, output)
    if not (project.parent / "voice" / "index.json").is_file():
        from finish_episode import export_picture
        export_picture(output / "visual.mp4", output / "delivery_1080p.mp4")
        print(output / "delivery_1080p.mp4")
