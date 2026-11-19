"""Check the local tools, ComfyUI nodes and model names without queuing a render."""
import argparse
import shutil
import subprocess

import requests

from run_workflow import BASE, build_graph, response_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref2va", action="store_true", help="Also check the Ref2VA checkpoint and Pika LoRA.")
    args = parser.parse_args()

    problems = []
    if not shutil.which("ffprobe"):
        problems.append("找不到 ffprobe。请安装 FFmpeg。")
    if shutil.which("ffmpeg"):
        encoders = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], text=True)
        for name in ("libx264", "aac"):
            if name not in encoders:
                problems.append(f"FFmpeg 缺少 {name} 编码器。")
    else:
        problems.append("找不到 ffmpeg。请安装 FFmpeg。")

    config = {"width": 1088, "height": 1920, "native_audio": True, "context_length": 5}
    segment = {"prompt": "", "num_frames": 56, "guide_audio": "guide.wav"}
    graph = build_graph(config, segment, "head.png", "tail.png", "check", 2, 1, "previous")
    nodes = {item["class_type"] for item in graph.values()}
    model_inputs = {"UNETLoader": "unet_name", "CLIPLoader": "clip_name",
                    "VAELoader": "vae_name", "LoraLoaderModelOnly": "lora_name"}
    models = [(item["class_type"], item["inputs"][model_inputs[item["class_type"]]])
              for item in graph.values() if item["class_type"] in model_inputs]
    if args.ref2va:
        nodes.update(("MiniMaxH3ReferenceToVideo", "ImageCrop", "SaveAudio"))
        models.extend([
            ("UNETLoader", "minimax_h3_ref2va_pruned_int8_convrot.safetensors"),
            ("LoraLoaderModelOnly", "pika_h3_flex_ref2va_comfy_bf16.safetensors"),
        ])

    with requests.Session() as session:
        session.trust_env = False
        try:
            registered = response_json(session.get(f"{BASE}/object_info", timeout=60))
        except (requests.RequestException, RuntimeError) as exc:
            raise SystemExit(f"无法读取 ComfyUI 节点。请检查 COMFYUI_URL 和服务日志。\n{exc}") from exc

    for name in sorted(nodes - registered.keys()):
        problems.append(f"ComfyUI 缺少节点 {name}。")
    for loader, filename in models:
        if loader in registered:
            choices = registered[loader]["input"]["required"][model_inputs[loader]][0]
            if filename not in choices:
                problems.append(f"{loader} 未找到 {filename}。")
    if problems:
        raise SystemExit("\n".join(problems))
    print(f"{BASE} 已注册 {len(nodes)} 个所需节点，识别到 {len(models)} 个权重，FFmpeg 编码器可用。")


if __name__ == "__main__":
    main()
