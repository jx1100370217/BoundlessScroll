#!/usr/bin/env python3
"""Export a project.json as a drag-and-drop ComfyUI workflow without queuing GPU work."""
import argparse
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests
from playwright.sync_api import sync_playwright

from run_workflow import BASE, build_graph, node, upload


def project_graph(cfg, frames):
    graph, positions, boxes = {}, {}, []

    def add(item, title, x, y):
        key = str(len(graph) + 1)
        graph[key] = dict(item, _meta={"title": title})
        positions[key] = [x, y]
        return key

    shared = {}
    tails = []
    for group_idx, group in enumerate(cfg["groups"], 1):
        previous = None
        for idx, segment in enumerate(group["segments"]):
            serial = len(tails)
            shot = segment.get("id") or f"{group_idx}.{idx + 1}"
            y = 650 + serial * 600
            if segment.get("guide_audio"):
                segment = dict(segment, guide_audio=frames[segment["guide_audio"]])
            local = build_graph(cfg, segment, frames[group["frames"][idx]],
                                frames[group["frames"][idx + 1]], "ui", idx + 1,
                                cfg.get("seed", 61025) + serial, None)
            ids = dict(shared)

            def insert(old, title, x, height):
                item = local[old]
                inputs = {name: [ids[value[0]], value[1]] if isinstance(value, list) else value
                          for name, value in item["inputs"].items()}
                ids[old] = add(node(item["class_type"], **inputs), title, x, height)

            if not shared:
                for old, title, x, h in [
                        ("1", "FL2VA 主模型", 0, 60), ("2", "H3 文本编码器", 0, 250),
                        ("3", "视频 VAE", 415, 250), ("4", "Turbo LoRA", 415, 60),
                        ("5", "H3 采样偏移", 825, 60), ("11", "Euler 采样器", 1235, 60),
                        ("12", "采样步数", 1235, 250)]:
                    insert(old, title, x, h)
                if "21" in local:
                    insert("21", "H3 音频 VAE", 825, 250)
                shared.update(ids)
                boxes.append(["共享模型与采样设置", -40, -55, 1700, 520])
            for old, title, x, dy in [
                    ("6", "首帧", 0, 0), ("7", "尾帧", 385, 0), ("8", "首尾帧与提示词", 770, 0),
                    ("9", "前段运动上下文" if previous else "场景起点", 1250, 0),
                    ("10", "固定种子", 1250, 310), ("13", "采样条件", 1650, 0),
                    ("14", "音视频采样", 2025, 0), ("15", "解码画面", 2395, 0),
                    ("16", "裁切重叠帧", 2760, 0)]:
                if old=="9" and "23" in local:
                    insert("23", "对白时间引导", 360, y+355)
                    if "25" in local:
                        insert("25", "裁去引导声的重叠时间", 540, y+410)
                    insert("24", "指定原生表演时刻", 775, y+335)
                if old=="16" and "22" in local:
                    insert("22", "解码 H3 同步原声", 2395, y+240)
                insert(old, title, x, y + dy)
            if previous:
                graph[ids["9"]]["inputs"]["context_latent"] = [previous, 0]
            previous = ids["14"]
            tails.append((ids["16"], [ids["16"],1] if "22" in local else None, y))
            boxes.append([f"场景 {group_idx} · 镜头 {shot}", -40, y - 105, 3150, 575])

    # Preserve frame order without repeatedly copying a growing prefix into each batch.
    x = 3230
    while len(tails) > 1:
        joined = []
        for i in range(0, len(tails) - 1, 2):
            (left, sound_left, ya), (right, sound_right, yb) = tails[i:i + 2]
            y = (ya + yb) / 2
            image = add(node("ImageBatch", image1=[left, 0], image2=[right, 0]), "按分镜顺序拼接", x, y)
            sound = None
            if sound_left:
                k = add(node("AudioConcat", audio1=sound_left, audio2=sound_right, direction="after"), "按同一顺序拼接原声", x, y+180)
                sound = [k,0]
            joined.append((image,sound,y))
        if len(tails) % 2:
            joined.append(tails[-1])
        tails = joined
        x += 335
    last, sound, y = tails[0]
    if cfg.get("native_audio"):
        from finish_episode import delivery_size
        width,height=delivery_size(cfg["width"],cfg["height"])
        if cfg["width"]>=width and cfg["height"]>=height:
            last=add(node("ImageCrop",image=[last,0],width=width,height=height,
                          x=(cfg["width"]-width)//2,y=(cfg["height"]-height)//2), "原生画幅中心裁切",x,y)
            x+=340
    video = add(node("CreateVideo", images=[last, 0], fps=24.0, **({"audio":sound} if sound else {})), "按 24 fps 合成", x, y)
    add(node("SaveVideo", video=[video, 0], format="mp4", **{"format.codec": "auto"},
             filename_prefix="h3_multi_fl2va/ui/sequence"), "保存完整画面", x + 340, y)
    return graph, positions, boxes


def native_episode_graph(episode):
    """Provide the actual shot graphs and their stereo previews without building a full-film image tensor."""
    graph, positions, boxes, contexts = {}, {}, [], {}
    layout = [(1,2,3,22), (4,5,11,12,10), (6,7,31,32), (8,23,35), (33,34,24), (29,13,14), (15,25,41,42), (17,18,26)]
    for index, row in enumerate(json.loads((episode / "episode.json").read_text())):
        name = row["id"]
        receipt = json.loads((episode / "render/receipts" / (name+".json")).read_text())
        nodes, params = receipt["graph"], receipt["parameters"]
        prefix = index * 100
        remap = lambda key: str(prefix + int(key))
        if params.get("previous_context"):
            nodes.pop("28")
        nodes.pop("30", None)
        trim = row.get("trim", 0)
        picture, audio = ["15", 0], ["25", 0]
        if trim:
            nodes["41"] = node("MiniMaxH3MotionContextTrim", images=picture, audio=audio,
                               trim_frames=trim, fps=24., match_tail=True)
            picture, audio = ["41", 0], ["41", 1]
        nodes["42"] = node("ImageCrop", image=picture, width=1080, height=1920, x=4, y=0)
        nodes["17"]["inputs"].update(images=["42", 0], audio=audio)
        for output in ("18", "26"):
            nodes[output]["inputs"]["filename_prefix"] = "native/ui-shots/" + name
        for key, item in nodes.items():
            inputs = {k: [remap(v[0]), v[1]] if isinstance(v, list) else v for k,v in item["inputs"].items()}
            if key == "29":
                inputs["context_latent"] = [contexts[params["previous_context"]], 0]
            graph[remap(key)] = node(item["class_type"], **inputs)
        if params.get("save_context"):
            contexts[params["save_context"]] = remap("14")
        top = index * 1860
        for x, column in enumerate(layout):
            for y, number in enumerate(column):
                if str(number) in nodes:
                    positions[remap(number)] = [x*510,top+y*348]
        boxes.append([name + "  " + row.get("text",""), -40, top-75, 4070,1770])
    return graph, positions, boxes


def export(project, output):
    cfg = json.loads(project.read_text(encoding="utf-8"))
    if cfg.get("native_episode"):
        graph, pos, boxes = native_episode_graph(project.parent / cfg["native_episode"])
        url = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8189")
        return write_canvas(graph, pos, boxes, output, url)
    context = cfg.get("context_length", 22)
    if (not cfg["groups"] or cfg.get("fps", 24) != 24 or context not in (5, 22, 39, 56)
            or any(cfg[k] <= 0 or cfg[k] % 32 for k in ("width", "height"))):
        raise ValueError("Expected nonempty groups, 24 fps, a supported context and a 32px canvas grid")
    if cfg.get("native_audio"):
        from finish_episode import delivery_scale
        delivery_scale(cfg["width"], cfg["height"])
    for group in cfg["groups"]:
        segments = group["segments"]
        if not segments or len(group["frames"]) != len(segments) + 1:
            raise ValueError("Each group needs one more keyframe than segments")
        for i, seg in enumerate(segments):
            n = seg["num_frames"]
            if not 5 <= n <= 3600 or n % 17 != 5 or (i and n <= min(context, segments[i-1]["num_frames"])):
                raise ValueError(f"Invalid FL2VA sampling length {n} for segment {i + 1}")
    with requests.Session() as http:
        http.trust_env = False
        files = dict.fromkeys(name for group in cfg["groups"] for name in group["frames"])
        files.update({segment["guide_audio"]: None for group in cfg["groups"]
                      for segment in group["segments"] if segment.get("guide_audio")})
        frames = {name: upload(http, (project.parent / name).resolve()) for name in files}
    graph, pos, boxes = project_graph(cfg, frames)
    return write_canvas(graph, pos, boxes, output, BASE)


def write_canvas(graph, pos, boxes, output, url):
    # The frontend rounds TrimAudioDuration's time widgets to 0.01 s. A frame count
    # divided by the shared frame rate survives save/reload exactly, including 5/24.
    trims = [(key,n) for key,n in graph.items() if n["class_type"]=="TrimAudioDuration"]
    if trims:
        fps = str(max(map(int,graph)) + 1)
        graph[fps] = node("PrimitiveInt", value=24)
        pos[fps] = [-1070,0]
        seconds = {}
        for key,item in trims:
            for field in ("start_index","duration"):
                value = item["inputs"][field]
                frames = round(value*24)
                if abs(value-frames/24)>1e-9:
                    raise ValueError(f"{key}: audio timing does not fall on a frame")
                if frames not in seconds:
                    calc = str(int(fps) + len(seconds) + 1)
                    graph[calc] = node("ComfyMathExpression", expression=f"{frames} / a", **{"values.a":[fps,0]})
                    pos[calc] = [-630,len(seconds)*168]
                    seconds[frames] = calc
                item["inputs"][field] = [seconds[frames],0]
    with sync_playwright() as engine:
        browser = engine.chromium.launch(args=["--disable-gpu"])
        page = browser.new_page(viewport={"width": 1500, "height": 960})
        page.goto(url, wait_until="networkidle", timeout=60000)
        page.wait_for_function("window.comfyAPI?.app?.app?.canvas?.graph && window.comfyAPI.app.app.rootGraph && document.querySelector('#graph-canvas')")
        result = page.evaluate("""async ({graph,pos,boxes,title}) => {
          const app = window.comfyAPI.app.app;
          await app.loadApiJson(graph, title);
          for (const [id, xy] of Object.entries(pos)) {
            const n = app.rootGraph.getNodeById(Number(id));
            n.pos = xy;
            if (n.type === 'LoadImage') n.size = [345, 350];
            if (n.type === 'MiniMaxH3ImageToVideo') n.size = [440, 450];
            if (n.type === 'MiniMaxH3MotionContext') n.size = [350, 230];
            if (n.type === 'RandomNoise') {
              const control = n.widgets.find(w => w.name === 'control_after_generate');
              if (control) control.value = 'fixed';
            }
          }
          for (const [name, x, y, w, h] of boxes) {
            const box = new window.LiteGraph.LGraphGroup(name);
            box.pos = [x, y]; box.size = [w, h]; app.rootGraph.add(box);
          }
          const {workflow} = await app.graphToPrompt();
          workflow.extra.ds = {scale: 0.32, offset: [95, 170]};
          await app.loadGraphData(workflow, true, true, title);
          return {workflow, replay: (await app.graphToPrompt()).output};
        }""", {"graph": graph, "pos": pos, "boxes": boxes, "title": output.stem})
        browser.close()
    # Verify every authored constant and edge after the same frontend deserialization as drag/drop.
    for key, item in graph.items():
        restored = result["replay"].get(key, {})
        if (restored.get("class_type") != item["class_type"] or
                any(restored.get("inputs", {}).get(k) != v for k, v in item["inputs"].items())):
            differences = {k:(v,restored.get("inputs", {}).get(k)) for k,v in item["inputs"].items()
                           if restored.get("inputs", {}).get(k) != v}
            raise RuntimeError(f"ComfyUI could not round-trip node {key} ({item['class_type']}): {differences}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=output.parent, delete=False, encoding="utf-8") as tmp:
        json.dump(result["workflow"], tmp, ensure_ascii=False, separators=(",", ":"))
        tmp.write("\n")
    Path(tmp.name).replace(output)
    print(f"{output} · {len(graph)} nodes, {len(result['workflow']['links'])} links", flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("project", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    export(args.project.resolve(), args.output.resolve())
