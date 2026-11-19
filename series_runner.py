"""Continue native H3 episode production and prepare each eligible cut for review."""
import argparse
from collections import defaultdict
from pathlib import Path
import subprocess
import time

from native_media import DialogueMismatch
from native_qa import ShotRejected
from package_episode import PACKAGE_LAYOUT_VERSION, CaptionPlacementError, digest, load, package
from production_progress import track
from render_series import THERE, render_series
from run_workflow import save_json


ROLES = {
    "顾承川": "盛川集团继承人", "林小满": "林家的养女", "陈莉": "顾总首席秘书",
    "林国富": "林家的家主", "王秀兰": "小满的养母", "林雪晴": "林家亲生女儿",
    "张少": "张家年轻少爷", "苏蔓": "苏家大小姐", "店长艾米": "婚纱馆店长",
    "艾米": "婚纱馆店长", "管家赵伯": "庄园老管家", "赵伯": "庄园老管家",
}


def slot_id(slot):
    return f'{slot["group"]:03}-{slot["segment"]:03}'


def prepare_cards(storyboard, timeline, time_jumps):
    fps = timeline["fps"]
    by_group = defaultdict(list)
    for slot in timeline["segments"]:
        by_group[slot["group"]].append(slot)
    occupied = set()
    final_slot = timeline["segments"][-1]
    time_cards = []
    for group, text in time_jumps:
        slot = by_group[group][0]
        anchor = slot_id(slot)
        time_cards.append({"id": f"time-{group}", "variant": "time", "anchor": anchor,
                           "offset": .12, "duration": min(2.35, (slot["frames"] - 4) / fps), "text": text})
        occupied.add(anchor)

    seen = set()
    names = []
    for group, run in enumerate(storyboard["runs"], 1):
        for actor in run.get("introductions", run["characters"]):
            if actor in seen or actor not in ROLES:
                continue
            slot = next((entry for entry in by_group[group] if entry is not final_slot
                         and slot_id(entry) not in occupied and entry["frames"] >= 46), None)
            if slot is None:
                continue
            anchor = slot_id(slot)
            names.append({"id": f"name-{len(names)+1}", "variant": "name", "anchor": anchor,
                          "offset": .12, "duration": min(2.3, (slot["frames"] - 4) / fps),
                          "side": "left" if len(names) % 2 == 0 else "right",
                          "top": .41, "text": actor, "subtitle": ROLES[actor]})
            occupied.add(anchor)
            seen.add(actor)
    return {"nameCards": names, "timeCards": time_cards}


def verify_render_paths(dest, timeline):
    for slot in timeline["segments"]:
        receipt = load((dest / slot["path"]).with_suffix(".json"))
        graph = receipt["graph"]
        request = graph["8"]["inputs"]
        if (request["width"], request["height"]) != (1088, 1920):
            raise ValueError("An episode shot was sampled at the wrong resolution")
        native_audio = (graph["22"]["inputs"]["samples"] == graph["15"]["inputs"]["samples"]
                        and graph["16"]["inputs"]["audio"] == ["22", 0]
                        and graph["17"]["inputs"]["audio"] == ["16", 1])
        if not native_audio:
            raise ValueError("An episode shot is missing the H3 jointly sampled sound")


def run(args):
    script = load(args.source / "script.json")
    time_jumps = load(args.story_notes).get("timeCards", {})
    state = THERE / "runtime/series-resume.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    report = load(state) if state.is_file() else {"episodes": {}}
    if args.wait_unit:
        while subprocess.run(["systemctl", "--user", "is-active", "--quiet", args.wait_unit]).returncode == 0:
            time.sleep(8)
    for episode in script["episodes"]:
        number = episode["number"]
        if not args.start <= number <= args.end:
            continue
        key = str(number)
        if report["episodes"].get(key, {}).get("status") == "waiting_for_episode_2_approval":
            print(f"Episode {number}: awaiting the user's approval of episode two", flush=True)
            return
        title = f'第{number:02d}集_{episode["title"]}'
        base_dir = THERE / f"renders/ep{number:02d}"
        out = args.output / f"{title}_字幕审片版.mp4"
        receipt_path = out.with_suffix(".packaging.json")
        if out.is_file() and receipt_path.is_file():
            receipt = load(receipt_path)
            if receipt.get("layout_version") == PACKAGE_LAYOUT_VERSION and receipt["sha256"] == digest(out):
                continue
        report["episodes"][key] = {"status": "producing", "title": episode["title"]}
        save_json(state, report)
        try:
            render_series(args.source, args.output / "原声母版", number, number, dialogue_retries=2)
        except (DialogueMismatch, ShotRejected) as error:
            report["episodes"][key].update(status="needs_shot_review", error=str(error))
            save_json(state, report)
            print(f"Episode {number}: waiting for shot review before starting the next episode", flush=True)
            return
        timeline = load(base_dir / "timeline.json")
        verify_render_paths(base_dir, timeline)
        project = THERE / f"examples/ep{number:02d}"
        cards = prepare_cards(load(project / "storyboard.json"), timeline, time_jumps.get(key, []))
        try:
            with track(project / "project.json", base_dir, "exporting", "exported", delivery=out):
                package(base_dir / f"{title}.mp4", timeline, script, number, cards,
                        args.brush_font, out, args.face_detector)
        except CaptionPlacementError as error:
            report["episodes"][key].update(status="needs_card_placement", error=str(error))
            save_json(state, report)
            print(f"Episode {number}: waiting for caption placement before starting the next episode", flush=True)
            return
        report["episodes"][key].update(status="needs_review", review=str(out))
        save_json(state, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "story-notes", "brush-font"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--start", type=int, default=2)
    parser.add_argument("--end", type=int, default=25)
    parser.add_argument("--face-detector", type=Path)
    parser.add_argument("--wait-unit")
    args = parser.parse_args()
    for name in ("source", "output", "story_notes", "brush_font"):
        setattr(args, name, getattr(args, name).resolve())
    run(args)
