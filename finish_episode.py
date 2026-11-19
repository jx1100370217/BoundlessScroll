#!/usr/bin/env python3
"""Finish an episode with its rendered frame timeline, indexed dialogue and music."""
import argparse
import difflib
import hashlib
import json
import math
import re
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ffmpeg(*args, cwd=None, log="error", capture=False):
    return subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", log,
                           *map(str, args)], cwd=cwd, check=True, text=True,
                          stderr=subprocess.PIPE if capture else None)


def probe(path, selector, entries):
    cmd = ["ffprobe", "-v", "error", "-select_streams", selector,
           "-show_entries", "stream=" + entries, "-of", "json", str(path)]
    return json.loads(subprocess.check_output(cmd))["streams"][0]


def character_times(text, words, duration):
    """Anchor transcript characters to Whisper word spans, including ASR substitutions."""
    target = "".join(c for c in text if c.isalnum())
    heard, spans = [], []
    for word in words:
        chars = [c for c in word["text"] if c.isalnum()]
        for i, char in enumerate(chars):
            a, b = word["start"], word["end"]
            heard.append(char)
            spans.append((a + (b-a)*i/len(chars), a + (b-a)*(i+1)/len(chars)))
    if not spans or not target:
        raise ValueError(f"Missing word timing for {text}")
    result = []
    for tag, i, j, a, b in difflib.SequenceMatcher(None, target, "".join(heard),
                                                    autojunk=False).get_opcodes():
        if j == i:
            continue
        if tag == "equal":
            result.extend(spans[a:b])
        else:
            start = spans[a][0] if a < len(spans) else spans[-1][1]
            end = spans[b-1][1] if b > a else start
            for n in range(j-i):
                result.append((start + (end-start)*n/(j-i), start + (end-start)*(n+1)/(j-i)))
    return [(min(a, duration), min(b, duration)) for a, b in result]


def caption_cues(text, times, start, duration, speed):
    # Retain leading ellipses, closing quotation marks and interjections from the script.
    clauses = re.split(r"(?<=[，,。！？!?；;：:、…])(?=[^，,。！？!?；;：:、…”’」])",text)
    grouped = []
    prefix = ""
    for clause in clauses:
        if not any(c.isalnum() for c in clause):
            if grouped:
                grouped[-1] += clause
            else:
                prefix += clause
            continue
        clause, prefix = prefix+clause, ""
        if grouped and len(grouped[-1]) < 5 and len(grouped[-1] + clause) <= 16:
            grouped[-1] += clause
        else:
            grouped.append(clause)
    cues = []
    offset = 0
    for item in grouped:
        n = sum(c.isalnum() for c in item)
        if not n:
            continue
        s, e = times[offset][0], times[offset+n-1][1]
        offset += n
        next_start = times[offset][0] if offset < len(times) else duration
        e = min(next_start, e + 0.16)
        cues.append((start + s/speed, start + e/speed, item))
    return cues


def clock(seconds, ass=False):
    unit = 100 if ass else 1000
    ticks = round(seconds*unit)
    h, rem = divmod(ticks, 3600*unit)
    m, rem = divmod(rem, 60*unit)
    s, ms = divmod(rem, unit)
    return (f"{h}:{m:02}:{s:02}.{ms:02}" if ass else f"{h:02}:{m:02}:{s:02},{ms:03}")


def captions(cues, folder, width, height):
    size = max(14, round(min(width*.054, height*.08)))
    margin = round(height*.11)
    sides = round(width*.045)
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes
[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Dialogue,Noto Sans CJK SC,{size},&H00FFFFFF,&H000000FF,&H00151515,&H88000000,-1,0,0,0,100,100,0,0,1,{max(1,round(size*.075))},0,2,{sides},{sides},{margin},1
[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    lines, events = [], []
    for i,(start,end,text) in enumerate(cues,1):
        if end <= start or (i>1 and start < cues[i-2][1]-.001):
            raise ValueError(f"Overlapping or empty subtitle at {text}")
        lines.append(f"{i}\n{clock(start)} --> {clock(end)}\n{text}\n")
        ass_text = text.replace("\\", "＼").replace("{", "｛").replace("}", "｝")
        events.append(f"Dialogue: 0,{clock(start,True)},{clock(end,True)},Dialogue,,0,0,0,,{ass_text}")
    (folder/"captions.srt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (folder/"captions.ass").write_text(header+"\n".join(events)+"\n",encoding="utf-8")


def placements(project, timeline, voice_dir):
    positions = {(g,s): clip["id"] for g,group in enumerate(project["groups"],1)
                 for s,clip in enumerate(group["segments"],1)}
    slots = {}
    project_segments = {s["id"]:s for g in project["groups"] for s in g["segments"]}
    cursor = 0
    fps = timeline["fps"]
    for segment in timeline["segments"]:
        key = positions[segment["group"],segment["segment"]]
        a,b = segment["start_frame"],segment["end_frame_exclusive"]
        if key in slots or a != cursor or b <= a or segment.get("id", key) != key:
            raise ValueError(f"Invalid render timeline at {key}")
        slots[key] = (a/fps,b/fps)
        cursor = b
    if len(slots) != len(positions) or cursor != timeline["total_frames"]:
        raise ValueError("Rendered timeline does not cover the entire project")
    asr = read(voice_dir/"asr.json")["segments"]
    voice_index = read(voice_dir/"index.json")
    if any("dialogue_parts" in clip for clip in project_segments.values()):
        return place_series_dialogue(slots, project_segments, voice_index, asr, voice_dir)

    tracks, cues = [], []
    for row in voice_index:
        for part in row["segments"]:
            shot = part["shot"]
            scene_a,scene_b = slots[shot]
            duration = part["end"]-part["start"]
            lead_in = part.get("lead_in", .20)
            if not 0 <= lead_in < scene_b - scene_a:
                raise ValueError(f"Dialogue starts outside its shot: {shot}")
            t = scene_a+lead_in
            available = scene_b-t-.06
            if available <= 0 or duration <= 0:
                raise ValueError(f"No room for dialogue at {shot}")
            speed = max(1.,duration/available)
            if speed > 1.30:
                raise ValueError(f"Extend {shot}: its dialogue needs {duration:.2f}s")
            tracks.append((voice_dir/row["file"],part["start"],part["end"],t,speed))
            times = character_times(part["text"],asr[shot]["words"],duration)
            cues.extend(caption_cues(part["text"],times,t,duration,speed))
    return tracks, sorted(cues)


def place_series_dialogue(slots, clips, index, asr, voice_dir):
    lines = {row["id"]:row for row in index}
    times = {key:character_times(row["text"],asr[key]["words"],row["duration"])
             for key,row in lines.items()}
    consumed = {k:0. for k in lines}
    tracks,cues = [],[]
    for shot,(scene_a,scene_b) in slots.items():
        parts = clips[shot].get("dialogue_parts",[])
        if not parts:
            continue
        t = scene_a+.20
        length = sum(part["end"]-part["start"] for part in parts)
        available = scene_b-t-.06-.13*(len(parts)-1)
        if available <= 0 or length > 1.3*available:
            raise ValueError(f"Extend {shot}: {length:.3f}s of indexed dialogue")
        speed = max(1.,length/available)
        for part in parts:
            key,a,b = part["id"],part["start"],part["end"]
            row = lines[key]
            if abs(a-consumed[key])>.003 or b<=a or b>row["duration"]+.003:
                raise ValueError(f"Noncontiguous or out-of-range audio slice at {key}: {a}–{b}")
            consumed[key] = b
            tracks.append((voice_dir/row["file"],a,b,t,speed))
            # Assign each original character to the slice that contains its ASR midpoint.
            full = times[key]
            first = sum((x+y)/2<a for x,y in full) if a>0 else 0
            last = (len(full) if abs(b-row["duration"])<.003
                    else sum((x+y)/2<b for x,y in full))
            if first<last:
                letters = [i for i,c in enumerate(row["text"]) if c.isalnum()]
                i = letters[first] if first else 0
                j = letters[last] if last<len(letters) else len(row["text"])
                text = row["text"][i:j]
                local = [(max(0.,x-a),min(b-a,y-a)) for x,y in full[first:last]]
                cues.extend(caption_cues(text,local,t,b-a,speed))
            t += (b-a)/speed+.13
    missed = [k for k,row in lines.items() if abs(consumed[k]-row["duration"])>.003]
    if missed:
        raise ValueError(f"Render plan does not cover complete lines: {', '.join(missed)}")
    cues = sorted(cues)
    if "".join(c[2] for c in cues) != "".join(row["text"] for row in index):
        raise ValueError("Subtitle text/order does not reproduce the complete episode dialogue")
    return tracks,cues


def raw_mix(tracks, music, duration, path):
    samples = round(duration*48000)
    inputs, filters, labels = [], [], []
    for i,(wav,a,b,t,speed) in enumerate(tracks):
        inputs.extend(["-i",wav])
        tempo = f"atempo={speed:.7f}," if speed > 1.000001 else ""
        filters.append(f"[{i}:a]atrim=start={a}:end={b},asetpts=PTS-STARTPTS,"
                       f"{tempo}aresample=48000,adelay={round(t*1000)}:all=1[v{i}]")
        labels.append(f"[v{i}]")
    filters.append("".join(labels)+f"amix=inputs={len(labels)}:normalize=0,"
                   f"apad=whole_len={samples},atrim=end_sample={samples},asetpts=N/SR/TB,"
                   "pan=stereo|c0=c0|c1=c0,asplit=2[voice][key]")
    music_dur = float(probe(music,"a:0","duration")["duration"])
    cross = min(1.,music_dur/4)
    copies = max(1,math.ceil((duration-cross)/(music_dur-cross)))
    inputs.extend(["-i",music])
    music_label = f"[{len(tracks)}:a]"
    if copies>1:
        filters.append(music_label+f"asplit={copies}"+"".join(f"[m{i}]" for i in range(copies)))
        music_label = "[m0]"
        for n in range(1,copies):
            filters.append(music_label+f"[m{n}]acrossfade=d={cross}:c1=qsin:c2=qsin[mc{n}]")
            music_label=f"[mc{n}]"
    fade=min(1.5,duration/4)
    filters.append(music_label+f"atrim=duration={duration},aresample=48000,"
                   "aformat=channel_layouts=stereo,loudnorm=I=-30:TP=-5:LRA=9,aresample=48000,"
                   f"afade=t=in:d={fade},afade=t=out:st={duration-fade}:d={fade}[music]")
    filters.append("[music][key]sidechaincompress=threshold=0.035:ratio=4:attack=12:release=350[bed]")
    filters.append("[voice][bed]amix=inputs=2:normalize=0:duration=first[master]")
    ffmpeg(*inputs,"-filter_complex_threads","2","-filter_complex",";".join(filters),
           "-map","[master]","-c:a","pcm_f32le","-ar","48000","-t",f"{duration:.9f}",
           "-fs",str(samples*8+65536),path)


def delivery_size(width, height):
    return (1080, 1920) if height > width else (1920, 1080)


def delivery_scale(source_width, source_height):
    width,height = delivery_size(source_width,source_height)
    if source_width < width or source_height < height:
        raise ValueError(f"Source {source_width}x{source_height} is below the native 1080p delivery size")
    if source_width - width < 32 and source_height - height < 32:
        return f"crop={width}:{height}:(iw-ow)/2:(ih-oh)/2,setsar=1"
    return (f"scale={width}:{height}:force_original_aspect_ratio=decrease:"
            f"force_divisible_by=2:flags=lanczos,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1")


def export_picture(master, output):
    info = probe(master, "v:0", "width,height,nb_frames")
    size = delivery_size(info["width"], info["height"])
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".picture-") as scratch:
        tmp = Path(scratch) / "delivery.mp4"
        ffmpeg("-i", master, "-map", "0:v:0", "-map", "0:a?", "-vf", delivery_scale(info["width"], info["height"]),
               "-c:v", "libx264", "-crf", "17", "-preset", "fast", "-threads", "6",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", tmp)
        v = probe(tmp, "v:0", "width,height,nb_frames")
        if (v["width"], v["height"], v["nb_frames"]) != (*size, info["nb_frames"]):
            raise ValueError("1080p export geometry or frame count differs from the delivery specification")
        tmp.replace(output)


def finish(project_path, render, music, output):
    project=read(project_path)
    if project.get("native_episode"):
        from finish_native import master as finish_native_episode
        return finish_native_episode(project_path.parent/project["native_episode"],output,project_path.parent/"storyboard.json")
    if project.get("native_audio"):
        from native_media import finish_tracks
        return finish_tracks(project, render, output)
    timeline=read(render/"timeline.json")
    master = render/"visual.mp4"
    voice = project_path.parent/"voice"
    if "dialogue_revision" in project:
        revision = project_path.parent / project["dialogue_revision"]
        expected = read(revision / "source.json")["visual_sha256"]
        with master.open("rb") as original:
            if hashlib.file_digest(original, "sha256").hexdigest() != expected:
                raise ValueError("The dialogue revision was prepared for a different picture master")
        master = revision / "visual.mp4"
        voice = revision / "voice"
    vid = probe(master,"v:0","nb_frames,r_frame_rate,width,height")
    if (int(vid["nb_frames"]) != timeline["total_frames"]
            or Fraction(vid["r_frame_rate"]) != timeline["fps"]
            or (vid["width"],vid["height"]) != (timeline["width"],timeline["height"])):
        raise ValueError("visual.mp4 and timeline.json disagree on frames, geometry or fps")
    tracks,cues = placements(project,timeline,voice)
    duration = timeline["total_frames"]/timeline["fps"]
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent,prefix=".finish-") as scratch:
        tmp=Path(scratch)
        width, height = delivery_size(vid["width"], vid["height"])
        captions(cues,tmp,width,height)
        raw_mix(tracks,music,duration,tmp/"mix.wav")
        target="loudnorm=I=-16:TP=-1.8:LRA=10"
        measure=ffmpeg("-i",tmp/"mix.wav","-af",target+":print_format=json","-f","null","-",
                       log="info",capture=True).stderr
        values=json.loads(measure[measure.rindex("{"):measure.rindex("}")+1])
        keys={"measured_I":"input_i","measured_TP":"input_tp","measured_LRA":"input_lra",
              "measured_thresh":"input_thresh","offset":"target_offset"}
        norm=target+"".join(f":{k}={values[v]}" for k,v in keys.items())+":linear=true"
        ffmpeg("-i",master,"-i",tmp/"mix.wav","-map","0:v:0","-map","1:a:0",
               "-vf",delivery_scale(vid["width"],vid["height"])+",ass=captions.ass","-af",norm+",aresample=48000","-c:v","libx264",
               "-preset","fast","-crf","17","-threads","6","-pix_fmt","yuv420p",
               "-c:a","aac","-b:a","192k","-t",f"{duration:.9f}","-movflags","+faststart",
               tmp/"final.mp4",cwd=tmp)
        (tmp/"captions.srt").replace(output.with_suffix(".srt"))
        (tmp/"final.mp4").replace(output)
    print(f"{output} ({duration:.3f}s, {len(tracks)} dialogue segments, {len(cues)} captions)")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project",type=Path)
    for name in ("render","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--music",type=Path,help="Music for legacy external-audio edits")
    args=parser.parse_args()
    music=args.music if args.music else args.project.parent/"music.flac"
    finish(args.project.resolve(),args.render.resolve(),music.resolve(),args.output.resolve())
