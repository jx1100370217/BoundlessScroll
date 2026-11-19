"""Compile a filmed episode from ordered camera setups and timed dialogue."""
from pathlib import Path
import argparse
import json
import math
import shutil
from PIL import Image, ImageOps

from finish_episode import character_times

FPS = 24
CONTEXT = 22

SPEAKERS = {
    "顾承川": "the young man with the calm, warm voice", "陈莉": "the tablet-carrying secretary",
    "林小满": "the young woman with a single braid", "林雪晴": "the daughter in the red dress",
    "林国富": "the older stocky father", "王秀兰": "the older woman with short hair",
    "张少": "the young man with a gold chain", "苏蔓": "the woman with the sleek bob",
    "店长艾米": "the boutique manager", "鉴定师": "the elderly appraiser",
    "亲戚甲": "the seated family guest", "保安队长": "the uniformed security lead",
}

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def split_line(line, words):
    # Use actual pauses at punctuation first, then divide only the remaining long clauses.
    length = line["duration"]
    timestamps = character_times(line["text"], words, length)
    boundaries = [0.]
    char_index = -1
    for char in line["text"]:
        if char.isalnum():
            char_index += 1
        if char not in "，。！？；、：,.!?;:" or not 0 <= char_index < len(timestamps) - 1:
            continue
        left, right = timestamps[char_index][1], timestamps[char_index + 1][0]
        pause = (left + right) / 2
        if pause - boundaries[-1] >= .8 and length - pause >= .65:
            boundaries.append(pause)
    boundaries.append(length)
    word_ends = [(a["end"] + b["start"]) / 2 for a, b in zip(words, words[1:])]
    cuts = [0.]
    for start, end in zip(boundaries, boundaries[1:]):
        count = math.ceil((end - start) / 3.68)
        for remaining in range(count - 1, 0, -1):
            ideal = cuts[-1] + (end - cuts[-1]) / (remaining + 1)
            choices = [time for time in word_ends if abs(time - ideal) <= .65
                       and .7 <= time - cuts[-1] <= 3.68 and end - time <= remaining * 3.68]
            cuts.append(min(choices, key=lambda time: abs(time - ideal)) if choices else ideal)
        cuts.append(end)
    return [dict(id=line["id"], start=round(a, 6), end=round(b, 6)) for a, b in zip(cuts, cuts[1:])]


def pack_parts(parts):
    chunks = []
    for part in parts:
        duration = part['end'] - part['start']
        if chunks:
            filled = sum(x['end']-x['start']+.13 for x in chunks[-1])
            if filled+duration<=3.65:
                chunks[-1].append(part)
                continue
        chunks.append([part])
    return chunks or [[]]

def sentence_fragment(part, line, words):
    times = iter(character_times(line['text'],words,line['duration']))
    result = []
    showing=False
    for char in line['text']:
        if char.isalnum():
            a,b=next(times)
            showing = part['start'] <= (a+b)/2 < part['end'] or (a==b==line['duration']==part['end'])
        if showing:
            result.append(char)
    return ''.join(result)

def format_prompt(action, count, open_tail, soundscape):
    if open_tail:
        instruction = ('For the target video, at 0.00 seconds into the target video, '
                       '<Picture 1> (from [Shot 1]) is fully referenced.')
    else:
        end = (count - 1) / FPS
        instruction = ('How the reference pictures align with the target video — Picture 1 (from Shot 1) '
                       'aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) '
                       f'aligns with the {end:.2f}-second mark of the target video.')
    return (f'{instruction}\n\nintegrated_multimodal_description: [Shot 1] {action}'
            f'\n\noverall_soundscape: {soundscape}\n\nnon_diegetic_music: N/A')


def write_voice_guide(voice_dir, lines, parts, count, continuation, output):
    import wave
    import numpy as np
    # The sampled H3 sound is delivered. This clip supplies the words, timing and voice as a guide.
    signal = np.zeros(count*1000, dtype='<i2')
    start = int(.20*24000) + (CONTEXT*1000 if continuation else 0)
    for part in parts:
        row = lines[part['id']]
        with wave.open(str(voice_dir/row['file']), 'rb') as wav:
            if (wav.getnchannels(), wav.getframerate(), wav.getsampwidth()) != (1,24000,2):
                raise ValueError(f"Expected 24 kHz mono PCM for {part['id']}")
            a,b = round(part['start']*24000),round(part['end']*24000)
            wav.setpos(a);data = np.frombuffer(wav.readframes(b-a),dtype='<i2')
        signal[start:start+len(data)] = data
        start += len(data) + round(.13*24000)
    output.parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(output),'wb') as wav:
        wav.setparams((1,2,24000,0,'NONE','not compressed'))
        wav.writeframes(signal.tobytes())


def prepare_keyframes(art_dir, directory, names, setup, preparation):
    if len(names) != 2:
        raise ValueError(f'{setup}: expected two anchors for the camera setup')
    files = []
    for index, name in enumerate(names, 1):
        source = art_dir / Path(name).name
        output = directory / f'{setup}-{index}.png'
        with Image.open(source) as image:
            source_size = list(image.size)
            ImageOps.fit(image.convert('RGB'), (2160, 3840), method=Image.Resampling.LANCZOS).save(output)
        preparation.append({'source': str(source), 'source_size': source_size,
                            'size': [2160, 3840], 'method': 'Lanczos'})
        files.append(output)
    return files


def compile_episode(source, number, destination):
    name=f'ep{number:02d}'
    art_dir=source/'art'/name
    artwork=read(art_dir/'index.json')
    voice=source/'voice'/name
    lines=read(voice/'index.json')
    by_id={row['id']:row for row in lines}
    asr=read(voice/'asr.json')['segments']
    episode=next(ep for ep in read(source/'script.json')['episodes'] if ep['number']==number)
    script=[beat for scene in episode['scenes'] for beat in scene['beats'] if beat['type']=='dialogue']
    if [(l['id'],l['text']) for l in lines]!=[(s['id'],s['text']) for s in script]:
        raise ValueError(f'{name}: cast dialogue differs from script')
    ids=[i for run in artwork['runs'] for i in run['line_ids']]
    if ids!=[line['id'] for line in lines]:
        raise ValueError(f'{name}: storyboard dialogue order/coverage differs from script: {ids}')
    frames=destination/'frames'
    frames.mkdir(parents=True,exist_ok=True)
    shutil.copytree(voice,destination/'voice',dirs_exist_ok=True)
    image_preparation = []
    write(destination/'scene-script.json', episode)
    groups=[]
    for idx,run in enumerate(artwork['runs'],1):
        setup=f'E{number:02d}C{idx:02d}'
        files = prepare_keyframes(art_dir, frames, run['frames'], setup, image_preparation)
        if 'dialogue_chunks' in run:
            chunks = run['dialogue_chunks']
        else:
            parts = [part for key in run['line_ids'] for part in split_line(by_id[key], asr[key]['words'])]
            chunks = pack_parts(parts)
        beats = run.get('camera_beats', [{} for _ in chunks])
        if len(beats) != len(chunks):
            raise ValueError(f'{setup}: one camera beat is required per dialogue chunk')
        pickup = run.get('pickup')
        cut_index = pickup['segment'] - 1 if pickup else len(chunks)
        pickup_end = pickup.get('through_segment', len(chunks)) if pickup else len(chunks)
        if pickup:
            if not 0 < cut_index < pickup_end <= len(chunks):
                raise ValueError(f'{setup}: pickup must cover existing continuation segments')
            pickup_files = prepare_keyframes(art_dir, frames, pickup['frames'], setup + '-pickup', image_preparation)
        chain = [files[0].name] * len(chunks) + [files[1].name]
        segs = []
        head = files[0]
        for j, (chunk, beat) in enumerate(zip(chunks, beats)):
            shot_id = f'{setup}S{j+1:02d}'
            in_pickup = cut_index <= j < pickup_end
            cut_in = j > 0 and (j in (cut_index, pickup_end) or bool(beat.get('frame')))
            continuing = j > 0 and not cut_in
            if j == cut_index:
                head = pickup_files[0]
            elif j == pickup_end:
                head = files[0]
            if beat.get('frame'):
                head, _ = prepare_keyframes(art_dir, frames, [beat['frame']] * 2, shot_id, image_preparation)
            endpoint = pickup_files[1] if in_pickup else files[1]
            actual_endpoint = (j + 1 == (pickup_end if in_pickup else len(chunks))
                               and (pickup['frames'][0] != pickup['frames'][1] if in_pickup
                                    else run['frames'][0] != run['frames'][1]))
            open_tail = beat.get('open_tail', not actual_endpoint)
            speaking = sum(p['end'] - p['start'] for p in chunk) + .13 * max(0, len(chunk) - 1)
            duration = max(2.5, speaking + .38) if chunk else 3.0
            overhead = CONTEXT if continuing else 0
            count = beat.get('num_frames', 5 + 17 * math.ceil((FPS * duration + overhead - 5) / 17))
            phrases = [sentence_fragment(part, by_id[part['id']], asr[part['id']]['words']) for part in chunk]
            speakers = [SPEAKERS.get(by_id[part['id']]['speaker'], 'the on-screen speaker') for part in chunk]
            action = beat.get('action_en')
            if not action:
                action = pickup['action_en'] if in_pickup else run['action_en']
                if continuing:
                    action += (' Start in the action already underway in the carried-over frames. '
                               'Advance that action from its current state while preserving direction and eyelines.')
                action += (' Visible background people make small independent actions already appropriate to this '
                           'setting, such as turning to a neighbor or adjusting a cup, at staggered times. '
                           'Continue the ongoing movement through the last frames.')
                if speakers:
                    action += (' The timed Mandarin speech is performed once in this order: '
                               + ' then '.join(speakers) + '. Only the active speaker articulates the words.')
            sound = beat.get('soundscape_en', 'Natural location sounds, distinct close voices and soft room tone.')
            segment = dict(id=shot_id, prompt=format_prompt(action, count, open_tail, sound), num_frames=count,
                           dialogue_parts=chunk, keyframes=['frames/' + p.name for p in (head, head if open_tail else endpoint)])
            if open_tail:
                segment['open_tail'] = True
            if cut_in:
                segment['cut_in'] = True
            if beat.get('seed') is not None:
                segment['seed'] = beat['seed']
            if beat.get('crowd_regions'):
                segment['crowd_regions'] = beat['crowd_regions']
            if anchor := beat.get('motion_anchor'):
                if not overhead < anchor['frame'] < count-1:
                    raise ValueError(f'{shot_id}: action anchor must be inside the sampled action')
                picture = next((f for f,n in zip(files,run['frames']) if n == anchor['image']),None)
                if picture is None:
                    raise ValueError(f'{shot_id}: action anchor must name one of the camera setup pictures')
                segment['motion_anchor'] = dict(anchor, image='frames/'+picture.name)
            if chunk:
                segment['speech_text'] = ''.join(phrases)
                guide = Path('guides') / (shot_id + '.wav')
                write_voice_guide(voice, by_id, chunk, count, continuing, destination / guide)
                segment['guide_audio'] = str(guide)
            segs.append(segment)
        groups.append({'frames': ['frames/' + file for file in chain], 'segments': segs})
    cfg=dict(native_audio=True,check_native_shots=True,check_motion=True,width=1088,height=1920,fps=FPS,steps=8,context_length=CONTEXT,seed=30000+100*number,
             master_prompt='Photorealistic live-action contemporary Chinese vertical drama. Match the anchor faces, wardrobe and location. Keep the on-camera performers and spatial continuity consistent in a single uninterrupted shot. Preserve a clean photographic frame.',groups=groups)
    artwork['frame_preparation'] = image_preparation
    write(destination/'storyboard.json', artwork)
    write(destination/'project.json',cfg)
    print(f'{name}: {len(lines)} original lines, {len(groups)} camera setups, {sum(len(g["segments"]) for g in groups)} H3 segments', flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('episode',type=int)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    compile_episode(args.source,args.episode,args.destination)
