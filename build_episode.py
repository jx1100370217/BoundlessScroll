"""Compile a filmed episode from ordered camera setups and timed dialogue."""
from pathlib import Path
import argparse
import json
import math
import shutil
from PIL import Image, ImageOps

from finish_episode import character_times

FPS = 24
CONTEXT = 5

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

def intermediate_frames(files, chunks, dest, label):
    first,last=files
    chain=[first.name]
    for k in range(1,chunks):
        use_first = k <= chunks/2
        path = first if use_first else last
        closeness = min(k,chunks-k) / (chunks/2)
        with Image.open(path) as im:
            # A tighter composition at a dialogue beat keeps the same actor and camera axis.
            w,h=im.size
            ratio=.055*closeness
            dx,dy=int(w*ratio),int(h*ratio)
            picture=im.crop((dx,dy,w-dx,h-dy)).resize((w,h),Image.Resampling.LANCZOS)
            name=f'{label}-detail-{k:02d}.png'
            picture.save(dest/name)
        chain.append(name)
    return chain+[last.name]

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
        files=[]
        for n,raw in enumerate(run['frames']):
            file=art_dir/Path(raw).name
            dest=frames/(f'{setup}-{n+1}.png')
            with Image.open(file) as im:
                source_size = list(im.size)
                ImageOps.fit(im.convert('RGB'), (2160, 3840),
                             method=Image.Resampling.LANCZOS).save(dest)
            image_preparation.append({'source': str(file), 'source_size': source_size,
                                      'size': [2160, 3840], 'method': 'Lanczos'})
            files.append(dest)
        if len(files)!=2:
            raise ValueError(f'{setup}: expected two anchors for the camera setup')
        if 'dialogue_chunks' in run:
            chunks = run['dialogue_chunks']
        else:
            parts = [part for key in run['line_ids'] for part in split_line(by_id[key], asr[key]['words'])]
            chunks = pack_parts(parts)
        chain=intermediate_frames(files,len(chunks),frames,setup)
        segs=[]
        for j,chunk in enumerate(chunks):
            speaking=sum(p['end']-p['start'] for p in chunk)+.13*max(0,len(chunk)-1)
            duration=max(2.7, speaking+.55) if chunk else 3.0
            count=5+17*math.ceil((FPS*duration+(CONTEXT if j else 0)-5)/17)
            end=(count-1)/FPS
            phrases = [sentence_fragment(part, by_id[part['id']], asr[part['id']]['words']) for part in chunk]
            speakers = [SPEAKERS.get(by_id[part["id"]]["speaker"], "the on-screen speaker")
                        for part in chunk]
            spoken = ("The timed acoustic cue is performed aloud in Mandarin exactly once, in this order: "
                      + " then ".join(speakers)
                      + ". Match the audible words and pauses. The active speaker moves their own lips "
                      "and jaw naturally. Everyone else listens silently.") if speakers else ""
            action=run['action_en']
            if j and len(chunks)>1:
                action+=' Continue the moment in the same space. Preserve actor positions and eyelines.'
            prompt=(f'How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the {end:.2f}-second mark of the target video.\n\n'
                    'integrated_multimodal_description: [Shot 1] '+action
                    +' Preserve the first and last framing. Natural breathing, purposeful hand movements and attentive reactions. '
                    +spoken + ' A clean photographic image shows the performers, props and physical setting. '
                    +'\n\noverall_soundscape: Natural location sounds; clothing and footsteps are close, voices are distinct.\n\nnon_diegetic_music: N/A')
            shot_id=f'{setup}S{j+1:02d}'
            segment=dict(id=shot_id,prompt=prompt,num_frames=count,dialogue_parts=chunk)
            if chunk:
                segment['speech_text'] = ''.join(phrases)
                guide=Path('guides') / (shot_id+'.wav')
                write_voice_guide(voice,by_id,chunk,count,j>0,destination/guide)
                segment['guide_audio']=str(guide)
            segs.append(segment)
        groups.append({'frames':['frames/'+f for f in chain], 'segments':segs})
    cfg=dict(native_audio=True,check_native_shots=True,width=1088,height=1920,fps=FPS,steps=8,context_length=CONTEXT,seed=30000+100*number,
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
