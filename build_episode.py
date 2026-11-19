"""Compile a filmed episode from ordered camera setups and timed dialogue."""
from pathlib import Path
import argparse
import json
import math
import shutil
from PIL import Image

from finish_episode import character_times

FPS = 24
CONTEXT = 5

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def split_line(line, words):
    length = line['duration']
    count = max(1, math.ceil(length / 3.45))
    ends = [0.]
    for i in range(1, count):
        ideal = ends[-1] + (length - ends[-1]) / (count - i + 1)
        choices = []
        for a, b in zip(words, words[1:]):
            time = (a['end'] + b['start']) / 2
            if abs(time - ideal) < .80 and time-ends[-1] <= 3.68 and length-time <= (count-i)*3.68:
                pause = max(0., b['start'] - a['end'])
                choices.append((abs(time-ideal) - min(pause, .4)*2, time))
        for a, b in zip(line.get('parts',[]), line.get('parts',[])[1:]):
            time = (a['end']+b['start'])/2
            if abs(time-ideal)<.90 and time-ends[-1] <= 3.68 and length-time <= (count-i)*3.68:
                choices.append((abs(time-ideal)-1, time))
        ends.append(round(min(choices)[1] if choices else ideal, 6))
    ends.append(length)
    return [dict(id=line['id'], start=a, end=b) for a,b in zip(ends, ends[1:])]

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
    write(destination/'storyboard.json', artwork)
    write(destination/'scene-script.json', episode)
    groups=[]
    for idx,run in enumerate(artwork['runs'],1):
        setup=f'E{number:02d}C{idx:02d}'
        files=[]
        for n,raw in enumerate(run['frames']):
            file=art_dir/Path(raw).name
            with Image.open(file) as im:
                im.verify()
            dest=frames/(f'{setup}-{n+1}.png')
            shutil.copyfile(file,dest)
            files.append(dest)
        if len(files)!=2:
            raise ValueError(f'{setup}: expected two anchors for the camera setup')
        parts=[]
        for key in run['line_ids']:
            parts.extend(split_line(by_id[key],asr[key]['words']))
        chunks=pack_parts(parts)
        chain=intermediate_frames(files,len(chunks),frames,setup)
        segs=[]
        for j,chunk in enumerate(chunks):
            speaking=sum(p['end']-p['start'] for p in chunk)+.13*max(0,len(chunk)-1)
            duration=max(2.7, speaking+.55) if chunk else 3.0
            count=5+17*math.ceil((FPS*duration+(CONTEXT if j else 0)-5)/17)
            end=(count-1)/FPS
            speakers=[by_id[part['id']]['speaker'] for part in chunk]
            spoken=('The spoken Mandarin follows the timed audio cue. In order, the speakers are '
                    +' then '.join(speakers)+'. Their lips follow their own phrases.') if speakers else ''
            action=run['action_en']
            if j and len(chunks)>1:
                action+=' Continue the moment in the same space. Preserve actor positions and eyelines.'
            prompt=(f'How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the {end:.2f}-second mark of the target video.\n\n'
                    'integrated_multimodal_description: [Shot 1] '+action
                    +' Preserve the first and last framing. Natural breathing, purposeful hand movements and attentive reactions. '
                    +spoken
                    +'\n\noverall_soundscape: Natural location sounds; clothing and footsteps are close, voices are distinct.\n\nnon_diegetic_music: N/A')
            shot_id=f'{setup}S{j+1:02d}'
            segment=dict(id=shot_id,prompt=prompt,num_frames=count,dialogue_parts=chunk)
            if chunk:
                segment['speech_text']=''.join(sentence_fragment(part,by_id[part['id']],asr[part['id']]['words']) for part in chunk)
                guide=Path('guides') / (shot_id+'.wav')
                write_voice_guide(voice,by_id,chunk,count,j>0,destination/guide)
                segment['guide_audio']=str(guide)
            segs.append(segment)
        groups.append({'frames':['frames/'+f for f in chain], 'segments':segs})
    cfg=dict(native_audio=True,width=1088,height=1920,fps=FPS,steps=8,context_length=CONTEXT,seed=30000+100*number,
             master_prompt='Photorealistic live-action contemporary Chinese vertical drama. Match the anchor faces, wardrobe and location. Keep the on-camera performers and spatial continuity consistent in a single uninterrupted shot. Preserve a clean photographic frame.',groups=groups)
    write(destination/'project.json',cfg)
    print(f'{name}: {len(lines)} original lines, {len(groups)} camera setups, {sum(len(g["segments"]) for g in groups)} H3 segments', flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('episode',type=int)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    compile_episode(args.source,args.episode,args.destination)
