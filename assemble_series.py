"""Assemble the verified episode masters with frame-aligned chapter and subtitle times."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
from fractions import Fraction

def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams',
                            '-show_chapters','-show_format','-of','json',str(path)]))
def call(args, **kwargs):
    subprocess.run(['ffmpeg','-hide_banner','-v','error','-y',*args],check=True,**kwargs)
def milliseconds(s):
    h,m,s,ms=map(int,re.split('[:,]',s))
    return ((h*60+m)*60+s)*1000+ms
def timestamp(ms):
    h,ms=divmod(ms,3600000);m,ms=divmod(ms,60000);s,ms=divmod(ms,1000)
    return f'{h:02}:{m:02}:{s:02},{ms:03}'

def assemble(episodes, script_file, output):
    script=json.loads(script_file.read_text())['episodes']
    if [e['number'] for e in script]!=list(range(1,26)):
        raise ValueError('Expected all 25 episodes in script order')
    output.parent.mkdir(parents=True,exist_ok=True)
    total=0
    cues=[]
    sizes=[]
    chapters=[';FFMETADATA1','title=嫌我穷退婚？我转头娶走受气包养女']
    with tempfile.TemporaryDirectory(dir=output.parent,prefix='.assemble-') as scratch:
        tmp=Path(scratch)
        playlist=[]
        for e in script:
            name=f'第{e["number"]:02d}集_{e["title"]}'
            file=episodes/(name+'.mp4')
            sub=episodes/(name+'.srt')
            info=probe(file)
            v=next(s for s in info['streams'] if s['codec_type']=='video')
            a=next(s for s in info['streams'] if s['codec_type']=='audio')
            if (v['width'],v['height'],Fraction(v['r_frame_rate']),v['codec_name'],int(a['sample_rate']),a['channels'])!=(1080,1920,24,'h264',48000,2):
                raise ValueError(f'Incompatible streams: {name}')
            n=int(v['nb_frames']);sizes.append(n)
            with (tmp/'dialogue.pcm').open('ab') as pcm:
                call(['-i',str(file),'-map','0:a','-af',f'apad,atrim=end_sample={n*2000}',
                      '-f','s16le','-ar','48000','-ac','2','pipe:1'],stdout=pcm)
            link=tmp/f'{e["number"]:02d}.mp4'
            call(['-i',str(file),'-map','0:v:0','-an','-c:v','copy',str(link)])
            playlist.extend([f"file '{link.name}'",f'duration {n/24:.9f}'])
            offset=round(total*1000/24)
            for block in sub.read_text().strip().split('\n\n'):
                _,t,words=block.split('\n',2)
                begin,end=t.split(' --> ')
                cues.append(f'{len(cues)+1}\n{timestamp(milliseconds(begin)+offset)} --> {timestamp(milliseconds(end)+offset)}\n{words}\n')
            chapters.extend(['[CHAPTER]','TIMEBASE=1/24',f'START={total}',f'END={total+n}',f'title=第{e["number"]}集 {e["title"]}'])
            total+=n
        if (tmp/'dialogue.pcm').stat().st_size!=total*2000*4:
            raise ValueError('Combined PCM duration differs from picture')
        (tmp/'clips.txt').write_text('\n'.join(playlist)+'\n')
        (tmp/'chapters.ffmeta').write_text('\n'.join(chapters)+'\n')
        call(['-f','concat','-safe','1','-i',str(tmp/'clips.txt'),'-f','s16le','-ar','48000','-ac','2','-i',str(tmp/'dialogue.pcm'),
              '-f','ffmetadata','-i',str(tmp/'chapters.ffmeta'),'-map','0:v:0','-map','1:a:0','-map_metadata','2','-map_chapters','2',
              '-c:v','copy','-c:a','aac','-b:a','192k','-movflags','+faststart',str(tmp/'full.mp4')])
        info=probe(tmp/'full.mp4')
        streams={s['codec_type']:s for s in info['streams']}
        if (int(streams['video']['nb_frames'])!=total or len(info['chapters'])!=len(script)
                or abs(float(streams['audio']['duration'])-total/24)>.05
                or any(abs(float(streams[k]['start_time']))>.001 for k in ('video','audio'))):
            raise ValueError('Combined movie failed frame, duration, or chapter validation')
        offset=0
        for chapter,n in zip(info['chapters'],sizes):
            if abs(float(chapter['start_time'])-offset/24)>.002:
                raise ValueError('Chapter timeline does not match the individual episode')
            offset+=n
        (tmp/'full.srt').write_text('\n'.join(cues)+'\n')
        (tmp/'full.srt').replace(output.with_suffix('.srt'))
        (tmp/'full.mp4').replace(output)
    print(f'{output}: 25 chapters, {total} frames, {total/24:.3f} seconds')
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['episodes','script','output']:
        parser.add_argument(name,type=Path)
    a=parser.parse_args();assemble(a.episodes,a.script,a.output)
