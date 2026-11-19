"""Build and render the individual episodes with the local H3 service."""
import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from build_episode import compile_episode
from finish_episode import delivery_size, finish
from production_progress import track
from run_workflow import run

THERE=Path(__file__).resolve().parent
MUSIC=('cold_distance','pressure','pressure','cold_distance','pressure','repair','truth','warm_home',
       'warm_home','pressure','truth','cold_distance','repair','pressure','truth','warm_home','pressure',
       'pressure','truth','cold_distance','pressure','truth','cold_distance','wedding','wedding')

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def validate(movie, timeline, lines):
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(movie)]))
    streams={s['codec_type']:s for s in info['streams']}
    v,a=streams['video'],streams['audio']
    if (int(v['nb_read_frames']),v['r_frame_rate'],v['width'],v['height'])!=(timeline['total_frames'],'24/1',*delivery_size(timeline['width'], timeline['height'])):
        raise ValueError(f'Invalid video geometry or frame count: {movie}')
    if a['channels']!=2 or a['sample_rate']!='48000' or abs(float(a['duration'])-timeline['total_frames']/24)>.055:
        raise ValueError(f'Invalid episode sound: {movie}')
    words=''.join(b.split('\n',2)[2] for b in movie.with_suffix('.srt').read_text().strip().split('\n\n'))
    expected=''.join(line['text'] for line in lines)
    chars=lambda text: ''.join(ch for ch in text if ch.isalnum())
    if chars(words)!=chars(expected):
        raise ValueError(f'Final captions differ from the script: {movie}')

def render_series(source,episodes,start,end):
    script=read(source/'script.json')['episodes']
    episodes.mkdir(parents=True,exist_ok=True)
    for ep in script[start-1:end]:
        number=ep['number']
        folder=f'ep{number:02d}'
        title=f'第{number:02d}集_{ep["title"]}'
        project=THERE/'examples'/folder
        dest=THERE/'renders'/folder
        ready=[source/'art'/folder/'index.json',source/'voice'/folder/'index.json',source/'voice'/folder/'asr.json']
        if number!=1:
            with track(project/'project.json',dest,'preparing','planned',fresh=True) as progress:
                while not all(p.is_file() for p in ready):
                    progress.update('waiting_assets')
                    print(f'Waiting for {folder} assets',flush=True)
                    time.sleep(30)
                compile_episode(source,number,project)
        lines=read(project/'voice/index.json')
        out=dest/(title+'.mp4')
        # Running the sampler checks every graph signature and completed clip before reusing it.
        run((project/'project.json').resolve(),dest.resolve())
        music=THERE/'examples'/'music'/(MUSIC[number-1]+'.flac')
        finish((project/'project.json').resolve(),dest.resolve(),music.resolve(),out.resolve())
        config=read(project/'project.json')
        if config.get('native_episode'):
            report=read(project/config['native_episode']/'native-release.json')
            timeline=dict(total_frames=report['frames'],width=report['source_size'][0],height=report['source_size'][1])
        else:
            timeline=read(dest/'timeline.json')
        with track(project/'project.json',dest,'validating','exported',delivery=out):
            validate(out,timeline,lines)
            for item in [out,out.with_suffix('.srt')]:
                tmp=episodes/(item.name + '.partial')
                shutil.copyfile(item,tmp)
                tmp.replace(episodes/item.name)
        print(f'EPISODE VERIFIED {number:02d} {title} {timeline["total_frames"]/24:.3f}s',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('episodes',type=Path)
    parser.add_argument('--start',type=int,default=2)
    parser.add_argument('--end',type=int,default=25)
    a=parser.parse_args()
    if not 1<=a.start<=a.end<=25:parser.error('Choose episodes from 1 through 25')
    render_series(a.source.resolve(),a.episodes.resolve(),a.start,a.end)
