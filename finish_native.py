"""Master native H3 picture and its own stereo audio at portrait 1080p, with frame-aligned captions."""
from pathlib import Path
from tempfile import NamedTemporaryFile,TemporaryDirectory
import argparse,hashlib,json,math,re,subprocess
import numpy as np,soundfile as sf
from scipy.signal import resample_poly
from math import gcd
base=Path(__file__).resolve().parent
from finish_episode import captions,character_times,caption_cues
def info(path,count=False):
 args=['ffprobe','-v','error']
 if count:args+=['-count_frames']
 return json.loads(subprocess.check_output([*args,'-show_streams','-show_format','-of','json',str(path)]))
def ff(*args,cwd=None):
 subprocess.run(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-y',*map(str,args)],cwd=cwd,check=True)
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def clean(text):return ''.join(c for c in text if c.isalnum())
def master(root,out,source):
 episode=read(root/'episode.json');script=read(source)['source_utterances'];expected=''.join(v['text'] for v in script)
 if clean(''.join(x.get('text','') for x in episode))!=clean(expected):raise ValueError('Edit does not cover the whole screenplay dialogue')
 out.parent.mkdir(parents=True,exist_ok=True);cues=[];samples=[];edits=[];overhangs=[];cursor=0
 with TemporaryDirectory(dir=out.parent,prefix='.native-cut-') as tmp:
  tmp=Path(tmp);playlist=[]
  for i,item in enumerate(episode):
   name=item['id'];record=read(root/'accepted'/(name+'.json'));rec=record['receipt'];graph=rec['graph']
   current=read(root/'render/receipts'/(name+'.json'))
   if current.get('job')!=rec['job']:
    raise ValueError(f'{name}: the current take still needs dialogue and picture review')
   video=root/'render'/rec['video'];audio=root/'render'/rec['audio']
   node=graph['8'];assert node['inputs']['width']==1088 and node['inputs']['height']==1920
   assert graph['17']['inputs']['audio']==['25',0] and graph['25']['class_type']=='VAEDecodeAudio' and graph['25']['inputs']['samples']==['14',0]
   assert graph['15']['inputs']['samples']==['14',0] and graph['17']['inputs']['images']==['15',0]
   streams=info(video)['streams'];v=next(s for s in streams if s['codec_type']=='video');a=next(s for s in streams if s['codec_type']=='audio')
   n=int(v['nb_frames']);trim=item.get('trim',0);delivery=n-trim
   assert (v['width'],v['height'],v['r_frame_rate'],n)==(1088,1920,'24/1',item['frames']) and a['channels']==2
   assert delivery>0
   clipdir=tmp/f'{i:02d}';clipdir.mkdir()
   words=record['words'];text=item.get('text','');local=[]
   adjust=record.get('audio_advance_seconds',0.)
   if abs(adjust)>.16:raise ValueError(f'{name}: time correction exceeds the reviewed small alignment change')
   if text:
    if words['expected']!=text:raise ValueError(f'{name}: wrong script')
    if words['phonetic_distance'] and not record.get('reviewed_text'):raise ValueError(f'{name}: dialogue verification pending')
    timed=character_times(text,words['words'],delivery/24)
    local=caption_cues(text,timed,0,delivery/24,1.)
    local=[(max(0,x-adjust),min(delivery/24,y-adjust),line) for x,y,line in local]
    if clean(''.join(c[2] for c in local))!=clean(text):raise ValueError(f'{name}: lost captions')
    cues.extend((x+cursor/24,y+cursor/24,t) for x,y,t in local)
   vf=f'trim=start_frame={trim}:end_frame={n},setpts=PTS-STARTPTS,crop=1080:1920:4:0,setsar=1'
   if local:
    captions(local,clipdir,1080,1920);vf+=',ass=captions.ass'
   clip=tmp/f'{i:02d}.mp4'
   ff('-i',video,'-map','0:v:0','-an','-vf',vf,'-threads','5','-c:v','libx264','-preset','fast','-crf','15','-pix_fmt','yuv420p','-movflags','+faststart',clip,cwd=clipdir)
   if int(info(clip)['streams'][0]['nb_frames'])!=delivery:raise ValueError(f'{name}: missing picture frames')
   playlist.extend([f"file '{clip.name}'",f'duration {delivery/24:.9f}'])
   wave,sr=sf.read(audio,dtype='float32',always_2d=True)
   if wave.shape[1]!=2 or sr!=32000 or abs(len(wave)/sr-n/24)>.035:raise ValueError(f'{name}: native AV duration inconsistent')
   g=gcd(sr,48000);wave=resample_poly(wave,48000//g,sr//g,axis=0)
   wave=wave[trim*2000:n*2000]
   gain_db=float(record.get('dialogue_gain_db',0.))
   if not -12 <= gain_db <= 3:raise ValueError(f'{name}: dialogue gain exceeds the reviewed mix')
   if gain_db:wave*=10**(gain_db/20)
   if len(wave)<delivery*2000:wave=np.pad(wave,((0,delivery*2000-len(wave)),(0,0)))
   preserve=bool(record.get('preserve_boundary_audio') and adjust)
   fade=np.sin(np.linspace(0,np.pi/2,192,dtype='float32'))[:,None]**2
   if adjust:
    delta=round(abs(adjust)*48000)
    if preserve:
     fragment=(wave[:delta] if adjust>0 else wave[-delta:]).copy()
     if adjust>0:fragment[:192]*=fade
     else:fragment[-192:]*=fade[::-1]
     at=cursor*2000-delta if adjust>0 else (cursor+delivery)*2000
     overhangs.append((at,fragment))
    if adjust>0:wave=np.pad(wave[delta:],((0,delta),(0,0)))
    else:wave=np.pad(wave[:-delta],((delta,0),(0,0)))
   # A 4 ms de-click at editorial joins leaves all speech and action in their original positions.
   if not preserve or adjust<0:wave[:192]*=fade
   if not preserve or adjust>0:wave[-192:]*=fade[::-1]
   samples.append(wave)
   edits.append({'id':name,'source_shot':item.get('source_shot',name),'job':rec['job'],'video':rec['video'],'audio':rec['audio'],'motion_context_trim':trim,'native_audio_advance_seconds':adjust,
                 'source_frames':n,'start_frame':cursor,'end_frame_exclusive':cursor+delivery,'native_size':[1088,1920],'dialogue_gain_db':gain_db,'preserve_boundary_audio':preserve,
                 'source_video_sha256':sha(video),'source_audio_sha256':sha(audio),'text':text,'recognized':words.get('recognized',''),'sync':record['sync']})
   cursor+=delivery;print('conformed',name,delivery,flush=True)
  if clean(''.join(c[2] for c in cues))!=clean(expected):raise ValueError('Final captions omit spoken words')
  raw=np.concatenate(samples,axis=0);assert len(raw)==cursor*2000
  for start,part in overhangs:
   end=start+len(part)
   if start<0 or end>len(raw):raise ValueError('Audio overhang outside the episode')
   bed=raw[start:end]
   if np.sqrt(np.mean(bed**2))>.016:raise ValueError('Audio overhang overlaps audible dialogue at the cut')
   bed+=part
  peak=float(np.max(abs(raw)));gain=min(1.,.89/max(peak,.00001))
  sf.write(tmp/'h3-stereo.wav',raw*gain,48000,subtype='PCM_24')
  (tmp/'frames.txt').write_text('\n'.join(playlist)+'\n')
  captions(cues,tmp,1080,1920)
  dest=tmp/'finished.mp4'
  ff('-f','concat','-safe','1','-i',tmp/'frames.txt','-i',tmp/'h3-stereo.wav','-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-b:a','256k','-metadata','title=装穷赴婚约 · H3原生1080p','-metadata:s:a:0','language=zho','-movflags','+faststart',dest)
  d=info(dest,True);v,a=d['streams']
  if (v['width'],v['height'],v['r_frame_rate'],int(v['nb_read_frames']))!=(1080,1920,'24/1',cursor) or abs(float(v['duration'])-float(a['duration']))>.045 or abs(float(v['start_time']))>.001:raise ValueError('Final container failed verification')
  report=dict(source_size=[1088,1920],delivery_size=[1080,1920],crop_left=4,crop_right=4,frames=cursor,seconds=cursor/24,fps=24,shots=edits,
              audio_source='Native H3 decoded stereo waveform from the same sampler as each image sequence',sound_gain=gain,caption_count=len(cues),file_sha256=sha(dest))
  with NamedTemporaryFile('w',dir=root,prefix='.native-release-',suffix='.json',delete=False,encoding='utf-8') as pending:
   pending.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
  dest.replace(out)
  (tmp/'captions.srt').replace(out.with_suffix('.srt'))
  Path(pending.name).replace(root/'native-release.json')
 print(out,cursor,cursor/24,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('--script',type=Path,default=base/'examples/ep01/storyboard.json')
 a=p.parse_args();master(a.root.resolve(),a.output.resolve(),a.script.resolve())
