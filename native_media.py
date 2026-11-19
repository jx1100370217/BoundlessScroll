"""Frame-exact concatenation and subtitle delivery of H3's own audio/video samples."""
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import subprocess

class DialogueMismatch(ValueError):
    def __init__(self, shot, recognized):
        self.shot = shot
        super().__init__(f"{shot}: native dialogue differs from the script: {recognized}")


def join_native(clips, output):
    """Each input MP4 has already had both sides of the motion-context overlap removed."""
    from finish_episode import ffmpeg, probe
    with TemporaryDirectory(dir=output.parent, prefix='.native-join-') as temporary:
        work = Path(temporary)
        entries = []
        total = 0
        with (work/'stereo.pcm').open('wb') as pcm:
            for i,(clip, frames) in enumerate(clips):
                video = work/f'{i:04}.mp4'
                ffmpeg('-i',clip,'-map','0:v:0','-an','-c:v','copy',video)
                sound = probe(clip,'a:0','channels')
                if sound.get('channels')!=2:
                    raise ValueError(f'Missing native stereo audio at {clip}')
                subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(clip),'-map','0:a:0',
                                '-af',f'aresample=48000,apad,atrim=end_sample={frames*2000}',
                                '-f','s16le','-ac','2','pipe:1'], check=True,stdout=pcm)
                entries.extend([f"file '{video.name}'",f'duration {frames/24:.9f}'])
                total += frames
        if (work/'stereo.pcm').stat().st_size != total*2000*4:
            raise ValueError('Native sound length does not match the delivered video frames')
        (work/'videos.txt').write_text('\n'.join(entries)+'\n')
        ffmpeg('-f','concat','-safe','1','-i',work/'videos.txt','-f','s16le','-ar','48000',
               '-ac','2','-i',work/'stereo.pcm','-map','0:v:0','-map','1:a:0','-c:v','copy',
               '-c:a','aac','-b:a','256k','-movflags','+faststart',output)


def finish_tracks(project, render, output):
    """Time captions to the native sampled dialogue and keep the native stereo stream."""
    import numpy as np
    from faster_whisper import WhisperModel
    from finish_episode import captions, caption_cues, character_times, delivery_scale, delivery_size, ffmpeg, probe
    from native_words import canonical, distance
    from pypinyin import lazy_pinyin
    timeline=json.loads((render/'timeline.json').read_text())
    clips=[seg for group in project['groups'] for seg in group['segments']]
    if len(clips)!=len(timeline['segments']):
        raise ValueError('Shot list and sampled timeline disagree')
    source=render/'visual.mp4';v=probe(source,'v:0','width,height,nb_frames,r_frame_rate')
    if int(v['nb_frames'])!=timeline['total_frames'] or v['r_frame_rate']!='24/1':
        raise ValueError('Native master and frame timeline disagree')
    vf=delivery_scale(v['width'],v['height']);w,h=delivery_size(v['width'],v['height'])
    model=None;cues=[]
    for clip,slot in zip(clips,timeline['segments']):
        text=clip.get('speech_text','')
        if not text:continue
        video_path = render / slot["path"]
        receipt_path = video_path.with_suffix(".json")
        receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
        checked = receipt.get("quality", {})
        if checked.get("status") == "checked" and checked.get("expected") == text:
            words = checked["words"]
        else:
            if model is None:
                model=WhisperModel('mobiuslabsgmbh/faster-whisper-large-v3-turbo',device='cpu',
                                   compute_type='int8',cpu_threads=5)
            raw=subprocess.check_output(['ffmpeg','-nostdin','-v','error','-i',str(video_path),
                                         '-vn','-ar','16000','-ac','1','-f','f32le','pipe:1'])
            transcription=list(model.transcribe(np.frombuffer(raw,dtype=np.float32),language='zh',beam_size=5,
                               temperature=0,word_timestamps=True,condition_on_previous_text=False)[0])
            recognized=''.join(s.text for s in transcription)
            if distance(lazy_pinyin(canonical(text)),lazy_pinyin(canonical(recognized))):
                raise DialogueMismatch(clip["id"], recognized)
            words=[dict(text=word.word,start=word.start,end=word.end) for s in transcription for word in s.words]
        duration=slot['frames']/24;timing=character_times(text,words,duration)
        cues.extend(caption_cues(text,timing,slot['start_frame']/24,duration,1.))
    output.parent.mkdir(exist_ok=True,parents=True)
    with TemporaryDirectory(dir=output.parent,prefix='.native-finish-') as temporary:
        tmp=Path(temporary)
        if cues:
            captions(cues, tmp, w, h, serif=project.get("check_native_shots", False))
            vf += ",ass=captions.ass"
        ffmpeg('-i',source,'-map','0:v:0','-map','0:a:0','-vf',vf,'-c:v','libx264','-crf','15',
               '-preset','fast','-threads','5','-pix_fmt','yuv420p','-c:a','copy','-movflags','+faststart',tmp/'movie.mp4',cwd=tmp)
        vd=probe(tmp/'movie.mp4','v:0','width,height,nb_frames')
        if (vd['width'],vd['height'],vd['nb_frames'])!=(w,h,v['nb_frames']):
            raise ValueError('Native finish geometry failed verification')
        if cues:(tmp/'captions.srt').replace(output.with_suffix('.srt'))
        (tmp/'movie.mp4').replace(output)
