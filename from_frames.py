"""Prepare an H3 run from an ordered directory of keyframes, as in the author's workflow."""
import argparse
import json
import os
from pathlib import Path
import re

from finish_episode import delivery_scale

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('folder', type=Path)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--master', default='')
parser.add_argument('--prompts', type=Path, help='Plain text, one line per adjacent pair')
parser.add_argument('--lengths', default='73', help='One 17n+5 frame count, or comma-separated counts')
parser.add_argument('--size', type=int, nargs=2, default=(1088,1920), metavar=('WIDTH','HEIGHT'))
parser.add_argument('--start-index', type=int, default=0)
parser.add_argument('--count', type=int)
parser.add_argument('--loop', action='store_true')
args = parser.parse_args()
if args.start_index < 0 or (args.count is not None and args.count < 2):
    parser.error('start-index must be nonnegative and count must select at least two images')
if any(n < 32 or n % 32 for n in args.size):
    parser.error('Both image dimensions must be positive multiples of 32')
try:
    delivery_scale(*args.size)
except ValueError as exc:
    parser.error(str(exc))
def numerical_order(path):
    return [int(piece) if i % 2 else piece.casefold()
            for i, piece in enumerate(re.split(r'(\d+)', path.name))]
ext = {'.png','.jpg','.jpeg','.webp'}
images = sorted((p for p in args.folder.iterdir() if p.is_file() and p.suffix.lower() in ext),
                key=numerical_order)
images = images[args.start_index:][:args.count]
if len(images) < 2:
    parser.error('Select at least two keyframes')
if args.loop:
    images.append(images[0])
n = len(images) - 1
prompts = args.prompts.read_text().splitlines() if args.prompts else [''] * n
lengths = [int(v) for v in args.lengths.split(',')]
if len(lengths) == 1:
    lengths *= n
if len(prompts) != n or len(lengths) != n:
    parser.error(f'This selection requires {n} segment prompts and lengths')
if any(k < 22 or k % 17 != 5 for k in lengths):
    parser.error('Use 17n+5 frames per segment (22, 39, 56, 73, ...)')
group = {'frames':[os.path.relpath(p.resolve(), args.output.resolve().parent) for p in images],
         'segments':[{'id':f'S{i:03d}', 'prompt':s, 'num_frames':k}
                     for i,(s,k) in enumerate(zip(prompts,lengths),1)]}
cfg = {'width':args.size[0], 'height':args.size[1], 'fps':24, 'steps':8,
       'native_audio':True, 'context_length':5, 'master_prompt':args.master, 'groups':[group]}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n')
print(f'{args.output} ({n} segments)')
