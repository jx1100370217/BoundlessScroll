#!/usr/bin/env python3
"""Generate the storyboard's original frames with the local novel-h3-studio image provider."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from PIL import Image

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('image_plan', type=Path)
parser.add_argument('--studio', type=Path, default=Path(__file__).resolve().parent.parent/'novel-h3-studio')
args = parser.parse_args()
plan = args.image_plan.resolve()
assets = plan.parent / 'assets'
frames = plan.parent / 'frames'
assets.mkdir(exist_ok=True)
frames.mkdir(exist_ok=True)
studio = args.studio.resolve()
for task in json.loads(plan.read_text()):
    target = assets / task['file']
    if not target.exists():
        request = dict(prompt=task['prompt'], output_path=str(target),
                       reference_paths=[str(assets / name) for name in task['references']],
                       image_model='gpt-image-2.5-sunburst')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json') as payload:
            json.dump(request, payload, ensure_ascii=False)
            payload.flush()
            result = subprocess.run([str(studio / '.venv-openmontage/bin/python'),
                                     'studio.py', 'tool', 'codex_image', '@' + payload.name],
                                    cwd=studio, check=True, text=True, stdout=subprocess.PIPE)
            response = json.loads(result.stdout)
            if not response['success']:
                raise RuntimeError(response.get('error', 'Image generation failed'))
    with Image.open(target) as image:
        image.load()
        cells = task.get('cells', [])
        for n, name in enumerate(cells):
            if name is None:
                continue
            cols = task['columns']
            rows = len(cells) // cols
            x, y = n % cols, n // cols
            w, h = image.size
            # Trim the two-pixel contact-sheet boundary before encoding each anchor.
            box = (round(w*x/cols)+2, round(h*y/rows)+2,
                   round(w*(x+1)/cols)-2, round(h*(y+1)/rows)-2)
            image.crop(box).save(frames / (name + '.png'))
    print(target.name, flush=True)
