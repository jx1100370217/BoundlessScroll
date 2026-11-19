"""Measure movement on native takes and identify safe trims at the cuts."""
import math


FPS = 24


def read_frames(path):
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(cv2.resize(frame, (192, 340), interpolation=cv2.INTER_AREA))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f'Could not decode {path}')
    return np.stack(frames)


def stationary_edges(frames):
    import cv2
    import numpy as np

    differences = np.array([cv2.absdiff(a, b).mean() for a, b in zip(frames, frames[1:])])
    if not len(differences):
        return 0, 0, 0.
    # Require effectively the same image, even when a few pixels flicker during encoding.
    threshold = min(.55, max(.32, float(np.median(differences)) * .32))
    def count(values):
        return next((index for index, value in enumerate(values) if value > threshold), len(values))
    return count(differences), count(differences[::-1]), round(float(np.median(differences)), 3)


def stabilize(previous, current):
    import cv2
    import numpy as np

    points = cv2.goodFeaturesToTrack(previous, 180, .012, 5)
    if points is None or len(points) < 8:
        return None
    targets, valid, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None)
    if targets is None:
        return None
    valid = valid.ravel() != 0
    if int(valid.sum()) < 8:
        return None
    transform, support = cv2.estimateAffinePartial2D(points[valid], targets[valid], ransacReprojThreshold=1.2)
    if transform is None or support.mean() < .35 or not np.isfinite(transform).all():
        return None
    h,w = current.shape
    return cv2.warpAffine(previous, transform, (w,h), borderMode=cv2.BORDER_REPLICATE)


def crowd_motion(frames, regions):
    import cv2
    import numpy as np

    height,width = frames.shape[1:]
    activity = [[] for _ in regions]
    boxes = [(round(x*width),round(y*height),round(right*width),round(bottom*height))
             for x,y,right,bottom in regions]
    if any(not (1 <= a < c <= width-1 and 1 <= b < d <= height-1) for a,b,c,d in boxes):
        raise ValueError('Crowd regions must be inside the visible picture')
    for offset in range(6, len(frames), 6):
        current = frames[offset]
        before = stabilize(frames[offset-6], current)
        if before is None:
            continue
        delta = cv2.absdiff(before, current)
        for samples,(x,y,right,bottom) in zip(activity,boxes):
            samples.append(round(float((delta[y:bottom,x:right] >= 8).mean()),4))
    result = []
    for region, samples in zip(regions, activity):
        moving = sum(value >= .02 for value in samples)
        result.append({'region': region, 'compared_windows': len(samples), 'moving_windows': moving,
                       'moving_area': samples, 'needs_visual_review': len(samples) < 4 or moving < 2})
    return result


def motion_edits(timeline, path, receipts, regions):
    import cv2

    analyses = []
    for shot, quality, crowd in zip(timeline, receipts, regions, strict=True):
        frames = read_frames(path / shot['path'])
        if len(frames) != shot['frames']:
            raise ValueError(f"Motion check changed the frame count of {shot['path']}")
        still_head,still_tail,median = stationary_edges(frames)
        words = quality.get('words', [])
        lead = max(0, math.floor(words[0]['start'] * FPS) - 5) if words else 0
        tail = max(0, len(frames) - math.ceil(words[-1]['end'] * FPS) - 6) if words else 0
        head_cut = min(still_head, lead, 18)
        tail_cut = min(still_tail, tail, 18)
        analyses.append({'head': head_cut if head_cut >= 4 else 0, 'tail': tail_cut if tail_cut >= 4 else 0,
                         'head_frames': frames[:19], 'tail_frames': frames[-19:],
                         'near_static_head_frames': still_head, 'near_static_tail_frames': still_tail,
                         'median_frame_change': median, 'crowd': crowd_motion(frames, crowd) if crowd else []})
    links = []
    if analyses:
        # Keep the planned opening and the end of an episode. Trim only around internal joins.
        analyses[0]['head'] = analyses[-1]['tail'] = 0
    for index, (previous, current) in enumerate(zip(analyses, analyses[1:])):
        continuous = timeline[index+1].get('continues', False)
        left = previous['tail_frames']; right = current['head_frames']
        difference = float(cv2.absdiff(left[-1], right[0]).mean())
        trimmed = float(cv2.absdiff(left[-1-previous['tail']], right[current['head']]).mean())
        if continuous and trimmed > difference + .7:
            previous['tail'] = current['head'] = 0
            trimmed = difference
        links.append({'from': timeline[index]['path'], 'to': timeline[index+1]['path'],
                      'same_camera': continuous, 'grey_difference': round(difference,3),
                      'edited_grey_difference': round(trimmed,3),
                      'tail_trim_frames': previous['tail'], 'head_trim_frames': current['head']})
    elapsed = 0
    shots = []
    for slot,row in zip(timeline, analyses):
        trim_in,trim_out = row.pop('head'),row.pop('tail')
        row.pop('head_frames'); row.pop('tail_frames')
        count = slot['frames'] - trim_in - trim_out
        slot.update(source_start_frame=trim_in, source_end_frame_exclusive=slot['frames']-trim_out,
                    frames=count, start_frame=elapsed, end_frame_exclusive=elapsed+count)
        elapsed += count
        shots.append(dict(row, path=slot['path']))
    return {'fps': FPS, 'frames': elapsed, 'shots': shots, 'joins': links}
