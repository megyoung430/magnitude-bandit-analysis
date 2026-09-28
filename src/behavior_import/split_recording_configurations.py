"""Keep restarted recordings with different arm sets in separate segments."""
import json


def _signature(frame):
    rows = frame.loc[(frame['type'] == 'variable') & (frame['subtype'] == 'run_start'), 'content']
    if rows.empty:
        return None, None
    raw = rows.iloc[0]
    state = raw if isinstance(raw, dict) else json.loads(raw)
    magnitudes = state.get('current_reward_magnitudes', state.get('curr_rew_mag'))
    initiation = state.get('current_initiation_tower', state.get('curr_initiation_tow', state.get('initiation_tower')))
    return (tuple(sorted(magnitudes)) if magnitudes else None), initiation


def split_recording_configurations(data):
    """Split only multi-file sessions whose known tower configuration changes.

    Retains both recordings and their original session key as provenance. The
    suffixed segment keys sort in recording order and allow existing problem
    grouping to assign each configuration correctly. Same-task restarts are
    still merged as before. Calling this repeatedly is safe.
    """
    for mouse,sessions in data.items():
        for key,session in list(sessions.items()):
            frames = session.get('data')
            if not isinstance(frames,list) or len(frames) < 2:
                continue
            groups, current, previous = [], [], None
            for i,frame in enumerate(frames):
                signature = _signature(frame)
                if previous is not None and signature[0] is not None and previous[0] is not None and signature != previous:
                    groups.append(current)
                    current = []
                current.append(i)
                previous = signature
            groups.append(current)
            if len(groups) == 1:
                continue
            del sessions[key]
            for part,indices in enumerate(groups,1):
                segment_key = f'{key}_part-{part:02d}'
                if segment_key in sessions:
                    raise ValueError(f'Segment key already exists: {segment_key}')
                segment = dict(session)
                segment['source_session'] = key
                for field in ('data','df'):
                    values = session.get(field)
                    if isinstance(values,list):
                        selected = [values[i] for i in indices]
                        segment[field] = selected[0] if len(selected)==1 else selected
                towers,initiation = _signature(frames[indices[0]])
                if towers is not None:segment['choice_towers'] = set(towers)
                if initiation is not None:segment['initiation_tower'] = initiation
                sessions[segment_key] = segment
            print(f'[INFO] Split {mouse} {key} into {len(groups)} recording segments because the tower configuration changed.')
    return data
