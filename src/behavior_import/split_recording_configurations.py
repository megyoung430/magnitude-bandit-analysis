"""Keep restarted recordings with different arm sets in separate segments."""

from src.behavior_import.import_data import (
    RunStartError,
    read_run_start,
    settings_evidence,
    session_task_type,
    source_path,
)


def _signature(frame, mouse, key):
    """Sorted tower names and initiation tower of a recording's first ``run_start``.

    ``(None, None)`` if the recording has no ``run_start`` row; a corrupt row raises a
    ``ValueError`` naming the mouse, session and file.
    """
    is_run_start = (frame['type'] == 'variable') & (frame['subtype'] == 'run_start')
    if not is_run_start.any():
        return None, None
    try:
        state = read_run_start(frame)
    except RunStartError as err:
        raise ValueError(f'{mouse} {key} ({source_path(frame)}): {err}') from err
    magnitudes = state.get('current_reward_magnitudes', state.get('curr_rew_mag'))
    initiation = state.get('current_initiation_tower', state.get('curr_initiation_tow', state.get('initiation_tower')))
    return (tuple(sorted(magnitudes)) if magnitudes else None), initiation


def _segment_evidence(session, segment_frames):
    """Settings evidence of one segment; one starting without ``run_start`` keeps the session's."""
    first = segment_frames[0]
    if not ((first['type'] == 'variable') & (first['subtype'] == 'run_start')).any():
        return session['settings_evidence']
    return settings_evidence(segment_frames, read_run_start(first))


def split_recording_configurations(data):
    """Split only multi-file sessions whose known tower configuration changes.

    Retains both recordings and their original session key as provenance. The
    suffixed segment keys sort in recording order and allow existing problem
    grouping to assign each configuration correctly. Same-task restarts are
    still merged as before. Calling this repeatedly is safe. Segments do not inherit the
    parent's ``problem`` or ``problem_provisional``; ``group_sessions_by_problem`` assigns
    problems. The first segment keeps the parent's ``problem_flag``: ``import_data`` numbered
    the session by its first recording and warned about that flag, so
    ``group_sessions_by_problem`` must not warn about it again. The other segments carry none.

    Each segment is a shallow copy of the parent session with its own ``"data"`` / ``"df"``
    (one DataFrame, or a list if the segment has several recordings), ``"settings_evidence"``,
    ``"task_name"``, ``"choice_towers"``, ``"initiation_tower"`` and, when the parent had task
    types classified, ``"task_type"``. Each carries ``"source_session"`` (the parent's key).
    A split session's key ``ses-<N>_date-<YYYYMMDD>`` is replaced by
    ``ses-<N>_date-<YYYYMMDD>_part-01``, ``_part-02`` and so on.

    Args:
        data (dict): ``{subject: {session_key: session_dict}}`` as returned by
            :func:`src.behavior_import.import_data.import_data`. Mutated in place.

    Returns:
        dict: The same *data* object, with each split session replaced by its segments.

    Raises:
        ValueError: If a recording's first ``run_start`` row is not valid JSON (naming mouse,
            session and file), if a segment key already exists, or if a segment's task name
            cannot be classified when task types are classified.
    """
    for mouse,sessions in data.items():
        for key,session in list(sessions.items()):
            frames = session.get('data')
            if not isinstance(frames,list) or len(frames) < 2:
                continue
            groups, current, previous = [], [], None
            for i,frame in enumerate(frames):
                signature = _signature(frame, mouse, key)
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
                # The parent's problem number does not describe a segment; it is assigned
                # afresh by group_sessions_by_problem. import_data numbered the parent by its
                # first recording, so only the first segment continues its flag.
                segment.pop('problem', None)
                segment.pop('problem_provisional', None)
                if part > 1:
                    segment.pop('problem_flag', None)
                for field in ('data','df'):
                    values = session.get(field)
                    if isinstance(values,list):
                        selected = [values[i] for i in indices]
                        segment[field] = selected[0] if len(selected)==1 else selected
                segment['settings_evidence'] = _segment_evidence(session, [frames[i] for i in indices])
                segment['task_name'] = segment['settings_evidence']['task_name']
                if session.get('task_types_classified'):
                    # a segment is a recording of its own task, not of the parent's first one
                    segment['task_type'] = session_task_type(
                        mouse, segment_key, segment['task_name'],
                        [source_path(frames[i]) for i in indices], skip_pretraining=True,
                    )
                towers,initiation = _signature(frames[indices[0]], mouse, key)
                if towers is not None:segment['choice_towers'] = set(towers)
                if initiation is not None:segment['initiation_tower'] = initiation
                sessions[segment_key] = segment
            print(f'[INFO] Split {mouse} {key} into {len(groups)} recording segments because the tower configuration changed.')
    return data
