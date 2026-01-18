import numpy as np

def group_trials_by_good_block(session_data):
    """
    Groups trials in the session data by their respective blocks.

    Parameters:
    session_data (list of dict): List containing trial data dictionaries.

    Returns:
    dict: A dictionary where keys are block numbers and values are lists of trials in that block.
    """
    # --- Find Reversals ---
    good_idx = event_indices_from_cumulative(session_data.get("good_reversals"))
    bad_idx  = event_indices_from_cumulative(session_data.get("bad_reversals"))

def concatenate_blocks(all_session_data):
    """
    Concatenates trials from all sessions into a single list.

    Parameters:
    all_session_data (list of list of dict): List containing session data lists.

    Returns:
    list of dict: A single list containing all trials from all sessions.
    """
    concatenated_data = []
    for session in all_session_data:
        concatenated_data.extend(session)
    return concatenated_data

# --- Reversal Identification Helper Function ---
def event_indices_from_cumulative(counter, N=None):
    """
    counter: list of cumulative counts per trial.
    Returns trial indices where the counter increments (event occurs at that trial index).
    """
    if counter is None:
        return []
    c = np.asarray(counter, dtype=float)
    if N is not None:
        c = c[:N]
    if len(c) < 2:
        return []
    d = np.diff(c)
    return list(np.where(d > 0)[0] + 1)