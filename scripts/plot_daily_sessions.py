import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
from src.behavior_import.import_data import *
from src.behavior_import.extract_trials import *
from src.behavior_visualization.plot_single_session import *

root = "../data/cohort-2/rawdata/"
subjects_data = import_data(root)
subjects_trials = extract_trials(subjects_data)

all_subjects = list(subjects_trials.keys())
for current_subject in all_subjects:
    subject_sessions = subjects_trials[current_subject]
    all_sessions = list(subject_sessions.keys())
    for current_session in all_sessions:
        print(f"Plotting {current_subject} | {current_session}")
        if current_subject == "MY_05_N" and current_session == "ses-1_date-20260111":
            print("Skipping session with no trials")
            continue
        ses = subjects_trials[current_subject][current_session]
        ses = unpack_reward_magnitudes(ses)
        ses = unpack_choices(ses)
        curr_save_path = Path(f"../results/figures/single_sessions/{current_subject}/{current_session}/Session Plot")
        plot_single_session(ses, title=f"{current_subject} | {current_session}", save_path=curr_save_path)