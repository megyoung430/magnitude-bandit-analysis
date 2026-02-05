import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
from src.behavior_import.import_data import *
from src.behavior_import.extract_trials import *
from src.behavior_visualization.plot_single_session import *

cohort = "cohort-02"
problem_number = 4
problem = f"problem-{problem_number:02d}"
root = f"../data/{cohort}/{problem}/rawdata/"
subjects_data = import_data(root)
subjects_trials = extract_trials(subjects_data)

all_subjects = list(subjects_trials.keys())
for current_subject in all_subjects:
    subject_sessions = subjects_trials[current_subject]
    all_sessions = list(subject_sessions.keys())
    for current_session in all_sessions:
        if not subject_sessions[current_session]["trial_info"]:
            print(f"Skipping {current_subject} | {current_session}. There are no trials.")
            continue
        print(f"Plotting {current_subject} | {current_session}")
        curr_save_path = Path(f"../results/figures/{cohort}/{problem}/single-sessions/{current_subject}/{current_session}/Session Plot")
        plot_single_session(subject_sessions[current_session], title=f"{current_subject} | {current_session}", save_path=curr_save_path)