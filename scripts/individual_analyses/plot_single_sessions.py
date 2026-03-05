import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
from src.behavior_import.import_data import *
from src.behavior_import.extract_trials import *
from src.behavior_visualization.plot_single_session import *
sys.path.append(str(Path(__file__).resolve().parent.parent))
from fix_grid_maze_cohort_02_problems import *

task = "grid-maze"
# task = "open-field"

folder_name = None
cohort = None
if task == "grid-maze":
    cohort = "cohort-02"
    folder_name = "3x3_maze_blocked_reward_bandit"
elif task == "open-field":
    cohort = "cohort-01"
    folder_name = "3x3_field_blocked_reward_bandit"
root = f"/Volumes/behrens/meg/{folder_name}/{cohort}/rawdata/"

subjects_data = import_data(root)
subjects_trials_by_problem = extract_trials_grouped_by_problem(subjects_data)
if task == "grid-maze" and cohort == "cohort-02":
    subjects_trials_by_problem = fix_grid_maze_cohort_02_problems(subjects_trials_by_problem)

for problem_number in subjects_trials_by_problem.keys():
    print(problem_number)
    problem = f"problem-{problem_number:02d}"

    subjects_trials = subjects_trials_by_problem[problem_number]

    all_subjects = list(subjects_trials.keys())
    for current_subject in all_subjects:
        subject_sessions = subjects_trials[current_subject]
        all_sessions = list(subject_sessions.keys())
        for current_session in all_sessions:
            if not subject_sessions[current_session]["trial_info"]:
                print(f"Skipping {current_subject} | {current_session}. There are no trials.")
                continue
            print(f"Plotting {current_subject} | {current_session}")
            curr_save_path = Path(f"../results/figures/{task}/{cohort}/{problem}/single-sessions/{current_subject}/{current_session}/Session Plot")
            plot_single_session(subject_sessions[current_session], title=f"{current_subject} | {current_session}", save_path=curr_save_path)