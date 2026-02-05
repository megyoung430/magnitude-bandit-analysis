import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
from src.behavior_import.import_data import *
from src.behavior_import.extract_trials import *
from src.behavior_visualization.plot_num_reversals import *

cohort = "cohort-02"
problem_number = 4
problem = f"problem-{problem_number:02d}"
root = f"../data/{cohort}/{problem}/rawdata/"
subjects_data = import_data(root)
subjects_trials = extract_trials(subjects_data)
curr_save_path = Path(f"../results/figures/{cohort}/{problem}/reversal-stats/Total Reversals")
plot_num_reversals(subjects_trials, save_path=curr_save_path)

curr_save_path = Path(f"../results/figures/{cohort}/{problem}/reversal-stats/Cumulative Reversals Over Time")
plot_num_reversals_over_time(subjects_trials, threshold=10, save_path=curr_save_path)