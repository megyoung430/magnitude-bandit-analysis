import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
from src.behavior_import.import_data import *
from src.behavior_import.extract_trials import *
from src.behavior_analysis.get_task_statistics import *
from src.behavior_visualization.plot_task_statistics import *

cohort = "cohort-02"
root = f"../data/{cohort}/rawdata/"
annotate_y = 300 

subjects_data = import_data(root)
subjects_trials = extract_trials(subjects_data)

curr_save_path = Path(f"../results/figures/{cohort}/task-stats/Median Block Lengths")
block_lengths_summary = get_block_lengths(subjects_trials, boundary="good")
plot_block_lengths(block_lengths_summary, annotate_y=annotate_y, save_path=curr_save_path)