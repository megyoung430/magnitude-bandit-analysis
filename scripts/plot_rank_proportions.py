import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
from src.behavior_import.import_data import *
from src.behavior_import.extract_trials import *
from src.behavior_analysis.get_good_reversal_info import *
from src.behavior_analysis.get_rank_counts_by_good_reversal import *
from src.behavior_visualization.plot_rank_proportions import *

cohort = "cohort-02"
root = f"../data/{cohort}/rawdata/"
subjects_data = import_data(root)
subjects_trials = extract_trials(subjects_data)

reversal_windows = get_good_reversal_info(subjects_trials, include_first_block=True)
rank_counts_by_good_reversal = get_rank_counts_by_good_reversal(reversal_windows)

save_path = f"../results/figures/{cohort}/choice-stats/Rank Proportions"
plot_rank_proportions(rank_counts_by_good_reversal, save_path=save_path)