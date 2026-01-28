import numpy as np 

def get_num_long_pokes(subjects_trials):
    all_subjects = list(subjects_trials.keys())
    avg_num_pokes_across_subjects = {}
    for subject in all_subjects:
        for session in subjects_trials[subject].keys():
            try:
                num_long_pokes = subjects_trials[subject][session]["num_long_pokes"]
                avg_num_pokes_across_subjects.setdefault(subject, {})
                avg_num_pokes_across_subjects[subject][session] = np.mean(num_long_pokes)
            except:
                print(f"Missing num_long_pokes for {subject} {session}")
    for session in avg_num_pokes_across_subjects[all_subjects[0]].keys():
        session_avg = np.mean([avg_num_pokes_across_subjects[subject][session] for subject in all_subjects if session in avg_num_pokes_across_subjects[subject]])
        avg_num_pokes_across_subjects.setdefault("average_across_subjects", {})
        avg_num_pokes_across_subjects["average_across_subjects"][session] = session_avg
    return avg_num_pokes_across_subjects