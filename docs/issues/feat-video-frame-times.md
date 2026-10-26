---
type: feature
complexity: small
status: in review
---

# Imaging: pyControl time of each frame of a run's behaviour video

The mini2p repo's movies can show the behaviour video beside the green and red imaging (`behavior = true` in
`[movies]`). They pair imaging and video frames by pyControl time, so each run needs the time of every video frame
alongside the time of every imaging frame (`frame_times.npz`).

## What it adds

- **Which camera files belong to a run** (`src/funcimg_import/runs.py`, `session_video_files`, deciding in the pure
  `match_videos`): every run has its own video. The camera table (`behav/*_GPIO_data.csv`, one row per frame, rsync on
  `GPIO1`) is the one whose pulses align with the run's `.tsv`. A camera sees a pulse at its next frame, so its intervals
  are off by up to one frame period: the match (`video_match_accepted`) needs 90 % of the pulses of *both* trains
  paired (so a camera that ran through two runs matches neither), a median interval error ≤ 50 ms (as for the
  photometry videos) and none over three frame periods, instead of the DAQ's 10 ms maximum. A run matching no table or
  several, a table matching two runs, or a malformed table (named, with the reason) stops the session.
- **Which video** (`video_beside`): the file named like the table without `_GPIO_data.csv`, with a video extension
  (`.avi`, `.mp4`, `.mov`, `.mkv`). None, or more than one, is refused.
- **`run-XX/video_frame_times.npz`** (`write_video_frame_times`, `load_video_frame_times` in
  `src/funcimg_analysis/frame_times.py`): `time_s` per video frame (NaN outside the matched pulses; on average half a
  frame period early, because each pulse is seen at the next frame), and the video, table and `.tsv` paths and the sync
  quality. The run's TIFF is checked against the batch record first, as for `frame_times.npz`, and the camera's match
  to the run's `.tsv` is checked again.
- Checked on two real sessions (MY_71_R, 5 and 7 Aug 2026): each 60 Hz camera matched its own `.tsv` (98 % of pulses,
  median 4.7 ms, max 16 ms) and not the other's; videos are named like their tables (`.mp4`).
- **Notebook** (`scripts/funcimg/reward_responses.ipynb`, step 1b): writes it for every run, with a table of the match.

## Not done

- No check that the video holds as many frames as the table has rows; the mini2p movie code stops if the video runs out.
