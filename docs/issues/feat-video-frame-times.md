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

- **Which camera files belong to a run** (`src/funcimg_import/runs.py`, `run_video_files`): every run has its own video.
  The camera table (`behav/*_GPIO_data.csv`, one row per frame, rsync on `GPIO1`) is the one whose pulses align with the
  run's `.tsv`; exactly one must, or the run is refused with the tables listed. A camera sees a pulse at its next frame,
  so its intervals are off by up to one frame period: the match needs 90 % of pulses paired (as for the DAQ) and a
  median interval error ≤ 50 ms (as for the photometry videos) instead of the DAQ's 10 ms maximum.
- **Which video** (`video_beside`): the file named like the table without `_GPIO_data.csv`, with a video extension
  (`.avi`, `.mp4`, `.mov`, `.mkv`). None, or more than one, is refused.
- **`run-XX/video_frame_times.npz`** (`write_video_frame_times`, `load_video_frame_times` in
  `src/funcimg_analysis/frame_times.py`): `time_s` per video frame (NaN outside the matched pulses), and the video, table
  and `.tsv` paths and the sync quality. The run's TIFF is checked against the batch record first, as for
  `frame_times.npz`.
- **Notebook** (`scripts/funcimg/reward_responses.ipynb`, step 1b): writes it for every run, with a table of the match.

## Not done

- No check that the video holds as many frames as the table has rows; the mini2p movie code stops if the video runs out.
