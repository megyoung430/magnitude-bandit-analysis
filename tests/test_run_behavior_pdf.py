"""The aligned runs' mean images over their behavior (run_behavior_pdf)."""

import json

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.funcimg_analysis import run_behavior_pdf as module  # noqa: E402

TOWERS = ["A1", "B1", "C1"]
VARIANT = "with_distortion_correction/with_destriping"
SUBJECT = "sub-03_id-MY_71_R"


def write_behavior(folder, name, n_trials, reversal_at=None):
    """A pyControl recording of *n_trials* bandit trials, B1 best until *reversal_at*."""
    run_start = {"curr_choice_tows": TOWERS, "curr_rew_mag": dict.fromkeys(TOWERS, 1),
                 "curr_initiation_tow": "B2", "num_t_found": 0, "found_ports": False}  # fmt: skip
    rows = [("0.000", "info", "subject_id", "MY_71_R"),
            ("0.002", "variable", "run_start", json.dumps(run_start))]  # fmt: skip
    for trial in range(n_trials):
        reversed_ = reversal_at is not None and trial >= reversal_at
        magnitudes = {"A1": 4 if reversed_ else 1, "B1": 1 if reversed_ else 4, "C1": 0}
        content = {
            "num_trials": trial + 1, "num_blocks": 1, "num_good_reversals": int(reversed_),
            "chosen_tower": "A1" if trial % 2 else "B1",
            "current_reward_magnitudes": magnitudes, "current_choice_towers": TOWERS,
            "chosen_rank": "best", "rank_counts": {"best": trial + 1, "second": 0, "third": 0},
        }  # fmt: skip
        rows.append((f"{10 + trial:.3f}", "variable", "print", json.dumps(content)))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    pd.DataFrame(rows, columns=["time", "type", "subtype", "content"]).to_csv(
        path, sep="\t", index=False
    )
    return path


def write_plane(variant_dir, rng, with_red=True):
    plane = variant_dir / "aligned_functional" / "suite2p" / "plane0"
    plane.mkdir(parents=True)
    ops = {"meanImg": rng.random((32, 40))}
    if with_red:
        ops["meanImg_chan2"] = rng.random((32, 40))
    np.save(plane / "ops.npy", ops, allow_pickle=True)


@pytest.fixture
def subject(tmp_path):
    """Session 5 with runs 1 and 2 (their own behavior files), and run 3 with no mean image;
    a non-aligned TIFF too."""
    rng = np.random.default_rng(0)
    raw, der = tmp_path / "raw" / SUBJECT, tmp_path / "der" / SUBJECT
    session = "ses-005_date-20260301"
    behav = raw / session / "behav"
    tsvs = [
        write_behavior(behav, "MY_71_R-2026-03-01-100000.tsv", 12, reversal_at=6),
        write_behavior(behav, "MY_71_R-2026-03-01-120000.tsv", 5),
        write_behavior(behav, "MY_71_R-2026-03-01-140000.tsv", 5),
    ]
    rows = []
    for number, tsv in enumerate(tsvs, start=1):
        run = f"run-{number:02d}"
        if number < 3:
            write_plane(der / session / "funcimg" / run / VARIANT, rng)
        rows.append({"session": session, "run": run, "tiff": f"/raw/t{number}.tif", "h5": "",
                     "tsv": str(tsv), "aligned": True, "reason": ""})  # fmt: skip
    rows.append({"session": session, "run": "", "tiff": "/raw/test.tif", "h5": "", "tsv": "",
                 "aligned": False, "reason": "no behavior file"})  # fmt: skip
    listing = tmp_path / "aligned_runs.csv"
    pd.DataFrame(rows).to_csv(listing, index=False)
    return listing, der, tsvs


def test_each_run_with_a_mean_image_gets_its_own_behavior_file_s_trials(subject, tmp_path):
    listing, der, tsvs = subject
    output = tmp_path / "meanImg_aligned_runs_behavior.pdf"
    pages, skipped = module.write_run_behavior_pdf(listing, der, VARIANT, output)
    assert [(p.run.run, p.run.tsv) for p in pages] == [("run-01", tsvs[0]), ("run-02", tsvs[1])]
    assert [len(p.behavior["reward_magnitudes_by_tower"]["A1"]) for p in pages] == [12, 5]
    assert [s.run.run for s in skipped] == ["run-03"]
    assert output.stat().st_size > 0
    caption = output.with_suffix(".caption.txt").read_text()
    assert "2 pages" in caption and "run-03" in caption


def test_the_behavior_panel_is_the_single_session_top_panel(subject):
    listing, der, _ = subject
    page = module.load_run_page(module.load_listing(listing)[0], der, VARIANT)
    fig = module.draw_run_page(page, module.DEFAULT_CONTRAST)
    behavior_ax = fig.axes[2]
    assert behavior_ax.get_ylabel() == "Reward Magnitude"
    assert behavior_ax.get_title().endswith(page.run.tsv.name)
    assert len(behavior_ax.collections[0].get_offsets()) == 12  # one choice dot per trial
    assert behavior_ax.get_legend().get_frame_on() is False


def test_a_registration_without_a_red_mean_image_gets_no_page(tmp_path):
    rng = np.random.default_rng(1)
    variant_dir = tmp_path / "run-01" / VARIANT
    write_plane(variant_dir, rng, with_red=False)
    with pytest.raises(KeyError, match="meanImg_chan2"):
        module.load_mean_images(variant_dir)


def test_an_unreadable_behavior_file_is_shown_on_the_page(subject, tmp_path):
    listing, der, _ = subject
    run = module.load_listing(listing)[0]
    missing = module.ListedRun(run.session, run.run, run.tiff, tmp_path / "sub-1" / "gone.tsv")
    page = module.load_run_page(missing, der, VARIANT)
    assert page.behavior is None and page.behavior_problem
    module.draw_run_page(page, module.DEFAULT_CONTRAST)


def test_an_existing_pdf_is_kept_unless_asked(subject, tmp_path):
    listing, der, _ = subject
    output = tmp_path / "out.pdf"
    output.write_text("keep")
    with pytest.raises(FileExistsError):
        module.write_run_behavior_pdf(listing, der, VARIANT, output)
    assert output.read_text() == "keep"


def test_a_listing_naming_a_run_twice_is_refused(subject, tmp_path):
    listing, der, _ = subject
    table = pd.read_csv(listing)
    table.loc[1, "run"] = "run-01"
    table.to_csv(listing, index=False)
    with pytest.raises(ValueError, match="twice"):
        module.load_listing(listing)


def test_the_script_writes_the_pdf(subject, tmp_path, capsys):
    from scripts.funcimg.aligned_runs_behavior_pdf import main

    listing, der, _ = subject
    output = tmp_path / "runs.pdf"
    main([str(listing), str(der), str(output), "--contrast", "1", "99.8", "1"])
    out = capsys.readouterr().out
    assert f"Wrote {output}: 2 pages" in out and "left out ses-005_date-20260301 run-03" in out
