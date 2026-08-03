# World Model Interpretability

Do world models learn physics, or shortcuts? We build a small autoregressive Transformer (via TransformerLens) that predicts the next state of a MiniGrid `DoorKey-6x6` world, with a deliberately injected spurious confound: during training, the background border color is perfectly correlated with which quadrant of the grid the agent occupies. We test whether the model relies on this shortcut (an out-of-distribution test with the color mapping flipped), diagnose it with linear probes and visualization tools, and try to causally remove it — via activation ablation at inference time, and via adversarial regularization during training.

## Status

This started as a graduate mechanistic-interpretability seminar assignment; **the assignment has been submitted.** Work is now continuing independently, aimed at a workshop paper. No rubric constraints apply going forward — the experiment is free to evolve based on what the data actually shows.

## Repo layout

- **`WMIP.ipynb`** — the main notebook. Runs top-to-bottom on either a local Mac (via the `wm-interp` venv) or Google Colab; auto-detects which via `IN_COLAB` and installs/imports accordingly. Narrative markdown, top-level driver calls, and all plotting live here; shared logic lives in `wm_lib.py` (below).
- **`wm_lib.py`** — shared library: env/confound, tokenization, data collection, model training, probing, ablation, and path-patching code. Imported by `WMIP.ipynb` and by `run_confound_sweep.py`, so pipeline logic exists in exactly one place.
- **`run_confound_sweep.py`** — Phase 2 driver: reruns the pipeline across a confound-strength × seed grid (see `findings.md` §7.1). Long-running (multi-hour, CPU-only); run it standalone, not from the notebook.
- **`findings.md`** — the living research log. Every bug found, methodology change made, and result produced gets recorded here as it happens — this is the source material the paper will be written from, not just an after-the-fact summary.
- **`results_<tag>_<device>.json`** — headline numbers from a given run (e.g. `results_mac_cpu.json`), tagged by environment/hardware so Mac and Colab runs can be diffed directly. `results_confound_sweep_mac_cpu.json` holds the Phase 2 grid.
- **`artifacts/probing_curve.png`, `artifacts/multi_direction_ablation.png`, etc.** — figures produced by the notebook, referenced from `findings.md`.
- **`wm-interp/`** — local Python 3.11 virtual environment (Mac only; not needed/used on Colab). Recreate with:
  ```
  python3 -m venv wm-interp && source wm-interp/bin/activate
  pip install --upgrade pip
  pip install gymnasium minigrid torch transformer_lens sae-lens scikit-learn matplotlib seaborn
  ```

## Running it

1. Local Mac: `source wm-interp/bin/activate`, register the kernel if needed, then run `WMIP.ipynb` top-to-bottom (Restart & Run All) in Jupyter/VS Code.
2. Colab: upload **both** `WMIP.ipynb` and `wm_lib.py` (two-file upload — the notebook imports from `wm_lib.py`, so it's no longer a single-file upload). The first cells detect the Colab environment and install dependencies automatically.
3. Headless (background) execution:
   ```
   jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=7200 \
     --output WMIP_executed.ipynb WMIP.ipynb
   ```
4. Phase 2 (confound-strength sweep), standalone — not run from the notebook:
   ```
   nohup ./wm-interp/bin/python3 run_confound_sweep.py > confound_sweep.log 2>&1 &
   ```
   Multi-hour job; writes `results_confound_sweep_mac_cpu.json`, checkpointed after every grid point.

For what's been tried, what worked, what didn't, and what's currently being investigated, see `findings.md`.
