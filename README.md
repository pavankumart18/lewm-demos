# LeWorldModel Demo Page

A standalone HTML report exploring Yann LeCun's LeWorldModel (LeWM) — a 15M-parameter Joint-Embedding Predictive Architecture (JEPA) that learns physics from raw pixels.

## Files

| File | Description |
|------|-------------|
| `index.html` | Main report page |
| `rollout_0.mp4` | Successful planning rollout — agent pushes T-block to goal |
| `rollout_11.mp4` | Failure case — model fails to align block angle |
| `rollout_19.mp4` | Additional rollout example |
| `voe_demo.png` | Violation-of-Expectation surprise score chart |
| `probe_demo.png` | Predicted vs. Actual scatter plots for latent probing |

## Usage

Open `index.html` directly in a browser. All media files must be in the same directory.

If the media files are missing, copy them from `~/.stable_worldmodel/pusht/`:

```bash
cp ~/.stable_worldmodel/pusht/rollout_0.mp4 .
cp ~/.stable_worldmodel/pusht/rollout_11.mp4 .
cp ~/.stable_worldmodel/pusht/voe_demo.png .
cp ~/.stable_worldmodel/pusht/probe_demo.png .
```

## What's Covered

- **Demo 1 — Planning:** LeWM achieves 96% success rate on Push-T by planning action sequences purely from pixel observations using the Cross-Entropy Method (CEM).
- **Demo 2 — Anomaly Detection:** The model detects teleportation (3.87× surprise spike) while ignoring color changes (0.88×), proving it learned physics, not appearance.
- **Demo 3 — Physics Probing:** A linear regression on the 192-dimensional latent embeddings recovers object positions with R² = 0.975.

## References

- Paper: [arXiv:2603.19312](https://arxiv.org/abs/2603.19312)
- Code: [github.com/lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)
- Weights: [huggingface.co/quentinll/lewm-pusht](https://huggingface.co/quentinll/lewm-pusht)
