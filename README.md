# When Does Self-Supervision Beat ImageNet?
Domain distance governs label-efficient pretraining in medical imaging.

MD6329 - Deep Learning for Healthcare AI, final mini-project.

## Research question

When labelled medical images are scarce, is in-domain self-supervised pretraining
(SimCLR) or ImageNet transfer the better use of compute, and does the answer depend on
how far the imaging modality sits from natural images?

Four initialisation arms share one unmodified ResNet-18 backbone, so the only
difference between arms is the initialisation of the weights:

| arm | meaning |
|---|---|
| `scratch` | random initialisation |
| `imagenet` | torchvision `IMAGENET1K_V1` weights |
| `ssl` | SimCLR pretraining on the unlabelled in-domain training images |
| `ssl_imnet` | the same SimCLR pretraining started from ImageNet weights |

Evaluated at label budgets {1, 5, 10, 25, 100}% x 3 seeds x 2 datasets
(PneumoniaMNIST, DermaMNIST from MedMNIST v2), 210 supervised runs in total
(120 fine-tune + 90 frozen linear probes; the scratch arm is not probed because a
frozen random backbone is not a meaningful representation).

## Headline results

- The ranking **inverts** with domain distance. On grayscale chest X-ray at a 1% budget
  (47 labelled images) in-domain SimCLR reaches 0.971 AUROC vs 0.917 for ImageNet
  transfer and 0.929 for scratch (+0.042 over scratch, paired t-test p = 0.011).
  On RGB dermatoscopy ImageNet transfer leads at every budget and in-domain SimCLR
  never separates from scratch.
- Pretraining buys **label efficiency, not peak accuracy**: with 100% of labels no arm
  beats scratch significantly on either task (all p > 0.14).
- A **frozen** in-domain SimCLR backbone reaches 0.989 AUROC on chest X-ray, higher
  than fine-tuning the same weights.
- **Colour jitter must be modality-specific**: removing it improves dermatoscopy
  (0.798 -> 0.823) and degrades radiography (0.982 -> 0.959), because colour is a
  diagnostic criterion in pigmented lesions but only an intensity augmentation on film.
- In-domain SimCLR is the best-calibrated and most corruption-robust arm; the composed
  `ssl_imnet` arm collapses below chance under severe additive noise on chest X-ray,
  a failure clean accuracy hides.

## Reproducing

Environment: Python 3.11, PyTorch 2.11 + CUDA 12.8, torchvision, kornia, scikit-learn,
pandas, matplotlib, reportlab (for the PDF), lectrace (for the presentation).

```
# 1. data (MedMNIST v2, 64px) -> data/
python download_data.py

# 2. SimCLR checkpoints (12 = 2 datasets x 6 variants, 400 epochs each, ~80 min)
python run_pretrain.py --epochs 400

# 3. experiments
python run_baselines.py        # pixel logistic regression + random forest
python run_sweep.py --probe    # label-budget grid, fine-tune + linear probe
python run_ablation.py         # SimCLR augmentation ablation at 10% labels
python run_robustness.py       # test-time corruption robustness
python run_gradcam.py          # Grad-CAM attribution maps

# 4. figures, tables, report, presentation
python make_data_fig.py       # dataset stats + Figure 1
python make_figures.py        # Figures 2-5
python make_views_fig.py
python make_tables_tex.py
python report/build_pdf.py
python -c "from lectrace.cli import main; main()" serve lectrace_presentation.py
```

Seeds {0, 1, 2} control initialisation, label subsampling and augmentation sampling.
Model selection uses validation AUROC with patience 8; the test split is evaluated
exactly once per run with the best-validation weights.

## Layout

```
src/                     shared code
  bootstrap.py           sandbox environment fixes (see Notes)
  data.py                MedMNIST loading, nested stratified label subsets, CPU transforms
  gpu_aug.py             batched GPU augmentation (kornia), GPU-resident splits
  models.py              ResNet-18 backbone, SimCLR projection head, classifier
  train.py               NT-Xent, SimCLR pretraining, fine-tuning, linear probing
  metrics.py             AUROC/AUPRC/balanced accuracy/sensitivity/specificity/ECE
run_*.py                 experiment drivers (see above)
make_data_fig.py         dataset statistics tables and Figure 1
make_figures.py          result figures (2-5)
make_tables_tex.py       LaTeX tables generated from the CSVs (no hand-typed numbers)
report/report.tex        NeurIPS-format source (compiles on Overleaf)
report/report.pdf        rendered report
report/build_pdf.py      LaTeX-subset renderer used to produce the PDF locally
overleaf/                Overleaf-ready copy for the official neurips_2024.sty
make_overleaf.py         regenerates overleaf/ from report/report.tex
lectrace_presentation.py execution-explanation presentation (~10 min, 5 states)
results/                 every metric produced, one row per run
figs/                    figures
ckpt/                    SimCLR checkpoints
```

## Notes on the compute environment

The sandbox used for development denies named-pipe creation, so multi-process
`DataLoader` workers and joblib parallelism are unavailable (`n_jobs=1` throughout).
Both datasets are therefore held resident in GPU memory and augmented in batch with
kornia, which reduced SimCLR pretraining from ~10 s to ~0.8-1.2 s per epoch.
`src/bootstrap.py` additionally sets the torch cache directory (torch's cache helper
calls the POSIX-only `pwd` module via `getpass.getuser()`) and fixes `sys.path` order
so one NumPy/OpenMP runtime is used. The NT-Xent loss is computed with autocast
disabled because the self-similarity mask overflows float16.

No TeX engine can run in the sandbox, so `report/report.pdf` is rendered by
`report/build_pdf.py` from `report/report.tex`; the `.tex` is the canonical source and
compiles unchanged on Overleaf with the official NeurIPS style file.

## Attribution

Datasets: MedMNIST v2 (Yang et al., Scientific Data 2023), derived from HAM10000
(Tschandl et al. 2018) and the paediatric chest X-ray collection (Kermany et al. 2018).
Methods: SimCLR (Chen et al. 2020), ResNet (He et al. 2016), Grad-CAM (Selvaraju et al.
2017), ECE (Guo et al. 2017), corruption robustness (Hendrycks & Dietterich 2019).
Libraries: PyTorch, torchvision, kornia, scikit-learn, lectrace. No third-party
project implementation was copied; all experiment code in this repository was written
for this project.

Generative AI (Claude) was used for code scaffolding, debugging, literature orientation
and draft text, as disclosed in the report. All experimental design decisions, validity
checks and interpretations are the author's.
