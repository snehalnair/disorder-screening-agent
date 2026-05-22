# Disorder-Aware Dopant Screening

Disorder-aware substitutional dopant screening for battery cathode materials.
Quantifies how chemical disorder (modelled with SQS ensembles + the MACE-MP-0
MLIP) changes which dopants a screening pipeline selects, across nine oxides
spanning six structure types.

## Repository layout

```
.
├── __main__.py            CLI entry point (python -m / disorder-screening)
├── pyproject.toml         Package metadata (flat-layout package: disorder-screening)
├── pytest.ini             Test config
├── requirements.txt
│
├── config/                Pipeline configuration + per-material targets   ┐
├── stages/                Screening stages (chemical prune → Stage 5 sim) │ core
├── pipeline_io/           I/O, templates, checkpointing                    │ package
├── ranking/               Candidate ranking logic                         │ (imported as
├── graph/                 LangGraph pipeline orchestration                 │  top-level
├── db/                    Run/checkpoint database                          ┘  modules)
├── tests/                 Test suite (pytest -m "not gpu" by default)
│
├── data/                  Reference inputs
│   ├── structures/        Parent CIFs
│   ├── known_dopants/     Literature dopant ground truth
│   ├── experimental_measurements/
│   ├── hea_validation/    HEA MACE-vs-DFT validation (raw 452 MB CSV gitignored)
│   └── *.json/*.csv       Shannon radii, element costs/metadata
│
├── notebooks/             Colab/Kaggle run notebooks (per-material runners)
│   └── archive/           Superseded notebooks (smoketests, old eval, reruns)
│
├── results/               Per-material screening outputs (downloaded checkpoints)
│   ├── lco/ lmo/ lno/ sto/ ceo2/    one JSON per dopant
│   └── lfp_screening_results.json
│
├── evaluation/            Pipeline evaluation outputs + figures
│
├── paper/                 Manuscript, analysis scripts, figures, web app
│   ├── draft_v2.md                  current manuscript
│   ├── supplementary_information.md current SI
│   ├── make_figures_v2.py           current figure generator
│   ├── *_screening.py               per-system screening (LFP, NASICON, NCO, NMC…)
│   ├── streamlit_app.py             disorder-risk web tool
│   └── archive/                     draft_v1, old supplementary, old make_figures
│
├── dft/                   DFT validation of the disorder voltage effect
│   ├── sqs_validation/    QE inputs + PSLibrary pseudopotentials (LiCoO₂, 5 dopants)
│   └── quantum_espresso/  QE 7.2 source tree + tarball (gitignored, third-party)
│
├── docs/                  Project documentation
├── scripts/               Helper scripts
└── archive/               Superseded top-level files (see archive/README.md)
```

## Quick start

```bash
pip install -e .            # installs the disorder-screening package
pytest                      # runs the non-GPU test suite
disorder-screening --help   # CLI usage
```

Production simulations run on Colab A100 via the notebooks in `notebooks/`;
analysis and figures are regenerated from `results/` by `paper/make_figures_v2.py`.

## Notes

- The core package uses a **flat layout** — `config/`, `stages/`, `pipeline_io/`,
  `ranking/`, `graph/`, `db/` are imported as top-level modules. Do not move them
  into a `src/` directory without updating all imports.
- Analysis scripts read per-material results from `results/<material>/`.
