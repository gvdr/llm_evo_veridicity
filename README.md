# Task Ecologies and the Evolution of World-Tracking Representations in Large Language Models

This repository contains the manuscript, simulation code, and experiment outputs for the paper by Giulio V. Dalla Riva.

Preprint: arXiv (forthcoming).

## Summary

We adapt the ecological veridicality framework to frozen autoregressive transformers. The governing object is the task ecology induced by the training distribution: a separation condition on world-state pairs that determines which distinctions a Bayes-optimal encoding must preserve. We prove a cross-entropy decomposition into an irreducible entropy floor and a Jensen--Shannon excess controlled by the induced partition, a minimum-complexity theorem selecting the coarsest zero-excess encoding, and a split-threshold characterising which distinctions survive simplicity pressure. A two-ecology extension separates the token-prediction ecology from the evaluation ecology and locates the gap pairs where post-training selection matters. All theoretical quantities are tested in a model-organism regime using a 27K-parameter frozen transformer.

## Repository structure

```
llm/
  src/            Julia package: microgpt, measurement, population dynamics
  scripts/        Experiment scripts, figure generation, corpus preparation
  output/         Experiment results and interpretation notes
  latex/          Manuscript source and compiled figures
    main.tex        Paper source
    refs.bib        Bibliography
    jmlr2e.sty      JMLR style file
    figures/        PDF figures referenced by the manuscript
    main.pdf        Compiled paper
  data/           Corpora (gitignored; see below)
  Project.toml    Julia project definition
  Manifest.toml   Julia dependency lock file
```

## Requirements

Julia >= 1.10. Dependencies are declared in `Project.toml`:

- CairoMakie.jl, DataFrames.jl, JLD2.jl, ProgressMeter.jl, StatsBase.jl
- LinearAlgebra, Statistics, Random (stdlib)

Install with:

```bash
cd llm
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Corpora

Training corpora are derived from public-domain sources and are not tracked in the repository. The scripts that download and normalise them are provided:

| Corpus | Source | Preparation script |
|---|---|---|
| Alice in Wonderland (5 languages) | Project Gutenberg | `scripts/clean_alice.jl` |
| Dante's Commedia (8 languages) | Project Gutenberg, Wikisource | `scripts/clean_dante.jl` |
| Communist Manifesto (10 languages) | Project Gutenberg | `scripts/clean_manifesto.jl` |
| Voynich EVA transcription | voynich.nu | `scripts/clean_voynich.jl` |
| Practical Common Lisp (bracket corpus) | gigamonkeys.com | `scripts/clean_lisp.jl` |

Run the preparation scripts before the experiments:

```bash
julia --project=. scripts/clean_alice.jl
julia --project=. scripts/clean_dante.jl
julia --project=. scripts/clean_manifesto.jl
julia --project=. scripts/clean_voynich.jl
julia --project=. scripts/clean_lisp.jl
```

## Experiments

All experiments use the microgpt model organism: a 2-layer, 32-dimensional frozen autoregressive transformer with approximately 27K parameters. Scripts are run from `llm/`:

```bash
julia -t 16 --project=. scripts/exp0_synthetic_exact.jl
julia -t 16 --project=. scripts/exp1_static.jl
julia -t 16 --project=. scripts/exp2_split_merge.jl
julia -t 16 --project=. scripts/exp3_population.jl
julia -t 16 --project=. scripts/exp4_off_ecology.jl
julia -t 16 --project=. scripts/exp6_bracket_ecology.jl
julia -t 16 --project=. scripts/exp7_transfer_ecology.jl
```

| Experiment | Tests | Main result |
|---|---|---|
| Exact synthetic ecology | CE decomposition, split-threshold identities | Exact verification on constructed ecologies |
| Static optimality (Alice, Manifesto) | CE decomposition on empirical ecologies | Separation count monotone in T, ratio = 1 |
| Split-merge path (Commedia, Manifesto) | Split-threshold on empirical ecologies | 12/13 transitions match predicted ordering |
| Wright--Fisher population (Alice, 5 langs) | Conditional selection diagnostics | rms z = 1.05, coverage 0.96 |
| Off-ecology failure (Voynich) | Off-ecology excess and non-identifiability | CE and JS excess rise off-ecology |
| Balance checking (Lisp) | Ecology injection, leaky evaluation design | Static sweep confirms injection threshold |
| Code validation (Lisp) | Ecology injection, recipe-trait selection | Indirect transfer, population selects on recipes |

Theorem sanity checks (339K cases, all pass) are run separately:

```bash
julia --project=. scripts/check_exact_static.jl
julia --project=. scripts/check_geometry.jl
julia --project=. scripts/check_priority2.jl
julia --project=. scripts/check_remaining.jl
julia --project=. scripts/check_finite_sample.jl
julia --project=. scripts/check_generalized_faithful.jl
julia --project=. scripts/check_two_ecology.jl
```

## Compiling the manuscript

```bash
cd latex
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
