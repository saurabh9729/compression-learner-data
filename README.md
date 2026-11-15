# compression-learner-data

Pipelines that pull large public datasets on GitHub's servers and keep only compact, preprocessed outputs. No raw dumps are ever stored here.

## Layout
- `pipelines/<dataset>/`: scripts and a `run.env` settings file per dataset.
- `.github/workflows/<dataset>.yml`: the GitHub Actions job that downloads, filters and preprocesses (triggered by pushing a change to `run.env`, or by hand).
- `processed/<dataset>/`: outputs committed by the job (each file < 95 MB).

## Rules
1. Raw data is streamed, never committed.
2. Every processed file is reproducible from `pipelines/` + `run.env` + the source URL.
3. Each dataset folder has a README: source, licence, what was kept, why.
4. Analyses do not live here. They live in the main repo (`compression-learner`) under `online-data/exploratory/<dataset>/` (first looks) and `online-data/main/<dataset>/` (promoted, pre-registered analyses).

## Datasets
| Dataset | Source | Licence | Status |
|---|---|---|---|
| Lichess | database.lichess.org monthly dumps | CC0 | pipeline ready; first month 2018-06 |
