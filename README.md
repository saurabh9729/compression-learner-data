# compression-learner-data

Pipelines that pull large public datasets on GitHub's servers and keep only compact, preprocessed outputs. No raw dumps are ever stored here.

## Layout
- `pipelines/<dataset>/`: scripts and a `run.env` settings file per dataset.
- `.github/workflows/<dataset>.yml`: the GitHub Actions job that downloads, filters and preprocesses (triggered by pushing a change to `run.env`, or by hand).
- `processed/<dataset>/`: outputs committed by the job (each file < 95 MB).

## ⚠ This repo is PUBLIC
Everything committed here, and every Actions log, can be read by anyone.

**What may go in:** only data that is (a) under an open licence that allows redistribution and (b) already public, with people identified at most by pseudonyms, which we replace by salted anonymous IDs. No dates, links or free text that point back to a person.

**What never goes in:**
- **Education data** (EdNet, PISA, school or student datasets, anything about learners or minors). It is handled separately, in a private place, after checking its licence and data-use terms. Not here, not even processed or aggregated, unless we decide that explicitly per dataset and write it in this README first.
- Chat or text data written by people (e.g. WildChat).
- Any lab or experiment data (Zhao et al., our own pilots).
- Secrets, tokens, salts or e-mail addresses.

**Every pipeline must:** replace names with salted IDs (salt in a GitHub secret, never in code); keep raw and intermediate files in the runner's temp folder; commit only named, anonymised output files; run a safety check that fails the job if a name appears; print only totals to the logs.

## Rules
1. Raw data is streamed, never committed.
2. Every processed file is reproducible from `pipelines/` + `run.env` + the source URL.
3. Each dataset folder has a README: source, licence, what was kept, why.
4. Analyses do not live here. They live in the main repo (`compression-learner`) under `online-data/exploratory/<dataset>/` (first looks) and `online-data/main/<dataset>/` (promoted, pre-registered analyses).

## Datasets
| Dataset | Source | Licence | Status |
|---|---|---|---|
| Lichess | database.lichess.org monthly dumps | CC0 | pipeline ready; first month 2018-06. Usernames → salted IDs; no game links, dates or opponent names |
| EdNet, PISA, other education data | — | — | **not allowed in this repo** (see above) |
