# LIP-Audit

Code, logs and stored predictions for

> Le, T.-G., Xu, X., Mai, X.-D. *Auditing Label Consistency and Person-Level Variance in Video-Based Engagement Recognition Benchmarks: Evidence From DAiSEE and EngageNet.* Submitted to IEEE Access, 2026.

LIP-Audit is a three-instrument audit of a clip-level video benchmark that needs no new data: **L**abel consistency across adjacent clips of one recording, **I**dentity decodability of a frozen representation by a retrained probe, and **P**erson-level variance (the within-/between-person decomposition of labels and model scores). The paper applies it to DAiSEE and EngageNet and reports two different failure modes.

**Neither dataset, nor its labels, is redistributed here.** DAiSEE is available from its authors on request; EngageNet under the terms of the EmotiW 2023 release. This repository holds no video, no face crops, no OpenFace descriptors or MARLIN tensors, and no labels of either corpus. The stored prediction files carry identifiers and class probabilities only: EngageNet chunk names in `engagenet/probs/*_probs.npz`; DAiSEE clip identifiers in the `ids` array of `logs/TEST_FINAL_probs.npz`, `logs/repro_test_probs_*.npz` and `logs/val_probs/*_val_probs.npz`. The analysis scripts join the labels from the corpora's own label files on those identifiers (`code/label_join.py`): set `ENGAGENET_ROOT` to the directory holding `train_engagement_labels.xlsx`, `validation_engagement_labels.xlsx` and `test_engagement_labels.csv` (and `DAISEE_ROOT` to the DAiSEE release for the scripts that read it through `code/p4_labels.py`), and pass the DAiSEE `Labels` directory where a script asks for it. A split whose label file is not supplied is skipped. Trained checkpoints, cached embeddings and saved adversary heads are not deposited: they are derived from the videos, and a retrained probe recovers participant identity from the embeddings (the paper's Sections V-C and V-D); scripts that need them (`p2_probe.py`, `p5_probe_recdisjoint.py`, `p6_probe_controls.py`, `p7_daisee_probe_controls.py`) are kept as the record of what was run.

## Layout

| Path | Contents |
|---|---|
| `audits/p2_session.py` | Session-neighbour smoothing of the stored sealed-test predictions (Supplementary S3.4). Its gap-two pairing is not the rule of Table 2. |
| `audits/p2_decompose.py` | Audit 2 on DAiSEE labels and stored predictions. |
| `code/p2_*.py`, `code/*.sh` | The DAiSEE modeling campaign: face extraction, the twelve interventions (`p2_train*.py`), retrained identity probes (`p2_probe.py`, `p2_id_restricted.py`, `p2_proto69.py`, `p2_adv_acc.py`), the one-shot sealed test (`p2_test_final.py`), auxiliary streams and ensembling. |
| `code/train_repro.py`, `repro_prep.py`, `repro_audit2.py` | Re-run of the reference ResNet-TCN from its released code and Audit 2 on its predictions (Supplementary S10). |
| `code/ci_sealed_test.py` | Recomputes every entry of the sealed-test table (Table 7) from `logs/TEST_FINAL_probs.npz`, `logs/Test_ids.npy` and `TestLabels.csv` of the DAiSEE release. |
| `code/p4_labels.py`, `p4_e1_harmonise.py`, `p4_e2_uncertainty.py` | Audits 1 and 2 on EngageNet with recording-clustered bootstraps and both permutation nulls (Tables 3 and 4). |
| `code/p4_reduce_openface.py`, `p4_reduce_stream.py`, `p4_check_frames.py`, `p4_verify_frames.py`, `p4_partial_clips.py`, `p4_timebase_by_subject.py`, `p4_frame_*.py` | EngageNet preprocessing and the frame-count, timebase and face-detection checks (Supplementary S7). |
| `code/p4_train_of.py`, `p4_train_marlin.py`, `p4_train_trainval.py`, `p4_pack_marlin.py`, `p4_pack_test.py` | Reimplemented EngageNet baselines (LSTM, CNN-LSTM, TCN, Transformer, MARLIN Transformer, MLP control), the frozen-checkpoint and Train+Val arms. |
| `code/p4_eval_test.py`, `p4_e79_test.py`, `p4_e10_summary.py`, `p4_e79_summary.py` | The registered once-only EngageNet test read and its summaries (Table 5). |
| `code/p4_e6_probe.py`, `p4_e67_checks.py`, `p4_e7_decompose.py`, `p4_e89.py` | EngageNet identity probes, centering controls, variance decomposition and ranking sensitivity. |
| `code/p4_tcct_build.py`, `p4_e4_tcct.py` | TCCT-Net inputs rebuilt from our OpenFace pass and the run of the released weights. |
| `code/p5_*.py` | Analyses added during the internal reviews before submission: DAiSEE ordering control (S3), strict-adjacency and pair-marginal recomputation (S3), paired participant-level bootstrap (S8), participant-clustered person-level intervals (S8), recording-disjoint probe (S9). |
| `code/p6_controls.py`, `code/p6_probe_controls.py` | Post-hoc controls: session clock, leave-one-out oracle, kappa re-analysis, reference-run peak estimate (S3, S8, S10); descriptor, MARLIN, fixed-pool and acquisition-metadata probes (S7, S9). |
| `code/p7_audit1_daisee.py` | Audit 1 on DAiSEE, the analysis of record for Table 2 and Fig. 2 (`--labels <DAiSEE>/Labels --out results/p7_audit1_daisee.json`); reproduces every figure of Table 2 from the released label files under the rule of Supplementary S3.1, with the ten-character, gap-two and sorted-neighbour variants, split- and pair-marginal chance terms, clustered bootstrap and both permutation nulls (10,000 draws). |
| `code/p7_nested_decomp.py` | Nested participant / recording / chunk decomposition of EngageNet labels and scores (Supplementary Table S16), from the EngageNet label files (under `ENGAGENET_ROOT`) and `engagenet/probs/`. |
| `code/p7_person_variance_ci.py` | Supplementary Table S15: participant-clustered intervals of the person-level quantities, seed mean inside each replicate, from `engagenet/probs/`. |
| `code/p7_acc_ci.py` | Table 5 accuracy intervals, same construction (participant-clustered, seed mean inside each replicate), from `engagenet/probs/`. |
| `figs/make_figs.py` | Figs. 2, S1, S2 and S3 from the printed values. |
| `code/p7_daisee_probe_controls.py` | Session-disjoint identity probe on DAiSEE and the adversary heads read on the probe's pool and held-out clips (Supplementary Tables S17 and S18); needs the cached embeddings and saved head weights, which are not redistributed. |
| `code/p7_daisee_extras.py` | Leave-one-out constant-per-subject oracle and alternative weightings of the within-person AUC on DAiSEE (Supplementary S4). |
| `code/p4_fig_contrast.py` | Figure 3 (writes `figs/figG_contrast.pdf`). |
| `code/p8_kappa_heterogeneity.py` | Between-split test of the DAiSEE Audit 1 kappas on the pairing of Table 2 (Supplementary S3.5; `results/p8_kappa_heterogeneity.json`). |
| `code/p8_daisee_audit2.py` | DAiSEE Audit 2 (within-person AUC_23, rho_b) recomputed from the deposited, label-free prediction files: five-window and action-unit validation runs, sealed-test ensemble, reference re-run (`results/p8_daisee_audit2.json`). |
| `code/label_join.py` | Joins the corpora's labels to the deposited prediction files on their identifiers. |
| `logs/` | DAiSEE per-epoch training curves, matched comparisons, the sealed-test output, `TEST_FINAL_probs.npz` (per-clip probabilities of the single sealed-test evaluation, with `ids`), `Test_ids.npy`, and the reference re-run's peak- and final-epoch probabilities. |
| `logs/val_probs/` | DAiSEE validation probabilities of the five-window model (three seeds, trained on the training split) and of the action-unit stream, with `ids` and no labels. |
| `engagenet/probs/` | Stored per-chunk probabilities of every EngageNet run: `*_val_probs.npz` (validation) and `*_test_probs.npz` (the once-only test read); both carry chunk identifiers and no labels. |
| `engagenet/logs/` | EngageNet per-epoch training curves (`*_log.jsonl`), extraction and campaign logs. |
| `engagenet/runs/` | Per-run summaries with the validation-selected epoch of every frozen checkpoint. |
| `engagenet/analysis/` | Raw output and logs of the analysis scripts as run on the server. |
| `results/` | JSON output of every audit and analysis referenced in the paper and its supplement (`E*.json`, including `E1_results.json`/`E2_results.json`, the EngageNet Audit 1 with kappa intervals; `p5_*.json`, `p6_*.json`, `p7_*.json`, `review_recompute.json`), the `w5fix` ablation logs, and `p2_RESULTS.csv`. |
| `RESULTS_P2.csv` | The collected DAiSEE run-level results table. |

## Reproducing the tables

- **Table 2 and Fig. 2** (DAiSEE Audit 1): `python code/p7_audit1_daisee.py --labels <DAiSEE>/Labels --out results/p7_audit1_daisee.json`; a few minutes (10,000 permutation draws). `python code/p5_review_recompute.py` gives the strict-adjacency and pair-marginal variants of Supplementary Table S7. Fig. 2 is drawn by `figs/make_figs.py` from the printed values.
- **Table 3 and Table 4, label rows** (EngageNet Audit 1 and 2): `code/p4_e1_harmonise.py` and `code/p4_e2_uncertainty.py` on the released EngageNet label files (`code/p4_labels.py` is the shared core).
- **Table 5** (reproduced EngageNet systems): `results/E10_test.json` and `results/E10_test_trainval.json` hold the registered test read; `code/p4_eval_test.py` is the script that produced them once. Validation figures are in `engagenet/runs/*_summary.json`. Participant-clustered accuracy intervals: `python code/p7_acc_ci.py`.
- **Table 6** (identity suppression): `code/p2_probe.py` on the frozen embeddings; `RESULTS_P2.csv` columns `id69`, `id88`, `adv_top1`.
- **Table 7** (sealed test): `python code/ci_sealed_test.py --labels <DAiSEE>/Labels`.
- **Supplementary S3** (ordering control, kappa re-analysis): `results/p5_daisee_order_curve.json`, `results/p6_controls.json`; the between-split kappa test: `python code/p8_kappa_heterogeneity.py --labels <DAiSEE>/Labels --out results/p8_kappa_heterogeneity.json`.
- **Supplementary S4 and the DAiSEE within-person band of Sections V-E and VI** (Audit 2 on DAiSEE models): `python code/p8_daisee_audit2.py --labels <DAiSEE>/Labels`.
- **Supplementary S8** (paired bootstrap, person-level intervals): `python code/p5_paired_bootstrap.py`, `python code/p5_person_variance_ci.py`, `python code/p7_person_variance_ci.py` (Table S15, seed mean inside each replicate) and `python code/p7_nested_decomp.py` (Table S16) on `engagenet/probs/` (validation rows need the EngageNet validation label file, test rows the test label file; `code/label_join.py` joins them).
- **Supplementary S9** (probe robustness): `engagenet/analysis/p5_probe_recdisjoint_b.json`, `engagenet/analysis/p5_probe_recdisjoint_e.json`, `results/p6_probe_controls.json`, `results/p7_daisee_probe_controls.json`; scripts `code/p5_probe_recdisjoint.py`, `code/p6_probe_controls.py` need the frozen checkpoints and the training-split descriptors, which are not redistributed.
- **Supplementary S10** (reference re-run): `logs/repro_log.jsonl`, `logs/repro_test_probs_*.npz` (clip identifiers in `ids`; labels from `TestLabels.csv`), `code/repro_audit2.py` (as run on the server, on files that then carried labels; `code/p8_daisee_audit2.py` recomputes its within-person AUC and rho_b from the deposited files).

Training runs used a single NVIDIA L4 (24 GB), PyTorch 2.13, bfloat16; about 1.5 to 2 GPU-hours per DAiSEE run and minutes per EngageNet descriptor model.

## What runs from this repository as deposited

Given the corpora's label files (the DAiSEE `Labels` directory; the EngageNet label files under `ENGAGENET_ROOT`), these scripts run from the repository as deposited: `code/ci_sealed_test.py`, `code/p8_daisee_audit2.py`, `code/p8_kappa_heterogeneity.py`, `code/p5_paired_bootstrap.py`, `code/p5_person_variance_ci.py`, `code/p7_person_variance_ci.py`, `code/p7_acc_ci.py`, `code/p7_nested_decomp.py`, `code/p7_audit1_daisee.py`, `code/p4_e1_harmonise.py` and `code/p4_e2_uncertainty.py`, `figs/make_figs.py` and `code/p4_fig_contrast.py`; the kappa re-analysis of `code/p6_controls.py` reads its label directory from the `LAB` constant. Several of them write their JSON into `results/` or `engagenet/analysis/`, over the deposited copies, and a run without the test label file writes validation rows only: run them in a copy of the repository. `code/p4_e1_harmonise.py` and `code/p4_e2_uncertainty.py` read the corpora through `DAISEE_ROOT` and `ENGAGENET_ROOT` and include the EngageNet test split unless `--no-test` is given. With the labels joined this way, the output of `ci_sealed_test.py` and the validation rows of `p5_paired_bootstrap.json`, `p5_person_variance_ci.json`, `p7_acc_ci.json`, `p7_person_variance_ci.json` and `p7_nested_decomp.json` were checked on 28 September 2026 against the values of record (identical, or within 1e-9 floating-point noise); `p8_daisee_audit2.py` reproduces the DAiSEE within-person AUC_23 values .607, .589, .572 and .543 and rho_b .70, .50 and .59 quoted in the paper. `audits/p2_decompose.py`, `audits/p2_session.py` and `code/repro_audit2.py` are the server-side records of those first readings and do not run on the label-free files. The remaining scripts (training, extraction, probes that need checkpoints or descriptors, anything that reads `~/daisee` or `~/p2` on the server) are kept as the record of the campaign and need the server-side artefacts.

## Provenance

The DAiSEE material (`audits/`, `code/p2_*`, `logs/`, `RESULTS_P2.csv`) dates from August and September 2026; the EngageNet material and the `p5_*`, `p6_*` and `p7_*` analyses were added on 25 September 2026; `code/label_join.py`, the `p8_*` analyses and the label-free prediction files on 28 September 2026. The repository was republished with a fresh history on 16 October 2026, so that no earlier commit carries label arrays. Paths under the server account's home directory are written `~` in the logs and scripts. The laboratory journal with the prospectively specified criteria and the EngageNet test registration file are available from the corresponding author on request.

## Analyses added in the first review round (October 2026)

These analyses answer the first-round review of the IEEE Access submission. All were specified in `r1/REGISTRATION_R1_20261026.md` before any was run; its SHA-256 is `ccc9f9065932a67cb8937a0cb10415350c583522f1a1d4ff9e6e9b35386fdf84`. Departures from it are listed in `r1/DEVIATIONS_R1.md`. The outputs are in `results/r1/`, and Supplementary Section S11 reports every registered quantity.

No new script reads a test split.
- EngageNet: the server scripts read copies of the training and validation files only, and the workstation scripts take EngageNet labels from a directory with no test file.
- DAiSEE: the released `TestLabels.csv`, already read by the label-only Audit 1 of Table 2, is read again only by the label-only analyses of `r1_label_analyses.py` (`kappa`, `coding`). No model is evaluated on any test split.

| Path | Contents |
|---|---|
| `code/r1_label_analyses.py` | Participant-clustered Audit 1 intervals (DAiSEE; EngageNet copied from `results/E2_results.json`), coding sensitivity with a cumulative-logit mixed model and its simulation check, equivalence bounds and minimum detectable differences from the stored paired intervals, within-participant residuals by true label (S11.1–S11.4) |
| `code/r1_controlled_sim.py` | Synthetic Audit 1 on the real recording structure; semi-synthetic Audit 2 with two score models of equal accuracy (S11.5, S11.6) |
| `code/r1_engagenet.py` | EngageNet training and validation only: identity overlap against disjointness at matched size, label-balanced training, retraining with shuffled sub-windows, frozen checkpoints read with shuffled sub-windows (S11.7–S11.9) |
| `code/r1g_frames_prep.py`, `code/r1g_frames_train.py`, `code/r1g_frames_eval.py` | ResNet-TCN frame ablation (16 consecutive, 16 spread, 50 spread frames), DAiSEE training split, validation read every epoch; seeds 1 and 2 of the 50-frame arm were added under departure 6 of `r1/DEVIATIONS_R1.md` (S11.10) |
| `code/r1_make_tables.py` | The S11 tables, built from the result files |
| `results/r1/*.json` | Outputs of the scripts above |

The probability files written by `r1_engagenet.py` carry labels and are not deposited. The frame-ablation validation probabilities (`results/r1/r1_frames_val_probs_<arm>.npz`, every epoch) are deposited with clip identifiers and without labels; `r1g_frames_eval.py` joins the labels from the DAiSEE validation label file.

## Licence

GNU General Public License v3.0, see `LICENSE`.

## Citation

To be added when the article has a DOI.
