# Registration of the analyses and experiments added in response to the first-round review (Access-2026-49377)

Written 26 October 2026, before any of the analyses below was run. Its SHA-256 and date are recorded in `REGISTRATION_R1_20261026.sha256`. Every result will be reported, including null results and results against the stated direction. Any departure from this file will be logged with its date and reason in `DEVIATIONS_R1.md` in this folder.

## 0. Data that may and may not be read

- **EngageNet test split.** Neither its labels nor its features are loaded, by any script, for any purpose. Guards:
  - On the server, the new scripts read from a fresh directory `~/r1/data/` that holds copies of the train and validation label and descriptor files only.
  - Each script asserts that every split it loads is `train` or `validation` and that no loaded path contains the string `test`.
  - Locally, `ENGAGENET_ROOT` points to a directory holding only the train and validation label files.
  - The existing `p4_e1_harmonise.py` and `p4_e2_uncertainty.py`, which include the test split by default, are not re-run.
- **DAiSEE test split.** No model is evaluated on it. Its released labels (`TestLabels.csv`), already read by the label-only Audit 1 of Table 2, may be read again for the label-only analyses of Sections 1a and 1b. This was decided by the corresponding author on 26 October 2026.
- **DAiSEE model runs (Section 4G).** These train on the training split and evaluate on the validation split only. No test CSV is created.
- **Stored test predictions and results** (`p5_paired_bootstrap.json` and similar) are used only through the intervals already computed, in Section 2. No test label is re-joined to them.

## 1. Label-only re-analyses (Reviewer 1, comments 1 and 2)

### 1a. Participant-clustered intervals for Audit 1

- Quantity: κ of adjacent-clip agreement, with the pairing rule of record:
  - DAiSEE: Table 2, Supplementary S3.1;
  - EngageNet: `E1_results.json`, four-level and decisive axes.
- Interval: percentile bootstrap, 2,000 replicates. Each replicate resamples participants with replacement and keeps all of each participant's recordings and pairs; the chance term is recomputed in each replicate.
- Splits:
  - DAiSEE: train, validation, test;
  - EngageNet: train, validation.
- No directional hypothesis. The intervals are reported beside the recording-clustered intervals of record.

### 1b. Coding sensitivity of the between-person share of labels

The share is computed under four codings:
- (i) the equal-interval 0–3 coding of record;
- (ii) the binary decisive coding of record;
- (iii) ridit scores, the mid-rank of each level in the pooled split distribution;
- (iv) the latent-scale intraclass correlation of a cumulative-logit mixed model.

For (iv), the random intercept is per participant on DAiSEE, and per participant plus per recording within participant on EngageNet, so that it lines up with Table S16. The latent ICC is σ²_participant / (σ²_participant + σ²_recording + π²/3).

The estimator for (iv) is our own implementation, by adaptive Gauss–Hermite quadrature. It is first checked on simulated data with known latent ICCs (0.1, 0.3, 0.5) and the participant and recording counts of each split, and the check is reported.

Splits:
- DAiSEE: train, validation, test;
- EngageNet: train, validation.

**H1.** Under every coding, the EngageNet share exceeds the DAiSEE share on the same split type.

## 2. Equivalence bounds and minimum detectable differences (Reviewer 1, comment 3)

Source: the 95% paired participant-clustered bootstrap intervals in `results/p5_paired_bootstrap.json`, all pairs in its four blocks (validation best and last epoch, frozen-checkpoint test, Train+Val test), for accuracy and UAR. No label is read.

- **Equivalence bound** δ = max(|lower|, |upper|) of the 95% interval. Differences larger than δ in either direction are excluded at the 2.5% level per side. This is a conservative two one-sided test.
- **Minimum detectable difference** at 80% power and α = .05: (1.96 + 0.84) × SE, with SE = (upper − lower)/3.92. This is a normal approximation and is labeled as one.

## 3. Within-participant residuals (Reviewer 1, comment 4)

Validation splits only:
- DAiSEE: stored validation probabilities of the five-window model (three seeds) and the action-unit stream;
- EngageNet: the fifteen OpenFace-model validation files at the validation-selected epoch.

Score = P(level 3) on the decisive axis, among clips at level 2 or above. This is the definition used in `p4_e7_decompose.py`.

- **(a) Residuals by label.** Within-participant residual = score − participant mean score. Report the mean residual by true decisive label with participant-clustered intervals.
- **(b) Label-matched share.** Reweight each participant's clips so that the participant's decisive-label distribution equals the pooled distribution, then recompute the between-person share of scores.

**H2.** On EngageNet, the between-person share of scores after matching stays above zero, with a participant-clustered interval that excludes zero. If it does not, the paper's statement that scores carry person-level structure beyond label composition will be revised.

## 4. Controlled conditions (Reviewer 2, comment 2)

Metric definitions throughout are the functions of the existing scripts, imported unchanged:
- `within_person_auc`, `eta2_between` and the decisive axis from `p4_e7_decompose.py`;
- ρ_b as in the main text.

### A. Synthetic Audit 1

- Model: a latent state persists between adjacent clips and switches with probability r, to a level drawn from the split's marginal. Each observed label is the state, or with probability q a different level drawn from the marginal. This is the model behind Eq. (4).
- Recording lengths and class marginals: those of DAiSEE train and EngageNet train.
- Grid: r, q ∈ {0, .05, .1, .2, .3}, 200 replicate datasets per cell.
- Audit 1 code of record applied to each dataset.
- A boundary-transition variant draws state switches only between adjacent ordinal levels.

**H3.** κ decreases monotonically in r and in q.

**H4.** Distinct (r, q) pairs on the same iso-agreement curve give κ values whose replicate distributions overlap. In other words, κ alone does not identify r and q. This is demonstrated, not tested.

### B. Semi-synthetic Audit 2: equal accuracy, different behavior

- Data: real EngageNet training labels on the decisive axis, 91 participants, with their recording structure.
- Two synthetic score models, each with one tuning parameter set so that accuracy at the 0.5 threshold matches the other model's within 0.5 points:
  - **S (person shortcut):** score = participant's mean decisive label (leave-one-clip-out) + Gaussian noise. The score carries no within-person information.
  - **W (within-person):** score = 0.5 + c·(label − participant mean label) + Gaussian noise. The score carries no person offset.
- A mixture grid between S and W is also reported.

**H5.** At matched accuracy:
- within-person AUC of W exceeds that of S, and S is within .02 of .5;
- the between-person share of scores and ρ_b of S exceed those of W.

### C. Identity overlap against identity disjointness at matched training size

- Pool: EngageNet training participants with at least two recordings.
- Participants are split into five folds. For each fold, the held-out set H is one recording per participant of the fold, the same recording for both conditions. The models then train in two ways:
  - **Overlap:** on all other recordings of all pool participants, including the other recordings of fold participants.
  - **Disjoint:** on all recordings of participants outside the fold.
- The overlap training set is subsampled to the disjoint set's size, by removing chunks of participants outside the fold.
- Families and seeds: MLP and TCN (training loop and hyperparameters of `p4_train_of.py`), three seeds each. The epoch count is fixed at the median validation-selected epoch of each family in `engagenet/runs/`, so no held-out data is used for selection.
- Outcomes on H, pooled over folds, with participant-clustered intervals of the overlap − disjoint difference: accuracy, UAR, within-person AUC (decisive), between-person share of scores.

**H6.** Accuracy is higher under overlap, with an interval that excludes zero.

**H7.** The between-person share of scores is higher under overlap.

No directional hypothesis is registered for within-person AUC.

### D. A condition in which identity cannot help

- Task: binary decisive (Highly-Engaged against Engaged). This is the target of training and evaluation alike.
- Data: EngageNet training participants with at least eight chunks of each decisive class.
- Balanced condition: each participant is subsampled to equal numbers of the two classes, so that the label is independent of the participant.
- Comparison condition: the same participants, the same total size, each participant's natural class proportions.
- Participant-disjoint five-fold evaluation; MLP and TCN, three seeds.
- Outcomes: within-person AUC; between-person share of scores; and identity decodability of the penultimate representation, read by a retrained logistic probe on held-out chunks of training participants. ρ_b is not used here, because participant mean labels are constant under balancing.

**H8.** The between-person share of scores is lower in the balanced condition.

**H9.** Identity remains decodable far above chance (more than ten times chance) in both conditions. In other words, decodability does not require that identity be used for the target.

### E. A known person-only model

The leave-one-out constant-per-subject oracle of Supplementary S4 is reported in the same table as a model that is person-based by construction. This needs no new run.

### F. Temporal order

Families: LSTM, CNN-LSTM, TCN and Transformer, with the MLP as the anchor without a temporal axis.

- **(i) Retrained with shuffled order.** Each family is retrained with three seeds, the nine sub-windows of every chunk being permuted at random in each epoch. The model is read on validation at the fixed epoch count of Section 4C.
  - The ordered-trained comparison runs are retrained in the same script at the same fixed epoch count, not taken from the stored runs, so that only the order differs.
  - Primary outcome: shuffled-trained − ordered-trained validation accuracy, UAR and within-person AUC, paired over seeds, with participant-clustered intervals.
- **(ii) Frozen checkpoints, shuffled at inference.** The twelve stored checkpoints are read on validation with 20 random permutations and with reversed order. This arm is secondary: it mixes use of order with a distribution shift, because the TCN reads the last step and the Transformer has position embeddings.

**H10.** For each family, the interval of the primary difference includes zero, in line with the main text's finding that no sequence model separates from the non-temporal control.

### G. ResNet-TCN frame ablation (Reviewer 1, comments 5 and 7)

The article describes its input as follows: "The videos were down-sampled, temporally and spatially, to get 50 × 3 × 224 × 224 (L × C × H × W) tensors" (Abedi and Khan, Section IV-B). The released code instead reads the first 16 consecutive frames.

Arms:
- **A16c (primary):** the first 16 consecutive frames, the released configuration, using the frames already extracted for the re-run.
- **A50u (primary):** 50 frames spread uniformly over the clip, frame index ⌊i·n/50⌋ for i = 0..49, where n is the decoded frame count.
- **A16u (secondary):** 16 frames spread uniformly, index ⌊i·n/16⌋. This arm separates frame count from temporal span.

Everything else is the released code, unchanged:
- ResTCN, ImageNet-pretrained ResNet-18;
- SGD with learning rate 10⁻³ and momentum 0.9, batch size 4, 100 epochs, seed 0;
- unweighted cross-entropy;
- the 80:560 column crop and resizing to 224.

Data: the arms train on the DAiSEE training split only (5,358 clips) and are evaluated on the validation split (1,429 clips) after every epoch. The validation probabilities of every epoch are saved. The three arms run concurrently on one GPU.

- **Primary outcomes:** final-epoch validation accuracy, UAR and within-person AUC₂₃ (Audit 2 on the saved probabilities).
- **Secondary outcome:** the peak validation accuracy, reported as a maximum over 100 reads.

**Interpretation rule, fixed in advance.** The ablation estimates the effect of frame count and span under the released training recipe, on validation, with one seed. It does not estimate whether 63.9% is recoverable, because it differs from the article in training data (training split only), evaluation split and selection. It will not be described as a reproduction of the article.

No directional hypothesis is registered for accuracy. **H11:** within-person AUC₂₃ of A50u lies inside the 0.54–0.61 band of the main text. If it lies outside, the paper's statement that the tested DAiSEE representations share that band will be revised to say which configuration departs from it.

## 5. Reporting

All registered quantities go to the supplement, with a summary in the main text. Directional hypotheses are reported as supported or not supported, with intervals. The registration's SHA-256 is cited in the response to reviewers.
