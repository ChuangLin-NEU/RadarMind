# Cross-Radar Human Activity Recognition: When High Within-Radar Accuracy Fails to Generalize

Authors: [TODO]  
Affiliation: [TODO]

## Abstract

Radar human activity recognition (HAR) is commonly evaluated with high same-sensor accuracy, and that accuracy is then treated as evidence that a representation will transfer across carrier frequency and hardware [CITATION NEEDED]. We test the transfer assumption on 327 timestamp-paired CI4R acquisitions recorded at 10, 24 and 77 GHz (7 actions; acquisition-level split 229/49/49). Within-radar accuracy on official plotted spectrograms is 0.843 / 0.898 / 0.922, while the 24→77 transfer collapses to 0.039. A three-radar Radar-ID probe remains near ceiling after numerical Doppler reconstruction (0.993). For the two-radar 24/77 GHz diagnostic, mapping Doppler frequency onto a common radial-velocity axis reduces representation-level radar identification only from 1.000 to 0.939. A source-only model predicts part of the paired velocity-marginal response (24→77 $R^2$ 0.744), but a low-rank 2D translation oracle on the full time–velocity field reaches only $R^2$ 0.418. The results indicate that cross-radar failure cannot be explained by plotting, Doppler scaling, or simple calibration alone, and that current paired data lack the spatial and acquisition observables needed to disentangle motion from sensor-dependent measurement effects.

**Keywords:** radar HAR; cross-radar generalization; micro-Doppler; domain shift; measurement response; paired multi-frequency data

## 1. Introduction

Radar sensing of human motion has been studied for indoor activity recognition, gait analysis and related monitoring tasks [CITATION NEEDED]. Reported accuracies on single-sensor benchmarks are often high. That pattern can suggest, incorrectly, that the representation already isolates motion from the instrument.

Most published radar HAR evaluations stay within one radar, one processing chain and one plotted spectrogram convention [CITATION NEEDED]. Under those conditions a classifier may use any stable cue—window length, clutter filter, display crop, residual carrier-dependent scaling—without that cue being named as motion.

Cross-radar generalization is harder because a spectrogram is not a camera image of the body. It is the output of a sensing configuration and a processing chain. Carrier frequency, pulse repetition frequency, range gating, moving-target indication (MTI) and the time–frequency operator all shape the same physical movement into different arrays.

A common response is to treat radar shift as ordinary feature-domain shift and to apply adaptation, alignment or adversarial training [CITATION NEEDED]. Those tools assume that a shared motion factor is already present and mainly needs to be decoded. They do not test whether the arrays still contain a recoverable sensor identity after plotting conventions, Doppler-axis unification and first-order calibration are removed.

This paper asks a narrower question: how much of the observed cross-radar failure on a public paired 10/24/77 GHz corpus comes from plotted-image conventions, from numerical reconstruction, from physical Doppler-to-velocity scaling, from signal-processing mismatch, and from a residual measurement response that still depends on motion state? We do not claim that the question is closed by a new encoder.

Contributions. (i) A timestamp-paired 10/24/77 GHz HAR benchmark in which within-radar accuracy is high and several cross-radar directions collapse. (ii) A representation audit from official plotted images through cropped images and numerical Doppler maps to a 24/77 GHz radial-velocity grid, with Radar-ID and activity probes kept on explicitly labeled protocols. (iii) A decomposition of preprocessing, harmonized STFT/MTI, known-radar calibration, adapters and interpretable motion factors. (iv) Evidence that source-observable motion factors predict part of the paired velocity-marginal response, while low-rank 2D translation meets a low oracle ceiling, and that the public pairs lack spatial and acquisition fields needed to go further. A next-recording protocol is stated as Outlook, not as an experimental result.

## 2. Methods

### 2.1 Dataset and paired acquisition protocol

We use 327 CI4R acquisitions that share a timestamp across 10, 24 and 77 GHz, restricted to 7 actions (Away, Bend, Crawl, Kneel, Pick, Sit, Towards). The official corpus description states that activities were performed along the radar line of sight [CITATION NEEDED]. Filenames and the paired mapping do not provide per-trial distance, aspect, orientation, a subject identifier aligned to timestamp, or an environment identifier. We therefore treat the set as paired in time, not as a geometry-controlled factorial. Each radar contributes 327 reconstructions on this subset.

### 2.2 Evaluation protocols

The primary split is acquisition-timestamp based (229/49/49). The same timestamp never appears in two splits. We do not encode subject from a filename prefix. Cross-radar HAR numbers in the main text are train-on-one-radar, test-on-another, unless labeled otherwise.

Leave-one-radar-out (LORO) encoder numbers are domain generalization and are reported as supplementary. CORAL-style alignment is unsupervised domain adaptation (UDA) and is not DG; those numbers are also supplementary.

Radar-ID and activity probes are diagnostics, not proposed HAR systems. A three-radar probe (10/24/77 GHz) is never treated as interchangeable with a two-radar 24/77 GHz probe. Representation-level logistic probes, encoder-embedding probes and motion-factor probes are reported in separate compare groups.

### 2.3 Official plotted representations

The official representation is the public spectrogram PNG. A cropped, common-postprocessed plotted image is used only to test whether display conventions explain the domain gap. Both remain display products.

### 2.4 Numerical Doppler reconstruction

Numerical Doppler maps are $S(\tau, f_D)$ on a common Doppler band, built from native short-time processing rather than from a screenshot. Invertible normalization is a train-only z-score of clipped decibels.

### 2.5 Physical velocity alignment

For 24 GHz, $f_c=24.75$ GHz is taken from the raw header. For 77 GHz the official script value is 77.56365 GHz. Radial velocity is $v=c f_D/(2 f_c)$. The common interval is the physical intersection $[-3.028, 3.027]$ m/s, not a tuned crop. 10 GHz is excluded from all velocity maps because the effective carrier is unresolved; a published range-bin sampling rate must not be used as $f_c$.

### 2.6 Representation and Radar-ID probes

A linear Radar-ID probe (pooled $16\times16$ features, logistic regression) estimates how recoverable the sensor is. An activity probe on the same features estimates motion-related information. Pair separation is same-timestamp cosine minus different-timestamp cosine.

### 2.7 Signal-processing harmonization

Unified STFT rebuilds 24/77 velocity maps with a shared window, hop and FFT length while keeping official range/MTI choices. A further variant transplants a 24 GHz Butterworth operator at a physical cutoff. A unified range-metre gate is not available.

### 2.8 Sensor calibration

Known-radar affine and histogram matches are applied on numerical Doppler maps. Global and per-velocity affine maps are then tested on unified velocity maps. Gates were preregistered on validation statistics.

### 2.9 Motion-factor analysis

38 interpretable energy-weighted factors are computed with pre-registered Doppler bands. Action labels are not used as calibration inputs.

### 2.10 Motion-conditioned measurement response

On unified-STFT log-energy, $D_i(v)=\log P_{77,i}(v)-\log P_{24,i}(v)$ uses an energy mask with 69 valid bins. Source-only factors $q$ (no target radar, no action label) predict $D(v)$ by Ridge. The train-mean $D(v)$ is the baseline. This is paired known-radar regression, not unseen-radar DG.

### 2.11 Low-rank 2D paired-difference model

$\Delta_i(\tau,v)=X_{77,i}-X_{24,i}$ in log-energy. TRAIN-only PCA selects rank $K=20$ by validation reconstruction error. The TEST oracle projects true TEST $\Delta$ onto the TRAIN basis; that quantity is an upper bound, not a deployable translator. A second model predicts PCA scores from $q$.

## 3. Results

### 3.1 High within-radar accuracy masks poor cross-radar transfer

On official plotted spectrograms, within-radar 7-class accuracy is 0.843 (10 GHz), 0.898 (24 GHz) and 0.922 (77 GHz). Single-source 24→77 transfer is 0.039; 77→24 is 0.163. A reader who only sees the diagonal would conclude that the task is largely solved (Table 2, Fig. 2).

### 3.2 Preprocessing artifacts explain only part of the domain gap

Cropping official spectrograms raises 10→24 accuracy from 0.143 to 0.510, so display convention is a real confound. The three-radar Radar-ID probe remains 1.000 on official images and 1.000 after cropping. Other cross directions stay weak (Table 2). Plotting is not a sufficient explanation.

### 3.3 Numerical Doppler representations retain strong radar identity

After numerical Doppler reconstruction, the three-radar Radar-ID probe is 0.993; the corresponding representation-level activity probe is 0.946. Within-radar 7-class accuracy on these maps remains usable (0.776 / 1.000 / 0.878). Pairwise cosine is high for both same-timestamp and different-timestamp pairs (0.917 vs 0.913), so global similarity is not a substitute for identity removal. LORO 7-class accuracy on held-out radars is reported in the Supplementary (e.g. 10+24→77 = 0.224) and is domain generalization, not UDA.

### 3.4 Velocity alignment does not remove the 24/77 GHz domain shift

The following numbers are two-radar 24/77 GHz diagnostics and must not be read as a drop of the three-radar probe in Section 3.3. On R2-D restricted to 24/77 GHz, representation-level Radar-ID is 1.000. After mapping both radars onto the common radial-velocity axis, that two-class Radar-ID is 0.939. The matching activity probes are 0.878 and 0.857. Pair separation is 0.086 versus 0.087. Carrier-frequency Doppler scaling is a weak contributor relative to the remaining measurement difference. 10 GHz cannot join this comparison.

### 3.5 Harmonized signal processing and simple calibration remain insufficient

Still on the 24/77 GHz pair, unified STFT leaves Radar-ID at 0.939. Transplanting the 24 GHz MTI operator increases it to 0.980. Known-radar C0–C4 calibration does not pass the preregistered validation gate (best Radar-ID 0.980 on the three-radar probe). The strongest first-order per-velocity affine map has two-class Radar-ID 0.949 and velocity-marginal Radar-ID 0.929. Empirical $D(v)$ is not a stable hardware curve: transfer stability is 0.395 and action $\eta^2$ is 0.558.

### 3.6 Learned alignment does not yield sensor-invariant representations

A0, A1 and A2 are known-radar joint-task accuracies, not cross-radar zero-shot scores. Shared-encoder accuracy is 0.946. Adding radar-specific adapters and an alignment term yields 0.878 and 0.782. Radar-ID on A2 embeddings is 1.000. Known-radar pairing does not improve, and LORO adapters do not improve unseen-radar HAR (Supplementary). The arrays still support a radar classifier after the mapping that was meant to hide the sensor.

### 3.7 Radar response varies systematically with motion state

A linear probe on 38 motion factors gives activity-probe 0.687 and Radar-ID 1.000 (stable subset 0.490 / 0.959). These probes live on factor vectors, not on the spectrogram pixels of Sections 3.3–3.5. High activity information and high radar information coexist (Fig. 6).

Source-only Ridge predicts the velocity-marginal transfer $D(v)$ far above the train mean: 24→77 $R^2$ 0.744 (mean baseline 0.219), 77→24 $R^2$ 0.575. These $R^2$ values measure prediction of paired Doppler-energy differences, not HAR accuracy. The two directions are not interchangeable.

### 3.8 Current observables are insufficient for stable cross-radar measurement translation

Applying $\hat D(v\mid q)$ as a symmetric correction moves 24/77 Radar-ID from 0.939 to 1.000. Rank-20 PCA of the 2D paired difference explains 0.490 of TRAIN variance. The TEST oracle—true TEST $\Delta$ projected onto the TRAIN basis—has $R^2$ 0.418. That oracle is an upper bound on how much of an unseen 2D difference lies in the TRAIN subspace; it is not a deployed translator. The $q$-driven model reaches 0.300 (24→77) and 0.173 (77→24). Activity on predicted 24→77 maps falls from 0.898 to 0.735. We stop empirical sensor translation on this corpus.

## 4. Discussion

A representation can be informative for action and still be a reliable radar fingerprint. HAR accuracy measures the first; it does not measure the second. Calling a high-accuracy spectrogram encoder motion-centric is a category error unless radar identity has been shown to be unrecoverable under a stated probe protocol.

Paired correspondence (same timestamp, two radars) is not invariance. A map that makes a 24 GHz array look 77-like to a domain classifier can still fail to match the 2D measurement field and can discard activity cues. Cross-radar change is not a global style shift that a single filter can undo.

The same action label does not imply the same measurement operator. Transfer $D(v)$ varies with source-observable motion factors and with action in post-hoc residuals. That is compatible with a motion-dependent measurement response. It is not evidence that a missing range or aspect angle caused the residual: those fields were never recorded.

The binding constraint is missing spatial and acquisition metadata: physical range, subject–radar distance, aspect, orientation, subject and session identity, and a recoverable two-radar range axis. Without them, sensor, geometry and subject remain entangled.

Based on those limitations, a next synchronized 24/77 GHz acquisition protocol should record distance, aspect, orientation, raw IQ, the full radar configuration and video timestamps. Mock toolkit files produced while designing that protocol are not experimental outcomes and are not used as results in this paper.

## 5. Conclusions

### Version A

Within-radar HAR accuracy on plotted CI4R spectrograms is high, and several cross-radar transfers collapse. We traced the gap through cropping, numerical Doppler reconstruction, 24/77 velocity alignment, STFT/MTI harmonization, known-radar calibration, adapters and interpretable factors. Radar identity remains recoverable at each stage that we can justify from the files, provided three-radar and two-radar probes are not collapsed into one number. A source-only linear model predicts part of the paired velocity-marginal response, but a low-rank 2D translation oracle remains limited. The public pairs do not support a claim that sensor-invariant motion features have been learned.

The practical implication is not another translator on the same 327 timestamps. It is a recording design that factorizes sensor, motion, geometry and subject, with raw IQ and a documented range axis. Until those observables exist, statements about universal or geometry-aware radar encoders are not testable on this corpus.

### Version B

Strong same-sensor radar HAR can coexist with near-chance cross-radar transfer. After removing plotting artifacts and aligning 24/77 GHz velocity axes, radar identity is still easy to read, and empirical translation meets a low 2D ceiling. The missing piece is spatial and acquisition metadata, not an untried network.

## Data Availability

See `data_availability.md`. CI4R is a public third-party corpus [CITATION NEEDED]. Derived files used here live under the project `results/` tree. A public archive of those derived files: **TODO** (no DOI assigned).

## Code Availability

See `code_availability.md`. Research code and the acquisition-toolkit engineering module: repository **TODO**. No public URL or DOI is claimed.

## Author Contributions

[TODO]

## Conflict of Interest

[TODO]

## References

[CITATION NEEDED] Official CI4R dataset publication (bibliographic fields to be copied from the publisher; do not invent a DOI here).

[CITATION NEEDED] Additional radar HAR and domain-adaptation references to be added during manual writing.
