"""Full SCI manuscript blocks. All numbers come from the metric snapshot via D()."""
from __future__ import annotations


def D(snap, key: str) -> str:
    return snap["metrics"][key]["display"]


def pm(snap, mean_key: str, std_key: str) -> str:
    return f"{D(snap, mean_key)} ± {D(snap, std_key)}"


TITLES = [
    "RadarMind: Toward Generalizable Human Motion Understanding Across Radar Domains",
    "RadarMind: A Generalization-Centered Framework for Cross-Radar Human Motion Recognition, Continuous Understanding, Open-Set Detection, and Language-Based Interpretation",
    "RadarMind: From Radar Measurements to Structured Motion Intelligence and Language-Based Interpretation",
    "RadarMind: Diagnosing Cross-Radar Failure While Building a Radar-to-Semantics Pipeline",
    "RadarMind: Connecting Radar Recognition, Continuous Inference, Open-Set Detection, and Semantic Interaction",
]
SELECTED_TITLE = TITLES[0]


def abstract(snap) -> str:
    return (
        "Radar-based human activity recognition (HAR) provides privacy-preserving motion sensing, "
        "but conventional systems usually assume a fixed set of activity classes, discrete clips, "
        "and the same sensing domain at train and test time. This paper presents RadarMind, "
        "a multilayer framework that connects radar representations to activity recognition, "
        "cross-domain generalization analysis, continuous motion inference, open-set rejection, "
        "structured motion descriptors, and language-based semantic interaction. "
        "The study uses three public datasets that serve different roles rather than a single merged benchmark: "
        "CI4R multi-frequency spectrograms (10/24/77 GHz), mRI point clouds with official subject identities, "
        "and RadHAR point clouds under the official train/test split. "
        f"On mRI subject-independent recognition, PointNet-GRU reaches accuracy {D(snap,'MRI_SI_ACC')} "
        f"and macro-F1 {D(snap,'MRI_SI_F1')}. On RadHAR official XYZV, accuracy and macro-F1 are "
        f"{D(snap,'RADHAR_XYZV_ACC')} / {D(snap,'RADHAR_XYZV_F1')}. On CI4R official plotted spectrograms, "
        f"within-radar accuracies are {D(snap,'R0_10_WITHIN')} / {D(snap,'R0_24_WITHIN')} / {D(snap,'R0_77_WITHIN')}, "
        f"while 24→77 transfer falls to {D(snap,'R0_24_TO_77')}. Numerical Doppler reconstruction still yields "
        f"3-radar Radar-ID {D(snap,'RID_R2D_3CLASS')}; 24/77 velocity alignment reduces 2-radar Radar-ID from "
        f"{D(snap,'RID_R2D_2477')} to {D(snap,'RID_R2V_2477')}. Labeled-interval continuous replay on mRI "
        f"gives frame accuracy {D(snap,'CONT_FRAME_ACC')} and macro-F1 {D(snap,'CONT_FRAME_F1')}. "
        f"Leave-one-action-out prototype detection reaches AUROC {pm(snap,'PROTO_AUROC','PROTO_AUROC_STD')}. "
        "The language layer does not operate on raw radar. Structured Motion Intelligence and temporal events "
        "are passed to an online language-model service for explanation, summary, uncertainty communication, "
        "question answering, and interaction. RadarMind demonstrates a complete radar-to-semantics pipeline, "
        "while also showing that robust unseen-radar generalization remains limited by sensor-dependent "
        "measurement effects and missing spatial metadata."
    )


def blocks(snap) -> list[tuple]:
    """Return (kind, payload) blocks. kind in title, sub, meta, abs, h1, h2, p, fig, tab, cap, ref."""
    B = []
    add = B.append

    add(("title", SELECTED_TITLE))
    add(("sub", "Working draft compiled from frozen RadarMind experimental records"))
    add(("meta", "Authors and affiliations withheld in this working draft."))
    add(("h1", "Abstract"))
    add(("abs", abstract(snap)))
    add(("p", "Keywords: radar HAR; cross-radar generalization; subject-independent recognition; "
         "continuous motion; open-set detection; Motion Intelligence; language-based interpretation."))

    add(("h1", "1. Introduction"))
    add(("p",
         "Radar sensing records human motion without optical images. Frequency-modulated continuous-wave "
         "and pulsed radars return micro-Doppler spectrograms or sparse point clouds that remain usable "
         "when lighting is poor and when cameras are unacceptable. Those measurements have supported "
         "a large body of human activity recognition (HAR) work. The practical question is no longer "
         "whether a classifier can label a short clip on one radar. The question is whether the same "
         "sensing chain can describe motion when the person, the radar, the temporal structure, or the "
         "activity inventory changes."))
    add(("p",
         "A conventional radar HAR system has a short path: radar measurement, a representation "
         "(spectrogram or point cloud), a classifier, and a label from a closed set. That path is "
         "adequate for within-distribution clips. It is not a description of human motion. It does "
         "not say how long an action lasted, whether the decision is stable, what happened before "
         "and after the clip, or whether the observation lies outside the training classes."))
    add(("p",
         "Three limitations follow directly. First, high within-radar accuracy does not imply "
         "generalization. A model trained on one person or one radar can collapse on another. "
         "Subject-independent evaluation requires real subject identifiers; filename prefixes on "
         "CI4R are not subjects and must not be reported as such. Cross-radar evaluation requires "
         "paired acquisitions and an explicit protocol for unseen radars. Second, clip classification "
         "does not describe continuous behavior. Sliding windows on a recording produce a sequence, "
         "not a single label, and boundary localization is a different task from isolated recognition. "
         "Third, a closed set cannot handle unknown motion. Maximum softmax probability on known "
         "classes is not an open-set detector, and open-set rejection is not open-world or "
         "open-vocabulary recognition."))
    add(("p",
         "A fourth limitation is semantic. A model output such as Bend does not answer how long the "
         "action lasted, whether the confidence is high, what preceded it, how a continuous interval "
         "should be summarized, or whether an unseen motion occurred. Users and downstream tools "
         "need a structured intermediate layer—activity, confidence, uncertainty, motion descriptors "
         "that the representation actually supports, and a timeline of events—before a language model "
         "can explain the record. That layer is Motion Intelligence. It is not the language model."))
    add(("p",
         "RadarMind is organized in three layers. Layer I performs radar recognition: representation "
         "adapters, HAR models, subject-independent evaluation, and cross-radar transfer. Layer II "
         "performs motion understanding: shared-representation learning, labeled-interval continuous "
         "inference, leave-one-action-out open-set detection, Motion Intelligence, and timeline "
         "construction. Layer III performs semantic interaction: a language layer that reads only "
         "structured JSON, plus an interactive research application. The framework is a pipeline "
         "from radar measurements to increasingly structured, continuous, uncertainty-aware, and "
         "interpretable descriptions of motion. It is not a claim that a sensor-invariant or "
         "open-world representation has been obtained."))
    add(("p",
         "Cross-radar failure is the hardest scientific problem in this study and is not hidden. "
         "The paper tracks plotted spectrograms, numerical Doppler reconstruction, physical velocity "
         "alignment, STFT and MTI harmonization, first-order calibration, learned adapters, and "
         "motion-conditioned measurement response. Several interventions improve known-radar "
         "correspondence or 24/77 pair geometry. None removes radar identity on a held-out radar. "
         "Geometry is a plausible missing factor; it is not proven here as the residual cause."))
    add(("p",
         "Related radar HAR work typically reports within-dataset clip accuracy on micro-Doppler "
         "images or point clouds [CITATION NEEDED]. Domain-adaptation papers sometimes treat a "
         "second radar or a second room as a target domain under UDA [CITATION NEEDED]. Open-set "
         "and out-of-distribution scores are standard in vision [CITATION NEEDED] and have begun "
         "to appear in radar [CITATION NEEDED]. Language models have been attached to sensors "
         "mainly by sending images or free text [CITATION NEEDED]. RadarMind differs in two "
         "respects that are empirical, not rhetorical. First, the cross-radar matrix and the "
         "Radar-ID ladder are reported even when they fail. Second, the language layer is denied "
         "raw radar and is restricted to a typed intermediate record. Those choices keep the "
         "paper from collapsing recognition, physics, and wording into one unverifiable claim."))
    add(("p",
         "The contributions are five. (1) A unified RadarMind pipeline covering spectrogram and "
         "point-cloud inputs and multiple HAR datasets, without forcing those datasets into one "
         "benchmark. (2) A systematic generalization evaluation that includes within-distribution, "
         "subject-independent, cross-radar, and leave-one-radar-out protocols, with protocol names "
         "kept distinct. (3) A physics-informed investigation of cross-radar domain shift from "
         "plotted representations through numerical Doppler, velocity alignment, signal-processing "
         "harmonization, calibration, and measurement-response modeling. (4) An extension from "
         "discrete closed-set HAR to labeled-interval continuous inference and leave-one-action-out "
         "open-set detection. (5) A structured Motion Intelligence layer and a language layer that "
         "convert model outputs, descriptors, uncertainty, and temporal events into explanations, "
         "summaries, question answering, and application text. The language layer is not claimed to "
         "improve HAR accuracy."))

    add(("p",
         "The rest of the paper follows that order. Section 2 defines the architecture, datasets, "
         "adapters, evaluation protocols, physical operators, continuous and open-set procedures, "
         "Motion Intelligence, the language-layer contract, and the application. Section 3 reports "
         "recognition, subject and temporal ablations, the cross-radar matrix, shared representation, "
         "physical diagnosis, continuous replay, open-set scores, structured examples, language-layer "
         "cases, and application smoke. Section 4 discusses what the language layer can and cannot "
         "do and why the intermediate record is required. Section 5 states limitations and the "
         "next acquisition protocol. Section 6 concludes with the two-layer summary: a working "
         "radar-to-semantics stack, and an unsolved unseen-radar problem."))
    add(("h1", "2. Methods"))
    add(("h2", "2.1 RadarMind system architecture"))
    add(("p",
         "The implemented chain is: radar data; a sensor/representation adapter; a motion encoder; "
         "recognition, open-set scoring, or continuous window inference; Motion Intelligence; a "
         "timeline of merged events; a language layer; and the RadarMind application. "
         "Spectrogram and point-cloud paths share software interfaces (MotionInput, MotionPrediction, "
         "MotionIntelligence, MotionTimeline). They do not share a trained token vocabulary. "
         "Fully shared motion tokens across radar forms have not been realized. "
         "A Radar-ID probe in this paper is a linear or logistic classifier trained to recover "
         "sensor identity from a representation, using the same acquisition split as the HAR "
         "experiments. High Radar-ID means the representation still names the radar. It is not "
         "itself a HAR accuracy and is never substituted for one. An activity probe on the same "
         "feature family asks whether motion information survives after an operator is applied. "
         "The two probes together are the information-tradeoff diagnostic used in Fig. 6."))
    add(("p",
         "Leave-one-radar-out (LORO) trains on two radars and tests on the third with no target "
         "samples in training, validation, or normalization statistics. That is domain generalization. "
         "CORAL uses unlabeled target-train samples and is UDA. Known-radar joint encoders (M0–M3, "
         "A0–A2) see all three radars in training. Mixing those three protocols in one leaderboard "
         "would make a weak LORO number look like a known-radar success, or the reverse. Tables "
         "in this paper keep the protocol column explicit."))
    add(("fig", ("fig01_radarmind_framework.png",
                 "Fig. 1. RadarMind three-layer framework. Layer I maps radar representations to recognition, "
                 "subject generalization, and cross-radar tests. Layer II builds a shared representation, "
                 "labeled-interval continuous inference, open-set scores, Motion Intelligence, and a timeline. "
                 "Layer III exposes structured JSON to a language layer. The language layer does not receive raw radar. "
                 "The dashed box lists problems that remain outside the current evidence.")))

    add(("h2", "2.2 Datasets and radar representations"))
    add(("p",
         f"CI4R supplies multi-frequency spectrograms and, for 24/77 GHz, reconstructable raw records. "
         f"The paired cross-radar subset contains {D(snap,'N_PAIRED_ACQ')} timestamp-aligned acquisitions "
         f"with agreed action labels across 10, 24, and 77 GHz, {D(snap,'N_ACTIONS')} actions, and an "
         f"acquisition-timestamp split of {D(snap,'SPLIT_TRAIN')} / {D(snap,'SPLIT_VAL')} / {D(snap,'SPLIT_TEST')}. "
         f"An additional {D(snap,'N_CONFLICT')} timestamp triples were excluded because the three "
         "radars did not agree on the action label. "
         "This subset is the cross-radar benchmark. A separate CI4R 77 GHz 11-class setting is used only "
         "for within-radar random-record versus prefix-group-disjoint comparison. Prefix groups are "
         "acquisition-group-disjoint partitions. They are not subject-independent splits."))
    add(("p",
         f"mRI is a point-cloud corpus with {D(snap,'MRI_N_SUBJECTS')} official subject identities, "
         f"{D(snap,'MRI_N_SEGMENTS')} labeled segments, {D(snap,'MRI_N_ACTIONS')} normalized activities, "
         f"and window length T={D(snap,'MRI_T')} with P={D(snap,'MRI_P')} sampled points. "
         f"The subject-independent split uses {D(snap,'MRI_SI_N_TRAIN')} / {D(snap,'MRI_SI_N_VAL')} / "
         f"{D(snap,'MRI_SI_N_TEST')} windows. Test subjects are subject3, subject10, and subject20. "
         "mRI is the only local corpus that supports true subject-independent evaluation, labeled-interval "
         "continuous replay, and leave-one-action-out open-set folds."))
    add(("p",
         f"RadHAR is a point-cloud corpus with {D(snap,'RADHAR_N_ACTIONS')} actions and the official "
         f"train/test listing (T={D(snap,'RADHAR_T')}, P={D(snap,'RADHAR_P')}, "
         f"n_test={D(snap,'RADHAR_N_TEST')}). It has no subject metadata in the local package, so it "
         "is never reported as subject-independent. It supports official-split recognition and "
         "temporal/feature ablation. The three datasets are complementary. They are not treated as "
         "one aligned multi-radar, multi-person, multi-environment benchmark."))
    add(("tab", None))
    add(("fig", ("fig02_dataset_protocol_map.png",
                 "Fig. 2. Dataset–protocol map. CI4R paired 10/24/77 GHz supports cross-radar tests. "
                 "CI4R 77 GHz 11-class compares random-record and prefix-group-disjoint splits and is not subject-independent. "
                 "mRI is the only local corpus with official subject IDs, labeled-interval continuous replay, and LOAO open-set. "
                 "RadHAR uses the official train/test listing. The three sources are complementary, not one unified benchmark.")))
    add(("p",
         "Table 1 states the same roles in text: which representation each corpus supplies, "
         "which experiment it carries, and which protocol reading is forbidden. "
         "CI4R prefix-group-disjoint is not subject-independent. The RadHAR official listing is not subject-independent."))

    add(("h2", "2.3 Representation adapters"))
    add(("p",
         "SpectrogramAdapter maps a micro-Doppler image to a 224×224 tensor with the normalization "
         "used by the corresponding ResNet. PointCloudAdapter maps a T×P×C window and a validity mask "
         "to the PointNet-GRU input. Both adapters are software contracts. They do not implement a "
         "learned sensor-invariant tokenizer. Fields that a representation cannot support—direction "
         "on an uncalibrated Doppler sign, environment, or subject on RadHAR—remain UNKNOWN."))

    add(("h2", "2.4 Baseline HAR models"))
    add(("p",
         "CI4R official plotted spectrograms use ResNet18. mRI and RadHAR use PointNet-GRU. "
         "RadHAR also includes a voxel CNN-GRU comparison. Model choice is not the scientific claim. "
         "The models exist to hold the protocol fixed while datasets, splits, temporal length, and "
         "feature channels vary. Training details, seeds, and checkpoints remain in the frozen "
         "Phase 1 records and are not re-run for this manuscript."))

    add(("h2", "2.5 Cross-subject evaluation"))
    add(("p",
         f"mRI random-segment PointNet-GRU (XYZVI, T=32) reaches accuracy {D(snap,'MRI_RAND_ACC')} "
         f"and macro-F1 {D(snap,'MRI_RAND_F1')}. The same architecture under true subject-independent "
         f"partitioning (XYZ, T=32) reaches accuracy {D(snap,'MRI_SI_ACC')} and macro-F1 {D(snap,'MRI_SI_F1')}. "
         "The drop is the quantity that matters: people in the test set are absent from training. "
         f"CI4R 77 GHz 11-class random-record accuracy is {D(snap,'CI4R77_RAND_ACC')} "
         f"(macro-F1 {D(snap,'CI4R77_RAND_F1')}); prefix-group-disjoint accuracy is "
         f"{D(snap,'CI4R77_PREFIX_ACC')} (macro-F1 {D(snap,'CI4R77_PREFIX_F1')}). "
         "The prefix protocol is acquisition-group-disjoint. It is not written as cross-subject."))

    add(("p",
         "Point sampling is fixed once in the Phase 1 cache: mRI uses P="
         f"{D(snap,'MRI_P')} points and T={D(snap,'MRI_T')}; RadHAR uses P={D(snap,'RADHAR_P')} "
         f"and T={D(snap,'RADHAR_T')}. Empty frames are masked rather than imputed with a fake "
         "origin point. Intensity and Doppler/velocity occupy extra channels when the source file "
         "provides them. Channel ablations therefore disable a stored channel; they do not "
         "recompute a physical quantity that was never written."))
    add(("h2", "2.6 Temporal and feature ablation"))
    add(("p",
         f"On mRI subject-independent data, T=1 pooling yields accuracy {D(snap,'MRI_T1_ACC')} "
         f"(macro-F1 {D(snap,'MRI_T1_F1')}). Extending to T=32 raises accuracy to "
         f"{D(snap,'MRI_SI_ACC')} for XYZ, {D(snap,'MRI_XYZV_ACC')} for XYZV, and "
         f"{D(snap,'MRI_XYZVI_ACC')} for XYZVI. XYZ is the strongest mRI channel set in this grid. "
         f"On RadHAR official split, T=1 yields {D(snap,'RADHAR_T1_ACC')}; T=60 XYZV reaches "
         f"{D(snap,'RADHAR_XYZV_ACC')} (macro-F1 {D(snap,'RADHAR_XYZV_F1')}). XYZ alone is "
         f"{D(snap,'RADHAR_XYZ_ACC')}; XYZVI is {D(snap,'RADHAR_XYZVI_ACC')}; voxel CNN-GRU is "
         f"{D(snap,'RADHAR_VOXEL_ACC')}. Velocity contributes on RadHAR; intensity does not add a "
         "further gain in this grid. Isolated-frame geometry is not a substitute for temporal context."))

    add(("h2", "2.7 Cross-radar benchmark"))
    add(("p",
         f"The CI4R paired subset is the cross-radar testbed: {D(snap,'N_PAIRED_ACQ')} acquisitions, "
         f"{D(snap,'N_ACTIONS')} actions, split {D(snap,'SPLIT_TRAIN')}/{D(snap,'SPLIT_VAL')}/{D(snap,'SPLIT_TEST')}. "
         f"Within-radar official-plot accuracies are {D(snap,'R0_10_WITHIN')} (10 GHz), "
         f"{D(snap,'R0_24_WITHIN')} (24 GHz), and {D(snap,'R0_77_WITHIN')} (77 GHz). "
         f"Zero-shot transfers include 24→77 = {D(snap,'R0_24_TO_77')}, 77→24 = {D(snap,'R0_77_TO_24')}, "
         f"10→24 = {D(snap,'R0_10_TO_24')}, 77→10 = {D(snap,'R0_77_TO_10')}, "
         f"10→77 = {D(snap,'R0_10_TO_77')}, and 24→10 = {D(snap,'R0_24_TO_10')}. "
         "The same acquisition-timestamp split is used for all subsequent representation edits. "
         "No test radar sample enters source-only training. "
         "The pairing key is the acquisition timestamp, not a subject identifier and not a "
         "filename prefix. Two radars that share a timestamp and an agreed action label are "
         "treated as a paired view of the same physical event. That pairing is strong enough "
         "to support Radar-ID, retrieval, and difference-field analyses. It is not strong enough "
         "to identify distance or aspect, because those fields are absent."))
    add(("p",
         "Within-radar training uses only the source radar's train split. Cross-radar testing "
         "uses the target radar's test split. The same 49 or 51 test acquisitions therefore "
         "appear in every row of Fig. 4 that ends on that radar. A collapse on 24→77 is not "
         "explained by a change of action inventory: the seven classes are shared. The remaining "
         "explanations are representation, measurement physics, unrecorded geometry, and person "
         "differences that the acquisition-timestamp split does not isolate. Later sections "
         "remove plotting and frequency-axis mismatch. They do not obtain a spatial metadata "
         "table, so geometry stays a hypothesis. "
         f"Native Doppler lengths differ by radar (10/24/77 GHz: {D(snap,'NATIVE_N_10')} / "
         f"{D(snap,'NATIVE_N_24')} / {D(snap,'NATIVE_N_77')}). R2-V interpolates 24/77 onto a "
         f"common velocity interval [{D(snap,'VEL_LO')}, {D(snap,'VEL_HI')}] m/s after reading "
         f"the 24 GHz carrier as {D(snap,'FC_24_GHZ')} GHz. Those facts define what aligned means. "
         "They are not evidence that alignment has removed the sensor."))
    add(("p",
         "A 3-radar Radar-ID probe asks whether sensor identity is recoverable from a representation. "
         "A 2-radar 24/77 probe uses the same idea on a pair. Putting both numbers on one axis "
         "without a break would invent a protocol that was never run. Fig. 6 therefore keeps the "
         "3-radar probe and the 24/77 probe in separate panels. A point labeled R2-V is not comparable "
         "to a point labeled R0 as if they were one experiment with one identifiability score."))

    add(("h2", "2.8 Shared motion representation learning"))
    add(("p",
         "A joint encoder is trained on all three radars of the paired training acquisitions. "
         f"M0 is cross-entropy only (accuracy {D(snap,'M0_ACC')}). M1 adds InfoNCE "
         f"(accuracy {D(snap,'M1_ACC')}). M2 adds physics-informed augmentation "
         f"(accuracy {D(snap,'M2_ACC')}). Linear activity probes on the embeddings are "
         f"{D(snap,'M0_ACTP')} / {D(snap,'M1_ACTP')} / {D(snap,'M2_ACTP')}. "
         f"Same-action retrieval top-1 is {D(snap,'M0_RET')} / {D(snap,'M1_RET')} / {D(snap,'M2_RET')}. "
         f"Radar-ID on the same embeddings is {D(snap,'M0_RID')} / {D(snap,'M1_RID')} / {D(snap,'M2_RID_SHARE')}. "
         "Paired learning improves cross-radar semantic correspondence among known radars. "
         "It does not remove sensor identity."))

    add(("h2", "2.9 Domain alignment and sensor adapters"))
    add(("p",
         f"CORAL unsupervised domain adaptation on 77→10 moves source-only accuracy "
         f"{D(snap,'R0_77_TO_10')} to {D(snap,'CORAL_77_TO_10')}. The change is negative. "
         "CORAL is UDA: unlabeled target-train samples are available. It is not domain generalization. "
         f"Domain-adversarial M3 keeps Radar-ID at {D(snap,'M3_RID')} and does not improve leave-one-radar-out "
         f"transfer (10+24→77 M3 = {D(snap,'M3_LORO_77')}; baseline LORO 10+24→77 = {D(snap,'LORO_1024_TO_77')}). "
         f"RadarBridge adapters on known radars give A0 {D(snap,'A0_ACC')}, A1 {D(snap,'A1_ACC')}, "
         f"A2 {D(snap,'A2_ACC')}; A2 embedding Radar-ID remains {D(snap,'RID_A2_EMBED')}. "
         "A shared representation that still identifies the radar is not sensor invariance. "
         "A0 is a joint encoder without radar-specific adapters. A1 inserts a lightweight "
         "per-radar map adapter. A2 trains the encoder with an additional invariance term. "
         f"Known-radar accuracy declines from A0 {D(snap,'A0_ACC')} to A2 {D(snap,'A2_ACC')} while "
         f"embedding Radar-ID stays {D(snap,'RID_A2_EMBED')}. The adapter family therefore does "
         "not provide a free lunch in which activity rises and radar identity falls together."))
    add(("p",
         "LORO numbers belong in the same subsection because they are the test that would have "
         "justified a sensor-invariance sentence. "
         f"Baseline LORO 10+24→77 is {D(snap,'LORO_1024_TO_77')}; M3 on that split is "
         f"{D(snap,'M3_LORO_77')}. Other LORO cells in the frozen Phase 3 table remain in the "
         "same low range. Seeing two radars in training does not yield the third. That is the "
         "operational meaning of 'unseen radar' in this paper."))

    add(("h2", "2.10 Physical representation reconstruction"))
    add(("p",
         "Official plotted PNGs (R0) contain color scales, padding, and overlay conventions that "
         "a convolutional network can use as a radar name tag. Cropped plots (R1) remove some of "
         "those cues. R2-D rebuilds spectrograms from numerical Doppler on a common frequency grid "
         "so that subsequent probes are not reading a PNG palette. "
         f"Three-radar Radar-ID remains {D(snap,'RID_R2D_3CLASS')} on R2-D, with activity probe "
         f"{D(snap,'ACT_R2D_REP')}. Plotting is therefore not a sufficient explanation of the fingerprint."))
    add(("p",
         "R2-V maps 24 GHz and 77 GHz Doppler to a shared radial-velocity axis using the carrier "
         "frequencies recorded in the raw headers. Ten-gigahertz records in this subset do not "
         "provide an equivalent raw chain, so R2-V is a 24/77 operator. "
         f"Two-radar Radar-ID falls from {D(snap,'RID_R2D_2477')} to {D(snap,'RID_R2V_2477')}. "
         f"Paired cosine on R2-D is {D(snap,'PAIR_R2D_SAME')} (same acquisition) versus "
         f"{D(snap,'PAIR_R2D_DIFF')} (different); after R2-V the pair is {D(snap,'PAIR_R2V_SAME')} "
         f"versus {D(snap,'PAIR_R2V_DIFF')}, and the same-minus-different gap moves from "
         f"{D(snap,'SEP_R2D')} to {D(snap,'SEP_R2V')}. Those two Radar-ID numbers are not on the "
         "same probe as the 3-radar series and are not plotted as one continuous protocol."))
    add(("p",
         "Unified STFT (U1) and transplanted MTI (U2) ask whether remaining mismatch is a "
         "processing-window mismatch. "
         f"U1 Radar-ID is {D(snap,'U1_RID')} with activity probe {D(snap,'U1_ACT')} and pair "
         f"separation {D(snap,'U1_SEP')}. U2 Radar-ID is {D(snap,'U2_RID')} with activity "
         f"{D(snap,'U2_ACT')} and separation {D(snap,'U2_SEP')}. Harmonizing the operator does "
         "not turn a 24 GHz map into a 77 GHz map. First-order known-radar calibration "
         f"(C0 Radar-ID {D(snap,'C0_RID')}; best-of-C0–C4 Radar-ID {D(snap,'CBEST_RID')}) likewise "
         "fails a gate that would have allowed a simple canonicalization claim."))
    add(("p",
         f"Interpretable motion-factor vectors still carry radar identity (Radar-ID {D(snap,'FAC_RID')}; "
         f"activity probe {D(snap,'FAC_ACT')}; stable subset Radar-ID {D(snap,'FAC_RID_STABLE')}, "
         f"activity {D(snap,'FAC_ACT_STABLE')}). D(v|q) then models the paired difference as a "
         "function of velocity and a motion-state descriptor q estimated from the source map. "
         f"Ridge D(v|q) reaches R2 {D(snap,'D2477_B1_R2')} on 24→77 and {D(snap,'D7724_B1_R2')} "
         f"on 77→24, against train-mean baselines {D(snap,'D2477_B0_R2')} and {D(snap,'D7724_B0_R2')}. "
         "Those R2 values describe predictability of a difference profile. They are not HAR scores."))
    add(("p",
         "A low-rank 2D model of the paired difference field is fit on TRAIN only. Rank is chosen "
         f"on VAL reconstruction ({D(snap,'K_SELECTED')} components). TRAIN explained variance is "
         f"{D(snap,'K20_TRAIN_VAR')}. The TEST oracle "
         f"(R2 = {D(snap,'ORACLE_R2_2D')}) projects the true unseen difference onto that TRAIN basis. "
         f"A q-predicted map (B2) reaches R2 {D(snap,'B2_R2_2477')} (24→77) and "
         f"{D(snap,'B2_R2_7724')} (77→24). Activity probe on source maps is {D(snap,'B2_ACT_SRC')}; "
         f"on B2-predicted maps it is {D(snap,'B2_ACT_PRED')}. The oracle is an upper bound. "
         "It is not a translator that can be run on a radar that has no paired target."))

    add(("h2", "2.11 Continuous motion inference"))
    add(("p",
         "Continuous results are labeled-interval continuous replay on mRI, not full-stream inference. "
         "Each mRI recording is one person and one radar CSV with multiple labeled intervals in "
         "absolute frame order. The local cache stores those labeled windows. Unlabeled gaps have "
         "ground-truth metadata and no radar inference. True full-stream inference, including "
         "background on unlabeled frames, would require rereading the original radar CSV. That CSV "
         "is not on this machine, and this paper does not pretend the cache contains it."))
    add(("p",
         "Windows use T=32 and stride=16 from the existing cache. The classifier is the frozen "
         "subject-independent PointNet-GRU; it is not retrained for continuous evaluation. "
         "Test recordings are subject3, subject10, and subject20—the same identities as the "
         f"subject-independent test split—totaling {D(snap,'CONT_N_WINDOWS')} windows. "
         "Validation subjects are used only to choose a smoothing threshold."))
    add(("p",
         "Three post-processing modes are compared. Raw window labels are used as emitted. "
         "Isolated-flip removal replaces a singleton that differs from both neighbors. "
         f"Confidence-aware smoothing uses conf_min={D(snap,'SMOOTH_CONF')} chosen on validation "
         f"(val macro-F1 {D(snap,'SMOOTH_VAL_F1')}). On test recordings, raw window accuracy / F1 "
         f"are {D(snap,'CONT_WIN_ACC_RAW')} / {D(snap,'CONT_WIN_F1_RAW')}; isolated-flip "
         f"{D(snap,'CONT_WIN_ACC_FLIP')} / {D(snap,'CONT_WIN_F1_FLIP')}; confidence-aware "
         f"{D(snap,'CONT_WIN_ACC')} / {D(snap,'CONT_WIN_F1')}. Frame scores under confidence-aware "
         f"smoothing are {D(snap,'CONT_FRAME_ACC')} / {D(snap,'CONT_FRAME_F1')}."))
    add(("p",
         f"Segment F1 is {D(snap,'CONT_SEG_F1')} after confidence-aware merge, versus "
         f"{D(snap,'CONT_SEG_F1_RAW')} raw and {D(snap,'CONT_SEG_F1_FLIP')} after isolated-flip "
         "removal. Median boundary error, averaged across the three recordings, is "
         f"{D(snap,'CONT_BND_MED')} frames ({D(snap,'CONT_BND_SEC')} s). Adjacent pose_* intervals "
         "are the typical merge error. Smoothing reduces fragmentation. It does not recover "
         "a usable boundary localizer. Velocity and motion-energy curves are stored as "
         "PHYSICS_DERIVED descriptors; they are not used as a stand-alone segmenter."))

    add(("h2", "2.12 Open-set unknown motion detection"))
    add(("p",
         f"Open-set evaluation is leave-one-action-out on mRI: {D(snap,'OPEN_N_FOLDS')} folds. "
         "Each fold trains on 12 known classes under the subject-independent split. The held-out "
         "action never enters training, validation, or threshold selection. Thresholds, where used, "
         "are set on validation known scores only. The unknown class is not given a training label "
         "and is not an extra closed-set category."))
    add(("p",
         "Three scores are computed on the same frozen embeddings. MSP uses the maximum known-class "
         "softmax. Energy uses the log-sum-exp of logits. Prototype distance uses the distance to "
         "the nearest known-class mean in the embedding. None of these methods invents a name for "
         "the unknown action. A detection is a reject / accept decision, not an open vocabulary."))
    add(("p",
         "Mean ± sample standard deviation over folds: "
         f"MSP AUROC {pm(snap,'MSP_AUROC','MSP_AUROC_STD')}, AUPR {pm(snap,'MSP_AUPR','MSP_AUPR_STD')}, "
         f"FPR95 {pm(snap,'MSP_FPR','MSP_FPR_STD')}; Energy AUROC {pm(snap,'ENERGY_AUROC','ENERGY_AUROC_STD')}, "
         f"AUPR {pm(snap,'ENERGY_AUPR','ENERGY_AUPR_STD')}, FPR95 {pm(snap,'ENERGY_FPR','ENERGY_FPR_STD')}; "
         f"Prototype AUROC {pm(snap,'PROTO_AUROC','PROTO_AUROC_STD')}, AUPR {pm(snap,'PROTO_AUPR','PROTO_AUPR_STD')}, "
         f"FPR95 {pm(snap,'PROTO_FPR','PROTO_FPR_STD')}. Prototype is the strongest of the three. "
         "Absolute performance remains limited. FPR95 near 0.8 means most known samples would be "
         "rejected if the threshold were set for 95% unknown recall. This is open-set rejection, "
         "not open-world recognition and not open-vocabulary recognition. Low closed-set confidence "
         "is not treated as unknown."))

    add(("h2", "2.13 Motion Intelligence"))
    add(("p",
         "Motion Intelligence is a structured record produced by the recognition stack, not by the "
         "language model. Required fields are activity, confidence, and uncertainty. When the "
         "representation supports them, the record also includes mean, median, and peak "
         "Doppler/velocity, positive and negative fractions, motion energy, periodicity, motion state, "
         "and temporal context. Each field carries a source tag (MODEL, PHYSICS_DERIVED, METADATA, "
         "UNKNOWN). Direction is UNKNOWN unless a coordinate convention has been validated. "
         "Environment is UNKNOWN on all current local sets. Subject is UNKNOWN on RadHAR and CI4R. "
         "The system does not invent a value to fill an empty slot."))
    add(("p",
         "MODEL fields come from the checkpoint: the argmax class, the softmax confidence, and a "
         "normalized entropy. PHYSICS_DERIVED fields are computed from the current window only. "
         "On point clouds they use the stored Doppler/velocity channel and, when present, intensity. "
         "On spectrograms they use column energy and are not converted to metres per second unless "
         "a validated Hertz-to-velocity map exists for that record. METADATA fields copy dataset, "
         "radar name, protocol, and subject when those keys exist in the sample index. UNKNOWN is "
         "a first-class value. The application displays it as UNKNOWN rather than as a blank that "
         "a later stage might fill."))
    add(("p",
         "This record is the only numeric object the language layer is allowed to see. If a field "
         "is UNKNOWN, a question about that field must be answered as missing evidence. If activity "
         "is boxing and the user asks about walk, the reply must use the timeline, not a prior "
         "over common radar actions. If open-set state is unknown motion, activity in the language "
         "input is not replaced by the nearest known class. Those rules are tested in the stored "
         "online tasks and in the application smoke template. They are interface contracts, not "
         "HAR metrics."))

    add(("h2", "2.14 Timeline construction"))
    add(("p",
         "Each window yields a prediction. Isolated flips are removed, adjacent same-activity events "
         "are merged, and confidence-aware smoothing is applied with the validation-selected threshold. "
         "An event stores start, end, activity, confidence, uncertainty, and available motion descriptors. "
         "Continuous evaluation and the language layer both read this timeline. The timeline is not "
         "a segmentation network and is not evaluated as full-stream inference."))

    add(("h2", "2.15 LLM-based semantic motion understanding"))
    add(("p",
         "The language model never receives raw radar data. It does not read raw IQ, spectrograms, "
         "point clouds, or network feature tensors. It reads Motion Intelligence JSON and Timeline JSON. "
         "That boundary is part of the system design, not an implementation detail."))
    add(("h2", "2.15.1 Language-layer input schema"))
    add(("p",
         "The schema contains activity, confidence, uncertainty, motion descriptors that the current "
         "representation supports, temporal events, unknown-motion state when an open-set score is "
         "available, and model-supported metadata. Missing items are written UNKNOWN. The language "
         "layer is instructed not to replace UNKNOWN with a guessed number or class. "
         "A typical RadHAR point-cloud record can include PHYSICS_DERIVED velocity statistics "
         "because the local files store a Doppler/velocity channel with a stated unit. A typical "
         "CI4R spectrogram record can include column-energy statistics but not a validated m/s "
         "velocity. An mRI record can include subject identity from metadata. The language prompt "
         "is required to quote source tags rather than to flatten every field into an unmarked number."))
    add(("p",
         "The timeline fragment sent to the language layer is a list of events, not the full window "
         "tensor. Each event has start time, end time, activity, confidence, and a low-confidence "
         "flag. Smoothing parameters are attached as metadata so that a summary can mention that "
         "adjacent windows were merged. The language layer is not given class-probability vectors "
         "unless a task explicitly needs them; the default path uses the already-decided activity "
         "and uncertainty scalar."))
    add(("h2", "2.15.2 Semantic tasks"))
    add(("p",
         "Supported tasks are current-motion explanation, continuous behavior summary, uncertainty "
         "explanation, question answering, unknown-motion explanation, and application interaction. "
         "The language layer does not perform radar recognition and is not scored with HAR accuracy."))
    add(("h2", "2.15.3 Online language-model service"))
    add(("p",
         "When an online language-model service is available, structured JSON is sent in that mode. "
         "If the service, key, or network is unavailable, a deterministic template fallback is used. "
         "Credentials are not recorded in this manuscript."))
    add(("h2", "2.15.4 Hallucination control"))
    add(("p",
         "The language layer cannot invent unsupported radar observations. If the timeline contains "
         "no bend-and-pick event and a user asks whether that action occurred, the reply must state "
         "that the current timeline does not contain evidence for it. If the open-set state is "
         "unknown motion, the nearest known class must not be emitted as the activity. The text "
         "must say that the observation cannot be reliably assigned to a known class. Labels such "
         "as pose_1 and pose_2 have no validated everyday-name mapping in the local taxonomy and "
         "must not be paraphrased into an invented action. Low confidence requires conservative wording."))

    add(("h2", "2.16 RadarMind application"))
    add(("p",
         "The research interface is a Streamlit application. Pages cover replay, generalization, "
         "ablation, cross-radar, shared motion, continuous replay, open-set, radar invariance, "
         "the pipeline, research status, and Motion Intelligence / language interaction. "
         "The interface loads frozen checkpoints and result files. Replay calls the registered "
         "model on the current window; it does not substitute a mock label. "
         f"Application smoke tests record {D(snap,'SMOKE_V01')} / {D(snap,'SMOKE_V01')} checks in the "
         f"first integration pack and {D(snap,'SMOKE_V02')} / {D(snap,'SMOKE_V02')} checks after "
         f"continuous and open-set pages were attached. Continuous smoke records "
         f"{D(snap,'CONT_SMOKE')} checks, including unlabeled-interval handling."))
    add(("p",
         "Replay pages load a stored test window, run the registered checkpoint, and write "
         "Motion Intelligence for that window. Generalization and ablation pages read the frozen "
         "CSV/JSON tables rather than retraining. Cross-radar and shared-motion pages display "
         "the paired-matrix and probe files. Continuous pages seek by absolute frame number inside "
         "a labeled-interval recording. Open-set pages show fold-wise scores and do not relabel "
         "an unknown sample as its nearest known class. Research status states what is implemented "
         "and what is not. The application is a research interface, not a field deployment. "
         "Live radar capture, new subject enrollment, and environment tags are out of scope for "
         "the current pages. If those inputs arrive later, they should appear first as new "
         "protocol rows in the snapshot, not as rewritten captions on old figures."))
    add(("p",
         "Health-check and page-registration smoke items confirm that the continuous, open-set, "
         "cross-radar, and shared-motion pages resolve their frozen files at startup. A missing "
         "result file fails the smoke rather than rendering an empty chart with a guessed axis. "
         "That fail-closed behavior is the same rule used by the manuscript generator."))
    add(("h2", "2.17 Evaluation metrics and reporting rules"))
    add(("p",
         "Accuracy and macro-F1 are reported on the 0–1 scale to three decimals unless a count is "
         "an integer. Open-set AUROC, AUPR, and FPR95 are fold means with sample standard deviations. "
         "Continuous segment F1 and boundary error are unweighted means over the three test "
         "recordings so that one long recording does not dominate. Radar-ID is a logistic probe on "
         "a representation, not a HAR system. Activity probes on maps or embeddings are linear "
         "diagnostics. R2 values on D(v|q) and on 2D difference fields are reconstruction scores. "
         "Language-layer success is a task pass on stored JSON, not an automatic metric against "
         "a reference caption."))
    add(("p",
         "Every numeric claim in the generated tables and figures is resolved from "
         "metric_snapshot_full.json. That snapshot records the source path, field, and read time. "
         "If a source file or field is missing, generation fails instead of substituting a "
         "remembered number. This manuscript does not modify Phase 1–10 result files."))

    add(("h1", "3. Results"))
    add(("h2", "3.1 Baseline recognition across radar representations"))
    add(("p",
         f"Table 2 and Fig. 3 summarize closed-set recognition under the protocol that belongs to "
         f"each dataset. mRI subject-independent XYZ PointNet-GRU: accuracy {D(snap,'MRI_SI_ACC')}, "
         f"macro-F1 {D(snap,'MRI_SI_F1')} on {D(snap,'MRI_SI_N_TEST')} test windows and "
         f"{D(snap,'MRI_N_ACTIONS')} classes. RadHAR official XYZV: accuracy {D(snap,'RADHAR_XYZV_ACC')}, "
         f"macro-F1 {D(snap,'RADHAR_XYZV_F1')}. CI4R official plotted 7-class within-radar: "
         f"{D(snap,'R0_10_WITHIN')} / {D(snap,'R0_24_WITHIN')} / {D(snap,'R0_77_WITHIN')}. "
         "High RadHAR official-split accuracy and high CI4R within-radar accuracy are not evidence "
         "that the same models travel across people or radars."))
    add(("fig", ("fig03_baseline_temporal.png",
                 "Fig. 3. (a) Closed-set accuracy under each dataset’s own protocol. Groups are not a shared leaderboard. "
                 "(b) Temporal-context ablation on mRI (T=1 vs T=32) and RadHAR (T=1 vs T=60). "
                 "The comparison illustrates temporal context, not architecture superiority.")))

    add(("p",
         "CI4R 77 GHz 11-class random-record accuracy "
         f"{D(snap,'CI4R77_RAND_ACC')} is not interchangeable with the 7-class paired within-77 "
         f"cell {D(snap,'R0_77_WITHIN')}. The class sets, splits, and pairing constraints differ. "
         "Table 2 keeps them on separate rows. Voxel CNN-GRU on RadHAR "
         f"({D(snap,'RADHAR_VOXEL_ACC')} / {D(snap,'RADHAR_VOXEL_F1')}) is weaker than PointNet-GRU "
         "XYZV under the same official listing; the paper uses that comparison as a sanity check, "
         "not as an architecture contest."))
    add(("h2", "3.2 Subject-independent recognition reveals generalization loss"))
    add(("p",
         f"On mRI, moving from random-segment ({D(snap,'MRI_RAND_ACC')} / {D(snap,'MRI_RAND_F1')}) "
         f"to subject-independent ({D(snap,'MRI_SI_ACC')} / {D(snap,'MRI_SI_F1')}) lowers both accuracy "
         "and macro-F1. Per-class records in the frozen metrics show T pose at F1 0 on the "
         "subject-independent test set; several pose_* classes remain mid-range. The protocol uses "
         "official subject IDs. CI4R prefix-group-disjoint is reported in Table 2 only as an "
         "acquisition-group gap, not as a subject gap."))

    add(("h2", "3.3 Temporal context substantially improves point-cloud HAR"))
    add(("p",
         f"mRI T=1 accuracy {D(snap,'MRI_T1_ACC')} versus T=32 {D(snap,'MRI_SI_ACC')} shows that "
         "single-frame point geometry is a weak activity cue on this 13-class set. "
         "Several mRI classes are static or slowly changing poses; even those classes do not "
         "reach usable accuracy from one frame, which indicates that the sampled 64-point cloud "
         "is a thin observation rather than a complete skeleton. "
         f"RadHAR T=1 {D(snap,'RADHAR_T1_ACC')} versus T=60 XYZV {D(snap,'RADHAR_XYZV_ACC')} shows "
         "the same direction on a 5-class official split. Feature ablations do not reverse that "
         "ordering. Motion understanding in this paper is therefore treated as a temporal problem, "
         "not as static shape classification."))

    add(("h2", "3.4 High within-radar accuracy masks cross-radar failure"))
    add(("p",
         f"Fig. 4 is the central negative result for spectrogram HAR. Diagonals remain high "
         f"({D(snap,'R0_10_WITHIN')}, {D(snap,'R0_24_WITHIN')}, {D(snap,'R0_77_WITHIN')}). "
         f"The 24→77 cell is {D(snap,'R0_24_TO_77')}. Several other off-diagonal cells stay below 0.40. "
         f"A 3-radar Radar-ID probe on official plots is {D(snap,'RID_R0_3CLASS')}: the representation "
         "still names the sensor. Subsequent sections ask whether that identity is a plotting artifact "
         "or a measurement effect."))
    add(("fig", ("fig04_cross_radar_matrix.png",
                 "Fig. 4. Official-representation 7-class HAR on CI4R paired acquisitions. "
                 "Rows are train radars and columns are test radars. The split is by acquisition timestamp. "
                 "Filename prefix is not a subject. Diagonal cells are within-radar; off-diagonal cells are zero-shot cross-radar.")))
    add(("p",
         "Table 3 lists the same nine cells for citation. Fig. 4 is the visual collapse; "
         "the table is the number that should be quoted."))

    add(("p",
         "The same split is used when CORAL is applied. Unlabeled 10 GHz training windows are "
         "visible to the adapter and invisible to a LORO run. Reporting CORAL next to LORO without "
         "that sentence would mix UDA with domain generalization. The negative CORAL result is "
         "kept because a positive UDA number on this subset would still not have been unseen-radar "
         "generalization."))
    add(("h2", "3.5 Paired learning improves activity correspondence but retains radar identity"))
    add(("p",
         f"M1 raises known-radar accuracy from {D(snap,'M0_ACC')} to {D(snap,'M1_ACC')} and "
         f"same-action retrieval from {D(snap,'M0_RET')} to {D(snap,'M1_RET')}. M2 does not improve "
         f"further on accuracy ({D(snap,'M2_ACC')}) or retrieval ({D(snap,'M2_RET')}). "
         f"Radar-ID remains {D(snap,'M1_RID')} for M1. Leave-one-radar-out numbers in the Phase 3 "
         "table stay far below the known-radar diagonal. The correct reading is that contrastive "
         "pairing helps when all three radars are seen in training. It is not unseen-radar domain "
         "generalization. Fig. 5 shows the known-radar accuracy/retrieval gain beside Radar-ID, "
         "which remains 1.000. Table 4 adds M3, A0–A2, CORAL, and LORO under an explicit Protocol column."))
    add(("fig", ("fig05_shared_representation.png",
                 "Fig. 5. Known-radar shared representation. (a) Activity accuracy and same-action retrieval for M0–M2. "
                 "(b) 3-radar Radar-ID on the same embeddings. InfoNCE improves semantic correspondence among radars seen in training; "
                 "it does not remove sensor identity. This is not a leave-one-radar-out ranking.")))

    add(("h2", "3.6 Physical reconstruction and calibration do not remove radar signatures"))
    add(("p",
         f"R2-D removes plotting as the sole explanation: 3-radar Radar-ID is still {D(snap,'RID_R2D_3CLASS')}. "
         f"Velocity alignment on 24/77 moves 2-radar Radar-ID from {D(snap,'RID_R2D_2477')} to "
         f"{D(snap,'RID_R2V_2477')} and changes pair cosine separation from {D(snap,'SEP_R2D')} to "
         f"{D(snap,'SEP_R2V')}. That is a real physical correction and a real but incomplete reduction "
         f"of a 2-class probe. It is not 3-radar invariance. First-order calibration, unified STFT/MTI, "
         f"and D(v|q) (24→77 R2 = {D(snap,'D2477_B1_R2')}; 77→24 R2 = {D(snap,'D7724_B1_R2')}) describe "
         f"structured measurement differences. The 2D oracle R2 {D(snap,'ORACLE_R2_2D')} is an upper bound, "
         f"not a deployed map. Predicted 2D transfer R2 remains {D(snap,'B2_R2_2477')}. "
         "Fig. 7 reports those R² values and marks the oracle as an upper bound. "
         "Table 5 collects Radar-ID, activity probe, and R² in one place and labels the metric type. "
         "None of these operators is reported as a working unseen-radar translator. "
         f"Applying D(v|q) as a calibration map changes 24/77 Radar-ID from {D(snap,'CAL_RID_BEFORE')} "
         f"to {D(snap,'CAL_RID_AFTER')} and activity probe from {D(snap,'CAL_ACT_BEFORE')} to "
         f"{D(snap,'CAL_ACT_AFTER')}. Pair separation moves from {D(snap,'CAL_SEP_BEFORE')} to "
         f"{D(snap,'CAL_SEP_AFTER')}. The calibration does not pass a gate that would justify "
         "calling the residual a solved measurement bias."))
    add(("fig", ("fig06_radar_identity_ladder.png",
                 "Fig. 6. Radar-ID along the physical processing chain. (a) 3-radar probe. (b) 24/77 probe. "
                 "Hollow markers are activity probes on the same representation family. "
                 "The two panels are different probes and are not a single continuous trajectory.")))
    add(("fig", ("fig07_measurement_difference.png",
                 "Fig. 7. Motion-conditioned and low-rank 2D difference models. "
                 "Bars are R² of the difference field, not HAR accuracy. "
                 "TEST oracle is the true TEST Δ projected onto the TRAIN basis; it is an upper bound, not a deployable translator. "
                 "Measured 2D difference maps are not stored in the frozen Phase 10 files, so field panels are omitted.")))

    add(("h2", "3.7 Continuous labeled-interval inference remains substantially harder than isolated classification"))
    add(("p",
         f"Isolated mRI subject-independent accuracy is {D(snap,'MRI_SI_ACC')}. The same checkpoint "
         f"replayed as labeled-interval windows yields frame accuracy {D(snap,'CONT_FRAME_ACC')} and "
         f"frame macro-F1 {D(snap,'CONT_FRAME_F1')}. Window scores are {D(snap,'CONT_WIN_ACC')} / "
         f"{D(snap,'CONT_WIN_F1')}. Segment F1 {D(snap,'CONT_SEG_F1')} and median boundary error "
         f"{D(snap,'CONT_BND_MED')} frames ({D(snap,'CONT_BND_SEC')} s) show that merging windows into "
         f"events is the weak step. Raw segment F1 {D(snap,'CONT_SEG_F1_RAW')} and isolated-flip "
         f"{D(snap,'CONT_SEG_F1_FLIP')} confirm that smoothing helps fragmentation and does not fix "
         "boundaries. These numbers are Mode A: labeled intervals only."))
    add(("fig", ("fig08_continuous_inference.png",
                 "Fig. 8. Labeled-interval continuous replay on mRI recording subject20_0513. "
                 "Top: ground-truth labeled intervals, raw window predictions, and confidence-aware events. "
                 "Hatched intervals are unlabeled gaps and are not scored. "
                 "Bottom: segment F1 and median boundary error on separate scales. "
                 "This is labeled-interval replay, not full-stream inference.")))

    add(("h2", "3.8 Open-set detection exposes limitations of closed-set confidence"))
    add(("p",
         f"MSP AUROC {pm(snap,'MSP_AUROC','MSP_AUROC_STD')} sits near chance. Energy improves to "
         f"{pm(snap,'ENERGY_AUROC','ENERGY_AUROC_STD')}. Prototype distance reaches "
         f"{pm(snap,'PROTO_AUROC','PROTO_AUROC_STD')} and is the selected score for application display. "
         f"FPR95 remains high for all three ({pm(snap,'MSP_FPR','MSP_FPR_STD')}, "
         f"{pm(snap,'ENERGY_FPR','ENERGY_FPR_STD')}, {pm(snap,'PROTO_FPR','PROTO_FPR_STD')}). "
         "Closed-set confidence is a poor unknown-motion detector. Prototype is better and still weak. "
         "The protocol is open-set rejection under leave-one-action-out. It is not open-world recognition. "
         "Table 6 places the continuous scores and the LOAO scores in one table with two blocks."))
    add(("fig", ("fig09_open_set.png",
                 "Fig. 9. Thirteen-fold leave-one-action-out open-set rejection on mRI. "
                 "(a) Per-fold AUROC for MSP, Energy, and Prototype. The dashed line is chance (0.500). "
                 "(b) Mean FPR95. Prototype is better than MSP and remains weak in absolute terms. "
                 "The protocol is LOAO rejection, not open-world recognition.")))

    add(("h2", "3.9 Structured Motion Intelligence supports semantic interpretation"))
    add(("p",
         f"A RadHAR official-split boxing window (sample in the frozen online-language test) is predicted "
         f"as boxing with confidence {D(snap,'BOX_CONF')} and uncertainty {D(snap,'BOX_UNC')}. "
         f"Physics-derived descriptors on that window include mean velocity {D(snap,'BOX_MEAN_V')} m/s, "
         f"peak |v| {D(snap,'BOX_PEAK_V')} m/s, and motion energy {D(snap,'BOX_ENERGY')}. "
         f"Direction, periodicity, subject, and environment are UNKNOWN. The associated timeline merges "
         f"{D(snap,'BOX_NWIN')} windows into one boxing event with event confidence {D(snap,'BOX_EVT_CONF')}. "
         "Those values are copied from the stored JSON. No extra numeric claim is added."))

    add(("h2", "3.10 The language layer converts structured radar inference into readable interpretation"))
    add(("p",
         f"Online language tests on that structured record cover {D(snap,'LLM_N')} tasks, of which "
         f"{D(snap,'LLM_N_OK')} succeeded: current-motion explanation, continuous summary, uncertainty "
         "listing, and a grounded question. A dedicated six-item hallucination suite is not present "
         "in the frozen result files; this manuscript therefore reports the four stored tasks rather "
         "than an unaudited count. Evaluation is groundedness, consistency with UNKNOWN fields, "
         "uncertainty wording, and refusal to invent events—not BLEU and not HAR accuracy."))
    add(("p",
         "Case A. The model predicts boxing. The language output states boxing, reports the high "
         "confidence, restates the available velocity statistics, and says that direction, periodicity, "
         "subject, and environment are absent. Case B. The user asks whether the timeline contains "
         "bend-and-pick. The stored reply states that the timeline contains boxing only and that there "
         "is no bend-and-pick event. Case C. The application smoke record includes an unknown-motion "
         "template: the text says the motion cannot be reliably assigned to a known class and does "
         "not force the nearest known label. Those cases illustrate the interface contract. They are "
         "not a claim that the language layer repairs recognition errors."))
    add(("p",
         "The current-motion reply restates boxing, quotes confidence, and lists missing fields "
         "instead of inferring a punch rate or a facing direction. The summary reply names a "
         "single merged event from 0 s to 48 s and does not add a second activity. The uncertainty "
         "reply enumerates direction, periodicity, subject, and environment as UNKNOWN and notes "
         "that activity entropy is low. The QA reply answers the bend-and-pick question from the "
         "event list and then, separately, reports the velocity numbers that do exist. That "
         "separation is the point of the structured layer: a missing event and a present velocity "
         "are not merged into a fictional bend-and-pick speed."))
    add(("p",
         "Offline fallback is a template that fills the same fields without calling the online "
         "service. Smoke records include both the online path and the fallback path. The paper "
         "does not report a preference study, a rater agreement, or an automatic caption metric. "
         "If those evaluations are added later, they should be groundedness checks against the "
         "JSON, not similarity to a free-text gold caption that itself might hallucinate."))
    add(("fig", ("fig10_motion_intelligence_llm.png",
                 "Fig. 10. Language interaction on a stored RadHAR boxing record. "
                 "Left: model prediction and timeline event. Center: Motion Intelligence fields copied from llm_online_test.json. "
                 "Right: stored QA. Raw radar is isolated from the language layer; only structured JSON is passed.")))
    add(("p",
         "Table 7 records the stored fields, the UNKNOWN handling, the four online tasks, "
         "the raw-radar restriction, and the grounded QA rule. It is not an application status list."))

    add(("h2", "3.11 RadarMind provides an end-to-end radar-to-semantics workflow"))
    add(("p",
         "The implemented workflow is replay → checkpoint prediction → Motion Intelligence → timeline → "
         "open-set / uncertainty fields → language layer → UI. Online language mode and template "
         f"fallback are both exercised in smoke records. Application smoke is "
         f"{D(snap,'SMOKE_V02')} / {D(snap,'SMOKE_V02')}; the earlier pack is "
         f"{D(snap,'SMOKE_V01')} / {D(snap,'SMOKE_V01')}; continuous smoke is "
         f"{D(snap,'CONT_SMOKE')} checks. Completing the workflow does not imply that unseen-radar "
         "generalization, full-stream continuous inference, or strong unknown detection have been solved."))
    add(("fig", ("fig11_capability_summary.png",
                 "Fig. 11. Capability and limitation summary. Symbols mark supported, partially supported, and unresolved items "
                 "against the frozen evidence. The working path is radar → recognition → Motion Intelligence → timeline → language. "
                 "Unseen-radar generalization remains unresolved.")))

    add(("h1", "4. Discussion"))
    add(("h2", "4.1 What the pipeline does and does not settle"))
    add(("p",
         "RadarMind already connects spectrogram and point-cloud HAR, subject-independent evaluation, "
         "a paired cross-radar matrix, shared-representation probes, physical reconstruction, "
         "labeled-interval continuous replay, open-set scores, structured descriptors, a language "
         "interface, and an interactive application. That is a system result. The same records show "
         "that building the system does not close the generalization problem. Cross-radar sensing "
         "remains the main bottleneck: within-radar cells stay high, zero-shot cells collapse, and "
         "Radar-ID stays near ceiling after plotting is removed."))
    add(("h2", "4.2 Why the language layer is not a recognition fix"))
    add(("p",
         "The language layer does not solve cross-radar collapse, weak unknown detection, or poor "
         "boundary localization. It converts model outputs into readable, context-aware interaction. "
         "If the timeline is wrong, a fluent explanation of that timeline is still wrong. The "
         "scientific role of the language layer is therefore limited to groundedness and interface "
         "behavior. Claiming that it improves HAR accuracy would be unsupported."))
    add(("h2", "4.3 Why a structured intermediate layer is used"))
    add(("p",
         "Sending a spectrogram or a point cloud directly to a language model would mix sensing, "
         "recognition, and wording in one untraceable step. Motion Intelligence keeps source tags, "
         "UNKNOWN slots, and numeric units visible. Uncertainty can be quoted. A question about an "
         "absent event can be answered from the timeline rather than from model prior. Modality "
         "separation also avoids placing raw radar in an external service. Those are engineering "
         "and audit reasons, not a claim about language-model accuracy on radar pixels."))
    add(("h2", "4.4 Measurement effects versus missing spatial metadata"))
    add(("p",
         "Velocity alignment and pair-difference models show that 24/77 measurements are not the same "
         "function of radial velocity even after a common axis is chosen. Action-conditioned residuals "
         "remain. Distance, aspect, orientation, and range are not recorded on the paired CI4R subset "
         "used here, so geometry cannot be isolated as a proven cause. The correct statement is that "
         "sensor-dependent measurement effects persist after plotting, resampling, and first-order "
         "calibration, and that spatial metadata needed to test a geometry hypothesis are missing."))
    add(("h2", "4.5 Continuous and open-set limits are different from cross-radar limits"))
    add(("p",
         "Labeled-interval replay fails mainly at event merge and boundary time, not at the same "
         "mechanism as 24→77 collapse. The checkpoint is already a subject-independent classifier; "
         "the extra error is temporal. Open-set failure is different again: known-class confidence "
         "does not separate unseen pose_* actions that sit close in embedding space. Treating these "
         "three failures as one 'generalization' number would hide the fact that each needs a "
         "different next measurement. Cross-radar needs paired geometry and raw IQ. Continuous "
         "needs unlabeled-interval radar. Open-set needs either better reject scores or a broader, "
         "cleaner activity inventory than pose_1–pose_10."))
    add(("h2", "4.6 What a next paper can claim only after new data"))
    add(("p",
         "A later study may test whether residual 24/77 differences shrink after distance, aspect, "
         "and orientation are recorded and balanced. That test does not exist in the present files. "
         "A later study may train a background detector on unlabeled mRI frames if the raw CSV is "
         "restored. That experiment is not reported here. A later study may replace leave-one-action-out "
         "with a larger unknown inventory. Until those data are collected, phrases such as "
         "sensor invariance as a finished result, open-world recognition as a finished result, "
         "or open-vocabulary HAR are not available to this project."))
    add(("h2", "4.7 Protocol discipline"))
    add(("p",
         "Several numbers in this project are easy to misread. CI4R prefix-group-disjoint is not "
         "cross-subject. RadHAR official split is not subject-independent. Continuous scores are "
         "labeled-interval replay. Open-set AUROC is not open-world accuracy. CORAL is UDA. "
         "Leave-one-radar-out is domain generalization and remains weak. Known-radar encoder "
         "accuracy (M1, A0–A2) is not zero-shot transfer. 3-radar Radar-ID and 24/77 Radar-ID are "
         "different probes. The TEST 2D oracle is not a deployed translator. The manuscript keeps "
         "those qualifiers next to the numbers."))

    add(("h2", "4.8 Failure ladder, in scientific rather than phase order"))
    add(("p",
         "The cross-radar evidence can be read as a ladder. Official plots already separate radars "
         f"(Radar-ID {D(snap,'RID_R0_3CLASS')}) while within-radar HAR stays high. Numerical Doppler "
         f"keeps 3-radar Radar-ID at {D(snap,'RID_R2D_3CLASS')}. Velocity alignment lowers a 24/77 "
         f"probe from {D(snap,'RID_R2D_2477')} to {D(snap,'RID_R2V_2477')} and does not create a "
         "working 24↔77 classifier. Processing harmonization and first-order calibration do not "
         "open a gate to a simple correction. Shared encoders and adapters raise known-radar "
         "correspondence and leave Radar-ID at ceiling. Motion-conditioned and low-rank 2D models "
         "show that part of the paired difference is structured and still not available as a "
         "zero-shot map. The ladder is the reason this paper keeps physical diagnosis in the middle "
         "rather than as an appendix remark."))
    add(("p",
         "System breadth sits beside that ladder rather than replacing it. Subject-independent mRI, "
         "labeled-interval continuous replay, leave-one-action-out detection, Motion Intelligence, "
         "and the language interface answer a different question: what can be said about motion "
         "once a model has produced a window label. Those modules would still be needed if "
         "cross-radar transfer were stronger. They do not substitute for the missing paired "
         "geometry and raw-IQ protocol."))
    add(("h1", "5. Current limitations and next acquisition protocol"))
    add(("p",
         "The limitations are specific. They are not a generic call for more data. Each missing "
         "item below blocks a claim that a reader might otherwise infer from the existence of "
         "the application or from a high official-split number."))
    add(("p",
         "RadarMind has not produced a sensor-invariant representation, robust unseen-radar domain "
         "generalization, full-stream continuous understanding on unlabeled intervals, a strong "
         "unknown-motion detector, cross-environment generalization, or open-world / open-vocabulary "
         "motion understanding. The local mRI cache cannot support inference on unlabeled gaps. "
         "No validated environment label exists for a cross-environment split. pose_* names have no "
         "everyday semantic map. Acquisition Toolkit files in the repository are a recording protocol "
         "and demonstration layout, not experimental results in this paper. "
         "pose_1 through pose_10 remain index names from the mRI taxonomy. The language layer "
         "is forbidden to translate them into everyday verbs. T pose on the subject-independent "
         f"test set has F1 {D(snap,'MRI_TPOSE_F1')} in the frozen per-class table; that class-level failure is part of "
         "the subject-independent result, not a reason to drop the class from the protocol."))
    add(("p",
         "Open-set folds inherit that inventory. Holding out pose_4 is not the same scientific "
         "object as holding out a rare real-world action that never appears in any training "
         "taxonomy. The LOAO mean is therefore a lower bound on difficulty only inside this "
         "closed list. It is not a field unknown-motion rate. Continuous replay inherits the "
         "same list and the same subject-independent checkpoint; improving segment F1 would "
         "still leave unlabeled gaps untouched."))
    add(("p",
         "Future paired 24/77 collection needs the same action and the same geometry across radars, "
         "and the same radar and action across controlled geometry changes. Required fields include "
         "distance, aspect, orientation, range, raw IQ, and time-aligned video. Only with those "
         "observables can sensor, geometry, and person be separated. That protocol is specified so "
         "that the next dataset can test the hypotheses this manuscript cannot close."))

    add(("h1", "6. Conclusions"))
    add(("p",
         "RadarMind connects radar sensing, activity recognition, generalization analysis, continuous "
         "inference, open-set detection, structured motion descriptors, and language-based semantic "
         "interaction within a single research framework. The implemented pipeline runs from a stored "
         "radar window to a model prediction, a Motion Intelligence record, a timeline, and a language "
         "response that is constrained to that record."))
    add(("p",
         "The results also show that building an end-to-end system does not imply that the underlying "
         "generalization problem has been solved. Cross-radar sensing remains the main bottleneck. "
         "Future work requires controlled 24/77 acquisition with distance, aspect, orientation, range, "
         "raw IQ, and video before sensor-invariant and open-world motion understanding can be "
         "meaningfully pursued. Until those data exist, the honest summary is a working radar-to-semantics "
         "stack and a documented failure of unseen-radar transfer."))

    add(("h2", "6.2 Reporting discipline"))
    add(("p",
         "A within-radar number without a cross-radar matrix can be read as if the representation "
         "were already sensor-agnostic. A known-radar joint-encoder number can be read as if the "
         "third radar were unseen. An official train/test number on a corpus without subject IDs "
         "can be read as subject-independent. A sliding-window accuracy on labeled segments can "
         "be read as full-stream understanding. An MSP AUROC can be read as open-world recognition. "
         "This manuscript keeps those readings unavailable by placing the protocol in the same "
         "sentence as the number. The cost is a longer paper. The alternative is a short paper "
         "that cannot be cited without re-checking every table."))
    add(("p",
         "The RadarMind application exists for the same reason. A reader can open the frozen "
         "matrix, the continuous timeline, and the language JSON on one machine. That does not "
         "make the application a product. It makes the negative results inspectable. If a later "
         "revision changes a checkpoint, the smoke counts in the frozen reports will no longer match this "
         "draft and the draft should be regenerated from the snapshot rather than edited by hand."))
    add(("h1", "6.1 Summary of two-layer conclusions"))
    add(("p",
         "Layer one—what is implemented. RadarMind runs multi-representation HAR, subject-independent "
         "evaluation on mRI, a paired cross-radar benchmark, known-radar shared representation, "
         "a physical reconstruction and measurement-response audit, labeled-interval continuous "
         "replay, leave-one-action-out open-set scoring, structured Motion Intelligence, a language "
         "layer that reads only that structure, and an interactive research application whose "
         f"smoke record is {D(snap,'SMOKE_V02')} / {D(snap,'SMOKE_V02')}."))
    add(("p",
         "Layer two—what is not implemented. The same files show no sensor-invariant token, no "
         "robust unseen-radar domain generalization, no full-stream continuous inference on "
         "unlabeled intervals, no strong unknown-motion detector, no cross-environment split, "
         "and no open-world or open-vocabulary motion understanding. Those absences are results. "
         "They are the reason the recommended title uses Toward."))
    add(("p",
         "The selected title is therefore "
         "RadarMind: Toward Generalizable Human Motion Understanding Across Radar Domains. "
         "Longer candidates that list continuous understanding, open-set detection, and language-based "
         "interpretation remain accurate as subtitles. They are not used as the running title because "
         "a title that names every module can be misread as a claim that every module is solved. "
         "Toward is the only allowed completion word for the unfinished items."))
    add(("p",
         "If this draft is later shortened for a specific venue, the sections that must not be "
         "removed are the cross-radar matrix, the Radar-ID protocol split, the labeled-interval "
         "qualifier, the open-set versus open-world distinction, and the statement that the "
         "language layer does not read raw radar and does not improve recognition accuracy. "
         "Removing those sentences would return the paper to the scope error of the previous "
         "cross-radar-only draft."))
    add(("h1", "Data availability"))
    add(("p",
         "CI4R, mRI, and RadHAR are public datasets used under their original terms. The paired CI4R "
         "subset, subject-independent mRI split, and RadHAR official listing used here are recorded "
         "in the frozen split files of this repository. Acquisition Toolkit demonstration files are "
         "not part of the experimental tables."))
    add(("h1", "Code availability"))
    add(("p",
         "Training scripts, evaluation scripts, adapters, Motion Intelligence builders, the language "
         "interface, and the Streamlit application are in the RadarMind repository. This manuscript "
         "was generated from those frozen result files without retraining."))
    add(("h1", "Author contributions"))
    add(("p",
         "Working draft. Experimental records are the frozen Phase 1–10 and application-integration "
         "outputs. Author list to be completed for submission."))
    add(("h1", "Conflict of interest"))
    add(("p", "The authors declare no competing interests for this working draft."))
    add(("h1", "Acknowledgements"))
    add(("p",
         "This working draft compiles frozen experimental records. Public dataset authors of CI4R, "
         "mRI, and RadHAR are cited as [CITATION NEEDED] until bibliographic records are inserted. "
         "No new human-subject collection is reported here. "
         "Online language-model calls, when they occurred, used a hosted service under the "
         "project's existing account configuration. This manuscript does not reproduce keys, "
         "endpoints, or request logs."))
    add(("h1", "References"))
    add(("ref",
         "[1] CI4R multi-frequency radar HAR dataset. [CITATION NEEDED]\n"
         "[2] mRI mmWave point-cloud activity dataset. [CITATION NEEDED]\n"
         "[3] RadHAR point-cloud HAR dataset. [CITATION NEEDED]\n"
         "[4] CORAL: correlation alignment for unsupervised domain adaptation. [CITATION NEEDED]\n"
         "[5] InfoNCE / contrastive representation learning. [CITATION NEEDED]\n"
         "[6] Open-set recognition and energy-based out-of-distribution scoring. [CITATION NEEDED]\n"
         "[7] Maximum softmax probability as an out-of-distribution baseline. [CITATION NEEDED]\n"
         "[8] Micro-Doppler radar signatures of human motion. [CITATION NEEDED]\n"
         "[9] PointNet and temporal point-cloud HAR. [CITATION NEEDED]\n"
         "[10] Residual networks for image recognition (ResNet). [CITATION NEEDED]"))
    return B


def table_specs(snap):
    def d(k):
        return D(snap, k)

    t1 = {
        "title": "Table 1. Datasets, radar representations, and experimental roles in RadarMind.",
        "header": ["Dataset", "Representation", "Primary role", "Protocol boundary"],
        "numeric_cols": [],
        "rows": [
            [
                f"CI4R paired 10/24/77 ({d('N_PAIRED_ACQ')} agreed triples)",
                "spectrogram / reconstructable raw (24/77)",
                "cross-radar 7-class HAR",
                "acquisition-timestamp split; filename prefix is not a subject",
            ],
            [
                "CI4R 77 GHz 11-class",
                "spectrogram",
                "random-record vs prefix-group-disjoint",
                "prefix-group-disjoint is not subject-independent",
            ],
            [
                f"mRI ({d('MRI_N_SUBJECTS')} subjects, {d('MRI_N_ACTIONS')} actions)",
                "point cloud",
                "true subject-independent HAR; labeled-interval continuous; LOAO open-set",
                "not full-stream continuous inference",
            ],
            [
                f"RadHAR ({d('RADHAR_N_ACTIONS')} actions)",
                "point cloud",
                "official train/test; temporal and feature ablation",
                "official split is not subject-independent",
            ],
        ],
    }
    t2 = {
        "title": (
            "Table 2. Closed-set HAR and temporal or feature ablations under each dataset’s own protocol. "
            "Rows are not a shared leaderboard."
        ),
        "header": ["Dataset", "Setting", "Model", "Protocol", "Acc", "Macro-F1"],
        "numeric_cols": [4, 5],
        "group_starts": [0, 5, 10],
        "rows": [
            ["mRI", "SI XYZ T=32", "PointNet-GRU", "subject-independent", d("MRI_SI_ACC"), d("MRI_SI_F1")],
            ["", "random XYZVI T=32", "PointNet-GRU", "random-segment; not SI", d("MRI_RAND_ACC"), d("MRI_RAND_F1")],
            ["", "SI T=1 XYZVI", "PointNet-pool", "subject-independent; single frame", d("MRI_T1_ACC"), d("MRI_T1_F1")],
            ["", "SI XYZV T=32", "PointNet-GRU", "subject-independent", d("MRI_XYZV_ACC"), "—"],
            ["", "SI XYZVI T=32", "PointNet-GRU", "subject-independent", d("MRI_XYZVI_ACC"), d("MRI_XYZVI_F1")],
            ["RadHAR", "official XYZ T=60", "PointNet-GRU", "official train/test; not SI", d("RADHAR_XYZ_ACC"), "—"],
            ["", "official XYZV T=60", "PointNet-GRU", "official train/test; not SI", d("RADHAR_XYZV_ACC"), d("RADHAR_XYZV_F1")],
            ["", "official XYZVI T=60", "PointNet-GRU", "official train/test; not SI", d("RADHAR_XYZVI_ACC"), "—"],
            ["", "official T=1 XYZVI", "PointNet-pool", "official; single frame", d("RADHAR_T1_ACC"), d("RADHAR_T1_F1")],
            ["", "official voxel", "CNN-GRU", "official train/test; not SI", d("RADHAR_VOXEL_ACC"), d("RADHAR_VOXEL_F1")],
            ["CI4R paired", "10 GHz within", "ResNet18", "7-class; official plot", d("R0_10_WITHIN"), "—"],
            ["", "24 GHz within", "ResNet18", "7-class; official plot", d("R0_24_WITHIN"), "—"],
            ["", "77 GHz within", "ResNet18", "7-class; official plot", d("R0_77_WITHIN"), "—"],
            ["CI4R 77 11-class", "random-record", "ResNet18", "not the paired 7-class split", d("CI4R77_RAND_ACC"), d("CI4R77_RAND_F1")],
            ["", "prefix-group-disjoint", "ResNet18", "not subject-independent", d("CI4R77_PREFIX_ACC"), d("CI4R77_PREFIX_F1")],
        ],
    }
    t3 = {
        "title": (
            "Table 3. Within-radar and zero-shot cross-radar accuracy on the paired CI4R 7-class split. "
            "The split is by acquisition timestamp; filename prefix is not a subject."
        ),
        "header": ["Train \\ Test", "10 GHz", "24 GHz", "77 GHz"],
        "numeric_cols": [1, 2, 3],
        "rows": [
            ["10 GHz", d("R0_10_WITHIN"), d("R0_10_TO_24"), d("R0_10_TO_77")],
            ["24 GHz", d("R0_24_TO_10"), d("R0_24_WITHIN"), d("R0_24_TO_77")],
            ["77 GHz", d("R0_77_TO_10"), d("R0_77_TO_24"), d("R0_77_WITHIN")],
        ],
    }
    t4 = {
        "title": (
            "Table 4. Shared representation, unsupervised adaptation, and leave-one-radar-out generalization. "
            "The Protocol column separates known-radar learning, UDA, and DG; the rows are not one ranking."
        ),
        "header": ["Method", "Quantity", "Value", "Protocol"],
        "numeric_cols": [2],
        "section_rows": [0, 14, 17],
        "rows": [
            ["Known-radar learning", "", "", ""],
            ["M0 CE", "activity accuracy", d("M0_ACC"), "known-radar joint encoder"],
            ["M0 CE", "same-action retrieval top-1", d("M0_RET"), "known-radar; 6 directed pairs"],
            ["M0 CE", "Radar-ID", d("M0_RID"), "known-radar; 3-radar probe"],
            ["M1 CE+InfoNCE", "activity accuracy", d("M1_ACC"), "known-radar joint encoder"],
            ["M1 CE+InfoNCE", "same-action retrieval top-1", d("M1_RET"), "known-radar; 6 directed pairs"],
            ["M1 CE+InfoNCE", "Radar-ID", d("M1_RID"), "known-radar; 3-radar probe"],
            ["M2 +physics aug", "activity accuracy", d("M2_ACC"), "known-radar joint encoder"],
            ["M2 +physics aug", "same-action retrieval top-1", d("M2_RET"), "known-radar; 6 directed pairs"],
            ["M2 +physics aug", "Radar-ID", d("M2_RID_SHARE"), "known-radar; 3-radar probe"],
            ["M3 CE+InfoNCE+GRL", "activity accuracy", d("M3_ACC"), "known-radar adversarial"],
            ["M3 CE+InfoNCE+GRL", "Radar-ID", d("M3_RID"), "known-radar; 3-radar probe"],
            ["A0 / A1 / A2", "activity accuracy", f"{d('A0_ACC')} / {d('A1_ACC')} / {d('A2_ACC')}", "known-radar adapters"],
            ["A2", "Radar-ID on embeddings", d("RID_A2_EMBED"), "known-radar adapters"],
            ["UDA", "", "", ""],
            ["CORAL 77→10", "HAR accuracy", d("CORAL_77_TO_10"), "UDA; not unseen-radar DG"],
            ["CORAL 77→24", "HAR accuracy", d("CORAL_77_TO_24"), "UDA; not unseen-radar DG"],
            ["Domain generalization", "", "", ""],
            ["LORO 10+24→77", "HAR accuracy", d("LORO_1024_TO_77"), "DG; target radar unseen"],
            ["LORO 10+77→24", "HAR accuracy", d("LORO_1077_TO_24"), "DG; target radar unseen"],
            ["LORO 24+77→10", "HAR accuracy", d("LORO_2477_TO_10"), "DG; target radar unseen"],
        ],
    }
    t5 = {
        "title": (
            "Table 5. Physical reconstruction and measurement-response diagnostics. "
            "Radar-ID, activity probe, and difference-field R² are different quantities."
        ),
        "header": ["Stage", "Metric", "Value", "Protocol"],
        "numeric_cols": [2],
        "rows": [
            ["R0 official plot", "Radar-ID (3-radar)", d("RID_R0_3CLASS"), "3-radar probe"],
            ["R1 crop / normalize", "Radar-ID (3-radar)", d("RID_R1_3CLASS"), "3-radar probe"],
            ["R1 crop / normalize", "activity probe", d("ACT_R1_REP"), "3-radar feature family"],
            ["R2-D numerical Doppler", "Radar-ID (3-radar)", d("RID_R2D_3CLASS"), "3-radar probe"],
            ["R2-D numerical Doppler", "activity probe", d("ACT_R2D_REP"), "3-radar feature family"],
            ["R2-D 24/77", "Radar-ID (24/77)", d("RID_R2D_2477"), "2-radar probe; not the 3-radar series"],
            ["R2-D 24/77", "activity probe", d("ACT_R2D_2477"), "24/77 feature family"],
            ["R2-V velocity align", "Radar-ID (24/77)", d("RID_R2V_2477"), "2-radar probe"],
            ["R2-V velocity align", "activity probe", d("ACT_R2V_2477"), "24/77 feature family"],
            ["U1 unified STFT", "Radar-ID (24/77)", d("U1_RID"), "2-radar probe"],
            ["U2 transplanted MTI", "Radar-ID (24/77)", d("U2_RID"), "2-radar probe"],
            ["D(v|q) calibration", "Radar-ID after / before", f"{d('CAL_RID_AFTER')} / {d('CAL_RID_BEFORE')}", "2-radar probe"],
            ["D(v|q) 24→77", "R² (velocity-marginal)", d("D2477_B1_R2"), "not a HAR score"],
            ["D(v|q) 77→24", "R² (velocity-marginal)", d("D7724_B1_R2"), "not a HAR score"],
            ["2D TEST oracle", "R² (difference field)", d("ORACLE_R2_2D"), "TEST Δ on TRAIN basis; not a translator"],
            ["B2 predicted 24→77", "R² (difference field)", d("B2_R2_2477"), "predicted map; not oracle"],
            ["B2 predicted 77→24", "R² (difference field)", d("B2_R2_7724"), "predicted map; not oracle"],
        ],
    }
    t6 = {
        "title": (
            "Table 6. Labeled-interval continuous replay and 13-fold leave-one-action-out open-set scores. "
            "Continuous rows are not full-stream inference; open-set rows are not open-world recognition."
        ),
        "header": ["Quantity", "Value", "Protocol"],
        "numeric_cols": [1],
        "section_rows": [0, 6],
        "rows": [
            ["Continuous inference", "", ""],
            ["test windows", d("CONT_N_WINDOWS"), "labeled-interval; mRI SI checkpoint"],
            ["frame Acc / F1", f"{d('CONT_FRAME_ACC')} / {d('CONT_FRAME_F1')}", "labeled intervals only"],
            ["window Acc / F1", f"{d('CONT_WIN_ACC')} / {d('CONT_WIN_F1')}", "confidence-aware"],
            ["segment F1 raw / flip / conf.", f"{d('CONT_SEG_F1_RAW')} / {d('CONT_SEG_F1_FLIP')} / {d('CONT_SEG_F1')}", "unweighted mean of 3 recordings"],
            ["median boundary error", f"{d('CONT_BND_MED')} frames / {d('CONT_BND_SEC')} s", "mean of per-recording medians"],
            ["Open-set rejection", "", ""],
            ["folds", d("OPEN_N_FOLDS"), "leave-one-action-out; 12 known + 1 unseen"],
            ["MSP AUROC / FPR95", f"{pm(snap,'MSP_AUROC','MSP_AUROC_STD')} / {pm(snap,'MSP_FPR','MSP_FPR_STD')}", "13-fold LOAO; not open-world"],
            ["Energy AUROC / FPR95", f"{pm(snap,'ENERGY_AUROC','ENERGY_AUROC_STD')} / {pm(snap,'ENERGY_FPR','ENERGY_FPR_STD')}", "13-fold LOAO"],
            ["Prototype AUROC / FPR95", f"{pm(snap,'PROTO_AUROC','PROTO_AUROC_STD')} / {pm(snap,'PROTO_FPR','PROTO_FPR_STD')}", "13-fold LOAO; selected score"],
        ],
    }
    t7 = {
        "title": (
            "Table 7. Motion Intelligence fields, language-layer checks, and semantic safeguards. "
            "The language layer reads structured JSON and does not receive raw radar."
        ),
        "header": ["Item", "Recorded value / rule", "Source"],
        "numeric_cols": [],
        "rows": [
            ["Motion Intelligence", f"boxing; conf. {d('BOX_CONF')}; unc. {d('BOX_UNC')}; peak |v| {d('BOX_PEAK_V')} m s−1; energy {d('BOX_ENERGY')}", "llm_online_test.json"],
            ["UNKNOWN handling", "direction / periodicity / subject / environment = UNKNOWN", "stored UNKNOWN fields"],
            ["Raw radar isolation", "Language layer does not receive raw radar", "system contract"],
            ["QA grounding", "Unsupported event (bend-and-pick) must refuse; stored QA does", "llm_online_test.json task=qa"],
            ["Unknown-motion behavior", "Must not force a known class", "unknown-motion template"],
            ["Stored semantic tasks", f"{d('LLM_N_OK')} / {d('LLM_N')}", "current, summary, uncertainty, QA"],
        ],
    }
    return [t1, t2, t3, t4, t5, t6, t7]


CLAIMS = [
    ("LLM_C1", "LLM consumes structured Motion Intelligence, not raw radar.", "STRONG", "radarmind_v01/radarmind_v01_report.md", "input schema", "§2.15"),
    ("LLM_C2", "Semantic QA is grounded in Timeline evidence.", "STRONG", "radarmind_v01/llm_online_test.json", "task=qa", "§3.10"),
    ("LLM_C3", "Unknown-motion output is not forcibly mapped to a known activity.", "STRONG", "radarmind_v02/smoke_test_report.md", "DeepSeek/fallback unknown template", "§2.15.4; §3.10"),
    ("LLM_C4", "Language layer provides explanation/interactivity, not recognition accuracy improvement.", "STRONG", "manuscript policy", "no HAR metric attributed to LLM", "§4.2"),
    ("CONT_C1", "Continuous results are labeled-interval replay, not full-stream inference.", "STRONG", "phase4_continuous/phase4_continuous_report.md", "Mode A", "§2.11; §3.7"),
    ("OPEN_C1", "Current method performs open-set rejection, not open-world recognition.", "STRONG", "phase5_open_set/phase5_open_set_report.md", "LOAO", "§2.12; §3.8"),
    ("HAR_C1", "Closed-set HAR is reported under dataset-specific protocols.", "STRONG", "phase1_pointcloud/baselines", "MRI_SI_ACC / RADHAR_XYZV_ACC", "§3.1"),
    ("SUB_C1", "mRI subject-independent uses official subject IDs; CI4R prefix is not subject.", "STRONG", "phase1_pointcloud/mri_summary.json", "si_subjects", "§2.5"),
    ("XR_C1", "High within-radar accuracy coexists with cross-radar collapse.", "STRONG", "phase2_cross_radar/runs", "R0_24_WITHIN vs R0_24_TO_77", "§3.4"),
    ("SH_C1", "Paired learning improves known-radar correspondence and does not remove Radar-ID.", "STRONG", "phase3_shared_motion/representation_metrics.csv", "M1_ACC / M1_RID", "§3.5"),
    ("PHY_C1", "Numerical Doppler reconstruction leaves 3-radar Radar-ID near ceiling.", "STRONG", "phase8_physical_representation/answers.json", "RID_R2D_3CLASS", "§3.6"),
]
