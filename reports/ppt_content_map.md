# RadarMind PPT Content Map

Source of truth: `Draft-RadarMind.pdf` (34 pages).  
Companion check: `results/final_delivery/RadarMind_Full_Manuscript.md` and `results/final_delivery/metric_snapshot_full.json`.  
All quantitative values below are copied from the PDF without rounding or embellishment.

Scientific boundaries that every slide must keep:

- No universal sensor-invariant representation.
- Robust unseen-radar generalization remains unresolved.
- CI4R prefix-group-disjoint is **not** subject-independent.
- True subject-independent evaluation exists only on mRI (official subject IDs).
- Continuous experiment is labeled-interval replay, not full-stream continuous sensing.
- Open-set is leave-one-action-out rejection, not open-world / open-vocabulary.
- LLM never receives raw IQ, spectrogram, point cloud, or hidden features.
- LLM does not improve HAR accuracy, cross-radar transfer, open-set, or temporal localization.
- Motion-conditioned R² is reconstruction, not HAR accuracy.
- TEST oracle is an upper-bound diagnostic, not a deployable translator.
- Geometry / distance / aspect / orientation are plausible causes only.
- Three datasets are complementary instruments, not one leaderboard.

---

## Slide 1 — Cover

| Field | Content |
| --- | --- |
| Scientific claim | RadarMind is a radar-to-semantics research framework: recognition → generalization diagnosis → continuous / open-set / Motion Intelligence → language. Not a claim that generalization is solved. |
| Source | Title page; Abstract; §1; Fig. 1 |
| Numerical values | none |
| Protocol boundary | Title uses *Toward*; cover must not imply solved invariance. |
| Visual form | Dark navy cover; native vector pipeline: radar beam → human motion → spectrogram / point cloud → Motion Intelligence → language |

---

## Slide 2 — Radar HAR works until the world changes

| Field | Content |
| --- | --- |
| Scientific claim | High recognition accuracy ≠ generalizable motion understanding. Conventional path is Radar → Representation → Classifier → Activity Label. |
| Source | §1 Introduction (PDF pp. 3–4) |
| Numerical values | none |
| Protocol boundary | Problem statement only; no dataset mixing. |
| Visual form | Traditional pipeline + four change axes: Different Person / Radar / Continuous Motion / Unknown Activity |

---

## Slide 3 — Why radar sensing?

| Field | Content |
| --- | --- |
| Scientific claim | Radar is privacy-preserving, illumination-independent, and motion-sensitive via Doppler / micro-Doppler; the measurement is strongly sensor-dependent. |
| Source | §1 opening; Keywords |
| Numerical values | none |
| Protocol boundary | Suitability of radar ≠ solved HAR generalization. |
| Visual form | Three property cards + schematic spectrogram / point-cloud glyphs |

---

## Slide 4 — From Activity Recognition to Motion Understanding

| Field | Content |
| --- | --- |
| Scientific claim | Five scientific stages: closed-set classification → generalization → continuous inference → open-set recognition → semantic understanding. Classification ≠ Understanding. |
| Source | §1; contributions (1)–(5); §2.1 |
| Numerical values | none |
| Protocol boundary | Stages are questions, not claimed solved capabilities. |
| Visual form | Horizontal five-stage chain |

---

## Slide 5 — Four questions behind RadarMind

| Field | Content |
| --- | --- |
| Scientific claim | Q1 generalization (subject / radar / domain); Q2 window → timeline; Q3 closed-set → open-set; Q4 prediction → Motion Intelligence → language. Central challenge: unseen-radar generalization. |
| Source | §1; §4; Conclusions |
| Numerical values | none |
| Protocol boundary | Central challenge is marked unresolved, not as a solved pillar. |
| Visual form | Four Q cards + one central-challenge callout |

---

## Slide 6 — RadarMind pipeline (Fig. 1 redraw)

| Field | Content |
| --- | --- |
| Scientific claim | Three layers: I Recognition & Generalization; II Motion Understanding; III Semantic Interaction. Language model never receives raw radar. |
| Source | §2.1; Fig. 1; §2.15 |
| Numerical values | none |
| Protocol boundary | Explicit X between Raw IQ / Spectrogram / Point Cloud and the language model. Fully shared motion tokens across radar forms have not been realized. |
| Visual form | Three-layer native shape diagram + isolation box |

---

## Slide 7 — Three datasets, three roles

| Field | Content |
| --- | --- |
| Scientific claim | CI4R, mRI, and RadHAR are complementary experimental instruments, not one merged leaderboard. |
| Source | §2.2; Table 1; Fig. 2 |
| Numerical values | CI4R: 10/24/77 GHz; 327 aligned triples; 7 actions; split 229/49/49. mRI: 20 subjects; 13 actions; point cloud. RadHAR: 5 actions; official train/test. |
| Protocol boundary | CI4R filename prefix is not a subject. Prefix-group-disjoint ≠ subject-independent. RadHAR official split ≠ subject-independent. Only mRI supports true SI / labeled-interval continuous / LOAO. |
| Visual form | Three role cards + footer boundary |

---

## Slide 8 — Within-domain radar HAR works

| Field | Content |
| --- | --- |
| Scientific claim | Closed-set HAR is strong inside each dataset’s intended protocol. These numbers are not one benchmark. |
| Source | Abstract; §3.1; Table 2; Fig. 3a |
| Numerical values | mRI SI: Acc 0.478, Macro-F1 0.420. RadHAR official XYZV: Acc 0.949, Macro-F1 0.949. CI4R within-radar: 10 GHz 0.843; 24 GHz 0.898; 77 GHz 0.922. |
| Protocol boundary | Different datasets and protocols; not directly comparable. mRI is true subject-independent. RadHAR is official split, not SI. CI4R is 7-class official plot, acquisition-timestamp split. |
| Visual form | Three large metric panels |

---

## Slide 9 — Motion is temporal

| Field | Content |
| --- | --- |
| Scientific claim | Temporal integration provides a stronger and more consistent gain than simply adding point-cloud channels. Human motion is a spatiotemporal process. |
| Source | §2.6; §3.3; Table 2; Fig. 3b |
| Numerical values | mRI T=1 0.254 → T=32 0.478. RadHAR T=1 0.607 → T=60 0.949. |
| Protocol boundary | mRI T=1 is SI XYZVI pool; T=32 is SI XYZ. RadHAR T=1 is official XYZVI pool; T=60 is official XYZV. Not an architecture contest. |
| Visual form | Two temporal arrows with exact accuracies |

---

## Slide 10 — The model also recognizes the radar

| Field | Content |
| --- | --- |
| Scientific claim | Same activity inventory and paired acquisitions still collapse under zero-shot cross-radar transfer. Cross-radar domain shift dominates. |
| Source | §2.7; §3.4; Table 3; Fig. 4 |
| Numerical values | Within: 10→10 0.843; 24→24 0.898; 77→77 0.922. Cross: 10→24 0.143; 24→77 0.039; 77→24 0.163. |
| Protocol boundary | CI4R paired 7-class; acquisition-timestamp split; prefix is not a subject. Highlight 24→77 = 0.039 without implying it is the only failure. |
| Visual form | Within vs cross columns; 24→77 warning highlight |

---

## Slide 11 — Diagonal accuracy hides cross-radar failure

| Field | Content |
| --- | --- |
| Scientific claim | Official-plot 3×3 matrix plus Radar-ID = 1.000 show that the representation itself is strongly sensor-dependent. |
| Source | Table 3; Fig. 4; §3.4 |
| Numerical values | Matrix: 0.843 / 0.143 / 0.373; 0.353 / 0.898 / 0.039; 0.275 / 0.163 / 0.922. Radar-ID = 1.000 (3-radar probe on official plots). |
| Protocol boundary | Filename prefix is not a subject. Radar-ID is a probe, not HAR accuracy. |
| Visual form | Native 3×3 heatmap + Radar-ID callout |

---

## Slide 12 — Shared representation ≠ sensor invariance

| Field | Content |
| --- | --- |
| Scientific claim | Contrastive learning improves known-radar semantic correspondence, but radar identity remains perfectly recoverable. |
| Source | §2.8; §3.3; Table 4; Fig. 5 |
| Numerical values | M0: Acc 0.769, Retrieval 0.718, Radar-ID 1.000. M1: 0.932 / 0.935 / 1.000. M2: 0.878 / 0.864 / 1.000. |
| Protocol boundary | Known-radar joint encoder, not LORO / unseen-radar DG. Retrieval is same-action top-1 over 6 directed pairs. |
| Visual form | Three method cards or clustered bars; Radar-ID locked at 1.000 |

---

## Slide 13 — Is the shift only an image-processing artifact?

| Field | Content |
| --- | --- |
| Scientific claim | Removing plotting conventions and aligning physical velocity helps, but radar identity remains strongly recoverable. 3-radar and 24/77 probes are different experiments. |
| Source | §2.10; §3.4; Table 5; Fig. 6 |
| Numerical values | 3-radar: R0 1.000; R1 1.000; R2-D 0.993. 24/77: R2-D 1.000; R2-V 0.939; Unified STFT 0.939; Transplanted MTI 0.980. |
| Protocol boundary | Do **not** draw 3-radar and 24/77 as one continuous monotonic curve. 10 GHz has no equivalent raw chain for R2-V. |
| Visual form | Two separate visual ladders |

---

## Slide 14 — Structured but only partially predictable

| Field | Content |
| --- | --- |
| Scientific claim | Part of the radar response is predictable from motion state; current models do not form a universal radar translator. |
| Source | §2.10; §3.4; Table 5; Fig. 7 |
| Numerical values | D(v\|q) 24→77 R² = 0.744; 77→24 R² = 0.575. 2D TEST oracle R² = 0.418; predicted 24→77 0.300; predicted 77→24 0.173. |
| Protocol boundary | Reconstruction R² ≠ HAR accuracy. TEST oracle ≠ deployable translator. Train-mean baselines 0.219 are context, not a HAR score. |
| Visual form | Two R² groups with explicit warnings |

---

## Slide 15 — Windows ≠ events

| Field | Content |
| --- | --- |
| Scientific claim | Temporal smoothing reduces fragmentation, but accurate event boundaries remain unresolved. |
| Source | §2.11; §3.5; Table 6; Fig. 8 |
| Numerical values | Window Acc 0.494 / Macro-F1 0.419 (confidence-aware). Frame Acc 0.489 / Macro-F1 0.416. Segment F1 = 0.152. Median boundary error 190.8 frames / 20.6 s. |
| Protocol boundary | Labeled-interval replay ≠ full-stream continuous sensing. Same mRI SI checkpoint; test subjects 3/10/20; 1207 windows. |
| Visual form | Three-row schematic timeline + metric strip |

---

## Slide 16 — Closed-set confidence is not enough

| Field | Content |
| --- | --- |
| Scientific claim | Prototype distance is best among tested scores, but unknown-motion detection remains weak. Claim is open-set rejection, not open-world / open-vocabulary. |
| Source | §2.12; §3.5; Table 6; Fig. 9 |
| Numerical values | 13-fold LOAO. MSP 0.476 ± 0.156. Energy 0.540 ± 0.144. Prototype 0.628 ± 0.125. Prototype FPR95 0.791 ± 0.219. Chance line = 0.5. |
| Protocol boundary | 12 known + 1 unseen. Not open-world, not open-vocabulary, no semantic naming of unseen activities. |
| Visual form | Three AUROC bars + chance line + claim/not-claimed box |

---

## Slide 17 — Motion Intelligence

| Field | Content |
| --- | --- |
| Scientific claim | Structured record sits between radar inference and language. Unknown information remains UNKNOWN, not guessed. |
| Source | §2.13; §3.6; Table 7; Fig. 10 |
| Numerical values | Activity boxing [MODEL]  ; Confidence 0.999 [MODEL]; Uncertainty 0.004 [MODEL]; Mean velocity −0.035 m/s [PHYSICS_DERIVED]; Peak \|velocity\| 3.073 m/s [PHYSICS_DERIVED]; Motion energy 0.176 [PHYSICS_DERIVED]; Direction / Periodicity / Subject / Environment = UNKNOWN. |
| Protocol boundary | RadHAR official-split stored boxing window from `llm_online_test.json`. Source tags: MODEL / PHYSICS_DERIVED / METADATA / UNKNOWN. |
| Visual form | Structured record cards + four source-tag legend |

---

## Slide 18 — LLM explains evidence

| Field | Content |
| --- | --- |
| Scientific claim | Language layer improves semantic accessibility, not HAR accuracy. It never receives raw radar. |
| Source | §2.15; §3.6; §4.2; Table 7 |
| Numerical values | Stored semantic tasks 4/4 (explanation, summary, uncertainty, QA). No HAR-delta is claimed. |
| Protocol boundary | Raw IQ / Spectrogram / Point Cloud / Hidden Features are isolated from the LLM. LLM does not improve HAR / transfer / open-set / localization. |
| Visual form | Pipeline + four task cards + information-boundary bar |

---

## Slide 19 — What RadarMind has established

| Field | Content |
| --- | --- |
| Scientific claim | Radar-to-semantics stack is operational. Operational framework ≠ solved generalization problem. |
| Source | §3.7; Fig. 11; §6.1 |
| Numerical values | Maturity labels only (no new scores). Closed-set HAR ESTABLISHED; Subject Generalization EVALUATED; Cross-Radar Transfer DIAGNOSED; Continuous PARTIAL; Open-set PARTIAL; Motion Intelligence ESTABLISHED; Semantic Interaction OPERATIONAL. |
| Protocol boundary | DIAGNOSED / PARTIAL must not be drawn as solved. Unseen-radar remains unresolved. |
| Visual form | Capability chain with teal / gold / muted-red states |

---

## Slide 20 — The negative result is part of the contribution

| Field | Content |
| --- | --- |
| Scientific claim | Radar identity persists even after reconstruction, velocity alignment, signal-processing harmonization, calibration, and learned adaptation. |
| Source | §3.4; §4; Conclusions |
| Numerical values | none new (refers to prior slides) |
| Protocol boundary | Do not convert negative results into a claim of invariance. Do not say geometry is proven. |
| Visual form | Worked vs unresolved columns + center sentence |

---

## Slide 21 — What current public datasets lack

| Field | Content |
| --- | --- |
| Scientific claim | Residual radar dependence is observed, but cannot be uniquely attributed to frequency, hardware, geometry, processing, or their interactions. |
| Source | §2.7; §4.4; §5 |
| Numerical values | Current paired data: 327 timestamp-aligned triples with matched activity. Missing: controlled distance / aspect / orientation; complete geometry metadata; synchronized raw IQ across all sensors; full-stream background annotation. |
| Protocol boundary | Distance / aspect / orientation are plausible causes, not proven unique causes. |
| Visual form | Center human glyph + surrounding missing factors + Radar A / Radar B |

---

## Slide 22 — Next dataset the question requires

| Field | Content |
| --- | --- |
| Scientific claim | Next step is controlled paired multi-radar acquisition and three method tracks: sensor generalization, continuous understanding, open-world motion. |
| Source | §5; Conclusions |
| Numerical values | none (protocol plan, not a result) |
| Protocol boundary | Acquisition Toolkit files in the repo are a recording protocol / demo layout, not experimental results in this paper. |
| Visual form | Controlled acquisition schematic + Track A/B/C |

---

## Slide 23 — Three sentences

| Field | Content |
| --- | --- |
| Scientific claim | (1) Radar HAR works within a known sensing domain. (2) Changing the radar exposes a sensor-dependent domain shift that current representation learning and physical harmonization do not eliminate. (3) RadarMind extends recognition toward continuous, open-set, and grounded semantic understanding while making unresolved limits explicit. |
| Source | Abstract; §6; §6.1 |
| Numerical values | none |
| Protocol boundary | Final line is “toward understanding motion across radar domains,” not a claim of achieved invariance. |
| Visual form | Three numbered statements + closing sentence + thank you |

---

## Number checklist used in slides

| ID | Value | Slide |
| --- | --- | --- |
| MRI_SI_ACC / F1 | 0.478 / 0.420 | 8 |
| RADHAR_XYZV_ACC / F1 | 0.949 / 0.949 | 8 |
| CI4R within 10/24/77 | 0.843 / 0.898 / 0.922 | 8, 10, 11 |
| MRI T=1 → T=32 | 0.254 → 0.478 | 9 |
| RadHAR T=1 → T=60 | 0.607 → 0.949 | 9 |
| Cross 10→24 / 24→77 / 77→24 | 0.143 / 0.039 / 0.163 | 10 |
| Full 3×3 | see Table 3 | 11 |
| Official-plot Radar-ID | 1.000 | 11 |
| M0/M1/M2 Acc | 0.769 / 0.932 / 0.878 | 12 |
| M0/M1/M2 Retrieval | 0.718 / 0.935 / 0.864 | 12 |
| M0/M1/M2 Radar-ID | 1.000 / 1.000 / 1.000 | 12 |
| R0 / R1 / R2-D (3-radar) | 1.000 / 1.000 / 0.993 | 13 |
| R2-D / R2-V / STFT / MTI (24/77) | 1.000 / 0.939 / 0.939 / 0.980 | 13 |
| D(v\|q) R² | 0.744 / 0.575 | 14 |
| 2D R² | 0.418 / 0.300 / 0.173 | 14 |
| Continuous window / frame | 0.494/0.419 ; 0.489/0.416 | 15 |
| Segment F1 / boundary | 0.152 ; 190.8 frames / 20.6 s | 15 |
| Open-set AUROC | 0.476±0.156 / 0.540±0.144 / 0.628±0.125 | 16 |
| Prototype FPR95 | 0.791±0.219 | 16 |
| Boxing MI | 0.999 / 0.004 / −0.035 / 3.073 / 0.176 | 17 |
| Paired triples | 327 | 7, 21 |
