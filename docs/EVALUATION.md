# Evaluation

This page lists every number the README and the model cards report, how each was measured, and what was not
measured. Sections 1–2 are Glance-Stream-0.8B, section 3 is Glance-Stream-4B. All model numbers are for the release
weights: in both model repositories the rule adapter (`text_lora.safetensors`) and the box adapter and head
(`box_lora.safetensors`, `box_head.safetensors`) are bit-identical to the training checkpoints that were evaluated, and
loading both in one model changes neither output (checked on GPU, see "Composite check").

## 1. Cross-time rule false alarms (EPIC-KITCHENS-100, 22 streams)

| | baseline (same recipe without rule training) | Glance-Stream-0.8B |
|---|---:|---:|
| false alarms per rule-hour on cross-time rules, at 80% event recall | 289.9 | **173.4** |

- **Streams.** 22 native EPIC-KITCHENS-100 videos from 10 participants, streamed whole at 30 fps. None of these videos
  or participants is in the training data (training used Charades clips only).
- **Rules.** Generated per stream by the rule engine in `glance_stream/rules.py` from the human action annotations
  ("verb noun" segments): the five cross-time types (`after_then`, `b_without_a`, `a_not_followed`,
  `count_in_window`, `while_state`) plus `now` rules. Ground truth is the violation intervals the rule engine computes;
  no model or language model produces a label. Questions use the fixed templates (`Rule.template()`).
- **Readout.** 600 s memory window (all frame groups of the last 4 s, older groups up to 600 s on a 2 s grid, recurrent state
  over the whole video), every rule asked every 0.5 s, p(yes) from the first answer token, reference engine.
- **Alerts and scoring.** An alert fires after 2 consecutive readouts at or above the threshold, then a 5 s cooldown.
  A violation interval is hit if an alert falls inside it (with tolerance; none before the deadline for
  `a_not_followed`); further alerts in the same interval are duplicates; every other alert is a false alarm. The
  threshold is chosen for 80% event recall on the other participants (leave one participant out) and applied to the
  held-out participant. False alarms are counted on rules that are violated at least once and divided by their total
  rule time.
- **Comparison.** The baseline is trained with the same recipe, steps and data order but without the cross-time rule
  batches. The difference is significant over participants (paired exact sign-flip test, p = 0.002, the smallest
  attainable with 10 participants).
- **What it does not say.** This is a false-alarm number at fixed recall, pooled over the five cross-time types; it is
  not a per-type accuracy. Per type, `a_not_followed` rules are not significantly better than chance at ranking
  readouts, and `count_in_window` rules fire too often (see Limitations in the README).

## 2. Boxes (HC-STVG v2 test, 267 clips)

| | Glance-Stream-0.8B |
|---|---:|
| Acc@0.5 (box IoU ≥ 0.5 with the annotated person, readouts inside the annotated tube) | **48.5** |

- **Weights.** The released box adapter and head (1,500 training steps; the earlier 600-step version reached 43.7 on the same clips).
- **Clips.** 267 HC-STVG v2 test clips (a fixed list; the first 300 after a seeded shuffle, minus 33 clips whose video
  could not be obtained). Each clip is streamed at 2 fps, 256×256, with a 4–60 s window; one box readout per second.
- **Question.** `Where is the person described as "<HC-STVG caption>" right now? Answer with a box.` The box head
  reads the newest frame group. Acc@0.5 is over the readouts that fall inside the annotated time span (1,960
  readouts), not gated by the presence probability.
- **Scope.** HC-STVG describes people, so the head was trained and tested on person descriptions only. Objects ("the
  knife") are untested.

## 3. Glance-Stream-4B

The same recipe on Qwen3.5-4B, with only the model size changed. Every 4B comparison is against **the 4B baseline**
(the same recipe on Qwen3.5-4B without the cross-time rule batches), with both arms evaluated on the same hardware.
The tests and their thresholds were written down before any 4B result existed. Differences are means over
participants with an exact paired sign-flip test (the smallest attainable p is 0.031 with 6 participants and 0.002
with 10).

| | 4B baseline | Glance-Stream-4B | difference (p) | pre-registered test |
|---|---:|---:|---|---|
| cross-time false alarms per rule-hour at 80% event recall, 8 streams / 6 participants | 152.8 | 76.5 | −74, 5 of 6 lower (p = 0.0625) | primary: lower with p ≤ 0.05, **not met** |
| cross-time macro AUROC over the five types, same 8 streams | 0.701 | 0.793 | +0.073 (p = 0.031) | co-primary: ≥ +0.05 with p ≤ 0.05, met |
| cross-time false alarms per rule-hour at 80% event recall, 22 streams / 10 participants | 162.8 | 93.3 | −72, 9 of 10 lower (p = 0.004) | secondary |
| cross-time macro AUROC, 22 streams | 0.687 | 0.793 | +0.106 (p = 0.002) | secondary |
| "now" rule false alarms per rule-hour, 48 deployment streams | 42.6 | 37.3 | ratio 0.876 [0.771, 0.998] | no forgetting: upper bound ≤ 1.10, met |
| event presence AUROC / Charades-STA mIoU, short clips at 2 fps | 0.908 / 62.4 | 0.905 / 61.6 | −0.002 / −0.9 | no forgetting: drops ≤ 0.01 / ≤ 2, met |
| box Acc@0.5, HC-STVG v2 test, 267 clips | — | **45.7** | | |

- **Primary not met.** The 8-stream set is the original EPIC-KITCHENS-100 subset (6 participants) on which the 4B
  primary test was fixed. False alarms halved, but one participant went up (+12.6), so the exact test stops at
  p = 0.0625. The 22-stream set is the 0.8B main set (section 1); for 4B it is a secondary analysis.
- **Streams, rules, alerts, thresholds** as in section 1 (600 s window, every rule every 0.5 s at 30 fps, 2 consecutive
  readouts, 5 s cooldown, threshold for 80% recall chosen leaving one participant out). Differences in the table are
  means of per-participant differences, so they are not exactly the difference of the pooled columns.
- **Per type** (22 streams, AUROC, baseline → 4B): `after_then` 0.880 → 0.924, `b_without_a` 0.721 → 0.886,
  `while_state` 0.852 → 0.941, `count_in_window` 0.709 → 0.767, `a_not_followed` 0.465 → 0.629. `a_not_followed`
  remains the weakest type. On counting questions with true counts of 1–7 the mean signed error moves from −0.24
  (baseline) to +0.96: the rule-trained model over-counts.
- **Deployment streams.** 48 ten-minute streams spliced from Charades test videos, 16 "is someone X right now" rules
  each (at most 8 of them occur in the stream), labels from the human Charades annotations. The ratio's 95% interval
  is a bootstrap over source videos (a video's segments in all streams are resampled together), thresholds re-chosen
  in every resample.
- **Short-clip probes.** Charades test clips at 2 fps: presence AUROC of "is X happening now" questions, Charades-STA
  temporal grounding mIoU.
- **Boxes.** Same clips and protocol as section 2. The box adapter and head were trained for 1,500 steps (the 600-step
  version reached 38.4).
- **Training difference.** The 4B rule adapter was trained on 4 GPUs (data parallel, one clip per GPU, global batch 4);
  the 4B baseline on one GPU with a batch of 4. Steps, learning rate and data are the same.
- **Across sizes.** Sections 1 and 3 use the same evaluator and protocol, but the 0.8B and 4B runs were not on the same
  hardware, and no test compares the two sizes directly.

## Composite check (release weights)

On one 36 s kitchen clip at 30 fps with four rules and one box (reference engine; for 0.8B repeated for the released
1,500-step box weights on an A100, for 4B on an RTX A6000): the rule probabilities of
the released model are bit-identical to those of the rule-training checkpoint alone, and its boxes are bit-identical to
those of the box checkpoint alone. Box questions use their own adapter and rule questions theirs; the video path uses
neither, so the two cannot interact. The script that ran this check is not part of the package; its output is in the
model repositories (`checks/independence.json`).

## Not measured for this release

- Speed: this release reports no latency number; the optimised engine is planned for a later release.
- Any rule wording other than the templates, and Chinese rules (training included Chinese phrasings; evaluation did
  not).
- Boxes for objects, boxes at 30 fps (the box benchmark streams at 2 fps), and calibration of p(yes) for cross-time
  rules (the 0.5 default threshold is not tuned; the benchmark chose thresholds on held-out participants).
