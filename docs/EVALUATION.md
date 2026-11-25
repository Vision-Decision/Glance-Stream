# Evaluation

This page lists every number the README and the model card report, how each was measured, and what was not
measured. All model numbers are for the release weights: the rule adapter (`text_lora.safetensors`) and the box adapter
and head (`box_lora.safetensors`, `box_head.safetensors`) are bit-identical to the training checkpoints that were
evaluated, and loading both in one model changes neither output (checked on GPU, see "Composite check").

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

## Composite check (release weights)

On one 36 s kitchen clip at 30 fps with four rules and one box (reference engine; repeated for the released 1,500-step box weights on an A100): the rule probabilities of
the released model are bit-identical to those of the rule-training checkpoint alone, and its boxes are bit-identical to
those of the box checkpoint alone. Box questions use their own adapter and rule questions theirs; the video path uses
neither, so the two cannot interact. The script that ran this check is not part of the package; its output is in the
model repository (`checks/independence.json`).

## Not measured for this release

- Speed: this release reports no latency number; the optimised engine is planned for a later release.
- Any rule wording other than the templates, and Chinese rules (training included Chinese phrasings; evaluation did
  not).
- Boxes for objects, boxes at 30 fps (the box benchmark streams at 2 fps), and calibration of p(yes) for cross-time
  rules (the 0.5 default threshold is not tuned; the benchmark chose thresholds on held-out participants).
