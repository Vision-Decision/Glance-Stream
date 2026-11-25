# Training and evaluation data

No video is redistributed in this repository or in the model repository. The demo clip is a Pexels video (see the
README).

## Training

| Source | Used for | Licence / terms |
|---|---|---|
| [Charades](https://prior.allenai.org/projects/charades) (train split) | "now" questions, when / which-one / count questions on short clips, and the long training streams for cross-time rules (train clips spliced into 5–10 min streams) | Allen Institute for AI, non-commercial research use |
| Charades-STA (train) | temporal questions on Charades clips | derived from Charades |
| [RepCount-A](https://svip-lab.github.io/dataset/RepCount_dataset.html) (train) | repetition counting | **unclear**: we found no explicit licence in the public release; the authors distribute it for research, and we treat it as research use only |
| [HC-STVG v2](https://github.com/tzhhhh123/HC-STVG) (train) | box questions (person descriptions with a box per frame) | **unclear**: we found no explicit licence in the public release; the authors distribute it for research, and we treat it as research use only |
| Rule sentences | cross-time rules over the Charades training streams: the rule engine (`glance_stream/rules.py`) computes every label from the human Charades annotations; rule wordings (English and Chinese, colloquial and formal) were written by Claude Opus 5.5 from the rule's structured form and kept only if a blind back-parse recovered the same rule | labels: derived from Charades; wordings: generated |
| Box head initialisation | the box head started from the box head of a 0.8B image Glance model (trained on RefCOCO/+/g, gRefCOCO and COCO-based grounding data; see [Glance](https://github.com/Vision-Decision/Glance)) | includes non-commercial sources (gRefCOCO) |
| [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) | base model, frozen | Apache-2.0 |

Train and evaluation sets are separated by video id. Several sources are for non-commercial research use only, and
two (RepCount-A, HC-STVG) have no explicit licence we could find, so the weights are released under **CC BY-NC 4.0**
(non-commercial use only). The code is Apache-2.0. If you hold rights in one of these datasets and see a problem with
this release, please open an issue.

## Evaluation

| Source | Used for | Licence / terms |
|---|---|---|
| [EPIC-KITCHENS-100](https://epic-kitchens.github.io/) | 22 long kitchen streams (10 participants) with rules generated from the action annotations; evaluation only, never trained on | CC BY-NC 4.0 |
| HC-STVG v2 (test) | box accuracy, 267 clips | unclear, as above (research use only) |

## Demo footage

`assets/demo.*` show [Woman Chopping Bell Pepper](https://www.pexels.com/video/woman-chopping-bell-pepper-6617422/) by
Anna Shvets, used under the [Pexels License](https://www.pexels.com/license/) (free to use and modify, no attribution
required; the unmodified video may not be sold or redistributed as stock footage). The rendered demo overlays model
outputs on a downscaled copy; the original file is not included.
