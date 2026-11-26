# Release checklist (before making the repositories public)

Both repositories stay **private** until the paper is posted on arXiv; they are made public together with it.

## Timing and anonymity

- [ ] The paper is on arXiv (or the venue's rules allow a public non-anonymous release now).
- [ ] Check the target venue's anonymity and preprint policy once more on the day of release: whether a public code and
      model release with author organisation names is allowed during review, and whether the paper must cite it
      anonymously.
- [ ] The paper and this repository report the same numbers (cross-time false alarms 289.9 → 173.4; box Acc@0.5 48.5)
      and describe the same weights.

## Content

- [ ] No internal hostnames, IP addresses, home directories, account names, internal notes or internal benchmark names
      anywhere in the repository, its history, the model card or the Hugging Face file metadata (grep the tree and
      `git log -p`).
- [ ] Every link points to `github.com/Vision-Decision/Glance-Stream` and
      `huggingface.co/Vision-Decision/Glance-Stream-0.8B`.
- [ ] `assets/demo.*` and `assets/demo_predictions.json` come from the released weights (sha256 of the three
      safetensors files as in the model repository).
- [ ] README, README.zh-CN, docs and the model card agree; the limitations section is unchanged in substance.
- [ ] `python -m pytest tests` passes; `examples/run_rules.py` runs from a clean install against the Hugging Face repo.

## Licences

- [ ] Code Apache-2.0 (`LICENSE`); weights CC BY-NC 4.0 (model card front matter and README badge).
- [ ] `docs/DATA.md` lists every training and evaluation source with its terms, including the sources without an
      explicit licence (RepCount-A, HC-STVG).
- [ ] Demo footage credited (Anna Shvets / Pexels, Pexels License); the original video file is not in the repository.

## Glance-Stream-4B

- [ ] Hugging Face `Vision-Decision/Glance-Stream-4B` holds `text_lora`, `box_lora`, `box_head`, `config.json`,
      `checks/independence.json`, the model card and `assets/demo.gif`; `assets/demo_4B.*` were rendered from exactly
      these files (sha256 checked in the render job).
- [ ] The card says plainly that the pre-registered primary 4B test was not met (p = 0.0625) and keeps the boxes-worse
      line.
- [ ] Visibility → public at the same moment as the README links to it are pushed to `main`.
- [ ] The 0.8B model card's "0.8B only" limitation is replaced by a link to the 4B repository.

## Switching to public

- [ ] GitHub `Vision-Decision/Glance-Stream`: visibility → public; add topics and the Hugging Face link.
- [ ] Hugging Face `Vision-Decision/Glance-Stream-0.8B`: visibility → public; the model card renders the demo GIF.
- [ ] Optional: add the model to the organisation's collections and profile README.
