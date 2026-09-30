# MoReCMT: Multimodal Recurrent Cross-Modal Transformer-Based Feature Fusion for Emotional Recognition

MemoCMT classifies emotion one utterance at a time. Audio goes through HuBERT, the transcript goes
through BERT, a cross-modal transformer fuses the two, and a head predicts the label. It works, but
it throws the conversation away. In a two-person IEMOCAP dialogue, what the other person said thirty
seconds ago usually matters.

This repo bolts a DialogueRNN context layer on top of that. Each utterance still gets its fused
MemoCMT vector, but instead of classifying it directly, the fused vectors for a whole conversation
are fed in order through a DialogueRNN cell that carries three things: a global conversation state,
a per-speaker state, and an emotion state. The classifier reads the emotion state.

Results come from `scripts/run_experiments.py`: test metrics on IEMOCAP Session 5 (4-class), mean ±
std over 3 seeds, written to `experiments/summary.md`.

## What's in here

| Path | What it is |
|---|---|
| `MemoCMTDialogueRNN.ipynb` | The notebook I actually run. Built for Colab. |
| `MemoCMT.ipynb` | The earlier baseline notebook, plain upstream MemoCMT. Kept for comparison. |
| `MemoCMT/` | Fork of [tpnam0901/MemoCMT](https://github.com/tpnam0901/MemoCMT) with my changes. |
| `conv-emotion/` | declare-lab's repo, vendored as a reference for the original DialogueRNN. Nothing imports it. |

The parts of `MemoCMT/` that are mine:

- `src/models/dialogue_rnn.py`: `DialogueRNNCell`, DialogueRNN as in Majumder et al. (2019): a
  global GRU, attention over past global states, a per-speaker party GRU, and an emotion GRU fed by
  the speaker's state. `EmotionClassifier` is its ReLU → linear output layer.
- `src/models/networks.py`: `MemoCMTDialogueRNN`, which wraps the original `MemoCMT` module and runs
  the context model over its fused output.
- `scripts/preprocess.py` and `src/data/iemocap_split.py`: the split. Session 5 is the test set, and
  val is 10% of the dialogues from Sessions 1–4, so no dialogue is spread across train, val, and
  test. Preprocessing also writes `meta.pkl` (session, dialogue, speaker, and start time for every
  utterance) and `split_info.json` (counts per split).
- `src/data/conversations.py` and `src/data/dataloader_dialogue.py`: regroup the flat pickles back
  into conversations, with turns ordered by their start time from `meta.pkl`. The filename counter
  (`F003`) is per speaker, so it can't give turn order.
- `src/trainer.py`: `DialogueTrainer`, which handles the extra time dimension and masks padded turns.
- `src/utils/metrics.py`: UA, WA, macro-F1, and weighted-F1, pooled over every utterance in a split.
- `scripts/run_experiments.py`: runs the window and modality grid over seeds and writes the summary.
- `src/configs/hubert_dialogue.py`: the config for all of this.
- `tests/`: unit tests for the pieces above. From `MemoCMT/`: `.venv/bin/pytest tests`.

## Running it

Everything runs in Colab off Google Drive. Local would work too, but the notebook assumes Drive.

1. Copy this repo into your Drive at `MyDrive/MemoCMTDialogueRNN/`.
2. Get IEMOCAP and preprocess it once:
   ```bash
   cd scripts
   python3 preprocess.py -ds IEMOCAP -dr /path/to/IEMOCAP_full_release
   ```
   That writes `train.pkl`, `val.pkl`, and `test.pkl` (test = Session 5), plus `meta.pkl` and
   `split_info.json`. Stash the result in Drive under a **new** folder name,
   `MyDrive/IEMOCAP_processed_session5/`, and point the notebook's copy cell at it. The notebook
   copies it back into `src/data/IEMOCAP` on every session. The old `IEMOCAP_processed/` folder has
   the random utterance split and no `meta.pkl`, and the dialogue loader refuses to use it.
3. Open `MemoCMTDialogueRNN.ipynb`, run the setup cells, then run the whole grid (window 1, 3, 6,
   and whole dialogue, plus text-only and audio-only, each with seeds 0, 1, 2):
   ```bash
   cd scripts
   python3 run_experiments.py
   ```
   Finished runs are skipped, so after a Colab disconnect you can just run it again. Point
   `OUT_DIR` in `run_experiments.py` at Drive (or symlink `experiments/`) so the results survive a
   session reset. Use `--only w1 w6` and `--seeds 0` to run part of the grid, and `--dry-run` to
   print the commands without running them. For a single run:
   ```bash
   python3 train.py -cfg ../src/configs/hubert_dialogue.py --set context_window=6 seed=0
   ```

Use a GPU runtime. The config assumes CUDA is there, and on CPU you will be waiting a very long time.

### Skip the ESD and MELD cells

They are not required and were not used in the experiments. They call `train.py -ds ESD -name ESD_train`, but `train.py` only takes `-cfg`, so
those two flags aren't recognized. Everything dataset-related lives in the config file now. If you
want ESD or MELD, preprocess them and point `data_name` / `data_root` at the result.

### Evaluating

There's no separate step. At the end of every run, `train.py` reloads the checkpoint with the best
validation UA, scores the test set once, and writes `<checkpoint>/results.json` (plus a copy at
`results_path` when set). `run_experiments.py` collects those into `experiments/summary.csv` and
`experiments/summary.md` as mean ± std over seeds.

`eval.py` is upstream MemoCMT's utterance-level evaluator and doesn't support the dialogue model. It
reads `cfg.log` out of the checkpoint folder to rebuild the config, which matters for the warning
below.

## Config knobs worth knowing

All in `src/configs/hubert_dialogue.py`. The ones I moved around the most:

- `context_window` (6): strict. To predict turn t, the model starts fresh and reads only turns
  t−w+1 through t, so `1` means no conversational context at all (same encoder and fusion), and
  `None` means the whole dialogue so far. This is the main lever.
- `dialogue_hidden_size` (128): width of the DialogueRNN states.
- `ablate_audio` (`False`): zeros out the audio waveform before it reaches HuBERT, leaving a
  text-only model with the full architecture still attached.
- `ablate_text` (`False`): replaces every transcript with just `[CLS][SEP]`, leaving an audio-only
  model.
- `seed` (0), `best_metric` (`"ua"`), `results_path` (`None`): the training seed, the validation
  metric that picks the checkpoint, and an extra place to write `results.json`.
- `batch_size` (2) and `memo_chunk_size` (16): batch size counts conversations, not utterances, so
  it blows up fast. `memo_chunk_size` is how many utterances go through the HuBERT/BERT stack at a
  time before the fused vectors are stitched back together. Drop it if you're hitting OOM.
- `text_unfreeze` / `audio_unfreeze`: currently BERT trains and HuBERT is frozen.
- `learning_rate` (5e-6), `fusion_learning_rate` (3e-5), `dialogue_learning_rate` (1e-4). Three
  separate rates, because the pretrained encoders need a much gentler one than the freshly
  initialized DialogueRNN.

Any key can be overridden from the command line without editing the file:
`train.py -cfg ... --set context_window=None seed=1`.

Also check `checkpoint_dir` in `src/configs/hubert_base.py`. It's currently hardcoded to one of my
Drive paths and you'll want to change it.

## Checkpoints and logs

All the training runs, checkpoints, and logs:

https://drive.google.com/drive/folders/1vD4Sw9k2AlWMRuzVvYMP9Pr7fXMLRvr3?usp=drive_link

- `20251207-1457`: context window 6, 81.51%.
- `20251207-191239`: 81.49%.
- `context_1`, `context_3`: the smaller context windows.

One warning. For a handful of runs in the middle, `cfg.log` bugged out and saved the wrong
config. The weights are fine, the recorded hyperparameters aren't. Since `eval.py` rebuilds the
config from that same `cfg.log`, evaluating one of those checkpoints can silently construct the
wrong model. If the numbers look off, that's probably why. Cross-check against the training log in
the same folder.

## Credit

- MemoCMT: [tpnam0901/MemoCMT](https://github.com/tpnam0901/MemoCMT), "MemoCMT: Cross-Modal
  Transformer-Based Multimodal Emotion Recognition System."
- DialogueRNN: Majumder et al., via [declare-lab/conv-emotion](https://github.com/declare-lab/conv-emotion).
