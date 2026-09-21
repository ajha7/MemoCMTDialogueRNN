# MemoCMT + DialogueRNN

MemoCMT classifies emotion one utterance at a time. Audio goes through HuBERT, the transcript goes
through BERT, a cross-modal transformer fuses the two, and a head predicts the label. It works, but
it throws the conversation away. In a two-person IEMOCAP dialogue, what the other person said thirty
seconds ago usually matters.

This repo bolts a DialogueRNN context layer on top of that. Each utterance still gets its fused
MemoCMT vector, but instead of classifying it directly, the fused vectors for a whole conversation
are fed in order through a DialogueRNN cell that carries three things: a global conversation state,
a per-speaker state, and an emotion state. The classifier reads the emotion state.

Best run so far is 81.51% accuracy on IEMOCAP (4-class) with a context window of 6.

## What's in here

| Path | What it is |
|---|---|
| `MemoCMTDialogueRNN.ipynb` | The notebook I actually run. Built for Colab. |
| `MemoCMT.ipynb` | The earlier baseline notebook, plain upstream MemoCMT. Kept for comparison. |
| `MemoCMT/` | Fork of [tpnam0901/MemoCMT](https://github.com/tpnam0901/MemoCMT) with my changes. |
| `conv-emotion/` | declare-lab's repo, vendored as a reference for the original DialogueRNN. Nothing imports it. |

The parts of `MemoCMT/` that are mine:

- `src/models/networks.py`: `DialogueRNNCell` and `MemoCMTDialogueRNN`, which wraps the original
  `MemoCMT` module and runs the context RNN over its fused output.
- `src/data/dataloader_dialogue.py`: regroups the flat `train.pkl` / `val.pkl` lists back into
  conversations. Speaker ID and turn order get parsed out of the audio filename
  (`Ses01F_impro01_F003` gives conversation `Ses01F_impro01`, speakertion
  then gets sorted by turn index and padded into a batch.
- `src/trainer.py`: `DialogueTrainer`, which handles the extra time dturns.
- `src/configs/hubert_dialogue.py`: the config for all of this.

## Running it

Everything runs in Colab off Google Drive. Local would work too, but the notebook assumes Drive.

1. Copy this repo into your Drive at `MyDrive/MemoCMTDialogueRNN/`.
2. Get IEMOCAP and preprocess it once:
   ```bash
   cd scripts
   python3 preprocess.py -ds IEMOCAP -dr /path/to/IEMOCAP_full_release
   That writes train.pkl and val.pkl. Stash the result in Drive as My
   so you never have to do it again. The notebook copies it back into src/data/IEMOCAP on
   every session.
3. Open MemoCMTDialogueRNN.ipynb, run the setup cells, then:
cd scripts
python3 train.py -cfg ../src/configs/hubert_dialogue.py

Use a GPU runtime. The config assumes CUDA is there, and on CPU you will be waiting a very long time.

Skip the ESD and MELD cells

They don't work. They call train.py -ds ESD -name ESD_train, but train.py only takes -cfg, so
those two flags aren't recognized. Everything dataset-related lives iyou
want ESD or MELD, preprocess them and point data_name / data_root at the result.

Evaluating

cd scripts
python3 eval.py -ckpt /path/to/checkpoints/MemoCMT_bert_hubert_base/2

eval.py reads cfg.log out of the checkpoint folder to rebuild the con
warning below.

Config knobs worth knowing

All in src/configs/hubert_dialogue.py. The ones I moved around the most:

- context_window: how many past utterances the global-state attention can look at. This is the
  main lever. Runs at 1, 3, and 6 are all in the logs and 6 was best,n at
  3, so bump it if you're trying to match the best number.
- dialogue_hidden_size (128): width of the DialogueRNN states.
- ablate_audio: zeros out the audio waveform before it reaches HuBERT, leaving a text-only model
  with the full architecture still attached. This is the ablation stuIt's
  currently on, so flip it to False for a normal run.
- batch_size (2) and memo_chunk_size (16): batch size counts conversa
  it blows up fast. memo_chunk_size is how many utterances go through the HuBERT/BERT stack at a
  time before the fused vectors are stitched back together. Drop it i
- text_unfreeze / audio_unfreeze: currently BERT trains and HuBERT is frozen.
- learning_rate (5e-6), fusion_learning_rate (3e-5), dialogue_learnin
  separate rates, because the pretrained encoders need a much gentler one than the freshly
  initialized DialogueRNN.

Also check checkpoint_dir in src/configs/hubert_base.py. It's current
Drive paths and you'll want to change it.

Checkpoints and logs

All the training runs, checkpoints, and logs:

https://drive.google.com/drive/folders/1vD4Sw9k2AlWMRuzVvYMP9Pr7fXMLRvr3?usp=drive_link

Copy the folder into your own Drive to reproduce any of it. The runs worth looking at:

- 20251207-1457: context window 6, 81.51%. Best one.
- 20251207-191239: 81.49%. Basically tied, honorable mention.
- context_1, context_3: the smaller context windows, for the comparison.

One warning. For a handful of runs in the middle, cfg.log bugged out and saved the wrong
config. The weights are fine, the recorded hyperparameters aren't. Si
config from that same cfg.log, evaluating one of those checkpoints can silently construct the
wrong model. If the numbers look off, that's probably why. Cross-chec in
the same folder.

Credit

- MemoCMT: tpnam0901/MemoCMT, "MemoCMT: Cross-Modal
  Transformer-Based Multimodal Emotion Recognition System."
- DialogueRNN: Majumder et al., via declare-lab/conv-emotion.
