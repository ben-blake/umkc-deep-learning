# HW3: Speaker Classification (Transformer)

Classify which of 600 speakers an utterance belongs to, using a single transformer encoder layer over mel-spectrogram features. The data is a 600-speaker subset of VoxCeleb1, already preprocessed into 40-dim mel frames: 69,438 utterances, split 90/10 into 62,494 train and 6,944 validation.

## Data

Each utterance is stored as its own `.pt` tensor of shape (frames, 40). `myDataset` takes a random 128-frame crop out of each one, and `collate_batch` pads anything shorter with -20 (about log 10^-20). Only one utterance in the set is shorter than 128 frames (shortest 96, median 584), so the padding almost never comes into play. At batch size 32 an epoch is 1,952 steps.

## Model

`Linear(40, 80)` prenet, then one `TransformerEncoderLayer` with d_model 80, 2 heads (40 dims each) and a 256-wide feedforward. The output is mean-pooled over time, then goes through `Linear(80, 80)`, `ReLU`, `Linear(80, 600)` for one logit per speaker. 125,896 parameters total: 3,280 prenet, 67,536 encoder layer, 55,080 head.

The encoder layer uses the default `batch_first=False`, which is why `forward` permutes the input to (length, batch, d_model) before it and transposes back after.

AdamW with peak lr 1e-3, cross entropy loss, batch size 32, 20,000 steps (a little over 10 epochs). The lr warms up linearly over the first 1,000 steps, then decays to 0 along a half-cosine.

## Results

Best validation accuracy **0.5291** at step 18,000. With 600 speakers, chance is about 0.17%.

| Step   | Valid acc | Valid loss |
| ------ | --------- | ---------- |
| 2,000  | 0.20      | 3.88       |
| 4,000  | 0.31      | 3.16       |
| 6,000  | 0.36      | 2.89       |
| 8,000  | 0.41      | 2.69       |
| 10,000 | 0.45      | 2.47       |
| 12,000 | 0.48      | 2.32       |
| 14,000 | 0.50      | 2.23       |
| 16,000 | 0.52      | 2.15       |
| 18,000 | **0.53**  | 2.08       |
| 20,000 | 0.52      | 2.12       |

Accuracy goes up and loss goes down at every checkpoint through 18,000. The small drop at 20,000 is most likely noise. By step 18k the lr is down to about 2.7e-5, so the weights barely move over the last 2k steps, and validation also takes random 128-frame crops, so its numbers wobble a bit from pass to pass.

There's no sign of overfitting. Validation loss falls at every checkpoint through 18k, and the only uptick is the tiny one at 20k when the lr is basically zero. The training accuracy in the progress bar is a single batch of 32 with dropout on, so it's too noisy to compare directly against validation. If anything the model looks like it would keep improving with more training. The TODOs fix the architecture, and the template says raising `total_steps` (e.g. to 70,000) gets higher accuracy, but that line isn't a TODO, so it stayed at 20,000.

One caveat on the checkpoint. The provided loop does `best_state_dict = model.state_dict()`, which holds references to the live weights rather than a copy. So `model.ckpt` written at step 20,000 actually has the step-20,000 weights (0.52), not the step-18,000 ones the log reports as 0.5291. It's about a 0.01 gap here.

## Files

| File               | Contents                                                                    |
| ------------------ | --------------------------------------------------------------------------- |
| `HW3-formal.ipynb` | The notebook with outputs. Summary and AI use disclosure are at the bottom. |

The dataset (`Dataset_HW3.zip`) is not committed. It's 8.2 GB, well over GitHub's 100 MB file limit, so it stays in Google Drive and is listed in `.gitignore`. It's a preprocessed 600-speaker subset of [VoxCeleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)), provided with the assignment as mel-spectrogram features.

## Running it

Open in Colab on a GPU runtime (this run used a T4). Update the Drive path in the first code cell to wherever `Dataset_HW3.zip` lives, then Run All. It unzips to `./Dataset`, which is what `parse_args` expects. Training took about 14 minutes on the T4.

`n_workers` is 8, but the Colab runtime only has 2 CPUs, so PyTorch prints a warning. It doesn't stop anything; training still averaged about 28 steps/s.

No seed is set, so the train/validation split and the crops change every run, and accuracies will move a bit from the table above.

## AI Use Disclosure

Anthropic Claude Code was utilized on this assignment. Its contributions were: assistance with the explanatory comments next to updated TODO items and the summary. All code outside the TODO markers is the provided template, unmodified apart from the Google Drive mount / dataset unzip cell. I personally modified, reviewed, and ran the final notebook.

## Documentation Referenced

Data loading:

- `torch.utils.data.DataLoader` - https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
- `torch.utils.data.random_split` - https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
- `pad_sequence` - https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html

Model:

- `nn.TransformerEncoderLayer` - https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
- `nn.TransformerEncoder` - https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
- `nn.Linear` - https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
- `nn.ReLU` - https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
- Attention Is All You Need - https://arxiv.org/abs/1706.03762

Loss, training and evaluation:

- `nn.CrossEntropyLoss` - https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- `torch.optim.AdamW` - https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
- `LambdaLR` (warmup + cosine schedule) - https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html
- `torch.no_grad` - https://pytorch.org/docs/stable/generated/torch.no_grad.html
- `Tensor.item` - https://pytorch.org/docs/stable/generated/torch.Tensor.item.html
