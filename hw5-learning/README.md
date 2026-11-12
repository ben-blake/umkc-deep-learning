# HW5: Autoencoders for Anomaly Detection

Train four autoencoders (fully connected, convolutional, variational and ResNet-based) to reconstruct 64x64 images, as the first half of a reconstruction-error approach to anomaly detection. The template covers training only; it loads the test set but doesn't score it.

## Data

`hw5_data.zip` unzips to `hw5_data/` with two uint8 arrays: `trainingset.npy` (850 images) and `testingset.npy` (150 images), each 64x64x3. `CustomTensorDataset` permutes them to channels-first and maps the pixels to [-1, 1]. The batch size is 1024, which is more than 850, so the whole training set is one batch. Each epoch is a single optimizer step, and 20 epochs is 20 steps.

## Models

| Model  | Parameters | Bottleneck                                                    |
| ------ | ---------- | ------------------------------------------------------------- |
| FCN    | 3,176,419  | 3 values (12,288 → 128 → 64 → 12 → 3), Tanh output            |
| CNN    | 47,355     | 48x8x8, three k4/s2/p1 convs each way, Tanh output            |
| VAE    | 65,835     | mu and logvar heads, each 48x8x8, reparameterized, Tanh output |
| ResNet | 13,698,969 | 256-dim, from a resnet18 trunk through FC + BatchNorm1d layers |

The ResNet decoder's unpadded transpose convs go 4 → 9 → 19 → 39, so the last step bilinearly interpolates 39x39 up to 64x64. Its output goes through a Sigmoid, so it's in [0, 1].

Shared training settings: MSE loss (plus KL for the VAE), Adam with lr 1e-6, 20 epochs.

## Results

Mean training loss after the first and last epoch, on a T4:

| Model  | Epoch 1 | Epoch 20 | Time  |
| ------ | ------- | -------- | ----- |
| FCN    | 0.2716  | 0.2716   | ~3 s  |
| CNN    | 0.2676  | 0.2676   | ~3 s  |
| VAE    | 6710.31 | 6589.14  | ~5 s  |
| ResNet | 0.6184  | 0.6161   | ~24 s |

With the template's hyperparameters, none of the models really learns to reconstruct anything. For reference, an all-zeros output has MSE 0.2671 and the mean training image has 0.2548, so the FCN and CNN are at the level of a blank image. At lr 1e-6 over 20 steps, their loss changes don't show up in four decimals.

- **VAE:** `loss_vae` adds a KL term summed over every latent value in the batch to a per-pixel mean MSE, so KL is almost all of the loss. After training it's about 6,583 KL vs 0.274 MSE. The drop over training comes from the latents moving toward N(0, 1), not from better reconstructions.
- **ResNet:** its Sigmoid output can't produce the negative half of the [-1, 1] targets, which on its own sets an MSE floor of about 0.160. It also starts higher (0.618) because the untrained output sits around 0.5.

None of the lines that cause this are TODOs, so they stayed as the template has them. The obvious fixes, in order of impact:

1. Raise the lr to something like 1e-3 and drop the batch size to 64-128.
2. Average the VAE's KL term instead of summing it.
3. Swap the ResNet decoder's Sigmoid for a Tanh.
4. Add a cell that scores the test set by reconstruction error.

## Notes on the provided code

- Two of the ResNet TODOs (`bn1`, `fc_bn5`) have a trailing comma after the blank (`self.bn1 = ????? ,`). Left in, it would store a one-element tuple instead of a layer, so it was dropped.
- The training loops call `format_str('bold')`, but the current `qqdm` needs `format_str(style, s)`, so it fails with `TypeError: format_str() missing 1 required positional argument: 's'`. All four loops now use `format_str('bold', 'Description')`.
- `models.resnet18(pretrained=False)` and `torch.cuda.FloatTensor` in the VAE print deprecation warnings. Neither one breaks anything.
- Building `model_classes` in every model cell creates all four models each time, even though only one gets used.

## Files

| File           | Contents                                                                    |
| -------------- | --------------------------------------------------------------------------- |
| `HW5.ipynb`    | The notebook with outputs. Summary and AI use disclosure are at the bottom. |
| `hw5_data.zip` | The dataset (11 MB), provided with the assignment.                          |

## Running it

Open in Colab on a GPU runtime (this run used a T4). Update the Drive path in the unzip cell to wherever `hw5_data.zip` lives, then Run All. It unzips to `/content/hw5_data/`, which is what the loading cell reads. All four models together train in well under a minute.

The seed is set with `same_seeds(12345678)` (the template's value), and cuDNN is put in deterministic mode.

## AI Use Disclosure

Anthropic Claude Code was utilized on this assignment. Its contributions were: assistance with the explanatory comments next to updated TODO items and the summary. I personally modified, reviewed, and ran the final notebook.

## Documentation Referenced

Data loading:

- `numpy.load` - https://numpy.org/doc/stable/reference/generated/numpy.load.html
- `torch.utils.data` (`TensorDataset`, `DataLoader`, `RandomSampler`) - https://pytorch.org/docs/stable/data.html
- `torchvision.transforms` - https://pytorch.org/vision/stable/transforms.html
- PyTorch reproducibility - https://pytorch.org/docs/stable/notes/randomness.html

Model:

- `nn.Linear` - https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
- `nn.Conv2d` - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- `nn.ConvTranspose2d` - https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html
- `nn.BatchNorm1d` - https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
- `nn.ReLU` - https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
- `nn.Tanh` - https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html
- `F.interpolate` - https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
- `Tensor.view` - https://pytorch.org/docs/stable/generated/torch.Tensor.view.html
- `torchvision.models.resnet18` - https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- Auto-Encoding Variational Bayes - https://arxiv.org/abs/1312.6114
- Deep Residual Learning for Image Recognition - https://arxiv.org/abs/1512.03385

Loss and training:

- `nn.MSELoss` - https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html
- `torch.optim.Adam` - https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
- `qqdm` - https://github.com/KinglittleQ/qqdm
