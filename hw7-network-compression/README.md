# HW7: Network Compression (Knowledge Distillation)

Train a small CNN (under 100,000 parameters) to classify food-11 images, using a pretrained ResNet-18 as a teacher through knowledge distillation. The student is the template's architecture with the classifier filled in, and the teacher also pseudo-labels the unlabeled training images so the student has more to learn from.

## Data

`food-11.zip` unzips to `food-11/` with 3,080 labeled training images, 6,786 unlabeled ones, 660 for validation (60 per class) and 3,347 unlabeled test images, across 11 food classes. `DatasetFolder` labels them by folder.

Training images are resized to 142x142, then randomly flipped, rotated up to 15 degrees and randomly cropped to 128x128. Validation and test images are resized to 142 and center cropped to 128, with no augmentation.

The teacher labels the 6,786 unlabeled images with its argmax prediction, and those get concatenated with the 3,080 labeled ones, so each epoch is 9,856 images (154 batches of 64, last partial batch dropped).

## Models

Student (87,907 parameters): four 3x3 convs (3 to 32 to 32 to 64 to 100 channels), each followed by BatchNorm and ReLU, with 2x2 max pools after the second, third and fourth. Global average pooling leaves 100 features, and `Linear(100, 11)` maps them to the classes. Most of the parameters (57,700) are in the last conv.

Teacher: `torchvision` ResNet-18 with its fc swapped for `Linear(512, 11)`, loaded from the provided `ResNet_food11.pth` and kept in eval mode. It gets **0.7955** on the validation set and has about 11.2M parameters, roughly 127x the student.

## Loss and training

The KD loss is

loss = (1 - alpha) * CE(student, labels) + alpha * T^2 * KL(softmax(teacher / T) || softmax(student / T))

with alpha 0.5 and T 20. The KL term uses `F.kl_div` with the student as log-probs and `reduction='batchmean'`. Dividing by T softens both distributions so the teacher's small probabilities on similar classes carry signal, and the T^2 factor brings the soft term's gradients back to the same scale as the hard term's.

Adam with lr 3e-4 and weight decay 1e-5, gradient norm clipped at 10, batch size 64, 30 epochs (the template has 80, see the notes below).

## Results

|                          | Parameters | Valid accuracy |
| ------------------------ | ---------- | -------------- |
| Teacher (ResNet-18)      | ~11.2M     | 0.7955         |
| Student, epoch 30        | 87,907     | **0.5394**     |
| Student, best (epoch 27) | 87,907     | 0.5591         |

| Epoch | Train loss | Train acc | Valid loss | Valid acc |
| ----- | ---------- | --------- | ---------- | --------- |
| 1     | 5.4393     | 0.3125    | 5.0858     | 0.2364    |
| 5     | 4.5146     | 0.4627    | 4.5142     | 0.3545    |
| 10    | 4.0554     | 0.5237    | 4.2845     | 0.4197    |
| 15    | 3.7717     | 0.5591    | 3.8619     | 0.4667    |
| 20    | 3.5899     | 0.5849    | 3.5077     | 0.4955    |
| 25    | 3.4716     | 0.6002    | 4.0273     | 0.4955    |
| 27    | 3.3964     | 0.6096    | 3.3380     | **0.5591** |
| 30    | 3.3553     | 0.6209    | 3.5186     | 0.5394    |

Train accuracy and loss improve every few epochs without stalling, and validation follows them up with a lot of noise. With only 660 validation images, one epoch to the next can swing by 0.07 (0.4955 at epoch 25, 0.5591 at 27, 0.5000 at 29). The train/valid gap ends around 0.08 and isn't widening, so the student is still underfitting. Train accuracy is partly measured against the teacher's pseudo-labels, since two thirds of the training images are pseudo-labeled, so it's not a clean number either.

The student holds about two thirds of the teacher's accuracy at under 1% of its size. That's below the 0.599 simple baseline, but this run only had 30 epochs. An earlier 80-epoch attempt on the same code reached 0.59 to 0.61 validation accuracy around epochs 49 to 53 before the runtime disconnected, so the full schedule would most likely land near that baseline.

`predict.csv` has the epoch-30 predictions for all 3,347 test images. They lean toward classes 4 (694), 2 (600), 9 (486) and 5 (472). The template doesn't give test labels, so there's no test accuracy here.

## Notes on the provided code

- `n_epochs` is 80 in the template. Colab disconnected the runtime twice in the middle of 80-epoch runs (at epochs 55 and 15), and each time the weights were lost, so the final run uses 30 epochs to get through in one piece.
- `num_workers` was 0 in every DataLoader, so all the jpg decoding and augmentation happened on one CPU thread while the GPU waited. The first run took about 65 s per epoch on a T4. I set it to 2 (the T4 runtime only has 2 CPUs) in the train, validation and pseudo-label loaders, which brought it down to 24 to 44 s depending on the runtime. Nothing else about the loaders changed.
- The pseudo-labels come from `unlabeled_set`, which uses `train_tfm`, so the teacher labels a randomly augmented view of each image once. The labels are fixed after that, even though the student later sees different augmentations of the same images.
- `criterion` is defined but never called. The training loop uses `loss_fn_kd`, which computes its own cross entropy.
- `valid_loader` has `shuffle=True`. It doesn't change the accuracy, just the order.
- `models.resnet18(pretrained=False)` prints deprecation warnings. They don't break anything.
- The student is only kept at its final epoch. There's no checkpointing of the best validation epoch, so `predict.csv` comes from the epoch-80 weights.

## Files

| File                | Contents                                                                    |
| ------------------- | --------------------------------------------------------------------------- |
| `HW7.ipynb`         | The notebook with outputs. Summary and AI use disclosure are at the bottom. |
| `predict.csv`       | Student predictions for the 3,347 test images (`Id,Category`).              |
| `ResNet_food11.pth` | The provided teacher weights (43 MB).                                       |

The dataset (`food-11.zip`) is not committed. It's 963 MB, over GitHub's 100 MB file limit, so it stays in Google Drive and is listed in `.gitignore`.

## Running it

Open in Colab on a GPU runtime (this run used a T4). Put `food-11.zip` and `ResNet_food11.pth` in the same Drive folder and update the two paths in the unzip cell, then Run All. It unzips to `/content/food-11`, which is what the dataset cell expects. Pseudo-labeling takes 15 to 30 s and the 30 training epochs took about 22 minutes on the T4 (about 44 s per epoch on that runtime). Keep the tab in the foreground while it trains, since a backgrounded tab or a sleeping laptop is what disconnected the earlier runs.

No seed is set, so the augmentations, pseudo-labels and shuffling change between runs and the accuracies will move a few points. Validation is only 660 images, so it already swings by several points from one epoch to the next.

## AI Use Disclosure

Anthropic Claude Code was utilized on this assignment. Its contributions were: assistance with the explanatory comments next to updated TODO items and the summary. All code outside the TODO markers is the provided template, unmodified apart from the Google Drive mount and dataset unzip cells, the `num_workers` setting on the DataLoaders, and `n_epochs`. I personally modified, reviewed, and ran the final notebook.

## Documentation Referenced

Data loading:

- `torchvision.transforms` (`RandomHorizontalFlip`, `RandomRotation`, `RandomCrop`, `ToTensor`) - https://pytorch.org/vision/stable/transforms.html
- `torchvision.datasets.DatasetFolder` - https://pytorch.org/vision/stable/generated/torchvision.datasets.DatasetFolder.html
- `torch.utils.data` (`DataLoader`, `ConcatDataset`, `num_workers`) - https://pytorch.org/docs/stable/data.html

Model:

- `nn.Conv2d` - https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- `nn.BatchNorm2d` - https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
- `nn.AdaptiveAvgPool2d` - https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html
- `nn.Linear` - https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
- `torchvision.models.resnet18` - https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
- torchsummary - https://github.com/sksq96/pytorch-summary
- MobileNets (depthwise separable convolutions) - https://arxiv.org/abs/1704.04861

Loss and training:

- `F.cross_entropy` - https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html
- `F.kl_div` - https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html
- `F.log_softmax` - https://pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html
- `F.softmax` - https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html
- `nn.CrossEntropyLoss` - https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- `torch.optim.Adam` - https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
- `nn.utils.clip_grad_norm_` - https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
- `torch.no_grad` - https://pytorch.org/docs/stable/generated/torch.no_grad.html
- `nn.Module.train` / `nn.Module.eval` - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
- Distilling the Knowledge in a Neural Network - https://arxiv.org/abs/1503.02531
