# ---
# title: "A Comparison of Multilayer Perceptron and VGG-16 Models for Fashion Image Classification"
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Introduction

# %% [markdown]
# This notebook develops three models for image classification on a dataset containing images of clothes and accessories
# and was written as part of a university assignment on machine learning.
# Unfortunately, the original source of the dataset is unknown — I and other students were only given a Google Drive link to the dataset.
#
# The three models are as follows:
#
# - A multilayer perceptron (MLP)
#
# - A VGG-16 implemented from scratch
#
# - A fine-tuned VGG-16 from PyTorch pre-trained on the [ImageNet](https://www.image-net.org/) dataset
#
# All models were implemented in PyTorch.
#
# The development of the models followed a typical machine learning pipeline:
#
# 1. **Data analysis:** Quick exploration of the dataset to understand its structure, class distribution and potential issues such as class imbalance.
#
# 1. **Data preprocessing:** Preparation of the data by dividing the dataset into train, validation and test splits, normalising pixel values, label encoding the target feature and defining an undersampling procedure.
#
# 1. **Hyperparameter tuning:** Tuning of hyperparameters like learning rate, batch size and number of layers to optimise each model’s performance. Hyperparameter tuning was achieved using [Optuna](https://optuna.org/). Each model has different hyperparameters to tune — these will be discussed more extensively later.
#
# 1. **Training:** Training of the models on the train split using the best set of hyperparameters determined by the hyperparameter tuning procedure.
#
# 1. **Evaluation:** Evaluation of the models on the test split using confusion matrices and metrics like accuracy and F1-score. Additionally, we will analyse each model's loss curves using Tensorboard and then compare the models' performance.
#
# Note that this notebook will take *several hours* to run even with a GPU.

# %% [markdown]
# ## Preamble

# %% [markdown]
# We'll start by installing additional libraries we need:

# %%
# For downloading the dataset from Google Drive.
# !pip install gdown

# %% [markdown]
# On Google Colab, Optuna is not installed, so we need to install it. However, on Kaggle, Optuna is installed by default, so we make the installation conditional.

# %%
import importlib.util

spec = importlib.util.find_spec("optuna")
if spec is None:
    import subprocess
    subprocess.run(["pip", "install", "optuna"])

# %% [markdown]
# Now, we can import all the libraries and modules we need for the rest of the notebook:

# %%
import itertools
import gc
import math
import os
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional, Sequence

import gdown
import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
import sklearn
import sklearn.model_selection
import torch
import torchvision
import torchvision.transforms.v2
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import LabelEncoder
import tensorboard
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# %% [markdown]
# We'll define some constants as configuration for this notebook. Feel free to adjust these constants to reduce the running time of the notebook:

# %%
# Variables for downloading and locating the dataset.
DATASET_GDRIVE_URL = "https://drive.google.com/file/d/1nWRm-Npq_QE0j_sHyVVxVEx2Rb0Lc1zU/view"
DATASET_OUT_PATH = "data.zip"
DATASET_ROOT_PATH = "data/"

# Automatically determine the device to use for PyTorch.
# On Google Colab and Kaggle, make sure to enable an accelerator to use CUDA.
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Default batch size for training and evaluation.
# Note that the batch size is still a hyperparameter that will be tuned;
# this is just the default.
DEFAULT_BATCH_SIZE: int = 32

# Number of folds to use for hyperparameter tuning.
N_FOLDS: int = 3

# Number of trials for hyperparameter tuning.
# Note: Be careful not to reduce this too much as you can accidentally cause
#       all trials to be pruned! If you want to set this to a low number,
#       you should disable trial pruning by setting `PRUNE_TRIALS` (below) to `False`.
N_TRIALS: int = 25

# Max number of epochs for hyperparameter tuning.
N_EPOCHS_TUNE: int = 30

# Max number of epochs for the real training.
N_EPOCHS_TRAIN: int = 30

# Whether to prune trials during hyperparameter tuning.
PRUNE_TRIALS: bool = True

# %% [markdown]
# ## Data
#
# This section includes the downloading of the dataset, preliminary analysis and preprocessing steps.

# %% [markdown]
# ### Downloading the Dataset
#
# First, we need to download the dataset. We'll download it directly from Google Drive using the `gdown` library:

# %%
gdown.cached_download(DATASET_GDRIVE_URL, DATASET_OUT_PATH, fuzzy=True, postprocess=gdown.extractall)


# %% [markdown]
# To verify that the dataset was downloaded and extracted, we'll use the shell's `ls` command to list the files in the `data` directory.
# If the download fails for whatever reason (e.g., Google Drive link is down), please manually download and extract the dataset and set the `DATASET_ROOT_PATH` constant to the root directory of the dataset.
# The following command should show `test  train  valid` if the dataset was successfully downloaded and extracted:

# %%
# !ls data

# %% [markdown]
# In case the Google Drive link is unavailable,
# you may also find the dataset from this notebook's GitHub repository
# [here](https://github.com/dixslyf/mlp-vgg16-fashion/releases/latest/download/data.zip).

# %% [markdown]
# ### Custom `Dataset`

# %% [markdown]
# To load the images from the dataset, we need to define a custom PyTorch `Dataset`:

# %%
class ClothesDataset(Dataset):
    """
    A PyTorch `Dataset` class for loading the clothes and accessories dataset.

    Attributes:
        input_transform (Optional[Callable[[Tensor], Tensor]]): Transformation applied to input images.
        target_transform (Optional[Callable[[str], Any]]): Transformation applied to target labels.
        categories (set[str]): The unique categories (subdirectory names) present in the dataset.
    """

    def __init__(
        self,
        img_dirs: str | Iterable[str],
        input_transform: Optional[Callable[[Tensor], Tensor]] = None,
        target_transform: Optional[Callable[[str], Any]] = None,
    ):
        """
        Initialises the dataset with one or more directories containing the images,
        and with optional input and target transformations.

        The given directories are assumed to be organised such that images are put into subdirectories
        that correspond to the categories.

        Args:
            img_dirs (str | Iterable[str]): Directory path(s) containing the images.
            input_transform (Optional[Callable[[Tensor], Tensor]]): Transformation applied to input images.
            target_transform (Optional[Callable[[str], Any]]): Transformation applied to target labels.
        """
        self._img_dirs: list[str] = (
            [img_dirs] if type(img_dirs) is str else list(img_dirs)
        )

        self.input_transform = input_transform
        self.target_transform = target_transform

        # A list of pairs that maps the path of an image (path includes each `img_dir`)
        # to its category. We're intentionally not using a dictionary because we need
        # to access elements by index, which is not possible with a dictionary.
        #
        # To be lazy and avoid I/O in `__init__()`, this will only be populated on
        # the first call to `__getitem__()` or `__len__()`.
        self._img_label_map: Optional[list[tuple[str, str]]] = None

    @property
    def categories(self) -> set[str]:
        """
        Returns the unique categories in the dataset.

        Returns:
            set[str]: Set of unique categories in the dataset.
        """
        if self._img_label_map is None:
            self._init_img_label_map()

        return self._categories

    def targets(self, indices: Optional[Iterable[int]] = None) -> Iterable:
        """
        Returns the target labels for the specified indices or for all samples if indices are not provided.

        Args:
            indices (Optional[Iterable[int]]): Indices of the target labels to retrieve.

        Yields:
            Iterable: Transformed target labels or raw labels if no transformation is applied.
        """
        if indices is None:
            it = self._img_label_map
        else:
            it = (self._img_label_map[idx] for idx in indices)

        for path, cat in it:
            if self.target_transform is None:
                yield cat
            else:
                yield self.target_transform(cat)

    def _init_img_label_map(self):
        """
        Initialises the image-label mapping by listing all images and their corresponding categories
        from the specified directories. The subdirectory names represent the categories.
        """
        cats = {
            cat
            for img_dir in self._img_dirs
            for cat in os.listdir(img_dir)
            if os.path.isdir(os.path.join(img_dir, cat))
        }

        self._categories = cats
        self._img_label_map = []
        for img_dir, cat in itertools.product(self._img_dirs, cats):
            cat_dir = os.path.join(img_dir, cat)
            if not os.path.isdir(cat_dir):
                continue

            for img in os.listdir(cat_dir):
                img_path = os.path.join(cat_dir, img)
                self._img_label_map.append((img_path, cat))

    def __getitem__(self, idx: int) -> tuple[Tensor, str]:
        """
        Returns the image and corresponding label at the specified index.

        Args:
            idx (int): Index of the image-label pair to retrieve.

        Returns:
            tuple[Tensor, str]: A pair of the image tensor and its corresponding label, with optional
            transformations applied.
        """
        if self._img_label_map is None:
            self._init_img_label_map()

        img_path, label = self._img_label_map[idx]
        img = torchvision.io.read_image(img_path)

        if self.input_transform is not None:
            img = self.input_transform(img)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        if self._img_label_map is None:
            self._init_img_label_map()

        return len(self._img_label_map)


# %% [markdown]
# I also define a custom `Subset` to make it easier to split the dataset and access the target labels of a subset:

# %%
class ClothesSubset(torch.utils.data.Subset):
    """
    A subset of a `ClothesDataset` that maintains access to the dataset's target labels.

    Args:
        dataset (ClothesDataset): The `ClothesDataset` from which the subset is drawn.
        indices (Sequence[int]): The indices of the elements in the original dataset to include in the subset.
    """

    def __init__(
        self, dataset: torch.utils.data.Dataset, indices: Sequence[int]
    ) -> None:
        """
        Initialises the subset with the given dataset and a sequence of indices.

        Args:
            dataset (Dataset): The `ClothesDataset` to create a subset from.
            indices (Sequence[int]): Indices specifying which elements of the dataset
                should be included in this subset.
        """
        super().__init__(dataset, indices)

    def targets(self, indices: Optional[Iterable[int]] = None):
        """
        Returns the target labels for the specified indices, or for the entire subset if no indices are provided.

        This method is designed to work specifically with `ClothesDataset` and also handles
        the case where the subset is a subset of another subset.

        Args:
            indices (Optional[Iterable[int]]): Indices of the target labels to retrieve from the subset.
                If no indices are provided, returns the labels for all items in the subset.

        Yields:
            The target labels corresponding to the specified indices in the subset.
        """
        if indices is None:
            for target in self.dataset.targets(self.indices):
                yield target
        else:
            # In case we have a subset of a subset, use a set to handle index lookups efficiently.
            indices_set = set(indices)
            for idx, target in enumerate(self.dataset.targets(self.indices)):
                if idx in indices_set:
                    yield target



# %% [markdown]
# We can now load the dataset and check various statistics about it, such as the length and categories.
#
# **Note:** I intentionally ignore the test split because it is unlabelled and only has 8 samples, which makes it unuseful for evaluation. In the code cell below, I combine the train and validation splits together — later in the notebook, I perform my own train-validation-test split on this combined dataset.

# %%
ds = ClothesDataset((os.path.join(DATASET_ROOT_PATH, split) for split in ("train", "valid")))

# %%
print(f"Total number of samples: {len(ds)}")
print(f"Categories: {ds.categories}")

# %% [markdown]
# The dataset has 3849 samples, which is relatively small compared to other image datasets.

# %% [markdown]
# ### Preliminary Analysis

# %% [markdown]
# To start, I'll plot some of the images to have a feel of what the images look like:

# %%
rows = 4
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
for idx in range(rows * cols):
    img, label = ds[idx * 200] # Multiply by 200 so that we don't just see shorts.
    ax = axes[idx // cols, idx % cols]
    ax.imshow(img.permute(1, 2, 0))
    ax.set_title(f"Label: {label}")
plt.tight_layout()
plt.show()

# %% [markdown]
# Some observations:
#
# - The images are *coloured*, not grayscale.
#
# - The images appear to all be the same size (of course, this is just a small sample size of 16, so there's no guarantee).
#
# The dimensions of each image are likely 432 by 300:

# %%
# Dimensions of the first image.
print(f"Dimensions: {ds[0][0].shape}")

# %% [markdown]
# In case there are any images that have different dimensions (which would cause problems when feeding them to the models), we'll resize all images to be the same size later. 432 by 300 is quite large and could potentially exhaust GPU memory during hyperparameter tuning and training, so the images will be resized to smaller dimensions (the dimensions will be tuned during hyperparameter tuning).

# %% [markdown]
# Next, are the pixel values normalised? They do not seem to be. Instead, they seem to be in the typical range of 0 to 255. We can gain some assurance from the documentation for PyTorch's [`read_image()`](https://pytorch.org/vision/main/generated/torchvision.io.read_image.html): "The values of the output tensor are in uint8 in [0, 255] for most cases". We will need to normalise the pixel values later.

# %%
print(f"Pixel values of the first image:\n{ds[0][0]}")

# %% [markdown]
# Unfortunately, looking at the distribution of the classes, there is an imbalanced class problem, where the "shoes" and "tees" classes have a much higher number of samples than the others. This could affect the performance of the models and make them biased towards "shoes" and "tees". Thankfully, the other classes are fairly balanced, so we can easily resolve this imbalance by undersampling the "shoes" and "tees" classes. However, this would lead to less training data, so I will also try out various data augmentations (e.g., horizontal flips, colour jitter) later during hyperparameter tuning to introduce more varied data.

# %%
sns.displot(list(ds.targets()), height=8, aspect=10/8)


# %% [markdown]
# ### Preprocessing

# %% [markdown]
# #### Normalisation and Label Encoding
#
# Now, to prepare the data for use, we'll first normalise the pixel values. The raw images have pixel values that range from 0 to 255, so we can normalise them by simply dividing each value by 255:

# %%
# Normalise.
def input_transform(X):
  return X / 255

ds.input_transform = input_transform

# %% [markdown]
# To confirm that the pixel values were normalised correctly, we'll inspect the pixel values for one of the images:

# %%
print(f"Pixel values:\n{ds[0][0]}")

# %% [markdown]
# Next, because our models can only understand numbers, we need to encode the labels (i.e., the categories) into numbers. I use `scikit-learn`'s `LabelEncoder` for this:

# %%
target_le = LabelEncoder()
target_le.fit(list(ds.categories))

def target_transform(y):
  encoded = target_le.transform([y])
  return encoded[0]

ds.target_transform = target_transform

# %% [markdown]
# We'll check the label encoder's classes and ensure that we have 8 unique integers for the labels:

# %%
target_le.classes_

# %%
set(ds.targets())

# %% [markdown]
# #### Splitting the Dataset
#
# As briefly mentioned earlier, I do not use the test split provided by the dataset as it only contains 8 samples and does not contain labels. I instead perform my own train-validation-test split with a ratio of `70:15:15`.
#
# The test set will be used for the final evaluation of the models — this split will **not** be touched at all during hyperparameter tuning and training to avoid leaking data from it, which would lead to overly optimistic results.
#
# The validation set will be used to estimate the validation loss during training so that we can compare it to the training loss. This validation loss will also be used to stop training early when the generalisation loss stops improving to prevent the model from overfitting.
#
# The training set will, of course, be used for training. However, it will also be used for hyperparameter tuning. Note that the validation set will not be used for hyperparameter tuning. I prefer to use K-fold cross-validation to further split the training set into train and validation folds during hyperparameter tuning to get a more unbiased estimate of model performance — if I re-use the validation set for hyperparameter tuning, there is a risk of overfitting to it, especially since the dataset is rather small.

# %%
indices_train, indices_other = sklearn.model_selection.train_test_split(
    np.arange(len(ds)),
    train_size=0.7,
    random_state=0,
    shuffle=True,
    stratify=np.fromiter(ds.targets(), dtype=int),
)

indices_val, indices_test = sklearn.model_selection.train_test_split(
    indices_other,
    train_size=0.5,
    random_state=0,
    shuffle=True,
    stratify=np.fromiter(ds.targets(indices_other), dtype=int),
)

ds_train = ClothesSubset(ds, indices_train)
ds_val = ClothesSubset(ds, indices_val)
ds_test = ClothesSubset(ds, indices_test)

# %%
print(f"Size of train split: {len(ds_train)}")

# %%
print(f"Size of validation split: {len(ds_val)}")

# %%
print(f"Size of test split: {len(ds_test)}")


# %% [markdown]
# #### Undersampling
#
# As mentioned earlier, the dataset has a class imbalance problem, where "shoes" and "tees" have many more samples than the other classes. To combat this, I will use undersampling, by which "shoes" and "tees" samples have a lower chance of being sampled. We'll create a `WeightedRandomSampler` to give smaller weights to the "shoes" and "tee" classes. This sampler will be passed to a `DataLoader` later during training and hyperparameter tuning.
#
# **Note:** The weights are determined **only from the train split**. Otherwise, we would be leaking data from the test (and validation) split. We also need to be careful during hyperparameter tuning to not use the weights determined by the whole train split since each iteration of K-fold cross-validation will have its own train set. Hence, I define a function to create an undersampler from specified training labels so that each iteration of K-fold cross-validation can pass its own train split and create its own undersampler:

# %%
def make_undersampler(labels_train: torch.Tensor):
    # Count the number of samples in each class
    # of the train split.
    class_counts = torch.bincount(labels_train)

    # Use the inverses of the class counts as the weights
    # so that "shoes" and "tees" have lower weights.
    class_weights = 1. / class_counts.float()

    # Assign weights to each sample.
    sample_weights = class_weights[labels_train]

    return torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

undersampler = make_undersampler(torch.tensor(np.fromiter(ds_train.targets(), dtype=int)))


# %% [markdown]
# The undersampler created here (`undersampler`) is **only for the real training** process that will use the entire train split. Each iteration of K-fold cross-validation during hyperparameter tuning will call the `make_undersampler()` function with its own train set to create its own undersampler.

# %% [markdown]
# ## Model Definitions
#
# This section contains definitions for each model (MLP, VGG-16 and pretrained VGG-16).
#
# Note that the MLP and VGG-16 that I implemented from scratch use the *lazy* variants of PyTorch's neural network modules, which defer the initialisation of layer parameters. The rationale is so that (a) I do not need to manually calculate the dimensions of the inputs and outputs for each layer, which is error-prone, and (b) the model will automatically detect the shape of the inputs without having to specify it during creation of the model. These make the model definitions much more readable and easier to prototype with. The trade-offs are that there is a slight overhead for initialising the layers during the first forward pass, and shape mismatch errors are detected later instead of during model initialisation.
#
# **Note:** For visualisations of the architectures of the final models, please see the Training and Evaluation section. The model architectures defined here are "semi-fixed" — some architectural decisions are treated as hyperparameters that need to be tuned, so the models are not instantiated until then.

# %% [markdown]
# ### Multilayer Perceptron
#
# Since it is not obvious what MLP architecture would work best for the dataset, the `MultilayerPercecptron` class below provides a semi-fixed architecture and allows specifying the activation function to use and the number of layers and their corresponding sizes. These are treated as hyperparameters to be tuned.
#
# This MLP model adds a batch normalisation layer after each linear layer to stabilise training, reduce dependency on input scaling and serve as a regularisation effect. Ideally, whether to use batch normalisation or not would have been a hyperparameter to tune, but I decided not to make it configurable to reduce the hyperparameter search space and time.
#
# Other architectural options I *could* have explored include: use of dropout layers, use of L2 regularisation and which weight initialisation technique to use. However, these were left unexplored to reduce the hyperparameter search space and time. MLPs are not well-suited for image tasks anyway (they do not consider the spacial structure of the pixels — you could randomly rearrange the pixels and the MLP would still yield the same performance results), so trying to explore this architectural space will likely be unfruitful.
#
# **Note:** Flattening of the input images is done outside of the `MultilayerPerceptron` class so that the class can be reused for other inputs that do not require flattening.

# %%
class MultilayerPerceptron(torch.nn.Module):
    def __init__(
        self,
        layer_sizes: Iterable[int],
        activation_constructor: Callable[[], torch.nn.Module],
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                module
                for out_features in layer_sizes[:-1]
                for module in (
                    torch.nn.LazyLinear(out_features),
                    torch.nn.LazyBatchNorm1d(),
                    activation_constructor(),
                )
            ]
        )
        self.layers.append(torch.nn.LazyLinear(layer_sizes[-1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# %% [markdown]
# **Note:** For visualisation of the architecture of the final model, please see the Training and Evaluation section.

# %% [markdown]
# ### VGG-16
#
# The VGG-16 model follows the original VGG-16 architecture (configuration D) proposed in [Very Deep Convultional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556):
#
# ![vgg16-architecture.png](attachment:070ade36-eeb9-401c-b3d8-565d89047370.png)
#
# Although not depicted in the image above, the authors mention in the paper that dropout is applied to the first two linear layers with a probability of 0.5.
#
# While my implementation follows the original architecture closely, some minor adjustments / additions were made:
#
# - When the model is instantiated, you can specify whether to use batch normalisation after each convolutional layer. This is a hyperparameter that will be tuned. During my initial tests, I found that the model struggled to learn the features of the input images without batch normalisation (the loss very quickly plateaued). This could be because VGG-16 is a considerably deep model and thus experiences the vanishing / exploding gradient problem — batch normalisation may help stabilise the gradients.
#
# - The VGG-16 model provided by PyTorch adds an additional adaptive average pooling layer before flattening the last feature map for the linear layers, presumably to allow the network to handle variable input image sizes by ensuring that the dimensions of the feature map before the linear layers is always the same. I replicate that addition in my implementation for the same reason, and to match my implementation closer with PyTorch's.

# %% [markdown]
# To reduce code repetition, I define a function to create a single VGG block. This function is loosely based on a similar function from [d2l.ai](https://d2l.ai/chapter_convolutional-modern/vgg.html).

# %%
def vgg_block(n_conv_layers: int, out_channels: int, batch_norm: bool = True):
    conv_layers = (
        module
        for _ in range(n_conv_layers)
        for module in (
            torch.nn.LazyConv2d(out_channels, kernel_size=3, padding=1),
            torch.nn.LazyBatchNorm2d() if batch_norm else torch.nn.Identity(),
            torch.nn.ReLU(inplace=True),
        )
    )

    return torch.nn.Sequential(
        *conv_layers, torch.nn.MaxPool2d(kernel_size=2, stride=2)
    )


# %% [markdown]
# The implementation of VGG-16 is then as follows:

# %%
class VGG16(torch.nn.Module):
    def __init__(self, out_classes: int, batch_norm: bool = True):
        super().__init__()
        self.convs = torch.nn.Sequential(
            vgg_block(n_conv_layers=2, out_channels=64, batch_norm=batch_norm),
            vgg_block(n_conv_layers=2, out_channels=128, batch_norm=batch_norm),
            vgg_block(n_conv_layers=3, out_channels=256, batch_norm=batch_norm),
            vgg_block(n_conv_layers=3, out_channels=512, batch_norm=batch_norm),
            vgg_block(n_conv_layers=3, out_channels=512, batch_norm=batch_norm),
        )
        self.avg_pool = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((7, 7)),
            torch.nn.Flatten(),
        )
        self.fcs = torch.nn.Sequential(
            *(
                module
                for _ in range(2)
                for module in (
                    torch.nn.LazyLinear(out_features=4096),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Dropout(p=0.5),
                )
            ),
            torch.nn.LazyLinear(out_features=out_classes)
        )

    def forward(self, X):
        X = self.convs(X)
        X = self.avg_pool(X)
        return self.fcs(X)


# %% [markdown]
# **Note:** For visualisation of the architecture of the final model, please see the Training and Evaluation section.

# %% [markdown]
# ### VGG-16 Pretrained
#
# The architecture of VGG-16 has already been discussed in the previous section.
#
# For the pretrained VGG-16 model, we still need to perform a bit of surgery:
#
# - The pretrained VGG-16 was trained to perform classification on ImageNet, which has 1000 different classes. We need to replace the last linear layer with one whose number of outputs matches the number of classes in our dataset.
#
# - The pretrained VGG-16 expects inputs with 3 channels, so the first convolutional layer expects that. Since I want to experiment with different numbers of input channels (e.g., grayscale images), I replace the first layer with a lazy convolutional layer (lazy means that the layer will automatically determine the number of input channels based on the first input).
#
# Performing these adjustments does mean that we lose some of the pretrained weights. However, most of the model still retains the pretrained weights, so it should not be an issue.
#
# The following cell shows the untouched architecture of the pretrained VGG-16 model:

# %%
torchvision.models.vgg16(weights="DEFAULT", progress=True)


# %% [markdown]
# The function below performs the surgery operations described above:

# %%
def make_vgg16_pretrained(weights: Optional[str] = "DEFAULT", freeze_conv: bool = False) -> torchvision.models.vgg.VGG:
    vgg16_pretrained = torchvision.models.vgg16(weights=weights, progress=True)

    if freeze_conv:
        # Freeze the convolutional layers.
        for param in vgg16_pretrained.features.parameters():
            param.requires_grad = False

    vgg16_pretrained.features[0] = torch.nn.LazyConv2d(64, kernel_size=3, stride=1, padding=1)
    vgg16_pretrained.classifier[6] = torch.nn.LazyLinear(10)
    return vgg16_pretrained


# %% [markdown]
# We can verify that the function works as intended by manually inspecting the architecture of the output model:

# %%
make_vgg16_pretrained()


# %% [markdown]
# ## Definitions for Training and Evaluation
#
# Before proceeding further, we'll define the necessary functions and classes needed ahead of time for hyperparameter tuning, training and evaluation so that the notebook is more readable. I suggest reading the documentation for each class to get a better understanding of their capabilities.
#
# First, I define classes for evaluation:

# %%
@dataclass
class EvaluationResults:
    """
    Stores the evaluation results of a model.

    Attributes:
        mean_loss (float): The mean loss across the evaluation dataset.
        classification_report (dict): A dictionary containing precision, recall,
            f1-score, and support for each class. See scikit-learn's `classification_report()`.
        classification_report_str (str): A string representation of the classification report.
            See scikit-learn's `classification_report()`.
        confusion_matrix (numpy.ndarray): A confusion matrix indicating the performance
            of the model on the evaluation dataset. See scikit-learn's `confusion_matrix()`.
        predictions (numpy.ndarray): An array of predicted labels for the evaluation dataset.
    """

    mean_loss: float
    classification_report: dict
    classification_report_str: str
    confusion_matrix: np.ndarray
    predictions: np.ndarray


class Evaluator:
    """
    Evaluator for evaluating machine learning models on a dataset.

    Attributes:
        batch_size (int): The size of the batches used during evaluation. Default is
            set to `DEFAULT_BATCH_SIZE`.
        desc (str): A short description label for the evaluation process used for
            progress tracking with tqdm. Defaults to "test".
        zero_division (str or int): Controls the behavior of metrics when there is
            a zero division. Can be "warn", 0 or 1. Defaults to "warn".
        device (str): The device on which to perform evaluation.
    """

    def __init__(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        desc: str = "test",
        zero_division="warn",
        device: str = DEVICE,
    ):
        """
        Initialises the Evaluator with the specified parameters.

        Args:
            batch_size (int): The size of the batches to use during evaluation.
            desc (str): A description for the evaluation process.
            zero_division (str or int): Specifies the behavior when there is a zero division.
            device (str): The device for evaluation.
        """
        self.batch_size: int = batch_size
        self.desc = desc
        self.zero_division = zero_division
        self.device: str = device

    def evaluate(
        self,
        model: torch.nn.Module,
        ds_test: Dataset,
        loss_fn: torch.nn.Module,
        transform: Optional[torchvision.transforms.v2.Transform] = None,
        target_labels: Optional[Iterable[str]] = None,
    ) -> EvaluationResults:
        """
        Evaluates the specified model on the given test dataset.

        Args:
            model (torch.nn.Module): The model to be evaluated.
            ds_test (Dataset): The test dataset to evaluate the model on.
            loss_fn (torch.nn.Module): The loss function to compute the loss.
            transform (Optional[torchvision.transforms.v2.Transform]): Optional
                transformations to apply to the input data before passing to the model.
            target_labels (Optional[Iterable[str]]): Optional list of target labels
                for classification report.

        Returns:
            EvaluationResults: Contains the evaluation metrics and predictions.
        """
        model.eval()

        dl_test = DataLoader(ds_test, batch_size=self.batch_size, shuffle=True)

        # Sum of the losses from each batch for the current epoch.
        # Used to calculate the mean loss for the current epoch.
        total_loss: float = 0
        with torch.no_grad(), tqdm(total=len(dl_test.dataset)) as pbar:
            # Tensor of all predictions.
            truth_all: torch.Tensor = torch.tensor([], dtype=int)
            pred_all: torch.Tensor = torch.tensor([], dtype=int)

            for batch, (X, y) in enumerate(dl_test):
                pbar.set_description(f"{self.desc} [batch {batch + 1}]")

                truth_all = torch.cat((truth_all, y))

                X = X.to(self.device)
                y = y.to(self.device)

                if transform is not None:
                    X = transform(X)

                unnormalised_logits = model(X)
                pred_indices = unnormalised_logits.argmax(dim=1)
                pred_all = torch.cat((pred_all, pred_indices.cpu()))

                loss = loss_fn(unnormalised_logits, y)

                del X
                del y

                total_loss += loss.item()

                pbar.update(dl_test.batch_size)
                pbar.set_postfix({"loss": loss.item()})

            gc.collect()
            torch.cuda.empty_cache()

            mean_loss: float = total_loss / len(dl_test)

            report_kwargs = {"output_dict": True}
            if target_labels is not None:
                report_kwargs["target_names"] = target_labels
            report = sklearn.metrics.classification_report(
                truth_all.cpu(),  # sklearn can only work with CPU tensors.
                pred_all.cpu(),
                zero_division=self.zero_division,
                **report_kwargs,
            )

            pbar.set_postfix(
                {"mean loss": mean_loss, "macro f1": report["macro avg"]["f1-score"]}
            )

        report_str_kwargs = {}
        if target_labels is not None:
            report_str_kwargs["target_names"] = target_labels

        report_str = sklearn.metrics.classification_report(
            truth_all.cpu(),  # sklearn can only work with CPU tensors.
            pred_all.cpu(),
            zero_division=self.zero_division,
            **report_str_kwargs,
        )

        confusion_matrix = sklearn.metrics.confusion_matrix(
            truth_all.cpu(),  # sklearn can only work with CPU tensors.
            pred_all.cpu(),
        )

        return EvaluationResults(
            mean_loss,
            report,
            report_str,
            confusion_matrix,
            pred_all.detach().cpu().numpy(),
        )


# %% [markdown]
# For training (including training during hyperparameter tuning), I use early stopping based on the validation loss to prevent overfitting and save time if the model's performance starts to plateau. If the observed validation loss does not improve after a specified number of consecutive epochs (patience), the training process is stopped. The delta parameter specifies the minimum increase in validation loss for it to be counted as a lack of improvement.

# %%
class ValidationLossEarlyStopper:
    """
    Implements early stopping based on validation loss to prevent overfitting
    during model training.

    The training process is halted when no improvement in validation loss is observed
    for a specified number of consecutive epochs (patience).
    """

    def __init__(self, patience: int = 1, delta: float = 0.0):
        """
        Initialises the early stopper with the specified patience and delta values.

        Args:
            patience (int): Number of epochs to wait for an improvement before stopping.
                Default is 1.
            delta (float): The minimum increase in validation loss required to count as a
                deterioration. Default is 0.0.
        """
        self._patience: int = patience
        self._delta: float = delta
        self._lapse_count: int = 0
        self._min_val_loss: float = math.inf

    def should_stop(self, val_loss: float) -> bool:
        """
        Checks whether training should stop based on the given validation loss.

        Args:
            val_loss (float): The current validation loss for the current epoch.

        Returns:
            bool: True if the training process should stop due to lack of improvement in
            validation loss; otherwise, False.
        """
        if val_loss < self._min_val_loss:
            self._min_val_loss = val_loss
            self._lapse_count = 0
        elif val_loss > (self._min_val_loss + self._delta):
            # Validation loss went up by more than delta from the min.
            self._lapse_count += 1

            # If the number of times the validation loss got worse
            # exceeds the patience level, then we should stop.
            if self._lapse_count >= self._patience:
                return True
        return False


# %% [markdown]
# Finally, for the actual training of the models, I define a `Trainer` class. Notably, the `Trainer` class allows specifying transforms for data augmentation — this will be used to mitigate the class imbalance problem.

# %%
ParamsT = Iterable[torch.Tensor] | Iterable[dict[str, Any]]


class Trainer:
    """
    Trains a PyTorch model over multiple epochs.

    Attributes:
        epochs (int): The total number of epochs to train the model.
        batch_size (int): The number of samples per gradient update.
        augment_transform (Optional[torchvision.transforms.v2.Transform]): Transformations to apply
            for data augmentation during training.
        train_loss_hook (Optional[Callable[[float, int], None]]): A callback function that is called
            after each training epoch to log the training loss.
        val_results_hook (Optional[Callable[[EvaluationResults, int], None]]): A callback function that
            is called after each validation epoch to log validation results.
        device (str): The device to run the model on.
    """

    def __init__(
        self,
        epochs: int,
        batch_size: int = DEFAULT_BATCH_SIZE,
        augment_transform: Optional[torchvision.transforms.v2.Transform] = None,
        train_loss_hook: Optional[Callable[[float, int], None]] = None,
        val_results_hook: Optional[Callable[[EvaluationResults, int], None]] = None,
        device: str = DEVICE,
    ):
        """
        Initialises the Trainer with the specified parameters.

        Args:
            epochs (int): The total number of epochs to train the model.
            batch_size (int): The number of samples per gradient update. Default is `DEFAULT_BATCH_SIZE`.
            augment_transform (Optional[torchvision.transforms.v2.Transform]): Transformations to apply
                for data augmentation during training. Default is None.
            train_loss_hook (Optional[Callable[[float, int], None]]): A callback function to log
                training loss after each epoch. Default is None.
            val_results_hook (Optional[Callable[[EvaluationResults, int], None]]): A callback function
                to log validation results after each epoch. Default is None.
            device (str): The device to run the model on. Default is `DEVICE`.
        """
        self.epochs: int = epochs
        self.batch_size: int = batch_size

        self.augment_transform: Optional[torchvision.transforms.v2.Transform] = (
            augment_transform
        )
        self.train_loss_hook: Optional[Callable[[float, int], None]] = train_loss_hook
        self.val_results_hook: Optional[Callable[[EvaluationResults, int], None]] = (
            val_results_hook
        )
        self.device: str = device

    def train(
        self,
        model: torch.nn.Module,
        ds_train: Dataset,
        loss_fn: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        train_sampler: Optional[torch.utils.data.Sampler] = None,
        transform: Optional[torchvision.transforms.v2.Transform] = None,
        ds_val: Optional[Dataset] = None,
        scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,
        early_stop: Optional[ValidationLossEarlyStopper] = None,
    ):
        """
        Trains the given model for a specified number of epochs.

        Args:
            model (torch.nn.Module): The PyTorch model to train.
            ds_train (Dataset): The training dataset.
            loss_fn (torch.nn.Module): The loss function to use for training.
            optimiser (torch.optim.Optimizer): The optimiser for updating model weights.
            train_sampler (Optional[torch.utils.data.Sampler]): An optional sampler for the training dataset.
            transform (Optional[torchvision.transforms.v2.Transform]): Optional transformations to apply
                to the training data before feeding them to the model.
            ds_val (Optional[Dataset]): An optional validation dataset for validating the model.
            scheduler (Optional[torch.optim.lr_scheduler.ReduceLROnPlateau]): Optional learning rate scheduler
                to adjust the learning rate based on validation loss.
            early_stop (Optional[ValidationLossEarlyStopper]): An optional early stopper to terminate training
                if validation loss does not improve.
        """
        model.to(self.device)

        # Create the data loader for the training set.
        # If a sampler was specified, use it.
        dl_train = DataLoader(
            ds_train,
            batch_size=self.batch_size,
            sampler=train_sampler,
        )

        # Training loop.
        for epoch in range(self.epochs):
            train_loss = self._train_one_epoch(
                model,
                dl_train,
                loss_fn,
                optimiser,
                epoch,
                transform=transform,
            )

            if self.train_loss_hook is not None:
                self.train_loss_hook(train_loss, epoch)

            if ds_val is not None:
                val_results = self._validate_one_epoch(
                    model,
                    ds_val,
                    loss_fn,
                    epoch,
                    transform=transform,
                )

                if self.val_results_hook is not None:
                    self.val_results_hook(val_results, epoch)

                if early_stop is not None and early_stop.should_stop(
                    val_results.mean_loss
                ):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

                if scheduler is not None:
                    scheduler.step(val_results.mean_loss)

    def _train_one_epoch(
        self,
        model: torch.nn.Module,
        dl_train: DataLoader,
        loss_fn: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        epoch: int,
        transform: Optional[torchvision.transforms.v2.Transform] = None,
    ):
        """
        Trains the given model for one epoch on the provided dataloader.

        Args:
            model (torch.nn.Module): The model to train.
            dl_train (DataLoader): The dataloader for the training dataset.
            loss_fn (torch.nn.Module): The loss function to calculate the loss.
            optimiser (torch.optim.Optimizer): The optimiser for updating weights.
            epoch (int): The current epoch number.
            transform (Optional[torchvision.transforms.v2.Transform]): Optional transformations to apply
                to the training data before feeding them to the model.

        Returns:
            float: The mean loss for the training epoch.
        """
        model.train()

        # Sum of the losses from each batch for the current epoch.
        # Used to calculate the mean loss for the current epoch.
        total_loss: float = 0

        with tqdm(total=len(dl_train.dataset)) as pbar:
            for batch, (X, y) in enumerate(dl_train):
                pbar.set_description(f"train [epoch {epoch + 1} batch {batch + 1}]")

                X = X.to(self.device)
                y = y.to(self.device)

                if self.augment_transform is not None:
                    X = self.augment_transform(X)

                if transform is not None:
                    X = transform(X)

                pred = model(X)
                loss = loss_fn(pred, y)

                del X
                del y

                # Backpropagation.
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()

                total_loss += loss.item()

                pbar.update(dl_train.batch_size)
                pbar.set_postfix({"train loss": loss.item()})

            mean_loss = total_loss / len(dl_train)
            pbar.set_postfix({"mean train loss": mean_loss})

        gc.collect()
        torch.cuda.empty_cache()

        return mean_loss

    def _validate_one_epoch(
        self,
        model: torch.nn.Module,
        ds_val: Dataset,
        loss_fn: torch.nn.Module,
        epoch: int,
        transform: Optional[torchvision.transforms.v2.Transform] = None,
    ) -> EvaluationResults:
        """
        Validates the model for one epoch on the given validation dataset.

        Args:
            model (torch.nn.Module): The model to validate.
            ds_val (Dataset): The validation dataset.
            loss_fn (torch.nn.Module): The loss function to calculate the loss.
            epoch (int): The current epoch number.
            transform (Optional[torchvision.transforms.v2.Transform]): Optional transformations to apply
                to the validation data before feeding them to the model.

        Returns:
            EvaluationResults: The results of the validation.
        """
        evaluator = Evaluator(
            batch_size=self.batch_size,
            desc="validate",
            zero_division=0,
            device=self.device,
        )

        return evaluator.evaluate(
            model,
            ds_val,
            loss_fn,
            transform=transform,
        )


# %% [markdown]
# Now, to prepare for hyperparameter tuning, I define some convenience functions that each accepts a dictionary of hyperparameters determined by Optuna during hyperparameter tuning:

# %%
def augmentation_transforms_from_params(params: dict) -> Optional[torchvision.transforms.v2.Transform]:
    """
    Returns the augmentation transforms according to the given hyperparameter dictionary.
    """
    transforms = []

    if params["jitter"]:
        transforms.append(
            torchvision.transforms.v2.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
        )

    if params["horizontal_flip"]:
        transforms.append(torchvision.transforms.v2.RandomHorizontalFlip())

    if params["rotation"]:
        transforms.append(torchvision.transforms.v2.RandomRotation(30))

    # Compose does not allow an empty list of transformations.
    if not transforms:
        return None

    return torchvision.transforms.v2.Compose(transforms)


def transforms_from_params(params: dict) -> torchvision.transforms.v2.Transform:
    """
    Returns the preprocessing transformations according to the given hyperparameter dictionary.
    """
    transforms = []
    length = params["resize_length"]
    transforms.append(torchvision.transforms.v2.Resize((length, length)))

    if params["grayscale"]:
        transforms.append(torchvision.transforms.v2.Grayscale())

    return torchvision.transforms.v2.Compose(transforms)


# %% [markdown]
# The following function simply ties everything together into one function to reduce code duplication:

# %%
def train_eval_generic_params(
    model: torch.nn.Module,
    ds_train: torch.utils.data.Dataset,
    ds_val: torch.utils.data.Dataset,
    ds_test: torch.utils.data.Dataset,
    sampler: torch.utils.data.Sampler,
    params: dict,
    train_loss_hook: Optional[Callable[[float, int], None]] = None,
    val_results_hook: Optional[Callable[[EvaluationResults, int], None]] = None,
    transforms_override: Optional[torchvision.transforms.v2.Transform] = None,
    max_epochs: int = N_EPOCHS_TRAIN,
    device: str = DEVICE,
):
    # Train.
    trainer = Trainer(
        epochs=max_epochs,
        batch_size=params["batch_size"],
        augment_transform=augmentation_transforms_from_params(params),
        train_loss_hook=train_loss_hook,
        val_results_hook=val_results_hook,
        device=device,
    )

    optimiser = torch.optim.SGD(
        model.parameters(),
        lr=params["lr"],
        weight_decay=params["weight_decay"],
        momentum=params["momentum"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, factor=params["reduce_lr_factor"]
    )

    transforms = transforms_override if transforms_override is not None else transforms_from_params(params)
    trainer.train(
        model,
        ds_train,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimiser=optimiser,
        train_sampler=sampler,
        transform=transforms,
        ds_val=ds_val,
        scheduler=scheduler,
        early_stop=ValidationLossEarlyStopper(
            patience=params["early_stop_patience"],
            delta=0.0,
        ),
    )

    # Evaluate.
    evaluator = Evaluator(
        batch_size=params["batch_size"],
        zero_division=0,
        device=device,
    )

    return evaluator.evaluate(
        model,
        ds_test,
        loss_fn=torch.nn.CrossEntropyLoss(),
        transform=transforms,
        target_labels=target_le.classes_,
    )


# %% [markdown]
# ## Hyperparameter Tuning
#
# I'll be using the Optuna library for hyperparameter tuning. Optuna uses Bayesian optimisation (at least by default — there are other optimisation methods it provides, but I used Bayesian optimisation) to find the best set of hyperparameters. Based on the performance from past hyperparameter sets, it tries to select promising hyperparameters. Unlike the traditional grid search and random search, Bayesian optimisation can reduce the number of trials needed to find an optimal set of hyperparameters.
#
# We first need to define an objective to optimise. I chose to maximise the macro average F1-score.
# A quick explanation of what macro average F1 is:
#
# - Precision measures the accuracy of positive predictions (i.e., high precision means that the model tends to be correct when it predicts a positive instance).
#
# - Recall measures how well the model is able to identify positive instances (i.e., high recall means that model is able to identify most positive instances).
#
# - The F1 score is the harmonic mean of precision and recall, so it balances these two metrics into a single measure.
#
# - The macro average F1 score calculates the F1 score for each class independently and then takes the average, treating all classes equally. This is important in multi-class problems where class distribution may be imbalanced (which is the case for our dataset) as it ensures that the performance on each class contributes equally to the overall score.
#
# Why macro average F1?
#
# - It is one of the metrics that will be used for final evaluation with real-world implications.
#
# - Unlike accuracy, the macro average F1-score takes into account both precision and recall, so maximising it will likely lead to the model being less biased towards specific classes.
#
# - Minimising the loss does not always mean that the model meets real-world objectives.
#
# Although the models have the same objective to maximise, they have different hyperparameters. Some of these hyperparameters are general and applicable to all of the models (e.g., learning rate and batch size). These general hyperparameters are:
#
# - **Batch size:** The number of training samples to process in a single batch before updating the model's parameters. Smaller batch sizes means more frequent updates and potentially better generalisation at the cost of longer training time. Larger batch sizes can stabilise the training process but may require more memory and may lead to poorer generalisation. See [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836) for a more detailed explanation. Unfortunately, while I wanted to try larger batch sizes (64 and above), I frequently encountered scenarios in which the GPU ran out of memory; hence, the search space for the batch size is kept on the lower side (24 and 32).
#
# - **Learning rate:** Controls the size of the steps for gradient updates. A low learning rate means updates are more precise but requires more epochs to get comparable performance. A high learning rate speeds up training at the cost of risking *overshooting*, where the optimal model parameters are constantly missed, potentially causing failure to converge.
#
# - **Weight decay:** A regularisation parameter that prevents weights from getting too large, which helps to prevent overfitting. A higher weight decay can improve generalisation but may make learning too slow.
#
# - **Learning rate reduction factor:** Factor by which to reduce the learning rate when validation performance starts to stagnate. A smaller factor allows for finer adjustments to the learning rate, which can help the model converge better towards the end of training.
#
# - **Momentum:** A parameter for stochastic gradient descent that accelerates the gradient descent process using past gradients to smooth out updates, which can lead to faster convergence and prevent the model from getting stuck in local minima.
#
# - **Early stop patience:** The number of epochs to wait without improvement in validation performance before stopping training. A higher patience value allows more epochs for the model to improve but risks overfitting to the training set while a lower value may stop training prematurely if the model is slow to converge.
#
# - **Color jitter?:** Whether to randomly alter the colours in the input images (e.g., perturb brightness, contrast, saturation and hue). Color jittering is useful as a data augmentation technique and can improve the model's robustness to variations in lighting and color conditions.
#
# - **Random horizontal flip?:** Whether to randomly flip images horizontally during training. Serves as a data augmentation technique that may help the model become more generalisable with respect to left-right orientation. I kept the probability of randomly flipping an image to 0.5.
#
# - **Random rotation?:** Whether to randomly rotate input images. Serves as a data augmentation technique that may help the model generalise better with respect to orientation. In this case, I kept this is a simple Boolean toggle to reduce the search space and time — random rotations are always done within 30 degrees.
#
# For the MLP and custom VGG-16 models, but _not_ the pretrained VGG-16 model, the following additional hyperparameters were tuned:
#
# - **Resize length:** Specifies the target size for input images — images are resized to a square. Resizing makes input dimensions consistent and reduces computational load. However, if images are resized to dimensions that are too small, we may lose too much information. In my case, I had to keep the maximum resize length rather small (224) because the GPU kept running out of memory.
#
# - **Grayscale?:** Whether to convert images to grayscale. Using grayscale reduces input dimensionality and computational requirements, which is especially useful for the MLP model (since there are many fully-connected units). However, it removes colour information that could have been important.
#
# The reason these hyperparameters are not tuned for the pretrained VGG-16 model is because we need to use the same transformations that were used when it was trained on ImageNet. These transformations can be accessed from the pretrained weights, as documented [here](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html#torchvision.models.VGG16_Weights).

# %% [markdown]
# The function below creates and returns an objective function that suggests the general hyperparameters. It accepts a `model_constructor` callable argument that should wrap the creation of a model with model-specific hyperparameter suggestions.
#
# The function also implements K-fold cross-validation to get a more unbiased estimate of model performance. It works by dividing the training dataset into _K_ subsets called folds. During each iteration, the current fold becomes the test set while the others become the training set. Hence, each sample gets the chance to be used for both training (_K-1_ times) and testing (1 time), reducing variance in performance estimates. I prefer to use K-fold cross-validation in this case because the dataset is rather small — had I used a single validation set for hyperparameter tuning, there is a risk of overfitting to it.
#
# Finally, the hyperparameter tuning process will use *median pruning*, which compares the intermediate results of a trial (e.g., validation loss) against the median value of all previously completed trials at the same step. If a trial's results is worse than the median, it is unlikely to improve on the previous trials, so the trial is pruned, saving computational resources and time.

# %%
def make_objective(
    suggest_model_params: Callable[[optuna.trial.Trial], None],
    model_from_params: Callable[[dict], torch.nn.Module],
    ds_train: torch.utils.data.Dataset,
    prune: bool = PRUNE_TRIALS,
    n_folds: int = N_FOLDS,
    max_epochs: int = N_EPOCHS_TUNE,
    transforms_override: Optional[torchvision.transforms.v2.Transform] = None,
    device: str = DEVICE,
) -> Callable[[optuna.trial.Trial], tuple[float, float]]:
    def objective(trial: optuna.trial.Trial) -> tuple[float, float]:
        # General hyperparameters.
        trial.suggest_int("batch_size", 24, 32, step=8)
        trial.suggest_float("lr", 1e-5, 1e-3)
        trial.suggest_float("weight_decay", 0, 1e-2)
        trial.suggest_float("reduce_lr_factor", 0.01, 0.1)
        trial.suggest_float("momentum", 0.8, 0.99)
        trial.suggest_int("early_stop_patience", 1, 3)

        # Preprocessing input transforms.
        if transforms_override is None:
            trial.suggest_int("resize_length", 32, 224, step=32)
            trial.suggest_categorical("grayscale", [True, False])

        # Augmentation transforms.
        trial.suggest_categorical("jitter", [True, False])
        trial.suggest_categorical("horizontal_flip", [True, False])
        trial.suggest_categorical("rotation", [True, False])

        suggest_model_params(trial)

        print(f"Trial {trial.number} params: {trial.params}")

        f1_scores: list[float] = []

        # Use stratified K-fold cross-validation with 4 folds.
        kfold_cv = sklearn.model_selection.StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=0
        )
        for fold_idx, (cv_indices_other, cv_indices_test) in enumerate(
            kfold_cv.split(
                X=np.arange(len(ds_train)),
                y=np.fromiter(ds_train.targets(), dtype=int),
            )
        ):
            print(f"Trial {trial.number} fold {fold_idx + 1}")

            cv_ds_other = ClothesSubset(ds_train, cv_indices_other)
            cv_ds_test = ClothesSubset(ds_train, cv_indices_test)

            # The `other` set is 3 out of 4 folds.
            # Use 80% of the `other` set as the training set.
            # The remaining 20% are used as the validation set for early stopping.
            # The 4th fold is used as the test set for evaluation.
            # Note: These indices are indices into `cv_ds_other`.
            cv_indices_train, cv_indices_val = sklearn.model_selection.train_test_split(
                np.arange(len(cv_ds_other)),
                train_size=0.8,
                random_state=0,
                shuffle=True,
                stratify=np.fromiter(cv_ds_other.targets(), dtype=int),
            )

            cv_ds_train = ClothesSubset(cv_ds_other, cv_indices_train)
            cv_ds_val = ClothesSubset(cv_ds_other, cv_indices_val)

            # Create the undersampler using `cv_ds_train`'s labels.
            # To avoid leakage, we must not use `ds_train`.
            # The shape won't match anyway.
            undersampler_cv_train = make_undersampler(
                torch.tensor(np.fromiter(cv_ds_train.targets(), dtype=int))
            )

            # Hook onto the validation results for pruning trials.
            def val_results_hook(results: EvaluationResults, epoch: int):
                if not prune:
                    return

                if epoch == 4 and results.mean_loss >= 1.5:
                    raise optuna.TrialPruned("Mean validation loss is still above or equal to 1.5 at the 5th epoch.")

                step = fold_idx * max_epochs + epoch
                f1 = results.classification_report["macro avg"]["f1-score"]
                trial.report(f1, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            model = model_from_params(trial.params)
            results = train_eval_generic_params(
                model=model,
                ds_train=cv_ds_train,
                ds_val=cv_ds_val,
                ds_test=cv_ds_test,
                sampler=undersampler_cv_train,
                params=trial.params,
                train_loss_hook=None,
                val_results_hook=val_results_hook,
                max_epochs=max_epochs,
                transforms_override=transforms_override,
            )

            f1 = results.classification_report["macro avg"]["f1-score"]
            f1_scores.append(f1)

            del model
            gc.collect()
            torch.cuda.empty_cache()

        mean_f1_score = statistics.fmean(f1_scores)
        return mean_f1_score

    return objective


# %% [markdown]
# ### Multilayer Perceptron
#
# The hyperparameters for the MLP are as follows:
#
# - **Number of layers:** The number of layers the MLP has, including the input and output layers. A higher number of layers allows the model to capture more complex patterns. However, setting it too high can lead to overfitting, vanishing/exploding gradients and unrealistic computational requirements.
#
# - **Number of units in each layer:** The number of neurons in a specific layer. Each layer has its number of units tuned (except the first and last layers since those are fixed), so they may not always have the same number of units. Having more units increases the model's ability to learn complex patterns, but too many units can lead to overfitting or high computational requirements.
#
# - **Activation function:** The activation function to use after each linear layer. Affects how well the network captures nonlinear relationships.
#
# I would have liked to further explore other hyperparameters, such as whether to use dropout and regularisation, but the search space was starting to get too large to reasonably explore in time.

# %%
def suggest_mlp_params(trial: optuna.trial.Trial):
    n_layers: int = trial.suggest_int("mlp_n_layers", 3, 5)

    # Suggest size of the second layer.
    # This layer is given a smaller size than the rest because
    # it will be connected to every single pixel of the input image,
    # which can lead to the model having too many parameters
    # and hence exhaust GPU memory.
    trial.suggest_categorical("mlp_layer_1_size", [256, 512])

    # Exclude the first and last layers since their sizes are predetermined.
    # Also exclude the second layer since we already suggested a (smaller)
    # size for it.
    for idx in range(n_layers - 3):
        trial.suggest_categorical(f"mlp_layer_{idx + 2}_size", [256, 512, 1024, 2048])

    activation_f_str = trial.suggest_categorical(
        "mlp_activation_f",
        ["relu", "sigmoid", "softplus"],
    )


def mlp_from_params(params: dict) -> MultilayerPerceptron:
    n_hidden_layers = params["mlp_n_layers"] - 2
    layer_sizes = []
    for idx in range(n_hidden_layers):
        layer_sizes.append(params[f"mlp_layer_{idx + 1}_size"])
    layer_sizes.append(10)  # Final prediction layer.

    activation_f = {
        "relu": torch.nn.ReLU,
        "sigmoid": torch.nn.Sigmoid,
        "softplus": torch.nn.Softplus,
    }[params["mlp_activation_f"]]

    return torch.nn.Sequential(
        torch.nn.Flatten(),
        MultilayerPerceptron(layer_sizes, activation_f)
    )


# %%
mlp_study = optuna.create_study(
    study_name="MLP study",
    sampler=optuna.samplers.TPESampler(seed=0),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
    directions=["maximize"],
    storage=optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend("mlp_journal.log"),
    ),
    load_if_exists=True,
)

mlp_study.optimize(
    make_objective(
        suggest_mlp_params,
        mlp_from_params,
        ds_train,
    ),
    n_trials=N_TRIALS,
    gc_after_trial=True
)

# %%
mlp_study.best_params


# %% [markdown]
# Interestingly, the hyperparameter tuning process indicates that the MLP performed best without performing any data augmentation. However, images had to be resized to 64 by 64. The size of each layer also seems rather small.

# %% [markdown]
# ### VGG-16
#
# For the custom VGG16 model, there is only one model-specific hyperparameter to tune since the model's architecture is mostly fixed. The hyperparameter is whether to use batch normalisation. As mentioned earlier, I found that the model struggled to learn (loss would not decrease) when batch normalisation is not applied after each convolutional layer, which could be because the VGG-16 architecture is rather deep, so it is prone to the vanishing / exploding gradient problem. Batch normalisation may have helped to stabilise the gradients.

# %%
def suggest_vgg16_params(trial: optuna.trial.Trial):
    trial.suggest_categorical("vgg16_batch_norm", [True, False])

def vgg16_from_params(params: dict) -> VGG16:
    return VGG16(10, batch_norm=params["vgg16_batch_norm"])


vgg16_study = optuna.create_study(
    study_name="VGG16 study",
    sampler=optuna.samplers.TPESampler(seed=0),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
    directions=["maximize"],
    storage=optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend("vgg16_journal.log"),
    ),
    load_if_exists=True,
)

vgg16_study.optimize(
    make_objective(
        suggest_vgg16_params,
        vgg16_from_params,
        ds_train,
    ),
    n_trials=N_TRIALS,
    gc_after_trial=True
)

# %%
vgg16_study.best_params


# %% [markdown]
# Like the MLP, the best hyperparameters include resizing of the images to 64 by 64. However, this time, VGG-16 seems to perform best when some data augmentation is applied (colour jittering and random horizontal flip). Furthermore, batch normalisation should be used.

# %% [markdown]
# ### VGG-16 Pretrained
#
# No model-specific hyperparameters were tuned for the pre-trained VGG-16 model since its architecture is fixed. However, in the future, I may want to consider whether to freeze the convolutional layers.
#
# As mentioned earlier, no resizing or grayscaling was applied to the images since the pretrained model requires using the same transformations applied when it was trained on ImageNet. However, whether to use data augmentation is still part of the search space.

# %%
def vgg16_pretrained_from_params(params: dict) -> torchvision.models.vgg.VGG:
    return make_vgg16_pretrained()


vgg16_pretrained_study = optuna.create_study(
    study_name="VGG16 Pretrained study",
    sampler=optuna.samplers.TPESampler(seed=0),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5),
    directions=["maximize"],
    storage=optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend("vgg16_pretrained_journal.log"),
    ),
    load_if_exists=True,
)

vgg16_pretrained_study.optimize(
    make_objective(
        lambda trial: None,
        vgg16_pretrained_from_params,
        ds_train,
        transforms_override=torchvision.models.VGG16_Weights.IMAGENET1K_V1.transforms(),
    ),
    n_trials=N_TRIALS,
    gc_after_trial=True
)

# %%
vgg16_pretrained_study.best_params


# %% [markdown]
# Interestingly, the pretrained VGG-16 model performed best without any data augmentation, like the MLP.

# %% [markdown]
# ## Training and Evaluation
#
# Finally, with the best set of hyperparameters determined for each model, we can perform the real training and evaluation. I first define a helper function to display the evaluation results, including a classification report containing various metrics, a confusion matrix and a visualisation of some of the predictions.

# %%
def display_results(
    results: EvaluationResults,
    model: torch.nn.Module,
    ds: torch.utils.data.Dataset,
    transform: torchvision.transforms.v2.Transform,
    device: str = DEVICE,
):
    print(results.classification_report_str)
    print(f"mean test loss: {results.mean_loss}")
    disp = sklearn.metrics.ConfusionMatrixDisplay(results.confusion_matrix, display_labels=target_le.classes_)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    disp.plot(ax=ax)
    plt.show()

    # Show predictions for some samples.
    rows = 4
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    for idx in range(rows * cols):
        img, label = ds[idx]
        img_transformed = transform(img.detach().clone().to(device))

        # Add an additional first dimension to simulate batching.
        img_transformed = img_transformed[None, :, :, :]

        label = target_le.inverse_transform([label])[0]

        predicted_idx = model(img_transformed).argmax(dim=1).cpu()
        predicted = target_le.inverse_transform(predicted_idx)[0]

        ax = axes[idx // cols, idx % cols]
        ax.imshow(img.permute(1, 2, 0))
        ax.set_title(f"Truth: {label}\nPredicted: {predicted}")
    plt.tight_layout()
    plt.show()


# %% [markdown]
# For keeping track of the training and validation losses using Tensorboard, I need to use PyTorch's `SummaryWriter`. Training and validation losses will be logged for each model using the train and validation loss hooks in the `Trainer`.

# %%
writer = SummaryWriter()
writer.add_custom_scalars({
    "Multilayer Perceptron": {
        "Loss": ["Multiline", ["mlp/loss/train", "mlp/loss/validation"]],
    },
    "VGG16": {
        "Loss": ["Multiline", ["vgg16/loss/train", "vgg16/loss/validation"]],
    },
    "VGG16 Pretrained": {
        "Loss": ["Multiline", ["vgg16-pretrained/loss/train", "vgg16-pretrained/loss/validation"]],
    },
})

# %% [markdown]
# As a quick recap, training of the models will use an undersampler to undersample the shoes and tees classes to mitigate the class imbalance problem. Training will also use early stopping based on the validation loss — if the validation loss stops improving after a certain number of epochs, the training will be stopped.
#
# Evaluation will use the following metrics / tools:
#
# - **Accuracy:** The number of correct predictions out of the total number of predictions. Although accuracy provides a general indication of how well the model is performing, it can be misleading when there is a class imbalance (which is the case for our dataset).
#
# - **Precision:** The number of true positive predictions out of the total number of positive predictions. A high precision means the model is generally correct when it predicts a positive.
#
# - **Recall:** The number of true positive predictions out of the total number of actual positive samples. A high recall means the model is able to identify most positive samples.
#
# - **F1 score:** The harmonic mean of precision and recall. Can be interpreted as an average between precision and recall. F1 is especially useful when dealing with class imbalances since it combines both precision and recall.
#
# - **Macro average F1 score:** The mean of the F1 scores for each class, useful for assessing overall model performance in multi-class classification tasks.
#
# - **Confusion matrix:** A table that compares the actual target labels with the labels predicted by the model. From the confusion matrix, we can see what the model tends to misclassify samples as.
#
# **Note:** For each model, its architecture will be shown before the training and evaluation begin.

# %% [markdown]
# ### Multilayer Perceptron

# %%
mlp = mlp_from_params(mlp_study.best_params)
print(mlp)

mlp_results = train_eval_generic_params(
    model=mlp,
    ds_train=ds_train,
    ds_val=ds_val,
    ds_test=ds_test,
    sampler=undersampler,
    params=mlp_study.best_params,
    train_loss_hook=lambda loss, epoch: writer.add_scalar(
        "mlp/loss/train", loss, epoch
    ),
    val_results_hook=lambda results, epoch: writer.add_scalar(
        "mlp/loss/validation", results.mean_loss, epoch
    ),
)

# %%
torch.save(mlp.state_dict(), "mlp_state.pt")

# %%
display_results(
    mlp_results,
    mlp,
    ds_test,
    transforms_from_params(mlp_study.best_params)
)

# %% [markdown]
# **Train loss vs. validation loss curve from Tensorboard** (squared points are train loss; diamond points are validation loss):
#
# ![curve-mlp.png](attachment:4accfd39-e15f-4ffd-997f-6e8518d41c0d.png)
#
# The train loss starts higher than the validation loss, though they both decrease. However, the validation loss seems to have a lower rate of decrease than the train loss — eventually, the train loss becomes lower than the validation loss. The MLP model seems to struggle to generalise to the validation set despite learning from the training data, which could either be a sign of overfitting or perhaps the model is too simple to adequately capture the underlying patterns. It may also be that the resizing of images to 64 by 64 removed too much information, suggesting that more trials for hyperparameter tuning is needed.

# %% [markdown]
# Nonetheless, the performance metrics for the MLP model show strong results for some classes but mixed outcomes for others. For instance, categories like "shoes," "accessories," and "jeans" perform exceptionally well, with F1-scores of 99%, 96%, and 97% respectively. This suggests that the model is highly effective at correctly classifying these categories and avoiding misclassifications.
#
# However, for classes like "knitwear" and "shirts", the MLP model struggles, with F1-scores of 25% and 48% respectively. These results indicate both low precision and recall, meaning that the model often misclassifies these items and fails to identify them correctly. If we look at the confusion matrix, we see that the model appears to struggle with differentiating between knitwear, shirts and tees. For example, it misclassified 17 tees as jackets, 22 tees as knitwear and 14 tees as shirts. However, considering how similar these items can be, this is not unexpected, especially since MLPs do not account for the order of the pixels in the input images.
#
# The accuracy of 77% indicates that the model's overall performance is not bad. The macro average F1-score of 74% shows that the model's performance across all classes is somewhat balanced. However, there is clearly still some room for improvement considering the misclassifications.
#
# Future improvements could focus on improving the model's ability to distinguish between knitwear, shirts and tees, perhaps using methods like patch-based learning to allow the model to focus on the specific areas of the images (such as the sleeves).

# %% [markdown]
# ### VGG-16

# %%
vgg16 = VGG16(10)
print(vgg16)

vgg16_results = train_eval_generic_params(
    model=vgg16,
    ds_train=ds_train,
    ds_val=ds_val,
    ds_test=ds_test,
    sampler=undersampler,
    params=vgg16_study.best_params,
    train_loss_hook=lambda loss, epoch: writer.add_scalar(
        "vgg16/loss/train", loss, epoch
    ),
    val_results_hook=lambda results, epoch: writer.add_scalar(
        "vgg16/loss/validation", results.mean_loss, epoch
    ),
)

# %%
torch.save(vgg16.state_dict(), "vgg16_state.pt")

# %%
display_results(
    vgg16_results,
    vgg16,
    ds_test,
    transforms_from_params(vgg16_study.best_params)
)

# %% [markdown]
# **Train loss vs. validation loss curve from Tensorboard** (squared points are train loss; diamond points are validation loss):
#
# ![curve-vgg16.png](attachment:b324e84a-9bee-47b4-a1bd-04bd329ff750.png)
#
# The curve for the VGG-16 model looks good — both train and validation losses decrease and eventually converge. There does not seem to be any sign of overfitting as the validation loss does not increase to become higher than the train loss. This indicates that the learning rate and other hyperparameters related to the optimiser were well-suited.

# %% [markdown]
# The custom VGG-16 model achieved an overall accuracy of 83% and a macro average F1-score of 81%. This indicates generally balanced performance across classes. However, this is worse than I expected since the VGG-16 architecture is deep and should be able to handle fairly complex patterns. High-performing classes include "accessories", "jeans", "shoes", and "shorts", with F1-scores above 95%.
#
# The model struggles with "knitwear" and "shirts," showing F1-scores of 54% and 46% respectively (these classes had lower precision and recall). Looking at the confusion matrix, the model seems to have issues distinguishing between knitwear, shirts and tees — similar to the MLP model, but to a lesser extent. "Shirts", in particular, has both low precision (53%) and recall (41%). Improvements are needed for better handling of these underperforming categories.

# %% [markdown]
# ### VGG-16 Pretrained

# %%
vgg16_pretrained = make_vgg16_pretrained()
print(vgg16_pretrained)

vgg16_pretrained_results = train_eval_generic_params(
    model=vgg16_pretrained,
    ds_train=ds_train,
    ds_val=ds_val,
    ds_test=ds_test,
    sampler=undersampler,
    params=vgg16_pretrained_study.best_params,
    train_loss_hook=lambda loss, epoch: writer.add_scalar(
        "vgg16-pretrained/loss/train", loss, epoch
    ),
    val_results_hook=lambda results, epoch: writer.add_scalar(
        "vgg16-pretrained/loss/validation", results.mean_loss, epoch
    ),
    transforms_override=torchvision.models.VGG16_Weights.IMAGENET1K_V1.transforms(),
)

# %%
torch.save(vgg16_pretrained.state_dict(), "vgg16_pretrained_state.pt")

# %%
display_results(
    vgg16_pretrained_results,
    vgg16_pretrained,
    ds_test,
    torchvision.models.VGG16_Weights.IMAGENET1K_V1.transforms(),
)

# %% [markdown]
# **Train loss vs. validation loss curve from Tensorboard** (squared points are train loss; diamond points are validation loss):
#
# ![curve-vgg16-pretrained.png](attachment:43c837df-e3ec-4860-ba14-2271c80c6de5.png)
#
# Unfortunately, the pretrained VGG-16 model seems to have slightly overfitted to the training data as the validation loss starts to increase slightly. However, the training was stopped before it overfitted further. Interestingly, the validation loss decreases at a much slower rate than that of the custom VGG-16 model. This is likely because the pretrained model has already learned common patterns from its previous training — indeed, the validation loss was already nearly 0.4 after the first epoch, unlike the custom VGG-16 model, whose validation loss was around 0.9 after the first epoch.

# %% [markdown]
# The fine-tuned pretrained VGG-16 model achieved an impressive overall accuracy of 95% and a macro average F1-score of 93%. Classes such as "accessories," "jeans," "shoes," and "shorts" had near-perfect performance, with F1-scores around 99% or 100%. "Tees" and "jackets" also performed well, both achieving F1-scores above 90%, indicating strong precision and recall.
#
# However, "knitwear" was the lowest-performing category, with an F1-score of 72%. Looking at the confusion matrix, the model seems to sometimes misclassify knitwear as either jackets or tees. Similarly, the model sometimes misclassifies tees as knitwear. Despite this, the model shows excellent results overall, performing robustly across most categories with little variance in precision and recall between classes.

# %% [markdown]
# ### Tensorboard
#
# The following cell runs Tensorboard with the train and validation loss data within the notebook. Note that this will not work on Kaggle (but it does work on Google Colab) — if you are using Kaggle, you will have to download the logs and run Tensorboard locally. For convenience, the plots have already been included in this notebook as images in the previous sections.

# %%
# %load_ext tensorboard
# %tensorboard --logdir runs

# %% [markdown]
# ## Conclusion
#
# The pretrained VGG-16 model appears to have resulted in the best performance, with an overall accuracy of 95% and a macro average F1-score of 93%. The custom VGG-16 model had decent performance but falls behind, with an overall accuracy of 83% and a macro average F1-score of 81%. This discrepancy demonstrates the impact of using a pretrained network versus training a model from scratch as the latter may require more extensive tuning and data to achieve optimal results.
# Finally, the MLP model had an overall accuracy of 77% and a macro average F1-score of 74%.
#
# As expected, the pretrained VGG-16 model achieved the best results since it had already learned common patterns from its previous training. Furthremore, since MLPs are unable to understand the spacial structure of images (they essentially ignore the order of pixels), the fact that the MLP model performed the worst falls within expectations. For all models, however, there seems to be a trend where the model struggles to differentiate between certain classes (albeit to different extents), especially knitwear, shirts and tees. Considering how similar images in these classes can be, the confusion is not unexpected.
#
# For future work, the focus should be on improving model performance for the underperforming classes. Although hyperparameter tuning was done, there are still many hyperparameters that were unexplored (e.g., use of dropout and regularisation, additional data augmentation transformations). Furthermore, only 25 trials were used for hyperparameter tuning — increasing the number of trials may lead to better hyperparameters that improve model performance. We may also want to explore assigning higher penalties to misclassifications (cost-sensitive learning), especially for the underperforming classes.
