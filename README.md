# Fashion Image Classification: MLP vs. VGG-16

This project analyses the performance of three neural network models for fashion image classification.
It was developed as part of a university project.
The models compared are:

- A multilayer perceptron (MLP)

- A custom implementation of VGG-16

- A fine-tuned pre-trained VGG-16 model

All models are implemented in PyTorch.

The dataset used for this project was distributed to students via a Google Drive link.
Unfortunately, no source or documentation was provided, so its origin remains unknown.

## Notebook

All code is contained in a single Jupyter notebook.
It is stored in this repository in `py:percent` format (using Jupytext for conversion)
for easier version control.

The full notebook (with outputs), dataset, training logs and model files
are available in the [latest release](https://github.com/dixslyf/mlp-vgg-fashion/releases/latest).

You can view a rendered, static version of the notebook [here](https://dixslyf.github.io/mlp-vgg-fashion/).

## Run Environment

The notebook was executed on Kaggle with GPU acceleration using the following environment:
[v153 - GPU](https://github.com/Kaggle/docker-python/releases/tag/141219e230dab548ccc19aa4e62bcf805ed9de0b4d5112227e28f5f1a25991f8).

You can view the original runs [on Kaggle](https://www.kaggle.com/code/dixonseanlowyanfeng/fashion-image-classification-mlp-vs-vgg-16).
Please note that most of the written analysis and commentary was added after the notebook was run,
so the Kaggle version contains minimal explanation.

## Building the Site

The static site is generated using [Quarto](https://quarto.org/) and deployed via GitHub Pages.

To build and view the site locally:

1. Install the Quarto CLI.

2. Clone this repository:

    ```sh
    git clone https://github.com/dixslyf/mlp-vgg-fashion.git
    ```

3. Download the full-output notebook from the latest release.

4. Place the notebook at the following path: `<repo>/notebooks/mlp-vgg16-fashion.ipynb`.

5. Render the site:

    ```sh
    quarto render
    ```

    Or, to preview the site:

    ```sh
    quarto preview
    ```
