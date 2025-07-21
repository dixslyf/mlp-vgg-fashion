notebook-path := "notebooks/mlp-vgg16-fashion.ipynb"

setup-extensions:
    quarto add --no-prompt dixslyf/quarto-group-html-cell-outputs@v1.0.1

preview-site: setup-extensions
    quarto preview {{notebook-path}}

render-site: setup-extensions
    quarto render {{notebook-path}}
