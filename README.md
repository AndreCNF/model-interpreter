# Model Interpreter

An API for intuitive interpretation of machine learning models

## Requirements

Requires a modified version of the SHAP package: https://github.com/AndreCNF/shap
You can install with the following command line operation:
`pip install -e git+https://github.com/AndreCNF/shap.git@f0777334bd82a1bacad578eaf1931c3ecbf40ec6#egg=shap`

Using [Poetry](https://poetry.eustace.io/) for dependency management, saving the
required packages and their specific versions so as to facilitate
reproducibility. As such, it's recommendable to use ```poetry install``` in the
directory before proceeding.

Also, if on macOS and PyTorch isn't working, it might be needed to install
libomp through [homebrew](https://brew.sh/):
`brew install libomp`
