# Nimbus-Inference

[![Tests][badge-tests]][link-tests]
[![Documentation][badge-docs]][link-docs]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/angelolab/Nimbus-Inference/test.yaml?branch=main
[link-tests]: https://github.com/angelolab/Nimbus-Inference/actions/workflows/test.yml
[badge-docs]: https://img.shields.io/readthedocs/Nimbus-Inference

The Nimbus repo contains code for inference of a machine learning model that classifies cells into marker positive/negative for arbitrary protein markers and different imaging platforms.

## Installation instructions

Clone the repository

`git clone https://github.com/angelolab/Nimbus-Inference`


Make a conda environment for Nimbus and activate it

`conda create -n Nimbus python==3.10`

`conda activate Nimbus`

Install CUDA libraries if you have a NVIDIA GPU available 

`conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`

Install the package and all depedencies in the conda environment

`python -m pip install -e Nimbus-Inference`


Navigate to the example notebooks and start jupyter

`cd Nimbus-Inference/templates`

`jupyter notebook`


## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][scverse-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

> t.b.a

[scverse-discourse]: https://discourse.scverse.org/
[issue-tracker]: https://github.com/angelolab/Nimbus-Inference/issues
[changelog]: https://Nimbus-Inference.readthedocs.io/latest/changelog.html
[link-docs]: https://Nimbus-Inference.readthedocs.io
[link-api]: https://Nimbus-Inference.readthedocs.io/latest/api.html
