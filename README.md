<p align="center">
<img src="https://github.com/angelolab/Nimbus-Inference/blob/main/assets/nimbus_logo.png">
</p>

![CI](https://github.com/angelolab/Nimbus-Inference/actions/workflows/ci.yaml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/nimbus-inference/badge/?version=latest)](https://nimbus-inference.readthedocs.io/en/latest/?badge=latest)


The Nimbus repo contains code for inference of a machine learning model that classifies cells into marker positive/negative for arbitrary protein markers and different imaging platforms.

## Installation instructions

Clone the repository

`git clone https://github.com/angelolab/Nimbus-Inference`


Make a conda environment for Nimbus and activate it

`conda create -n Nimbus python==3.10`

`conda activate Nimbus`

Install CUDA libraries if you have a NVIDIA GPU available 

`conda install -c conda-forge cudatoolkit=11.8 cudnn=8.2.0`

Install the package and all depedencies in the conda environment

`python -m pip install -e Nimbus-Inference`


Navigate to the example notebooks and start jupyter

`cd Nimbus-Inference/templates`

`jupyter notebook`


## Release notes

See the [changelog][changelog].

## Contact

If you found a bug, please use the [issue tracker][issue-tracker].
For questions and help requests, you can also reach out in the [issue tracker][issue-tracker].

## Citation

```bash
@article{rum2024nimbus,
  title={Automated classification of cellular expression in multiplexed imaging data with Nimbus},
  author={Rumberger, J. Lorenz and Greenwald, Noah F. and Ranek, Jolene S. and Boonrat, Potchara and Walker, Cameron and Franzen, Jannik and Varra, Sricharan Reddy and Kong, Alex and Sowers, Cameron and Liu, Candace C. and Averbukh, Inna and Piyadasa, Hadeesha and Vanguri, Rami and Nederlof, Iris and Wang, Xuefei Julie and Van Valen, David and Kok, Marleen and Hollman, Travis J. and Kainmueller, Dagmar and Angelo, Michael},
  journal={bioRxiv},
  pages={2024--05},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
[issue-tracker]: https://github.com/angelolab/Nimbus-Inference/issues
[changelog]: https://Nimbus-Inference.readthedocs.io/latest/changelog.html
[link-docs]: https://Nimbus-Inference.readthedocs.io
[link-api]: https://Nimbus-Inference.readthedocs.io/latest/api.html
