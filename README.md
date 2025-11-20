<p align="center">
  <img src="https://raw.githubusercontent.com/angelolab/Nimbus-Inference/refs/heads/main/assets/nimbus_logo.png" alt="Nimbus Logo"/>
</p>

![CI](https://github.com/angelolab/Nimbus-Inference/actions/workflows/ci.yaml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/nimbus-inference/badge/?version=latest)](https://nimbus-inference.readthedocs.io/en/latest/?badge=latest)
[![PyPI Downloads](https://static.pepy.tech/badge/nimbus-inference)](https://pepy.tech/projects/nimbus-inference)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mLt2K9_rqUhr3Z4CLw_znS12KSUVSPzj?usp=sharing)
[![Nature Methods](https://img.shields.io/badge/Nature%20Methods-Read_Paper-red?style=flat&logo=doi&logoColor=white)](https://www.nature.com/articles/s41592-025-02826-9)

# Nimbus Inference

**Nimbus** is a deep learning model for automated classification of marker expression in multiplexed imaging data. This repository provides code for:
- **Inference** of the Nimbus model on your multiplexed images.
- **Finetuning** the Nimbus model on new data if desired.
- **Interactive exploration** of the Nimbus Gold Standard dataset.

The code for training the Nimbus model from scratch can be found in the [angelolab/Nimbus](https://github.com/angelolab/Nimbus) repository 

> **Installation (via pip)**  
> ```bash
> pip install Nimbus-Inference
> ```
> Create a Python environment with version 3.9–3.11, then install as shown above.

---

## Example Notebooks

We provide three Jupyter notebooks (in the `templates` folder), each with its own **example dataset** that is loaded from the Hugging Face Hub within the notebook:

- **[1_Nimbus_Predict.ipynb](https://github.com/angelolab/Nimbus-Inference/blob/main/templates/1_Nimbus_Predict.ipynb)**  
   - Guides you through performing inference with the Nimbus model on multiplexed imaging data.  
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1k-KnuALBRXbjbkpIP63tf7aEEALw9L1O/view?usp=sharing)  
     
- **[2_Nimbus_Finetuning.ipynb](https://github.com/angelolab/Nimbus-Inference/blob/main/templates/2_Nimbus_Finetuning.ipynb)**  
   - Shows you how to finetune the Nimbus model on a new dataset.  
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1rYYVJQ0nkpG2QE9UjIrzl2x266agdppN/view?usp=sharing)  

- **[3_interactive_viewer.ipynb](https://github.com/angelolab/Nimbus-Inference/blob/main/templates/3_interactive_viewer.ipynb)**  
   - Lets you interactively explore the Nimbus Gold Standard labeled dataset.  
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1LW0vHC3sKKA3TyvW_9FeIaHj3PonzhGS/view?usp=sharing)

Each notebook loads an example dataset directly from the Hugging Face Hub, so you can get hands-on with Nimbus right away.

---

## Datasets

We have released two main datasets on the Hugging Face Hub:

1. **Pan-Multiplex**  
   - A large, noisy labeled dataset used for training and validation.  
   - [https://huggingface.co/datasets/JLrumberger/Pan-Multiplex](https://huggingface.co/datasets/JLrumberger/Pan-Multiplex)

2. **Pan-Multiplex Gold-Standard**  
   - A smaller subset of Pan-Multiplex where every cell was manually annotated by experts.  
   - [https://huggingface.co/datasets/JLrumberger/Pan-Multiplex-Gold-Standard](https://huggingface.co/datasets/JLrumberger/Pan-Multiplex-Gold-Standard)

---

## What is Nimbus?

Nimbus is a deep learning model designed to make **human-like, visual classifications** of multiplexed imaging data by determining which protein markers each cell is positive or negative for. Unlike many existing workflows, Nimbus:
- Uses the **raw image** pixels (rather than purely integrated intensity) to classify marker expression.
- Generalizes across many tissue types, imaging platforms, and markers — without retraining.
- Can be integrated into downstream clustering or phenotyping pipelines to improve accuracy.

For more details, please see our [preprint](https://pmc.ncbi.nlm.nih.gov/articles/PMC11185540/).

---

## Repository organization

Our github is organized as follows: 
- The `README` file (which you're looking at now) provides an overview of the project
- The `.github` folder contains code automating jobs via github actions for testing and deployment
- The `assets` folder contains images that are displayed in the README
- The `docs` folder allows us to build and maintain updated documentation for the project
- The `src` folder contains the core code for running Nimbus
- The `templates` folder contains example notebooks to provide easy starting examples for new users
- The `tests` folder contains code for testing the codebase
  
For a more detailed look on how to get started, please check out our [documentation](https://nimbus-inference.readthedocs.io/en/latest/?badge=latest)

---

## Contact

If you have questions, find a bug, or need help:
- Please use the issue tracker.
- We welcome contributions or pull requests to improve Nimbus!

---

## Citation

If you use Nimbus in your work, please cite:

```bibtex
@article{rumberger2025automated,
  title={Automated classification of cellular expression in multiplexed imaging data with Nimbus},
  author={Rumberger, Josef Lorenz and Greenwald, Noah F and Ranek, Jolene S and Boonrat, Potchara and Walker, Cameron and Franzen, Jannik and Varra, Sricharan Reddy and Kong, Alex and Sowers, Cameron and Liu, Candace C and others},
  journal={Nature Methods},
  volume={22},
  pages={2161–-2170},
  year={2025},
  publisher={Nature Publishing Group US New York}
}
