<p align="center">
  <img src="https://github.com/angelolab/Nimbus-Inference/blob/main/assets/nimbus_logo.png" alt="Nimbus Logo"/>
</p>

![CI](https://github.com/angelolab/Nimbus-Inference/actions/workflows/ci.yaml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/nimbus-inference/badge/?version=latest)](https://nimbus-inference.readthedocs.io/en/latest/?badge=latest)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mLt2K9_rqUhr3Z4CLw_znS12KSUVSPzj?usp=sharing)

# Nimbus Inference

**Nimbus** is a deep learning model for automated classification of marker expression in multiplexed imaging data. This repository provides code for:
- **Inference** of the Nimbus model on your multiplexed images.
- **Finetuning** the Nimbus model on new data if desired.
- **Interactive exploration** of the Nimbus Gold Standard dataset.

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
   - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mLt2K9_rqUhr3Z4CLw_znS12KSUVSPzj?usp=sharing)  
     
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

## Release Notes

See our changelog for details of what has changed in recent releases.

---

## Contact

If you have questions, find a bug, or need help:
- Please use the issue tracker.
- We welcome contributions or pull requests to improve Nimbus!

---

## Citation

If you use Nimbus in your work, please cite:

```bibtex
@article{rum2024nimbus,
  title={Automated classification of cellular expression in multiplexed imaging data with Nimbus},
  author={Rumberger, J. Lorenz and Greenwald, Noah F. and Ranek, Jolene S. and Boonrat, Potchara and Walker, Cameron and Franzen, Jannik and Varra, Sricharan Reddy and Kong, Alex and Sowers, Cameron and Liu, Candace C. and Averbukh, Inna and Piyadasa, Hadeesha and Vanguri, Rami and Nederlof, Iris and Wang, Xuefei Julie and Van Valen, David and Kok, Marleen and Hollman, Travis J. and Kainmueller, Dagmar and Angelo, Michael},
  journal={bioRxiv},
  pages={2024--05},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
