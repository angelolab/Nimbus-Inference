============
Installation
============

Clone the repository

.. code-block:: bash

    git clone https://github.com/angelolab/Nimbus-Inference


Make a conda environment for Nimbus and activate it

.. code-block:: bash

    conda create -n Nimbus python==3.10
    conda activate Nimbus

Install CUDA libraries if you have a NVIDIA GPU available 

.. code-block:: bash

    conda install -c conda-forge cudatoolkit=11.8 cudnn=8.2.0

Install the package and all depedencies in the conda environment

.. code-block:: bash

    python -m pip install -e Nimbus-Inference

Navigate to the example notebooks and start jupyter

.. code-block:: bash

    cd Nimbus-Inference/templates

    jupyter notebook
