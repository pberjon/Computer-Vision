<img src="./img/logo.png" hspace="20%" width="60%">

## Introduction

- **CNNumpy** is a Convolutional Neural Network written in pure Numpy (educational purpose only).
- There are 2 implementation versions:
    - Slow: The naive version with nested for loops.
    - Fast: The im2col/col2im version.
- The [slow implementation][slow-implementation] takes around **4 hours for 1 epoch** where the [fast implementation][fast-implementation] takes only **6 min for 1 epoch**.
- For your information, with the same architecture using **Pytorch**, it will take around **1 min for 1 epoch**.


## Installation

- Create a virtual environment in the root folder using [virtualenv][virtualenv] and activate it.

```bash
# On Linux terminal, using virtualenv.
virtualenv myenv
# Activate it.
source myenv/bin/activate
```

- Install **requirements.txt**.

```bash
pip install -r requirements.txt
# Tidy up the root folder.
python3 setup.py clean
```

## Usage of demo notebooks

To play with the `demo-notebooks/` files, you need to make sure jupyter notebook can select your virtual environnment as a kernel.

- Follow **"Installation"** instructions first and make sure your virtual environment is still activated.
- Run the following line in the terminal.
```bash
python -m ipykernel install --user--name=myenv
```
