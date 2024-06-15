import argparse
import inspect
import os
import json
from enum import Enum

# Optionally add in Numba in the future(?)
class NBType(Enum):
    Numpy = 1
    Jax = 2

def dump_nb(p_name: str, mode: NBType):
    # Define the structure of a minimal Jupyter notebook
    if mode == NBType.Jax:
        _imp = "import jax.numpy as jnp"
        f_name = f"{p_name}_jax.ipynb"
    else:
        _imp = "import numpy as np"
        f_name = f"{p_name}_np.ipynb"
    notebook_content = {
        "cells": [],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5
    }

    # Add a Markdown cell
    notebook_content["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {' '.join(p_name.split()).title()}",
        ]
    })

    # Add a Code cell
    notebook_content["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            _imp
        ]
    })

    # Convert the dictionary to a JSON string
    notebook_json = json.dumps(notebook_content, indent=4)

    # Specify the output file name

    loc = f"./{p_name}/"
    # Write the JSON string to a file with .ipynb extension
    with open(loc + f_name, "w") as notebook_file:
        notebook_file.write(notebook_json)

    print(f"Notebook '{f_name}' has been created.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("project_name", type=str, help="Name of the project to instantiate")
    args = parser.parse_args()
    p_name = args.project_name

    try:
        os.makedirs(p_name)
    except Exception as e:
        print(e)

    loc = f"./{p_name}/"

    with open(loc + "README.md", "w") as f:
        f.write(f"# {' '.join(p_name.split()).title()}")

    dump_nb(p_name, NBType.Numpy)
    dump_nb(p_name, NBType.Jax)
