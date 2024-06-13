# Case Studies

In this case-studies directory we include examples of JAX and how it can be used for various tasks. These tasks are standalone and often 
involve either ML or differential equations.

## Adding new case studies

You should run the `setup.py` via `python setup.py X`, where `X` is the name of your project. Running the command will create three files:

- `README.md` introducing the equation we are modeling
- `X_np.ipynb` - containing the raw numpy implementation of X
- `X_jax.ipynb` - containing the jax (ideally optimized) version of X, with descriptions.

