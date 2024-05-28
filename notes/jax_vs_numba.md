# Jax vs. Numba

Both Jax and Numba use a `jit` (Just-In-Time) compilation system to speed up computations, so it is natural to ask when you should use one over the other. Here are some considerations:

1) Deployment location (CPU, GPU, TPU, etc.)
2) Amount of existing code
3) Use-case
4) Code maintainability

In general, I'd recommend using Jax if your code will interface with machine learning frameworks or if you need to deploy your code on a GPU (having said that, Numba **does** support GPU computation). The last time I looked at Numba was in 2018, but from what I remember, Jax in 2023 has more restrictions and [sharp edges](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html) than Numba in 2018.

## Deployment Location

Although Numba has [CUDA support](https://numba.readthedocs.io/en/stable/cuda/index.html), allowing you to write your own kernels, learning the syntax and code is far more legwork than Jax. Jax's integration with XLA (Accelerated Linear Algebra) simplifies the process of deploying on GPUs and TPUs.

## Existing Code

If you already have a large codebase that extensively leverages NumPy, I'd recommend trying a quick find-and-replace approach, **swapping out `np.X` methods for `jnp.X`** and sanity checking your results to ensure that things are still lined up. However, if your existing codebase relies heavily on native Python code with `for-loops` and native control-flow methods, it might be worth looking at Numba. Using Numba would require fewer changes to your existing code.

## Use-Case

**For extensive machine learning tasks or numerical simulations, I recommend using Jax.** Jax was designed to address modern machine learning problems of scaling training across many machines and has many utilities built-in to help you. However, for general computing or optimizing a few hotspots in your code, using Numba is probably the right move. **Numba is well-suited for accelerating numerical computations and integrating seamlessly with existing Python codebases.**

## Code Maintainability

Jax is more difficult to write compared to standard Numba (GPU Numba is a different beast). **Jax introduces its own syntax and conventions, such as handling random numbers and ensuring pure functions for JIT compilation.** This means that whoever ends up reading or extending your code will need to be familiar with these concepts, such as the requirement for pure code when using JIT, or ensuring that incoming arrays are the same size to avoid triggering recompilation. **In contrast, Numba's approach is more straightforward and closer to standard Python, which can make it easier for others to maintain and extend your code.**

In summary:

- **Choose Jax if**:
  - Your code interfaces with machine learning frameworks.
  - You need efficient GPU or TPU deployment.
  - You are performing extensive numerical simulations or machine learning tasks.

- **Choose Numba if**:
  - You have an existing codebase with heavy reliance on native Python and some NumPy.
  - Your code's hot-spots are native Python and use a lot of control flow.
  - You are performing general numerical and scientific computing.