# Numpy To Optimized Jax

Jax is often thought of as Numpy for the GPU, but it is so much more (both in terms of features, and sharp edges). The tutorials presented here—one aimed at a general audience and the other at computational neuroscientists—were inspired by a roadblock I encountered in my research. Specifically, I was working on a LIF simulation problem that, despite using vectorized Numpy, took excessively long to run. By incorporating Jax into my workflow and iterating on it, I managed to reduce the runtime from ~10 seconds to ~0.2 seconds.

## Contents:

The `exercises` folder contains the code structured as a series of exercises for you to work through to reinforce the concepts.

### L1: Jax Function Calls

### L2 Jax JIT: 

- Using `jit`

- Understanding **when** to use jit a.k.a why not jit everything?

- Timing `jax`

### L3 Jax loops: 

- reading haskell-like function signatures

- `fori_loop`, `while_loop`, `scan`

- optimizing

### L4: Misc. Using vmap

- make your code look more like the math described in the papers

### L5: Profiling your code

### L6: Jax's Sharp Edges

### L7: using Einsum for more readable code

## Bonus Content:

### NamedTuples

Using namedtuples to clean up your Jax code

---

# Work in progress:

- [ ] Randomness in JAX
  - reproducible randomness across machines across accelerators
- [ ] pmap
  - Introduction to parallel execution with examples using multiple devices. Having said that, this should still work even with a single device
- [ ] grad
  - and grad_and_val to get the gradient

## LIF Benchmarks

2020 Macbook Pro M1 chip with 32gb RAM

**Numpy**: 50s
**Jax-01**: 8.8s
**Jax-02-unoptimized**: 20.3s
**Jax-02-Correct**: 6.7s
**Jax-03**: 6.4s

## Citation

If you use this software in your research, please cite it as follows:

```bibtex
@misc{numpy_to_jax,
  title = {Numpy To Jax},
  author = {Ian Quah},
  year = {2024},
  url = {https://github.com/IanQS/numpy_to_jax},
  version = {1.0.0},
  note = {Jax is often thought of as Numpy for the GPU, but it is so much more (both in terms of features, and sharp edges). The tutorials presented here—one aimed at a general audience and the other at computational neuroscientists—were inspired by a roadblock I encountered in my research}
}