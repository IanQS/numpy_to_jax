# Numpy To Optimized Jax

**Motivation**: This repo was originally a series of lessons introducing Jax concepts, and it can still be used for that. However, it has also advanced beyond that to include 
example code solving various real-world problems. These case-studies themselves can be used as a follow-up lesson for those who have finished the first lesson.

**Blurb**: Jax is often thought of as Numpy for the GPU, but it is so much more (both in terms of features, and sharp edges). The tutorials presented here—one aimed at a general audience and the other at computational neuroscientists—were inspired by a roadblock I encountered in my research. Specifically, I was working on a LIF simulation problem that, despite using vectorized Numpy, took excessively long to run. By incorporating Jax into my workflow and iterating on it, I managed to reduce the runtime from ~10 seconds to ~0.2 seconds.

## Table of Contents
- [Exercises](#exercises) 
- [Case Studies](#case-studies)
- [WIP](#work-in-progress)
- [Citing this work](#citation)

## Exercises:

The `exercises` folder contains the code structured as a series of exercises for you to work through to reinforce the concepts.

### L1: Jax Function Calls

### L2 Jax JIT: 

- Using `jit`

- Understanding **when** to use jit a.k.a why not jit everything?

- Timing `jax`

### L3 Jax loops: 

- reading haskell-like function signatures

- `fori_loop`, `while_loop`, `scan`

### L4: Misc. Using vmap

- make your code look more like the math described in the papers

### L5: Profiling your code

- in prior notebooks we had introduced methods to speed up code, and the JIT compilation. Let's investigate if and how much they speed up code!

### L6: RNG

- learn the design decisions behind Jax's RNG implementation

### L7: Grad Basics

- learn the basics behind Jax's grad methods that will cover 90% of usecases

### L8: Grad Intermediate

- should probably be called grad manipulations, where we stop gradients, skip applications, and more

### L9: PyTrees

- learn how Jax internally handles data structures and how to add your own custom data structure to the model registry

### Bonus: using Einsum for more readable code

Einsum isn't specific to Jax, but it's still useful to know!

## Case Studies

Case studies build on the exercises and rely on concepts covered in the lessons. In the case studies we see the concepts applied to real-world problems.

---

# Work in progress:

- [ ] Grad Advanced, which will cover custom gradients, `jacrev` and `jacfwd`
- [ ] 3d Parallelism
  - [ ] data
  - [ ] pipeline
  - [ ] tensor parallelism

## Citation

If you use this software in your research, please cite it as follows:

```bibtex
@misc{numpy_to_jax,
  title = {Numpy To Jax},
  author = {Ian Quah, Bryan Quah},
  year = {2024},
  url = {https://github.com/IanQS/numpy_to_jax},
  version = {1.0.0},
  note = {Jax is often thought of as Numpy for the GPU, but it is so much more (both in terms of features, and sharp edges). The tutorials presented here—one aimed at a general audience and the other at computational neuroscientists—were inspired by a roadblock I encountered in my research}
}