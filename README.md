# Numpy to Optimized Jax 

LIF Implementation for a real-word example of how to go from a numpy version to a highly optimized jax. During the course of this tutorial, we will also discuss best practices for cleaner code, profiling, and checking the computation graph.

This tutorial is comprises lessons that are structured as part of the LIF theme, and outside that theme; there's no point in trying to shoe-horn in these lessons.

## Contents:

The `exercises` folder contains the code structured as a series of exercises for you to work through to reinforce the concepts.

### L1: Jax Function Calls

### L2 Jax JIT: 

- What is it?
- What do I need to look out for?

### L3 Jax Scan: 

- reading haskell-like function signatures
- optimizing 
- What do I need to look out for?

### L4: Cleaning up the code 

Using namedtuples to clean up your code

### L5: Misc. Using einsum

### L6: Misc. Using vmap

### L7: Misc. Using the profiler and visualizing the computation graph

## LIF BenchmarksSmall Benchmarks

**Numpy**: 50s
**Jax-01**: 8.8s
**Jax-02-unoptimized**: 20.3s
**Jax-02-Correct**: 6.7s
**Jax-03**: 6.4s