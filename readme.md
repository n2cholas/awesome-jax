# Awesome JAX [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

A curated list of awesome JAX libraries, projects, and other resources. Inspired by [Awesome TensorFlow](https://github.com/jtoy/awesome-tensorflow).

## What is JAX?

JAX brings automatic differentiation and the XLA compiler together through a numpy-like API for high performance machine learning research on accelerators like GPUs and TPUs. More info [here](https://github.com/google/jax).


## Table of Contents

- [Libraries](#libraries)
- [Models/Projects](#projects)
- [Videos](#videos)
- [Papers](#papers)
- [Blog Posts](#posts)
- [Community](#community)
- [Contribute](#contribute)


<a name="libraries" />

## Libraries

- Neural Network Libraries
    - [Flax](https://github.com/google/flax) - a flexible library with the largest user base of all JAX NN libraries.
    - [Haiku](https://github.com/deepmind/dm-haiku) - focused on simplicity, created by the authors of Sonnet at DeepMind.
    - [Objax](https://github.com/google/objax) - has an object oriented design similar to PyTorch.
    - [Elegy](https://poets-ai.github.io/elegy/) - implements the Keras API with some improvements.
    - [RLax](https://github.com/deepmind/rlax) - library for implementing reinforcement learning agent.
    - [Trax](https://github.com/google/trax) - a "batteries included" deep learning library focused on providing solutions for common workloads.
    - [Jraph](https://github.com/deepmind/jraph) - a lightweight graph neural network library.
- [NumPyro](https://github.com/pyro-ppl/numpyro) - probabilistic programming based on the Pyro library.
- [Chex](https://github.com/deepmind/chex) - utilities to write and test reliable JAX code.
- [Optax](https://github.com/deepmind/optax) - a gradient processing and optimization library.
- [JAX, M.D.](https://github.com/google/jax-md) - accelerated, differential molecular dynamics.

<a name="projects" />

## Models/Projects

- [Reformer](https://github.com/google/trax/tree/master/trax/models/reformer) - an implementation of the Reformer (efficient transformer) architecture.

<a name="videos" />

## Videos

- [Introduction to JAX](https://youtu.be/0mVmRHMaOJ4) - a simple neural network from scratch in JAX.
- [JAX: Accelerated Machine Learning Research | SciPy 2020 | VanderPlas](https://youtu.be/z-WSrQDXkuM) - JAX’s core design, how it’s powering new research, and how you can start using it.
- [Bayesian Programming with JAX + NumPyro — Andy Kitchen](https://youtu.be/CecuWGpoztw) - introduction to Bayesian modelling using NumPyro.

<a name="papers" />

## Papers

- [Compiling machine learning programs via high-level tracing. Roy Frostig, Matthew James Johnson, Chris Leary. _MLSys 2018_.](https://mlsys.org/Conferences/doc/2018/146.pdf) - this white paper describes an early version of JAX, detailing how computation is traced and compiled.
- [Reformer: The Efficient Transformer. Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya. _ICLR 2020_.](https://arxiv.org/abs/2001.04451) - introduces the Reformer architecture with O(nlogn) self attention via locality sensitive hashing, providing significant gains in memory efficiency and speed on long sequences.
- [JAX, M.D.: A Framework for Differentiable Physics. Samuel S. Schoenholz, Ekin D. Cubuk. _NeurIPS 2020_.](https://arxiv.org/abs/1912.04232) - introduces JAX, M.D., a differentiable physics library which includes simulation environments, interaction potentials, neural networks, and more.
- [Enabling Fast Differentially Private SGD via Just-in-Time Compilation and Vectorization. Pranav Subramani, Nicholas Vadivelu, Gautam Kamath. _arXiv 2020_.](https://arxiv.org/abs/2010.09063) - uses JAX's JIT and VMAP to achieve faster differentially private than existing libraries.

<a name="posts" />

## Blog Posts

- [Using JAX to accelerate our research](https://deepmind.com/blog/article/using-jax-to-accelerate-our-research) - describes the state of JAX and the JAX ecosystem at DeepMind.
- [Getting started with JAX (MLPs, CNNs & RNNs)](https://roberttlange.github.io/posts/2020/03/blog-post-10/) - neural network building blocks from scratch with the basic JAX operators.
- [Plugging Into JAX](https://medium.com/swlh/plugging-into-jax-16c120ec3302) - compared Flax, Haiku, and Objax on the Kaggle flower classification challenge.
- [Understanding Autodiff with JAX](https://www.radx.in/jax.html) - understand autodiff implementation while working with JAX

<a name="community" />

## Community

- [JAX GitHub Discussions](https://github.com/google/jax/discussions)
- [Reddit](https://www.reddit.com/r/JAX/)

<a name="contribute" />

## Contribute

Contributions welcome! Read the [contribution guidelines](contributing.md) first.
