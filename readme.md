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

<a id="kitaev2020"></a>
- [Reformer](https://github.com/google/trax/tree/master/trax/models/reformer) - an implementation of the Reformer (efficient transformer) architecture.

<a id="dosovitskiy2020"></a>
- [Vision Transformer](https://github.com/google-research/vision_transformer) - official implementation in Flax of _An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale_.

<a id="tancik2020"></a>
- [Fourier Feature Networks](https://github.com/tancik/fourier-feature-networks) - official implementation of [_Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains_](https://people.eecs.berkeley.edu/~bmild/fourfeat).

- [Flax Models](https://github.com/google-research/google-research/tree/master/flax_models) - collection of open-sourced Flax models.

- [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf) - implementation of [_NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis_](http://www.matthewtancik.com/nerf) with multi-device GPU/TPU support.

<a name="videos" />

## Videos

- [Introduction to JAX](https://youtu.be/0mVmRHMaOJ4) - a simple neural network from scratch in JAX.
- [JAX: Accelerated Machine Learning Research | SciPy 2020 | VanderPlas](https://youtu.be/z-WSrQDXkuM) - JAX’s core design, how it’s powering new research, and how you can start using it.
- [Bayesian Programming with JAX + NumPyro — Andy Kitchen](https://youtu.be/CecuWGpoztw) - introduction to Bayesian modelling using NumPyro.
- [JAX: Accelerated machine-learning research via composable function transformations in Python | NeurIPS 2019 | Skye Wanderman-Milne](https://slideslive.com/38923687/jax-accelerated-machinelearning-research-via-composable-function-transformations-in-python) - JAX intro presentation in [_Program Transformations for Machine Learning_](https://program-transformations.github.io) workshop.
- [JAX on Cloud TPUs | NeurIPS 2020 | Skye Wanderman-Milne and James Bradbury](https://drive.google.com/file/d/1jKxefZT1xJDUxMman6qrQVed7vWI0MIn/edit) - presentation of TPU host access with demo.
- [Deep Implicit Layers - Neural ODEs, Deep Equilibirum Models, and Beyond | NeurIPS 2020](https://slideslive.com/38935810/deep-implicit-layers-neural-odes-equilibrium-models-and-beyond) - tutorial created by Zico Kolter, David Duvenaud, and Matt Johnson with Colab notebooks avaliable in [_Deep Implicit Layers_](http://implicit-layers-tutorial.org).

<a name="papers" />

## Papers

- [__Compiling machine learning programs via high-level tracing__. Roy Frostig, Matthew James Johnson, Chris Leary. _MLSys 2018_.](https://mlsys.org/Conferences/doc/2018/146.pdf) - this white paper describes an early version of JAX, detailing how computation is traced and compiled.
- [__JAX, M.D.: A Framework for Differentiable Physics__. Samuel S. Schoenholz, Ekin D. Cubuk. _NeurIPS 2020_.](https://arxiv.org/abs/1912.04232) - introduces JAX, M.D., a differentiable physics library which includes simulation environments, interaction potentials, neural networks, and more.
- [__Reformer: The Efficient Transformer__. Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya. _ICLR 2020_.](https://arxiv.org/abs/2001.04451) - introduces the Reformer architecture with O(nlogn) self attention via locality sensitive hashing, providing significant gains in memory efficiency and speed on long sequences.[^](#kitaev2020)
- [__Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains__. Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, Ren Ng. _NeurIPS 2020_.](https://arxiv.org/abs/2006.10739) - applies a Fourier feature mapping in low-dimensional problem domains to enable a multilayer perceptron (MLP) to learn high-frequency functions.[^](#tancik2020)
- [__Enabling Fast Differentially Private SGD via Just-in-Time Compilation and Vectorization__. Pranav Subramani, Nicholas Vadivelu, Gautam Kamath. _arXiv 2020_.](https://arxiv.org/abs/2010.09063) - uses JAX's JIT and VMAP to achieve faster differentially private than existing libraries.
- [__An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale__. Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby. _arXiv 2020_.](https://arxiv.org/abs/2010.11929) - image classification with transformer architecture using sequences of image patches, with results comparable to state-of-the-art CNNs while requiring substantially fewer computational resources.[^](#dosovitskiy2020)

<a name="posts" />

## Blog Posts

- [Using JAX to accelerate our research](https://deepmind.com/blog/article/using-jax-to-accelerate-our-research) - describes the state of JAX and the JAX ecosystem at DeepMind.
- [Getting started with JAX (MLPs, CNNs & RNNs)](https://roberttlange.github.io/posts/2020/03/blog-post-10/) - neural network building blocks from scratch with the basic JAX operators.
- [Plugging Into JAX](https://medium.com/swlh/plugging-into-jax-16c120ec3302) - compared Flax, Haiku, and Objax on the Kaggle flower classification challenge.
- [Meta-Learning in 50 Lines of JAX](https://blog.evjang.com/2019/02/maml-jax.html) - intro to both JAX and Meta-Learning.
- [Normalizing Flows in 100 Lines of JAX](https://blog.evjang.com/2019/07/nf-jax.html) - concise implementation of [RealNVP](https://arxiv.org/abs/1605.08803).
- [Differentiable Path Tracing on the GPU/TPU](https://blog.evjang.com/2019/11/jaxpt.html) - tutorial on implementing path tracing.
- [Ensemble networks](http://matpalm.com/blog/ensemble_nets) - ensemble nets are a method of representing an ensemble of models as one single logical model.
- [Out of distribution (OOD) detection](http://matpalm.com/blog/ood_using_focal_loss) - implements different methods for OOD detection.

<a name="community" />

## Community

- [JAX GitHub Discussions](https://github.com/google/jax/discussions)
- [Reddit](https://www.reddit.com/r/JAX/)

<a name="contribute" />

## Contribute

Contributions welcome! Read the [contribution guidelines](contributing.md) first.
