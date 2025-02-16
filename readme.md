<!--lint ignore double-link-->
# Awesome JAX [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)[<img src="https://raw.githubusercontent.com/google/jax/master/images/jax_logo_250px.png" alt="JAX Logo" align="right" height="100">](https://github.com/google/jax)

<!--lint ignore double-link-->
[JAX](https://github.com/google/jax) brings automatic differentiation and the [XLA compiler](https://www.tensorflow.org/xla) together through a [NumPy](https://numpy.org/)-like API for high performance machine learning research on accelerators like GPUs and TPUs.
<!--lint enable double-link-->

This is a curated list of awesome JAX libraries, projects, and other resources. Contributions are welcome!

## Contents

- [Libraries](#libraries)
- [Models and Projects](#models-and-projects)
- [Videos](#videos)
- [Papers](#papers)
- [Tutorials and Blog Posts](#tutorials-and-blog-posts)
- [Books](#books)
- [Community](#community)

<a name="libraries" />

## Libraries

- Neural Network Libraries
    - [Flax](https://github.com/google/flax) - Centered on flexibility and clarity. <img src="https://img.shields.io/github/stars/google/flax?style=social" align="center">
    - [Flax NNX](https://github.com/google/flax/tree/main/flax/nnx) - An evolution on Flax by the same team <img src="https://img.shields.io/github/stars/google/flax?style=social" align="center">
    - [Haiku](https://github.com/deepmind/dm-haiku) - Focused on simplicity, created by the authors of Sonnet at DeepMind. <img src="https://img.shields.io/github/stars/deepmind/dm-haiku?style=social" align="center">
    - [Objax](https://github.com/google/objax) - Has an object oriented design similar to PyTorch. <img src="https://img.shields.io/github/stars/google/objax?style=social" align="center">
    - [Elegy](https://poets-ai.github.io/elegy/) - A High Level API for Deep Learning in JAX. Supports Flax, Haiku, and Optax. <img src="https://img.shields.io/github/stars/poets-ai/elegy?style=social" align="center">
    - [Trax](https://github.com/google/trax) - "Batteries included" deep learning library focused on providing solutions for common workloads. <img src="https://img.shields.io/github/stars/google/trax?style=social" align="center">
    - [Jraph](https://github.com/deepmind/jraph) - Lightweight graph neural network library. <img src="https://img.shields.io/github/stars/deepmind/jraph?style=social" align="center">
    - [Neural Tangents](https://github.com/google/neural-tangents) - High-level API for specifying neural networks of both finite and _infinite_ width. <img src="https://img.shields.io/github/stars/google/neural-tangents?style=social" align="center">
    - [HuggingFace Transformers](https://github.com/huggingface/transformers) - Ecosystem of pretrained Transformers for a wide range of natural language tasks (Flax). <img src="https://img.shields.io/github/stars/huggingface/transformers?style=social" align="center">
    - [Equinox](https://github.com/patrick-kidger/equinox) - Callable PyTrees and filtered JIT/grad transformations => neural networks in JAX. <img src="https://img.shields.io/github/stars/patrick-kidger/equinox?style=social" align="center">
    - [Scenic](https://github.com/google-research/scenic) - A Jax Library for Computer Vision Research and Beyond.  <img src="https://img.shields.io/github/stars/google-research/scenic?style=social" align="center">
    - [Penzai](https://github.com/google-deepmind/penzai) - Prioritizes legibility, visualization, and easy editing of neural network models with composable tools and a simple mental model.  <img src="https://img.shields.io/github/stars/google-deepmind/penzai?style=social" align="center">
- [Levanter](https://github.com/stanford-crfm/levanter) - Legible, Scalable, Reproducible Foundation Models with Named Tensors and JAX.  <img src="https://img.shields.io/github/stars/stanford-crfm/levanter?style=social" align="center">
- [EasyLM](https://github.com/young-geng/EasyLM) - LLMs made easy: Pre-training, finetuning, evaluating and serving LLMs in JAX/Flax.  <img src="https://img.shields.io/github/stars/young-geng/EasyLM?style=social" align="center">
- [NumPyro](https://github.com/pyro-ppl/numpyro) - Probabilistic programming based on the Pyro library. <img src="https://img.shields.io/github/stars/pyro-ppl/numpyro?style=social" align="center">
- [Chex](https://github.com/deepmind/chex) - Utilities to write and test reliable JAX code. <img src="https://img.shields.io/github/stars/deepmind/chex?style=social" align="center">
- [Optax](https://github.com/deepmind/optax) - Gradient processing and optimization library. <img src="https://img.shields.io/github/stars/deepmind/optax?style=social" align="center">
- [RLax](https://github.com/deepmind/rlax) - Library for implementing reinforcement learning agents. <img src="https://img.shields.io/github/stars/deepmind/rlax?style=social" align="center">
- [JAX, M.D.](https://github.com/google/jax-md) - Accelerated, differential molecular dynamics. <img src="https://img.shields.io/github/stars/google/jax-md?style=social" align="center">
- [Coax](https://github.com/coax-dev/coax) - Turn RL papers into code, the easy way. <img src="https://img.shields.io/github/stars/coax-dev/coax?style=social" align="center">
- [Distrax](https://github.com/deepmind/distrax) - Reimplementation of TensorFlow Probability, containing probability distributions and bijectors. <img src="https://img.shields.io/github/stars/deepmind/distrax?style=social" align="center">
- [cvxpylayers](https://github.com/cvxgrp/cvxpylayers) - Construct differentiable convex optimization layers. <img src="https://img.shields.io/github/stars/cvxgrp/cvxpylayers?style=social" align="center">
- [TensorLy](https://github.com/tensorly/tensorly) - Tensor learning made simple. <img src="https://img.shields.io/github/stars/tensorly/tensorly?style=social" align="center">
- [NetKet](https://github.com/netket/netket) - Machine Learning toolbox for Quantum Physics. <img src="https://img.shields.io/github/stars/netket/netket?style=social" align="center">
- [Fortuna](https://github.com/awslabs/fortuna) - AWS library for Uncertainty Quantification in Deep Learning. <img src="https://img.shields.io/github/stars/awslabs/fortuna?style=social" align="center">
- [BlackJAX](https://github.com/blackjax-devs/blackjax) - Library of samplers for JAX. <img src="https://img.shields.io/github/stars/blackjax-devs/blackjax?style=social" align="center">

<a name="new-libraries" />

### New Libraries

This section contains libraries that are well-made and useful, but have not necessarily been battle-tested by a large userbase yet.

- Neural Network Libraries
    - [FedJAX](https://github.com/google/fedjax) - Federated learning in JAX, built on Optax and Haiku. <img src="https://img.shields.io/github/stars/google/fedjax?style=social" align="center">
    - [Equivariant MLP](https://github.com/mfinzi/equivariant-MLP) - Construct equivariant neural network layers. <img src="https://img.shields.io/github/stars/mfinzi/equivariant-MLP?style=social" align="center">
    - [jax-resnet](https://github.com/n2cholas/jax-resnet/) - Implementations and checkpoints for ResNet variants in Flax. <img src="https://img.shields.io/github/stars/n2cholas/jax-resnet?style=social" align="center">
    - [Parallax](https://github.com/srush/parallax) - Immutable Torch Modules for JAX. <img src="https://img.shields.io/github/stars/srush/parallax?style=social" align="center">
- Nonlinear Optimization
    - [Optimistix](https://github.com/patrick-kidger/optimistix) - Root finding, minimisation, fixed points, and least squares. <img src="https://img.shields.io/github/stars/deepmind/optax?style=social" align="center">
    - [JAXopt](https://github.com/google/jaxopt) - Hardware accelerated (GPU/TPU), batchable and differentiable optimizers in JAX. <img src="https://img.shields.io/github/stars/google/jaxopt?style=social" align="center">
- [jax-unirep](https://github.com/ElArkk/jax-unirep) - Library implementing the [UniRep model](https://www.nature.com/articles/s41592-019-0598-1) for protein machine learning applications. <img src="https://img.shields.io/github/stars/ElArkk/jax-unirep?style=social" align="center">
- [flowjax](https://github.com/danielward27/flowjax) - Distributions and normalizing flows built as equinox modules. <img src="https://img.shields.io/github/stars/danielward27/flowjax?style=social" align="center">
- [jax-flows](https://github.com/ChrisWaites/jax-flows) - Normalizing flows in JAX. <img src="https://img.shields.io/github/stars/ChrisWaites/jax-flows?style=social" align="center">
- [sklearn-jax-kernels](https://github.com/ExpectationMax/sklearn-jax-kernels) - `scikit-learn` kernel matrices using JAX. <img src="https://img.shields.io/github/stars/ExpectationMax/sklearn-jax-kernels?style=social" align="center">
- [jax-cosmo](https://github.com/DifferentiableUniverseInitiative/jax_cosmo) - Differentiable cosmology library. <img src="https://img.shields.io/github/stars/DifferentiableUniverseInitiative/jax_cosmo?style=social" align="center">
- [efax](https://github.com/NeilGirdhar/efax) - Exponential Families in JAX. <img src="https://img.shields.io/github/stars/NeilGirdhar/efax?style=social" align="center">
- [mpi4jax](https://github.com/PhilipVinc/mpi4jax) - Combine MPI operations with your Jax code on CPUs and GPUs. <img src="https://img.shields.io/github/stars/PhilipVinc/mpi4jax?style=social" align="center">
- [imax](https://github.com/4rtemi5/imax) - Image augmentations and transformations. <img src="https://img.shields.io/github/stars/4rtemi5/imax?style=social" align="center">
- [FlaxVision](https://github.com/rolandgvc/flaxvision) - Flax version of TorchVision. <img src="https://img.shields.io/github/stars/rolandgvc/flaxvision?style=social" align="center">
- [Oryx](https://github.com/tensorflow/probability/tree/master/spinoffs/oryx) - Probabilistic programming language based on program transformations.
- [Optimal Transport Tools](https://github.com/google-research/ott) - Toolbox that bundles utilities to solve optimal transport problems.
- [delta PV](https://github.com/romanodev/deltapv) - A photovoltaic simulator with automatic differentation. <img src="https://img.shields.io/github/stars/romanodev/deltapv?style=social" align="center">
- [jaxlie](https://github.com/brentyi/jaxlie) - Lie theory library for rigid body transformations and optimization. <img src="https://img.shields.io/github/stars/brentyi/jaxlie?style=social" align="center">
- [BRAX](https://github.com/google/brax) - Differentiable physics engine to simulate environments along with learning algorithms to train agents for these environments. <img src="https://img.shields.io/github/stars/google/brax?style=social" align="center">
- [flaxmodels](https://github.com/matthias-wright/flaxmodels) - Pretrained models for Jax/Flax. <img src="https://img.shields.io/github/stars/matthias-wright/flaxmodels?style=social" align="center">
- [CR.Sparse](https://github.com/carnotresearch/cr-sparse) - XLA accelerated algorithms for sparse representations and compressive sensing. <img src="https://img.shields.io/github/stars/carnotresearch/cr-sparse?style=social" align="center">
- [exojax](https://github.com/HajimeKawahara/exojax) - Automatic differentiable spectrum modeling of exoplanets/brown dwarfs compatible to JAX. <img src="https://img.shields.io/github/stars/HajimeKawahara/exojax?style=social" align="center">
- [PIX](https://github.com/deepmind/dm_pix) - PIX is an image processing library in JAX, for JAX. <img src="https://img.shields.io/github/stars/deepmind/dm_pix?style=social" align="center">
- [bayex](https://github.com/alonfnt/bayex) - Bayesian Optimization powered by JAX. <img src="https://img.shields.io/github/stars/alonfnt/bayex?style=social" align="center">
- [JaxDF](https://github.com/ucl-bug/jaxdf) - Framework for differentiable simulators with arbitrary discretizations. <img src="https://img.shields.io/github/stars/ucl-bug/jaxdf?style=social" align="center">
- [tree-math](https://github.com/google/tree-math) - Convert functions that operate on arrays into functions that operate on PyTrees. <img src="https://img.shields.io/github/stars/google/tree-math?style=social" align="center">
- [jax-models](https://github.com/DarshanDeshpande/jax-models) - Implementations of research papers originally without code or code written with frameworks other than JAX. <img src="https://img.shields.io/github/stars/DarshanDeshpande/jax-modelsa?style=social" align="center">
- [PGMax](https://github.com/vicariousinc/PGMax) - A framework for building discrete Probabilistic Graphical Models (PGM's) and running inference inference on them via JAX. <img src="https://img.shields.io/github/stars/vicariousinc/pgmax?style=social" align="center">
- [EvoJAX](https://github.com/google/evojax) - Hardware-Accelerated Neuroevolution <img src="https://img.shields.io/github/stars/google/evojax?style=social" align="center">
- [evosax](https://github.com/RobertTLange/evosax) - JAX-Based Evolution Strategies <img src="https://img.shields.io/github/stars/RobertTLange/evosax?style=social" align="center">
- [SymJAX](https://github.com/SymJAX/SymJAX) - Symbolic CPU/GPU/TPU programming. <img src="https://img.shields.io/github/stars/SymJAX/SymJAX?style=social" align="center">
- [mcx](https://github.com/rlouf/mcx) - Express & compile probabilistic programs for performant inference. <img src="https://img.shields.io/github/stars/rlouf/mcx?style=social" align="center">
- [Einshape](https://github.com/deepmind/einshape) - DSL-based reshaping library for JAX and other frameworks. <img src="https://img.shields.io/github/stars/deepmind/einshape?style=social" align="center">
- [ALX](https://github.com/google-research/google-research/tree/master/alx) - Open-source library for distributed matrix factorization using Alternating Least Squares, more info in [_ALX: Large Scale Matrix Factorization on TPUs_](https://arxiv.org/abs/2112.02194).
- [Diffrax](https://github.com/patrick-kidger/diffrax) - Numerical differential equation solvers in JAX. <img src="https://img.shields.io/github/stars/patrick-kidger/diffrax?style=social" align="center">
- [tinygp](https://github.com/dfm/tinygp) - The _tiniest_ of Gaussian process libraries in JAX. <img src="https://img.shields.io/github/stars/dfm/tinygp?style=social" align="center">
- [gymnax](https://github.com/RobertTLange/gymnax) - Reinforcement Learning Environments with the well-known gym API. <img src="https://img.shields.io/github/stars/RobertTLange/gymnax?style=social" align="center">
- [Mctx](https://github.com/deepmind/mctx) - Monte Carlo tree search algorithms in native JAX. <img src="https://img.shields.io/github/stars/deepmind/mctx?style=social" align="center">
- [KFAC-JAX](https://github.com/deepmind/kfac-jax) - Second Order Optimization with Approximate Curvature for NNs. <img src="https://img.shields.io/github/stars/deepmind/kfac-jax?style=social" align="center">
- [TF2JAX](https://github.com/deepmind/tf2jax) - Convert functions/graphs to JAX functions. <img src="https://img.shields.io/github/stars/deepmind/tf2jax?style=social" align="center">
- [jwave](https://github.com/ucl-bug/jwave) - A library for differentiable acoustic simulations <img src="https://img.shields.io/github/stars/ucl-bug/jwave?style=social" align="center">
- [GPJax](https://github.com/thomaspinder/GPJax) - Gaussian processes in JAX.
- [Jumanji](https://github.com/instadeepai/jumanji) - A Suite of Industry-Driven Hardware-Accelerated RL Environments written in JAX. <img src="https://img.shields.io/github/stars/instadeepai/jumanji?style=social" align="center">
- [Eqxvision](https://github.com/paganpasta/eqxvision) - Equinox version of Torchvision. <img src="https://img.shields.io/github/stars/paganpasta/eqxvision?style=social" align="center">
- [JAXFit](https://github.com/dipolar-quantum-gases/jaxfit) - Accelerated curve fitting library for nonlinear least-squares problems (see [arXiv paper](https://arxiv.org/abs/2208.12187)). <img src="https://img.shields.io/github/stars/dipolar-quantum-gases/jaxfit?style=social" align="center">
- [econpizza](https://github.com/gboehl/econpizza) - Solve macroeconomic models with hetereogeneous agents using JAX. <img src="https://img.shields.io/github/stars/gboehl/econpizza?style=social" align="center">
- [SPU](https://github.com/secretflow/spu) - A domain-specific compiler and runtime suite to run JAX code with MPC(Secure Multi-Party Computation). <img src="https://img.shields.io/github/stars/secretflow/spu?style=social" align="center">
- [jax-tqdm](https://github.com/jeremiecoullon/jax-tqdm) - Add a tqdm progress bar to JAX scans and loops. <img src="https://img.shields.io/github/stars/jeremiecoullon/jax-tqdm?style=social" align="center">
- [safejax](https://github.com/alvarobartt/safejax) - Serialize JAX, Flax, Haiku, or Objax model params with 🤗`safetensors`. <img src="https://img.shields.io/github/stars/alvarobartt/safejax?style=social" align="center">
- [Kernex](https://github.com/ASEM000/kernex) - Differentiable stencil decorators in JAX. <img src="https://img.shields.io/github/stars/ASEM000/kernex?style=social" align="center">
- [MaxText](https://github.com/google/maxtext) - A simple, performant and scalable Jax LLM written in pure Python/Jax and targeting Google Cloud TPUs. <img src="https://img.shields.io/github/stars/google/maxtext?style=social" align="center">
- [Pax](https://github.com/google/paxml) - A Jax-based machine learning framework for training large scale models. <img src="https://img.shields.io/github/stars/google/paxml?style=social" align="center">
- [Praxis](https://github.com/google/praxis) - The layer library for Pax with a goal to be usable by other JAX-based ML projects. <img src="https://img.shields.io/github/stars/google/praxis?style=social" align="center">
- [purejaxrl](https://github.com/luchris429/purejaxrl) - Vectorisable, end-to-end RL algorithms in JAX. <img src="https://img.shields.io/github/stars/luchris429/purejaxrl?style=social" align="center">
- [Lorax](https://github.com/davisyoshida/lorax) - Automatically apply LoRA to JAX models (Flax, Haiku, etc.)
- [SCICO](https://github.com/lanl/scico) - Scientific computational imaging in JAX. <img src="https://img.shields.io/github/stars/lanl/scico?style=social" align="center">
- [Spyx](https://github.com/kmheckel/spyx) - Spiking Neural Networks in JAX for machine learning on neuromorphic hardware. <img src="https://img.shields.io/github/stars/kmheckel/spyx?style=social" align="center">
- Brain Dynamics Programming Ecosystem
    - [BrainPy](https://github.com/brainpy/BrainPy) - Brain Dynamics Programming in Python. <img src="https://img.shields.io/github/stars/brainpy/BrainPy?style=social" align="center">
    - [brainunit](https://github.com/chaobrain/brainunit) - Physical units and unit-aware mathematical system in JAX. <img src="https://img.shields.io/github/stars/chaobrain/brainunit?style=social" align="center">
    - [dendritex](https://github.com/chaobrain/dendritex) - Dendritic Modeling in JAX. <img src="https://img.shields.io/github/stars/chaobrain/dendritex?style=social" align="center">
    - [brainstate](https://github.com/chaobrain/brainstate) - State-based Transformation System for Program Compilation and Augmentation. <img src="https://img.shields.io/github/stars/chaobrain/brainstate?style=social" align="center">
    - [braintaichi](https://github.com/chaobrain/braintaichi) - Leveraging Taichi Lang to customize brain dynamics operators. <img src="https://img.shields.io/github/stars/chaobrain/braintaichi?style=social" align="center">
- [OTT-JAX](https://github.com/ott-jax/ott) - Optimal transport tools in JAX. <img src="https://img.shields.io/github/stars/ott-jax/ott?style=social" align="center">
- [QDax](https://github.com/adaptive-intelligent-robotics/QDax) - Quality Diversity optimization in Jax. <img src="https://img.shields.io/github/stars/adaptive-intelligent-robotics/QDax?style=social" align="center">
- [JAX Toolbox](https://github.com/NVIDIA/JAX-Toolbox) - Nightly CI and optimized examples for JAX on NVIDIA GPUs using libraries such as T5x, Paxml, and Transformer Engine. <img src="https://img.shields.io/github/stars/NVIDIA/JAX-Toolbox?style=social" align="center">
- [Pgx](http://github.com/sotetsuk/pgx) - Vectorized board game environments for RL with an AlphaZero example. <img src="https://img.shields.io/github/stars/sotetsuk/pgx?style=social" align="center">
- [EasyDeL](https://github.com/erfanzar/EasyDeL) - EasyDeL 🔮 is an OpenSource Library to make your training faster and more Optimized With cool Options for training and serving (Llama, MPT, Mixtral, Falcon, etc) in JAX <img src="https://img.shields.io/github/stars/erfanzar/EasyDeL?style=social" align="center">
- [XLB](https://github.com/Autodesk/XLB) - A Differentiable Massively Parallel Lattice Boltzmann Library in Python for Physics-Based Machine Learning. <img src="https://img.shields.io/github/stars/Autodesk/XLB?style=social" align="center">
- [dynamiqs](https://github.com/dynamiqs/dynamiqs) - High-performance and differentiable simulations of quantum systems with JAX. <img src="https://img.shields.io/github/stars/dynamiqs/dynamiqs?style=social" align="center">
- [foragax](https://github.com/i-m-iron-man/Foragax) - Agent-Based modelling framework in JAX.  <img src="https://img.shields.io/github/stars/i-m-iron-man/Foragax?style=social" align="center">
- [tmmax](https://github.com/bahremsd/tmmax) - Vectorized calculation of optical properties in thin-film structures using JAX. Swiss Army knife tool for thin-film optics research <img src="https://img.shields.io/github/stars/bahremsd/tmmax" align="center">
- [Coreax](https://github.com/gchq/coreax) - Algorithms for finding coresets to compress large datasets while retaining their statistical properties. <img src="https://img.shields.io/github/stars/gchq/coreax?style=social" align="center">
- [NAVIX](https://github.com/epignatelli/navix) - A reimplementation of MiniGrid, a Reinforcement Learning environment, in JAX <img src="https://img.shields.io/github/stars/epignatelli/navix?style=social" align="center">


<a name="models-and-projects" />

## Models and Projects

### JAX

- [Fourier Feature Networks](https://github.com/tancik/fourier-feature-networks) - Official implementation of [_Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains_](https://people.eecs.berkeley.edu/~bmild/fourfeat).
- [kalman-jax](https://github.com/AaltoML/kalman-jax) - Approximate inference for Markov (i.e., temporal) Gaussian processes using iterated Kalman filtering and smoothing.
- [jaxns](https://github.com/Joshuaalbert/jaxns) - Nested sampling in JAX.
- [Amortized Bayesian Optimization](https://github.com/google-research/google-research/tree/master/amortized_bo) - Code related to [_Amortized Bayesian Optimization over Discrete Spaces_](http://www.auai.org/uai2020/proceedings/329_main_paper.pdf).
- [Accurate Quantized Training](https://github.com/google-research/google-research/tree/master/aqt) - Tools and libraries for running and analyzing neural network quantization experiments in JAX and Flax.
- [BNN-HMC](https://github.com/google-research/google-research/tree/master/bnn_hmc) - Implementation for the paper [_What Are Bayesian Neural Network Posteriors Really Like?_](https://arxiv.org/abs/2104.14421).
- [JAX-DFT](https://github.com/google-research/google-research/tree/master/jax_dft) - One-dimensional density functional theory (DFT) in JAX, with implementation of [_Kohn-Sham equations as regularizer: building prior knowledge into machine-learned physics_](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.126.036401).
- [Robust Loss](https://github.com/google-research/google-research/tree/master/robust_loss_jax) - Reference code for the paper [_A General and Adaptive Robust Loss Function_](https://arxiv.org/abs/1701.03077).
- [Symbolic Functionals](https://github.com/google-research/google-research/tree/master/symbolic_functionals) - Demonstration from [_Evolving symbolic density functionals_](https://arxiv.org/abs/2203.02540).
- [TriMap](https://github.com/google-research/google-research/tree/master/trimap) - Official JAX implementation of [_TriMap: Large-scale Dimensionality Reduction Using Triplets_](https://arxiv.org/abs/1910.00204).

### Flax

- [DeepSeek-R1-Flax-1.5B-Distill](https://github.com/J-Rosser-UK/Torch2Jax-DeepSeek-R1-Distill-Qwen-1.5B) - Flax implementation of DeepSeek-R1 1.5B distilled reasoning LLM.
- [Performer](https://github.com/google-research/google-research/tree/master/performer/fast_attention/jax) - Flax implementation of the Performer (linear transformer via FAVOR+) architecture.
- [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf) - Implementation of [_NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis_](http://www.matthewtancik.com/nerf) with multi-device GPU/TPU support.
- [mip-NeRF](https://github.com/google/mipnerf) - Official implementation of [_Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields_](https://jonbarron.info/mipnerf).
- [RegNeRF](https://github.com/google-research/google-research/tree/master/regnerf) - Official implementation of [_RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs_](https://m-niemeyer.github.io/regnerf/).
- [JaxNeuS](https://github.com/huangjuite/jaxneus) - Implementation of [_NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction_](https://lingjie0206.github.io/papers/NeuS/)
- [Big Transfer (BiT)](https://github.com/google-research/big_transfer) - Implementation of [_Big Transfer (BiT): General Visual Representation Learning_](https://arxiv.org/abs/1912.11370).
- [JAX RL](https://github.com/ikostrikov/jax-rl) - Implementations of reinforcement learning algorithms.
- [gMLP](https://github.com/SauravMaheshkar/gMLP) - Implementation of [_Pay Attention to MLPs_](https://arxiv.org/abs/2105.08050).
- [MLP Mixer](https://github.com/SauravMaheshkar/MLP-Mixer) - Minimal implementation of [_MLP-Mixer: An all-MLP Architecture for Vision_](https://arxiv.org/abs/2105.01601).
- [Distributed Shampoo](https://github.com/google-research/google-research/tree/master/scalable_shampoo) - Implementation of [_Second Order Optimization Made Practical_](https://arxiv.org/abs/2002.09018).
- [NesT](https://github.com/google-research/nested-transformer) - Official implementation of [_Aggregating Nested Transformers_](https://arxiv.org/abs/2105.12723).
- [XMC-GAN](https://github.com/google-research/xmcgan_image_generation) - Official implementation of [_Cross-Modal Contrastive Learning for Text-to-Image Generation_](https://arxiv.org/abs/2101.04702).
- [FNet](https://github.com/google-research/google-research/tree/master/f_net) - Official implementation of [_FNet: Mixing Tokens with Fourier Transforms_](https://arxiv.org/abs/2105.03824).
- [GFSA](https://github.com/google-research/google-research/tree/master/gfsa) - Official implementation of [_Learning Graph Structure With A Finite-State Automaton Layer_](https://arxiv.org/abs/2007.04929).
- [IPA-GNN](https://github.com/google-research/google-research/tree/master/ipagnn) - Official implementation of [_Learning to Execute Programs with Instruction Pointer Attention Graph Neural Networks_](https://arxiv.org/abs/2010.12621).
- [Flax Models](https://github.com/google-research/google-research/tree/master/flax_models) - Collection of models and methods implemented in Flax.
- [Protein LM](https://github.com/google-research/google-research/tree/master/protein_lm) - Implements BERT and autoregressive models for proteins, as described in [_Biological Structure and Function Emerge from Scaling Unsupervised Learning to 250 Million Protein Sequences_](https://www.biorxiv.org/content/10.1101/622803v1.full) and [_ProGen: Language Modeling for Protein Generation_](https://www.biorxiv.org/content/10.1101/2020.03.07.982272v2).
- [Slot Attention](https://github.com/google-research/google-research/tree/master/ptopk_patch_selection) - Reference implementation for [_Differentiable Patch Selection for Image Recognition_](https://arxiv.org/abs/2104.03059).
- [Vision Transformer](https://github.com/google-research/vision_transformer) - Official implementation of [_An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale_](https://arxiv.org/abs/2010.11929).
- [FID computation](https://github.com/matthias-wright/jax-fid) - Port of [mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid) to Flax.
- [ARDM](https://github.com/google-research/google-research/tree/master/autoregressive_diffusion) - Official implementation of [_Autoregressive Diffusion Models_](https://arxiv.org/abs/2110.02037).
- [D3PM](https://github.com/google-research/google-research/tree/master/d3pm) - Official implementation of [_Structured Denoising Diffusion Models in Discrete State-Spaces_](https://arxiv.org/abs/2107.03006).
- [Gumbel-max Causal Mechanisms](https://github.com/google-research/google-research/tree/master/gumbel_max_causal_gadgets) - Code for [_Learning Generalized Gumbel-max Causal Mechanisms_](https://arxiv.org/abs/2111.06888), with extra code in [GuyLor/gumbel_max_causal_gadgets_part2](https://github.com/GuyLor/gumbel_max_causal_gadgets_part2).
- [Latent Programmer](https://github.com/google-research/google-research/tree/master/latent_programmer) - Code for the ICML 2021 paper [_Latent Programmer: Discrete Latent Codes for Program Synthesis_](https://arxiv.org/abs/2012.00377).
- [SNeRG](https://github.com/google-research/google-research/tree/master/snerg) - Official implementation of [_Baking Neural Radiance Fields for Real-Time View Synthesis_](https://phog.github.io/snerg).
- [Spin-weighted Spherical CNNs](https://github.com/google-research/google-research/tree/master/spin_spherical_cnns) - Adaptation of [_Spin-Weighted Spherical CNNs_](https://arxiv.org/abs/2006.10731).
- [VDVAE](https://github.com/google-research/google-research/tree/master/vdvae_flax) - Adaptation of [_Very Deep VAEs Generalize Autoregressive Models and Can Outperform Them on Images_](https://arxiv.org/abs/2011.10650), original code at [openai/vdvae](https://github.com/openai/vdvae).
- [MUSIQ](https://github.com/google-research/google-research/tree/master/musiq) - Checkpoints and model inference code for the ICCV 2021 paper [_MUSIQ: Multi-scale Image Quality Transformer_](https://arxiv.org/abs/2108.05997)
- [AQuaDem](https://github.com/google-research/google-research/tree/master/aquadem) - Official implementation of [_Continuous Control with Action Quantization from Demonstrations_](https://arxiv.org/abs/2110.10149).
- [Combiner](https://github.com/google-research/google-research/tree/master/combiner) - Official implementation of [_Combiner: Full Attention Transformer with Sparse Computation Cost_](https://arxiv.org/abs/2107.05768).
- [Dreamfields](https://github.com/google-research/google-research/tree/master/dreamfields) - Official implementation of the ICLR 2022 paper [_Progressive Distillation for Fast Sampling of Diffusion Models_](https://ajayj.com/dreamfields).
- [GIFT](https://github.com/google-research/google-research/tree/master/gift) - Official implementation of [_Gradual Domain Adaptation in the Wild:When Intermediate Distributions are Absent_](https://arxiv.org/abs/2106.06080).
- [Light Field Neural Rendering](https://github.com/google-research/google-research/tree/master/light_field_neural_rendering) - Official implementation of [_Light Field Neural Rendering_](https://arxiv.org/abs/2112.09687).
- [Sharpened Cosine Similarity in JAX by Raphael Pisoni](https://colab.research.google.com/drive/1KUKFEMneQMS3OzPYnWZGkEnry3PdzCfn?usp=sharing) -  A JAX/Flax implementation of the Sharpened Cosine Similarity layer.
- [GNNs for Solving Combinatorial Optimization Problems](https://github.com/IvanIsCoding/GNN-for-Combinatorial-Optimization) -  A JAX + Flax implementation of [Combinatorial Optimization with Physics-Inspired Graph Neural Networks](https://arxiv.org/abs/2107.01188).
- [DETR](https://github.com/MasterSkepticista/detr) - Flax implementation of [_DETR: End-to-end Object Detection with Transformers_](https://github.com/facebookresearch/detr) using Sinkhorn solver and parallel bipartite matching.

### Haiku

- [AlphaFold](https://github.com/deepmind/alphafold) - Implementation of the inference pipeline of AlphaFold v2.0, presented in [_Highly accurate protein structure prediction with AlphaFold_](https://www.nature.com/articles/s41586-021-03819-2).
- [Adversarial Robustness](https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness) - Reference code for [_Uncovering the Limits of Adversarial Training against Norm-Bounded Adversarial Examples_](https://arxiv.org/abs/2010.03593) and [_Fixing Data Augmentation to Improve Adversarial Robustness_](https://arxiv.org/abs/2103.01946).
- [Bootstrap Your Own Latent](https://github.com/deepmind/deepmind-research/tree/master/byol) - Implementation for the paper [_Bootstrap your own latent: A new approach to self-supervised Learning_](https://arxiv.org/abs/2006.07733).
- [Gated Linear Networks](https://github.com/deepmind/deepmind-research/tree/master/gated_linear_networks) - GLNs are a family of backpropagation-free neural networks.
- [Glassy Dynamics](https://github.com/deepmind/deepmind-research/tree/master/glassy_dynamics) - Open source implementation of the paper [_Unveiling the predictive power of static structure in glassy systems_](https://www.nature.com/articles/s41567-020-0842-8).
- [MMV](https://github.com/deepmind/deepmind-research/tree/master/mmv) - Code for the models in [_Self-Supervised MultiModal Versatile Networks_](https://arxiv.org/abs/2006.16228).
- [Normalizer-Free Networks](https://github.com/deepmind/deepmind-research/tree/master/nfnets) - Official Haiku implementation of [_NFNets_](https://arxiv.org/abs/2102.06171).
- [NuX](https://github.com/Information-Fusion-Lab-Umass/NuX) - Normalizing flows with JAX.
- [OGB-LSC](https://github.com/deepmind/deepmind-research/tree/master/ogb_lsc) - This repository contains DeepMind's entry to the [PCQM4M-LSC](https://ogb.stanford.edu/kddcup2021/pcqm4m/) (quantum chemistry) and [MAG240M-LSC](https://ogb.stanford.edu/kddcup2021/mag240m/) (academic graph)
tracks of the [OGB Large-Scale Challenge](https://ogb.stanford.edu/kddcup2021/) (OGB-LSC).
- [Persistent Evolution Strategies](https://github.com/google-research/google-research/tree/master/persistent_es) - Code used for the paper [_Unbiased Gradient Estimation in Unrolled Computation Graphs with Persistent Evolution Strategies_](http://proceedings.mlr.press/v139/vicol21a.html).
- [Two Player Auction Learning](https://github.com/degregat/two-player-auctions) - JAX implementation of the paper [_Auction learning as a two-player game_](https://arxiv.org/abs/2006.05684).
- [WikiGraphs](https://github.com/deepmind/deepmind-research/tree/master/wikigraphs) - Baseline code to reproduce results in [_WikiGraphs: A Wikipedia Text - Knowledge Graph Paired Datase_](https://aclanthology.org/2021.textgraphs-1.7).

### Trax

- [Reformer](https://github.com/google/trax/tree/master/trax/models/reformer) - Implementation of the Reformer (efficient transformer) architecture.

### NumPyro

- [lqg](https://github.com/RothkopfLab/lqg) - Official implementation of Bayesian inverse optimal control for linear-quadratic Gaussian problems from the paper [_Putting perception into action with inverse optimal control for continuous psychophysics_](https://elifesciences.org/articles/76635)


<a name="videos" />

## Videos

- [NeurIPS 2020: JAX Ecosystem Meetup](https://www.youtube.com/watch?v=iDxJxIyzSiM) - JAX, its use at DeepMind, and discussion between engineers, scientists, and JAX core team.
- [Introduction to JAX](https://youtu.be/0mVmRHMaOJ4) - Simple neural network from scratch in JAX.
- [JAX: Accelerated Machine Learning Research | SciPy 2020 | VanderPlas](https://youtu.be/z-WSrQDXkuM) - JAX's core design, how it's powering new research, and how you can start using it.
- [Bayesian Programming with JAX + NumPyro — Andy Kitchen](https://youtu.be/CecuWGpoztw) - Introduction to Bayesian modelling using NumPyro.
- [JAX: Accelerated machine-learning research via composable function transformations in Python | NeurIPS 2019 | Skye Wanderman-Milne](https://slideslive.com/38923687/jax-accelerated-machinelearning-research-via-composable-function-transformations-in-python) - JAX intro presentation in [_Program Transformations for Machine Learning_](https://program-transformations.github.io) workshop.
- [JAX on Cloud TPUs | NeurIPS 2020 | Skye Wanderman-Milne and James Bradbury](https://drive.google.com/file/d/1jKxefZT1xJDUxMman6qrQVed7vWI0MIn/edit) - Presentation of TPU host access with demo.
- [Deep Implicit Layers - Neural ODEs, Deep Equilibirum Models, and Beyond | NeurIPS 2020](https://slideslive.com/38935810/deep-implicit-layers-neural-odes-equilibrium-models-and-beyond) - Tutorial created by Zico Kolter, David Duvenaud, and Matt Johnson with Colab notebooks avaliable in [_Deep Implicit Layers_](http://implicit-layers-tutorial.org).
- [Solving y=mx+b with Jax on a TPU Pod slice - Mat Kelcey](http://matpalm.com/blog/ymxb_pod_slice/) - A four part YouTube tutorial series with Colab notebooks that starts with Jax fundamentals and moves up to training with a data parallel approach on a v3-32 TPU Pod slice.
- [JAX, Flax & Transformers 🤗](https://github.com/huggingface/transformers/blob/9160d81c98854df44b1d543ce5d65a6aa28444a2/examples/research_projects/jax-projects/README.md#talks) - 3 days of talks around JAX / Flax, Transformers, large-scale language modeling and other great topics.

<a name="papers" />

## Papers

This section contains papers focused on JAX (e.g. JAX-based library whitepapers, research on JAX, etc). Papers implemented in JAX are listed in the [Models/Projects](#projects) section.

<!--lint ignore awesome-list-item-->
- [__Compiling machine learning programs via high-level tracing__. Roy Frostig, Matthew James Johnson, Chris Leary. _MLSys 2018_.](https://mlsys.org/Conferences/doc/2018/146.pdf) - White paper describing an early version of JAX, detailing how computation is traced and compiled.
- [__JAX, M.D.: A Framework for Differentiable Physics__. Samuel S. Schoenholz, Ekin D. Cubuk. _NeurIPS 2020_.](https://arxiv.org/abs/1912.04232) - Introduces JAX, M.D., a differentiable physics library which includes simulation environments, interaction potentials, neural networks, and more.
- [__Enabling Fast Differentially Private SGD via Just-in-Time Compilation and Vectorization__. Pranav Subramani, Nicholas Vadivelu, Gautam Kamath. _arXiv 2020_.](https://arxiv.org/abs/2010.09063) - Uses JAX's JIT and VMAP to achieve faster differentially private than existing libraries.
- [__XLB: A Differentiable Massively Parallel Lattice Boltzmann Library in Python__. Mohammadmehdi Ataei, Hesam Salehipour. _arXiv 2023_.](https://arxiv.org/abs/2311.16080) - White paper describing the XLB library: benchmarks, validations, and more details about the library.

<!--lint enable awesome-list-item-->

<a name="tutorials-and-blog-posts" />

## Tutorials and Blog Posts

- [Using JAX to accelerate our research by David Budden and Matteo Hessel](https://deepmind.com/blog/article/using-jax-to-accelerate-our-research) - Describes the state of JAX and the JAX ecosystem at DeepMind.
- [Getting started with JAX (MLPs, CNNs & RNNs) by Robert Lange](https://roberttlange.github.io/posts/2020/03/blog-post-10/) - Neural network building blocks from scratch with the basic JAX operators.
- [Learn JAX: From Linear Regression to Neural Networks by Rito Ghosh](https://www.kaggle.com/code/truthr/jax-0) - A gentle introduction to JAX and using it to implement Linear and Logistic Regression, and Neural Network models and using them to solve real world problems.
- [Tutorial: image classification with JAX and Flax Linen by 8bitmp3](https://github.com/8bitmp3/JAX-Flax-Tutorial-Image-Classification-with-Linen) - Learn how to create a simple convolutional network with the Linen API by Flax and train it to recognize handwritten digits.
- [Plugging Into JAX by Nick Doiron](https://medium.com/swlh/plugging-into-jax-16c120ec3302) - Compares Flax, Haiku, and Objax on the Kaggle flower classification challenge.
- [Meta-Learning in 50 Lines of JAX by Eric Jang](https://blog.evjang.com/2019/02/maml-jax.html) - Introduction to both JAX and Meta-Learning.
- [Normalizing Flows in 100 Lines of JAX by Eric Jang](https://blog.evjang.com/2019/07/nf-jax.html) - Concise implementation of [RealNVP](https://arxiv.org/abs/1605.08803).
- [Differentiable Path Tracing on the GPU/TPU by Eric Jang](https://blog.evjang.com/2019/11/jaxpt.html) - Tutorial on implementing path tracing.
- [Ensemble networks by Mat Kelcey](http://matpalm.com/blog/ensemble_nets) - Ensemble nets are a method of representing an ensemble of models as one single logical model.
- [Out of distribution (OOD) detection by Mat Kelcey](http://matpalm.com/blog/ood_using_focal_loss) - Implements different methods for OOD detection.
- [Understanding Autodiff with JAX by Srihari Radhakrishna](https://www.radx.in/jax.html) - Understand how autodiff works using JAX.
- [From PyTorch to JAX: towards neural net frameworks that purify stateful code by Sabrina J. Mielke](https://sjmielke.com/jax-purify.htm) - Showcases how to go from a PyTorch-like style of coding to a more Functional-style of coding.
- [Extending JAX with custom C++ and CUDA code by Dan Foreman-Mackey](https://github.com/dfm/extending-jax) - Tutorial demonstrating the infrastructure required to provide custom ops in JAX.
- [Evolving Neural Networks in JAX by Robert Tjarko Lange](https://roberttlange.github.io/posts/2021/02/cma-es-jax/) - Explores how JAX can power the next generation of scalable neuroevolution algorithms.
- [Exploring hyperparameter meta-loss landscapes with JAX by Luke Metz](http://lukemetz.com/exploring-hyperparameter-meta-loss-landscapes-with-jax/) - Demonstrates how to use JAX to perform inner-loss optimization with SGD and Momentum, outer-loss optimization with gradients, and outer-loss optimization using evolutionary strategies.
- [Deterministic ADVI in JAX by Martin Ingram](https://martiningram.github.io/deterministic-advi/) - Walk through of implementing automatic differentiation variational inference (ADVI) easily and cleanly with JAX.
- [Evolved channel selection by Mat Kelcey](http://matpalm.com/blog/evolved_channel_selection/) - Trains a classification model robust to different combinations of input channels at different resolutions, then uses a genetic algorithm to decide the best combination for a particular loss.
- [Introduction to JAX by Kevin Murphy](https://colab.research.google.com/github/probml/probml-notebooks/blob/main/notebooks/jax_intro.ipynb) - Colab that introduces various aspects of the language and applies them to simple ML problems.
- [Writing an MCMC sampler in JAX by Jeremie Coullon](https://www.jeremiecoullon.com/2020/11/10/mcmcjax3ways/) - Tutorial on the different ways to write an MCMC sampler in JAX along with speed benchmarks.
- [How to add a progress bar to JAX scans and loops by Jeremie Coullon](https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/) - Tutorial on how to add a progress bar to compiled loops in JAX using the `host_callback` module.
- [Get started with JAX by Aleksa Gordić](https://github.com/gordicaleksa/get-started-with-JAX) - A series of notebooks and videos going from zero JAX knowledge to building neural networks in Haiku.
- [Writing a Training Loop in JAX + FLAX by Saurav Maheshkar and Soumik Rakshit](https://wandb.ai/jax-series/simple-training-loop/reports/Writing-a-Training-Loop-in-JAX-FLAX--VmlldzoyMzA4ODEy) - A tutorial on writing a simple end-to-end training and evaluation pipeline in JAX, Flax and Optax.
- [Implementing NeRF in JAX by Soumik Rakshit and Saurav Maheshkar](https://wandb.ai/wandb/nerf-jax/reports/Implementing-NeRF-in-JAX--VmlldzoxODA2NDk2?galleryTag=jax) - A tutorial on 3D volumetric rendering of scenes represented by Neural Radiance Fields in JAX.
- [Deep Learning tutorials with JAX+Flax by Phillip Lippe](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial2/Introduction_to_JAX.html) - A series of notebooks explaining various deep learning concepts, from basics (e.g. intro to JAX/Flax, activiation functions) to recent advances (e.g., Vision Transformers, SimCLR), with translations to PyTorch.
- [Achieving 4000x Speedups with PureJaxRL](https://chrislu.page/blog/meta-disco/) - A blog post on how JAX can massively speedup RL training through vectorisation.
- [Simple PDE solver + Constrained Optimization with JAX by Philip Mocz](https://levelup.gitconnected.com/create-your-own-automatically-differentiable-simulation-with-python-jax-46951e120fbb?sk=e8b9213dd2c6a5895926b2695d28e4aa) - A simple example of solving the advection-diffusion equations with JAX and using it in a constrained optimization problem to find initial conditions that yield desired result.

<a name="books" />

## Books

- [Jax in Action](https://www.manning.com/books/jax-in-action) - A hands-on guide to using JAX for deep learning and other mathematically-intensive applications.

<a name="community" />

## Community

- [JaxLLM (Unofficial) Discord](https://discord.com/channels/1107832795377713302/1107832795688083561)
- [JAX GitHub Discussions](https://github.com/google/jax/discussions)
- [Reddit](https://www.reddit.com/r/JAX/)

## Contributing

Contributions welcome! Read the [contribution guidelines](contributing.md) first.
