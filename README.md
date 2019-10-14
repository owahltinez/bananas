# Bananas

This library is a collection of tools designed to serve as the starting point for a machine learning
(ML) framework. It provides the basic building blocks that every ML framework should have, including:

- Interfaces for a common defition of `fit`, `predict`, `score`, etc. across frameworks.
- A comprehensive suite of tests that ensures compliance with those interfaces.
- Useful constructs for a data pipeline, such as
  [Pipeline](core/pipeline.html#bananas.core.pipeline.Pipeline) or
  [Transformer](transformers).
- Miscellaneous utilities, such as [sampling](sampling),
  [preprocessing](preprocessing) or
  [image processing functions](utils/images.html).


## Why another framework?

Why not instead use or extend [Scikit-Learn, Tensorflow, Pytorch, Keras...]? All of those projects
have their merits and tons of related utilities. However, each one of them has its own definition of
fundamental methods such as `fit`. For example, in Scikit-Learn, `fit` is expected to receive
**all** data to be learned. That is impractical with the size of datasets nowadays.

The goal of this project is to provide a starting point for the next iteration of ML frameworks, so
that they can be compatible with one another. Imagine a world where we can setup a data pipeline
using a data transformer from Scikit-Learn, followed by an embedding from Tensorflow, which is then
followed by a classifier from Pytorch. While technically possible, doing that currently would be a
tremendous amount of work -- which is why most frameworks reimplement the same common utilities over
and over, so developers can use them with ease.

Additionally, nothing prevents us from writing wrappers around current frameworks so they become
compliant with the interfaces defined by this library. For example, take a look at [TODO](example.com),
which is a wrapper around Pytorch imlementing compliant estimators and can take advantage of all the
tools built around this library.


## Design

### Input data format
Input features are expected to be in **column-first format**. In other words, rather than passing a
list of samples, ML frameworks compliant with this library are expected to handle a list of features.
This is at odds with most (if not all) other ML frameworks, but allows for handling of changing of
features at runtime as described in the [input change](#input-change) design principle. See the
[core module](core) or [change map](changemap) documentation for more details.

   Note: this design also ensures that images are of consistent shape, see the documentation for
   [high-dimensional data](core/mixins.html#bananas.core.mixins.HighDimensionalMixin).

### Input data batches
This library has made some design choices that set it apart from other ML tooling and frameworks.
Fundamentally, it expects users to feed data to esatimators in batches. Either by repeatedly calling
`fit` with each batch or, preferrably, by calling `train` with an input function that will draw
batches online. We rarely have to create our own input functions, that's what the [samplers]
(sampling) do.

### Input change
Because data comes in incrementally, not all possible values are seen in each batch. For example,
a continuous feature may be within a range of [0, .7] in the first few batches but eventually become
[-1, 1]. Or a classifier may encounter a label never seen before after the first handful of batches.
Some frameworks like Scikit-Learn overcome this by implementing a different training function,
`partial_fit`, which expects users to pass all possible values, including unseen ones, as a
parameter. This library follows the principle of making no assumptions about data but being able to
handle runtime changes. In some cases, changes result in outputs of different dimensions and hence
incompatible with other components downstream in the pipeline (like doing one-hot encoding and
finding a label never seen before). This library handles this by forcing all transformers and
estimators to implement ["input change" event handling](changemap) which allows for handling of
addition and removal of features at runtime, losing as little work as possible during training.

### Modularity
Even though this library is designed to be an ML framework, the tools included with it should be
useful for purposes beyond that. For example, a lot of the functionality contained within the
[sampling](sampling) and [transformers](transformers) modules should be reusable in projects with
different goals.

### Dependencies
This project aims to have as few dependencies as possible, keeping them optional wherever feasible.
For example, many image utility functions require the [Pillow](https://pypi.org/project/Pillow)
module but it is not a global requirement. The largest dependency of global scope currently is
the `numpy` numerical library; but turning that into an optional dependency is currently being
evaluated as an option.
