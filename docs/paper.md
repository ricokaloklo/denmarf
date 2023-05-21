---
title: 'denmarf: a Python package for density estimation using masked autoregressive flow'
tags:
  - Python
  - data analysis
  - statistics
  - astronomy

authors:
  - name: Rico K. L. Lo
    orcid: 0000-0003-1561-6716
    affiliation: 1
affiliations:
 - name: LIGO Laboratory, California Institute of Technology, Pasadena, California 91125, USA
   index: 1
date: 19 May 2023
bibliography: paper.bib

---

# Summary

Masked autoregressive flow (MAF) [@NIPS2017_6c1da886] is a state-of-the-art non-parametric density estimation technique. 
It is based on the idea (known as a normalizing flow) that a simple base probability distribution can be mapped into 
a complicated target distribution that one wishes to approximate, using a sequence of bijective transformations [@cms/1266935020; @https://doi.org/10.1002/cpa.21423]. The `denmarf` package provides a `scikit-learn`-like interface in Python for researchers
to effortlessly use MAF for density estimation in their applications to evaluate probability densities 
of the underlying distribution of a set of data and generate new samples from the data, on either a CPU or a GPU. The package also implements logistic transformations to facilitate the fitting of bounded distributions.

# Statement of need

There are a number of ways to perform density estimation in a non-parametric fashion, one of which is kernel density estimation (KDE). 
Suppose we have a set of $D$-dimensional data of size $N$, $\left( \vec{x}_{1}, \vec{x}_{2}, \dots, \vec{x}_{N} \right)$, i.e. $\vec{x}_{i}$ is a $D$-dimensional vector where $i \in \left[ 1, N \right]$ that follows the probability distribution $f(\vec{x})$ we wish to approximate. 
The kernel density estimate $\hat{f}_{\rm KDE}$ using those input data is given by 
\begin{equation}
\label{eq:KDE}
  \hat{f}_{\rm KDE}(\vec{x}) = \dfrac{1}{N} \sum_{i=1}^{N} K(\vec{x} - \vec{x}_{i}),
\end{equation}
where $K$ is the kernel function that depends on the distance between the evaluation point $\vec{x}$ and the input data point $\vec{x}_{i}$. 
There are many implementations of KDE in Python, such as `scipy.stats.gaussian_kde` [@2020SciPy-NMeth], `sklearn.neighbors.KernelDensity` [@scikit-learn] and `kalepy` [@Kelley2021].
The cost of $M$ such evaluations using \autoref{eq:KDE} is therefore $O(MND)$. This can be slow if we need to evaluate the KDE of a large data set (i.e. large $N$) many times (i.e. large $M$). Give an example here if word limit permits.

However with MAF, an evaluation of the estimated density is independent of $N$. Suppose $T(\vec{x})$ maps the target distribution $f(\vec{x})$ into the base distribution $u$, usually chosen as a $D$-dimensional standard normal distribution, then the density estimate using MAF $\hat{f}_{\rm MAF}$ is given by
\begin{equation}
  \hat{f}_{\rm MAF}(\vec{x}) = u(T(\vec{x}))|J_{T}(\vec{x})|,
\end{equation}
where $|J_{T}|$ is the Jacobian determinant of the mapping, and note that there is no summation over the $N$ input data. \autoref{fig:timing} shows the computational cost for $M = 1000$ evaluations of the density estimate from data of size $N$ using KDE and that using MAF respectively. 
We can see that the evaluation cost using KDE scales with $N$ while that using MAF is indeed independent of $N$.

![Computation cost for $M = 1000$ evaluations of the density estimate from data of size $N$ using KDE with `scikit-learn` and that using MAF with `denmarf` respectively. We can see that the evaluation cost using KDE scales with $N$ while that using MAF is independent of $N$. \label{fig:timing}](KDE_MAF_timing.pdf){ width=80% }

While it is relatively straightforward to implement a routine to perform density estimation using MAF with the help of deep learning libraries such as `TensorFlow` [@tensorflow2015-whitepaper] and `PyTorch` [@paszke2017automatic],
the technical hurdle of leveraging MAF for people not well-versed in those libraries remains high. 
The `denmarf` package is designed to be an almost drop-in replacement of the `sklearn.neighbors.KernelDensity` module 
to lower the technical barrier and enable researchers to apply MAF for density estimation effortlessly.

New samples $\vec{x}_{i}$ can be generated from the approximated distribution by first drawing samples $\vec{y}_{i}$ from the base distribution $u(\vec{y})$ and then transforming them with the inverse mapping $T^{-1}$, i.e.
\begin{equation}
  \vec{x}_{i} = T^{-1}(\vec{y}_{i}).
\end{equation}
Indeed, if the transformations are bijective (i.e. both surjective and injective) then we can always find $\vec{x}_{i}$ such that $\vec{y}_{i} = T(\vec{x}_{i})$. This could potentially be a problem for input data $\vec{x}_{i}$ that are bounded, since in MAF $T$ is only rescaling and shifting (i.e. an affine transformation) and $u$ is usually a normal distribution which is unbounded. To solve this problem, `denmarf` will logit-transform the input data first if the underlying distribution should be bounded, and the logit-transformed data become unbounded. `denmarf` will automatically include the extra Jacobian from the logit transformation during density evaluations and sample regenerations.

# Acknowledgements

# References
