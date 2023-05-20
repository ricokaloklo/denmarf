---
title: 'denmarf: density estimation using masked autoregressive flow'
tags:
  - Python
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

Masked autoregressive flow [@NIPS2017_6c1da886] is a state-of-the-art non-parametric density estimation technique. 
It is based on the idea (known as a normalizing flow) that a simple base probability distribution can be mapped into 
a complicated target distribution that one wishes to approximate, using a sequence of bijective transformations [@cms/1266935020; @https://doi.org/10.1002/cpa.21423]. The `denmarf` package provides a `scikit-learn`-like interface for researchers to effortlessly and seamlessly 
use masked autoregressive flow to evaluate probability densities of a set of data and generate new samples from the data, on either a CPU or a GPU.

# Statement of need
<!---
While it is relatively straightforward to implement these with the help of deep learning libraries such as `PyTorch`, 
the technical hurdles of leveraging these cutting-edge methods for people outside of the machine learning community that are not well-versed in those 
libraries remain high. The `denmarf` package is designed to lower the technical barrier and enable researchers to apply masked autoregressive flow 
for density estimation in their researches seamlessly by providing a `scikit-learn`-like interface.
-->

# Acknowledgements

# References
