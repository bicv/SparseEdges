[![PyPI version](https://badge.fury.io/py/SparseEdges.svg)](https://badge.fury.io/py/SparseEdges)
[![Research software impact](http://depsy.org/api/package/pypi/SparseEdges/badge.svg)](http://depsy.org/package/python/SparseEdges)

What is the SparseEdges package?
================================

Our goal here is to build practical algorithms of sparse coding for computer vision.

This class exploits the [SLIP](https://pythonhosted.org/SLIP/) and [LogGabor](https://pythonhosted.org/LogGabor/) libraries to provide with a sparse representation of edges in images.

This algorithm was presented in the following paper:

~~~~{.bibtex}
@inbook{Perrinet15bicv,
    author = {Perrinet, Laurent U.},
    booktitle = {Biologically-inspired Computer Vision},
    chapter = {13},
    citeulike-article-id = {13566753},
    editor = {Keil, Matthias and Crist\'{o}bal, Gabriel and Perrinet, Laurent U.},
    keywords = {anr-trax, bicv-sparse},
    posted-at = {2015-03-31 14:21:35},
    priority = {2},
    publisher = {Wiley, New-York},
    title = {Sparse models},
    year = {2015}
}
~~~~

This package gives a python implementation.

Moreover, it gives additional tools to compute useful stistics in images; first- and second order statistics of co-occurences in images.
More information is available @ http://nbviewer.ipython.org/github/bicv/SparseEdges/blob/master/SparseEdges.ipynb
Tests for the packages are available @ http://nbviewer.ipython.org/github/bicv/SparseEdges/blob/master/notebooks/test-SparseEdges.ipynb
