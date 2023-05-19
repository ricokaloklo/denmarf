Installation
============

Dependencies
------------

Before installation, make sure that you have the following dependencies installed

 * ``scipy-stack`` (``numpy``, ``scipy``, ``matplotlib``)
 * ``tqdm``
 * ``torch``
 * ``getdist`` (for visualization, optional)
 * ``CUDA`` (for GPU capability, optional)

From pypi
--------------------

To install the latest stable release from pypi, simply

.. code-block:: bash

   pip install denmarf

From github repository
----------------------

To install from the source code (to use bleeding-edge features, or to contribute to the codebase), do

.. code-block:: bash

    git clone git@github.com:ricokaloklo/denmarf.git
    cd denmarf
    pip install -e .