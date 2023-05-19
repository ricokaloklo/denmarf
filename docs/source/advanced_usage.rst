Advanced Usage
==============

Checkpointing the training of a model
-------------------------------------

While in most cases the training of a model can be done with 10-15 minutes with the help of a modern GPU,
sometimes it is desirable/necessary to checkpoint the training progress so that the training can be resumed later.
``denmarf`` does not have a native checkpointing feature implemented in ``.fit()`` but such a feature can be achieved
by combining ``.save()`` and ``.from_file()``. This is because ``denmarf`` will check if a model already exists 
in a ``DensityEstimate`` object before initializing a new one.

Suppose we want to checkpoint every 100 epoches out of a total of 2000 epoches, this can be done with

.. code-block:: python

    from denmarf import DensityEstimate

    # Initialize a model first
    de = DensityEstimate()

    for _ in range(0, 20):
        # Train using samples in X
        de.fit(X, num_epochs=100, ...)

        # Save the model for checkpointing
        de.save("model.pkl")


Restarting the training of a model
----------------------------------

It is possible to resume the training from a previously saved model, either from a checkpointed/saved model
with the same set of samples or even from a previously trained model with a new set of samples, with ``denmarf``. 
This is because, again, ``denmarf`` will check if a model already exists 
in a ``DensityEstimate`` object before initializing a new one.


.. code-block:: python

    from denmarf import DensityEstimate

    # Initialize a model first
    de = DensityEstimate()

    # Train with a set of samples in X first
    de.fit(X, num_epochs=1000, ...)

    # Train with another set of samples in Y
    de.fit(Y, num_epochs=1000, ...)

