import unittest

# For generating the test dataset
import numpy as np
from scipy.stats import multivariate_normal

from denmarf import DensityEstimate

# For making diagnostic plots
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt
import getdist
from getdist import MCSamples, plots

class Test2DGaussian(unittest.TestCase):
    # Define and initialize a 2D Gaussian with certain mean and covariance matrix
    gaussian_dist = multivariate_normal([3.5, -1.2], [[2.0, 0.3], [0.3, 0.5]])

    def test_fitting_samples(self):
        # Generate some samples from the true distribution
        xgen = self.gaussian_dist.rvs(size=10000)
        # Use denmarf to fit this distribution
        # NOTE: if GPU with CUDA support is present, it will be used automatically
        de = DensityEstimate().fit(
            xgen,
            num_blocks=2,
            num_hidden=5,
            num_epochs=100,
        )

        # Save to file
        de.save("2d_gaussian_test.pkl")
        # If this does not crash, the test is considered successful
        return True

    def test_loading_model_from_disk(self):
        # NOTE Here we used a hack that unittest orders functions alphabetically for exec order
        # Load the previously saved model to CPU
        de = DensityEstimate.from_file(
            "2d_gaussian_test.pkl",
            device="cpu",
            use_cuda=False,
        )
        
        # If this does not crash, the test is considered successful
        return True

    def test_logpdf_values(self):
        # Load the previously saved model again (but this time on GPU if available)
        de = DensityEstimate.from_file(
            "2d_gaussian_test.pkl",
            use_cuda=True,
        )

        # Generate some samples from the 2D Gaussian distribution
        xgen = self.gaussian_dist.rvs(size=1000)

        # Compute the logpdf using the exact form
        logpdf_truth = self.gaussian_dist.logpdf(xgen)
        # Compute the logpdf using the density estimate
        logpdf_nf = de.score_samples(xgen)

        # Test if the average difference in pdf is smaller than a pre-defined tolerance
        _tol = 1e-1
        return np.mean(np.abs(np.exp(logpdf_truth - logpdf_nf))) < _tol

    def test_sampling(self):
        # Load the previously saved model again and again (but this time on GPU if available)
        de = DensityEstimate.from_file(
            "2d_gaussian_test.pkl",
            use_cuda=True,
        )

        # Generate some samples from the 2D Gaussian distribution
        xgen = self.gaussian_dist.rvs(size=1000)

        samples_truth = MCSamples(samples=xgen, label="from 2D Gaussian distribution")
        samples_nf = MCSamples(samples=de.sample(10000), label="from denmarf")

        fig = plots.get_subplot_plotter()
        fig.triangle_plot([samples_truth, samples_nf], filled=False)
        fig.export("2d_gaussian_test_corner_plot.png")

        # It will always consider this test to be successful, a human should check
        # if the two distributions look alike
        return True

if __name__ == '__main__':
    unittest.main()