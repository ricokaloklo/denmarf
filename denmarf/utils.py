import torch
from getdist import MCSamples, plots

def determine_device(device, use_cuda):
    """Determine which device to use for training and evaluation.

    Parameters
    ----------
    device : str
        Device type to be used. Can be "cpu", "cuda", or "{cpu/cuda}:{device ordinal}".
    use_cuda : bool
        Whether to use CUDA if available.

    Returns
    -------
    pytorch_device : torch.device
        Device to be used.

    Notes
    -----
    If use_cuda is True and device is "cpu", the first available GPU will be used.

    """
    if use_cuda:
        if device == "cpu":
            # GPU not specified, use the first one if available
            pytorch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            pytorch_device = torch.device(device)
    else:
        pytorch_device = torch.device(device)

    return pytorch_device

def plot_corner(original_samples, density_estimate, parameter_labels=None, filename="corner_plot.png"):
    """Plot a corner plot of the original samples and the generated samples.

    Parameters
    ----------
    original_samples : numpy.ndarray
        Original samples.
    density_estimate : denmarf.DensityEstimate
        Density estimate.
    parameter_labels : list of str, optional
        Parameter labels.
    filename : str, optional
        Filename of the plot.
        
    """
    samples_orig = MCSamples(samples=original_samples, label="original samples", names=parameter_labels)
    samples_synth = MCSamples(samples=density_estimate.sample(n_samples=original_samples.shape[0]), label="generated samples", names=parameter_labels)

    g = plots.get_subplot_plotter()
    g.triangle_plot([samples_orig, samples_synth], filled=False)
    g.export(filename)
