import torch
from getdist import MCSamples, plots

def determine_device(device, use_cuda):
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
    samples_orig = MCSamples(samples=original_samples, label="original samples", names=parameter_labels)
    samples_synth = MCSamples(samples=density_estimate.sample(n_samples=original_samples.shape[0]), label="generated samples", names=parameter_labels)

    g = plots.get_subplot_plotter()
    g.triangle_plot([samples_orig, samples_synth], filled=False)
    g.export(filename)
