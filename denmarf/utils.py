import torch

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