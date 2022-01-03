import numpy as np
import copy
import pickle
import tqdm

import torch
torch.set_default_dtype(torch.float32)
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import flows as fnn
from transform import LogitTransform

# An almost drop-in replacement of scikit-learn KernelDensity using MAF
class DensityEstimate():
    def __init__(
        self,
        seed=None,
        device="cpu",
        use_cuda=True,
        bounded=False,
    ):
        if use_cuda:
            if device == "cpu":
                # GPU not specified, use the first one if available
                self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            else:
                self.device = torch.device(device)
        else:
            self.device = torch.device(device)

        if seed is not None:
            # Calling this will take care of everything
            torch.manual_seed(seed)

        self.model = None
        self.bounded = bounded

    def fit(
        self,
        X,
        lower_bounds=None,
        upper_bounds=None,
        num_blocks=64,
        num_hidden=256,
        num_epochs=1000,
        learning_rate=1e-3,
        weight_decay=1e-6,
        batch_size=1,
        p_train=0.5,
        p_test=0.1,
        verbose=True,
    ):
        X = np.asarray(X)
        assert len(X.shape) == 2, "X must be of shape (n_samples, n_features)"

        # Perform logit transformation if distribution is bounded
        if self.bounded:
            assert lower_bounds is not None, "lower_bounds must be specified for bounded distribution"
            assert upper_bounds is not None, "upper_bounds must be specified for bounded distribution"
            self.transformation = LogitTransform(lower_bounds, upper_bounds)

            X = self.transformation.logit_transform(X)

        num_features = X.shape[1]
        assert num_features > 1, "MADE does not work for 1D case"

        # Split the data set into training set, validation set and test set
        N_train = int(X.shape[0]*p_train)
        N_test = int(X.shape[0]*p_test)
        N_validate = int(X.shape[0]) - (N_train+N_test)

        X = data.TensorDataset(torch.from_numpy(X.astype(np.float32)))
        train_dataset, validate_dataset, test_dataset = data.random_split(X, (N_train, N_validate, N_test))

        train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        validate_dataloader = data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Build the pytorch model
        modules = []
        for _ in range(num_blocks):
            modules += [
                fnn.MADE(num_features, num_hidden, None, act="tanh"),
                fnn.BatchNormFlow(num_features),
                fnn.Reverse(num_features)
            ]
        model = fnn.FlowSequential(*modules)

        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    module.bias.data.fill_(0)

        model.to(self.device)

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        def train(epoch):
            model.train()
            train_loss = 0

            for batch_idx, data in enumerate(train_dataloader):
                if isinstance(data, list):
                    if len(data) > 1:
                        cond_data = data[1].float()
                        cond_data = cond_data.to(self.device)
                    else:
                        cond_data = None

                    data = data[0]
                data = data.to(self.device)
                optimizer.zero_grad()
                loss = -model.log_probs(data, cond_data).mean()
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

            for module in model.modules():
                if isinstance(module, fnn.BatchNormFlow):
                    module.momentum = 0

            with torch.no_grad():
                model(train_dataloader.dataset.dataset.tensors[0].to(data.device))

            for module in model.modules():
                if isinstance(module, fnn.BatchNormFlow):
                    module.momentum = 1
        
        def validate(epoch, model, loader):
            model.eval()
            val_loss = 0

            for batch_idx, data in enumerate(loader):
                if isinstance(data, list):
                    if len(data) > 1:
                        cond_data = data[1].float()
                        cond_data = cond_data.to(self.device)
                    else:
                        cond_data = None

                    data = data[0]
                data = data.to(self.device)
                with torch.no_grad():
                    val_loss += -model.log_probs(data, cond_data).sum().item()  # sum up batch loss

            return val_loss / len(loader.dataset)        

        # Start training the network
        best_validation_loss = float('inf')
        best_validation_epoch = 0
        best_model = model

        if verbose:
            pbar = tqdm.tqdm(total=num_epochs)
        for epoch in range(num_epochs):
            train(epoch)
            validation_loss = validate(epoch, model, validate_dataloader)

            if verbose:
                pbar.update()
                pbar.set_description("current average log likelihood: {:.3f}".format(-validation_loss))

            if validation_loss < best_validation_loss:
                best_validation_epoch = epoch
                best_validation_loss = validation_loss
                best_model = copy.deepcopy(model)

        best_validation_loss = validate(best_validation_epoch, best_model, test_dataloader)
        if verbose:
            print("best average log likelihood: {:.3f}".format(-best_validation_loss))
        self.model = best_model

    def save(self, filename="density_estimate.pickle"):
        # Move the model to cpu first before saving
        self.model.to(torch.device("cpu"))

        with open(filename, "wb") as f:
            pickle.dump(self, f)

        # Move back to the original device
        self.model.to(self.device)

    @classmethod
    def from_file(
            cls,
            filename="density_estimate.pickle",
            device="cpu",
            use_cuda=True,
        ):
        
        with open(filename, "rb") as f:
            saved_model = pickle.load(f)

        reconstructed = cls(
            device=device,
            use_cuda=use_cuda,
            bounded=saved_model.bounded,
        )
        reconstructed.model = saved_model.model
        reconstructed.model.to(reconstructed.device)
        reconstructed.transformation = saved_model.transformation

        return reconstructed

    def score_samples(self, X):
        assert len(X.shape) == 2, "X must be of shape (n_samples, n_features)"
        logpdf = np.zeros(X.shape[0])
        X = X.astype(np.float32)

        if self.bounded:
            # First compute the log jacobian from logit transformation
            logpdf += self.transformation.log_jacobian(X)
            # Then perform the transformation
            X = self.transformation.logit_transform(X)
        
        X_torch = torch.from_numpy(X).to(self.device)

        logpdf += self.model.log_probs(X_torch).detach().cpu().numpy()
        return logpdf

    def score(self, X):
        return np.sum(self.score_samples(X))

    def sample(self, n_samples=1):
        self.model.eval()

        # NOTE: always return samples to CPU
        with torch.no_grad():
            generated_samples = self.model.sample(n_samples).detach().cpu().numpy().astype(np.float32)

        if self.bounded:
            # Perform inverse transform
            generated_samples = self.transformation.inverse_logit_transform(generated_samples)

            # Replace inf and -inf with proper values
            generated_samples = np.where(generated_samples == np.inf, self.transformation.upper_bounds, generated_samples)
            generated_samples = np.where(generated_samples == -np.inf, self.transformation.lower_bounds, generated_samples)

        return generated_samples
