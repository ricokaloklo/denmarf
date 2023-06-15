import numpy as np
import copy
import pickle
import tqdm

import torch
torch.set_default_dtype(torch.float32)
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

from . import flows as fnn
from .transform import LogitTransform
from .utils import determine_device

# An almost drop-in replacement of scikit-learn KernelDensity using MAF
class DensityEstimate():
    def __init__(
        self,
        seed=None,
        device="cpu",
        use_cuda=True,
        bounded=False,
    ):
        """Initialize the density estimate.
        
        Parameters
        ----------
        seed : int, optional
            Random seed.
        device : str, optional
            Device to be used. Can be "cpu", "cuda", or "{cpu/cuda}:{device ordinal}".
        use_cuda : bool, optional
            Whether to use CUDA if available.
        bounded : bool, optional
            Whether the distribution is bounded. If True, the distribution will be transformed to the unbounded space
            using logistic transformation.
        
        """
        self.device = determine_device(device, use_cuda)

        if seed is not None:
            # Calling this will take care of everything
            torch.manual_seed(seed)

        self.model = None
        self.bounded = bounded

    @staticmethod
    def construct_model(
        num_features,
        num_blocks,
        num_hidden,
    ):
        """Construct the pytorch model.

        Parameters
        ----------
        num_features : int
            Number of features.
        num_blocks : int
            Number of blocks.
        num_hidden : int
            Number of hidden units.

        Returns
        -------
        model : torch.nn.Module
            Pytorch model.

        """
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

        return model

    def fit(
        self,
        X,
        bounded=None,
        lower_bounds=None,
        upper_bounds=None,
        num_blocks=32,
        num_hidden=128,
        num_epochs=1000,
        learning_rate=1e-3,
        weight_decay=1e-6,
        batch_size=None,
        p_train=0.5,
        p_test=0.1,
        verbose=True,
    ):
        """Fit the density estimate to the data.
        
        Parameters
        ----------
        X : numpy.ndarray
            Training samples.
        bounded : bool, optional
            Whether the distribution is bounded. If True, the distribution will be transformed to the unbounded space
            using logistic transformation.
        lower_bounds : numpy.ndarray, optional
            Lower bounds of the bounded distribution.
        upper_bounds : numpy.ndarray, optional
            Upper bounds of the bounded distribution.
        num_blocks : int, optional
            Number of blocks.
        num_hidden : int, optional
            Number of hidden units.
        num_epochs : int, optional
            Number of epochs.
        learning_rate : float, optional
            Learning rate.
        weight_decay : float, optional
            Weight decay.
        batch_size : int, optional
            Batch size.
        p_train : float, optional
            Percentage (0 < p_train < 1.0) of training samples.
        p_test : float, optional
            Percentage (0 < p_test < 1.0) of test samples. The rest of the samples will be used for validation.
        verbose : bool, optional
            Whether to print progress.

        """
        X = np.asarray(X)
        assert len(X.shape) == 2, "X must be of shape (n_samples, n_features)"

        # Set batch_size to something reasonable, maybe splitting the training samples into 10 sets?
        if batch_size is None:
            batch_size = int(0.1*X.shape[0])

        # For compatibility with older version
        # Deprecation notice: the bounded option will be moved to .fit() in newer versions
        if bounded is not None:
            self.bounded = bounded

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

        self.num_features = num_features
        self.num_blocks = num_blocks
        self.num_hidden = num_hidden

        if self.model is None:
            # Construct a new model
            model = self.construct_model(self.num_features, self.num_blocks, self.num_hidden)
        else:
            # Resume from previously saved model
            model = self.model
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

        return self

    def save(self, filename="density_estimate.pickle"):
        """Save the density estimate to a file.
        
        Parameters
        ----------
        filename : str
            Filename to save the density estimate to.
        
        """
        torch.save(self, filename)

    @staticmethod
    def from_file(
            filename="density_estimate.pickle",
            device="cpu",
            use_cuda=False,
        ):
        """Load a density estimate from a file.

        Parameters
        ----------
        filename : str
            Filename to load the density estimate from.
        device : str
            Device to load the density estimate to. Does not have
            to be the same architecture as the one used to train the model.
        use_cuda : bool
            Whether to use CUDA.
        
        Returns
        -------
        density_estimate : DensityEstimate
            The loaded density estimate.

        """
        pytorch_device = determine_device(device, use_cuda)
        de = torch.load(filename, map_location=pytorch_device)
        de.device = pytorch_device
        return de

    def score_samples(self, X):
        """Compute the log likelihood of each sample in X.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples to compute the log likelihood for.
            
        Returns
        -------
        log_likelihoods : array-like, shape (n_samples,)
            Log likelihoods of each sample in X.
            
        """
        assert len(X.shape) == 2, "X must be of shape (n_samples, n_features)"
        logpdf = np.zeros(X.shape[0])
        X = X.astype(np.float32)

        if self.bounded:
            # Check if X is within the bounds
            valid_indices_lower_bounds = np.less_equal(
                self.transformation.lower_bounds,
                X
            ).all(axis=1)
            valid_indices_upper_bounds = np.less(
                X,
                self.transformation.upper_bounds,
            ).all(axis=1)
            logpdf = np.where(valid_indices_lower_bounds & valid_indices_upper_bounds, logpdf, -np.inf)
            # First compute the log jacobian from logit transformation
            logpdf += self.transformation.log_jacobian(X)
            # Then perform the transformation
            X = self.transformation.logit_transform(X)
        
        X_torch = torch.from_numpy(X).to(self.device)
        logpdf += self.model.log_probs(X_torch).detach().cpu().numpy()

        return logpdf

    def score(self, X):
        """Compute the total log likelihood of samples in X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples to compute the log likelihood for.

        Returns
        -------
        log_likelihood : float
            Total log likelihood of samples in X.

        """
        return np.sum(self.score_samples(X))

    def sample(self, n_samples=1):
        """Sample from the density estimate.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        Returns
        -------
        samples : array-like, shape (n_samples, n_features)
            Samples from the density estimate.

        """
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

    def log_jacobian(self, X):
        """Compute the log jacobian for a set of samples
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Samples to compute the log jacobian for.
            
        Returns
        -------
        array-like, shape (n_samples,)
            The array of log_jacobian values for each sample
        """
        # Boilerplate from score_samples
        assert len(X.shape) == 2, "X must be of shape (n_samples, n_features)"
        log_jacob = np.zeros(X.shape[0])
        X = X.astype(np.float32)

        if self.bounded:
            # Check if X is within the bounds
            valid_indices_lower_bounds = np.less_equal(
                self.transformation.lower_bounds,
                X
            ).all(axis=1)
            valid_indices_upper_bounds = np.less(
                X,
                self.transformation.upper_bounds,
            ).all(axis=1)
            log_jacob = np.where(valid_indices_lower_bounds & valid_indices_upper_bounds, log_jacob, -np.inf)
            # First compute the log jacobian from logit transformation
            log_jacob += self.transformation.log_jacobian(X)
            # Then perform the transformation
            X = self.transformation.logit_transform(X)

        X_torch = torch.from_numpy(X).to(self.device)
        _, model_log_jacob = self.model(X_torch)
        model_log_jacob = torch.squeeze(model_log_jacob).detach().cpu().numpy()
        log_jacob += model_log_jacob

        return log_jacob