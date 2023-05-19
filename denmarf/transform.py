import numpy as np
from scipy.special import logit, expit

class LogitTransform():
    def __init__(self, lower_bounds=None, upper_bounds=None, _OFFSET=1e-4):
        """Initialize a logitistic transformation.
        
        Parameters
        ----------
        lower_bounds : None or np.ndarray
            Lower bounds for the transformed data. If None, no rescaling is performed.
        upper_bounds : None or np.ndarray
            Upper bounds for the transformed data. If None, no rescaling is performed.
        _OFFSET : float
            Offset to avoid overflow.

        """
        if lower_bounds is not None and upper_bounds is not None:
            self.rescaled = True
            self.lower_bounds = lower_bounds
            self.upper_bounds = upper_bounds
            self._OFFSET = _OFFSET
        else:
            self.rescaled = False

    @staticmethod
    def rescale(x, lower_bounds, upper_bounds, _OFFSET=1e-4):
        """Rescale data to (0, 1) interval.
        
        Parameters
        ----------
        x : np.ndarray
            Data to be rescaled.
        lower_bounds : np.ndarray
            Lower bounds for the transformed data.
        upper_bounds : np.ndarray
            Upper bounds for the transformed data.
        _OFFSET : float
            Offset to avoid overflow.
            
        Returns
        -------
        np.ndarray
            Rescaled data.
            
        """
        x = np.asarray(x)

        # Introduce a small offset to avoid overflow
        x = np.where(x <= lower_bounds, lower_bounds+_OFFSET, x)
        x = np.where(x >= upper_bounds, upper_bounds-_OFFSET, x)

        return (x - lower_bounds)/(upper_bounds-lower_bounds)

    @staticmethod
    def inverse_rescale(x, lower_bounds, upper_bounds):
        """Inverse rescale data from (0, 1) interval.
        
        Parameters
        ----------
        x : np.ndarray
            Data to be rescaled.
        lower_bounds : np.ndarray
            Lower bounds for the transformed data.
        upper_bounds : np.ndarray
            Upper bounds for the transformed data.
            
        Returns
        -------
        np.ndarray
            Rescaled data.
        
        """
        x = np.asarray(x)
        return (upper_bounds-lower_bounds)*x + lower_bounds

    def logit_transform(self, x):
        """Apply logit transformation to data.

        Parameters
        ----------
        x : np.ndarray
            Data to be transformed.
        
        Returns
        -------
        np.ndarray
            Transformed data.

        """
        x = np.asarray(x, dtype=np.float128)
        if self.rescaled:
            # Rescale to (0, 1)
            x = self.rescale(x, self.lower_bounds, self.upper_bounds, self._OFFSET)
        logit_x = logit(x)
        return logit_x.astype(np.float32)

    def inverse_logit_transform(self, logit_x):
        """Apply inverse logit transformation to data.
        
        Parameters
        ----------
        logit_x : np.ndarray
            Data that were logit-transformed.
        
        Returns
        -------
        np.ndarray
            Original untransformed data.
            
        """
        logit_x = np.asarray(logit_x, dtype=np.float64)
        x = expit(logit_x)
        if self.rescaled:
            x = self.inverse_rescale(x, self.lower_bounds, self.upper_bounds)
        return x.astype(np.float32)

    def log_jacobian(self, x):
        """Compute the log jacobian of the transformation.

        Parameters
        ----------
        x : np.ndarray
            Data to be transformed.

        Returns
        -------
        np.ndarray
            Log jacobian of the transformation.

        """
        lj = np.zeros(x.shape[0])

        if self.rescaled:
            p = self.rescale(x, self.lower_bounds, self.upper_bounds, self._OFFSET)
            lj -= np.ones(x.shape[0])*np.sum(np.log(self.upper_bounds - self.lower_bounds))
        
        lj -= np.sum(np.log(p) + np.log1p(-p), axis=1)

        return lj