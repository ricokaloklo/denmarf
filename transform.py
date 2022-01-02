import numpy as np

class LogitTransform():
    def __init__(self, lower_bounds=None, upper_bounds=None):
        if lower_bounds is not None and upper_bounds is not None:
            self.rescaled = True
            _OFFSET = 0
            self.lower_bounds = lower_bounds - _OFFSET
            self.upper_bounds = upper_bounds + _OFFSET
        else:
            self.rescaled = False

    @staticmethod
    def rescale(x, lower_bounds, upper_bounds):
        x = np.asarray(x)
        return (x - lower_bounds)/(upper_bounds-lower_bounds)

    @staticmethod
    def inverse_rescale(x, lower_bounds, upper_bounds):
        x = np.asarray(x)
        return (upper_bounds-lower_bounds)*x + lower_bounds

    def logit_transform(self, x):
        x = np.asarray(x)
        if self.rescaled:
            # Rescale to (0, 1)
            x = self.rescale(x, self.lower_bounds, self.upper_bounds)
        logit_x = np.log(x) - np.log1p(-x)
        return logit_x

    def inverse_logit_transform(self, logit_x):
        logit_x = np.asarray(logit_x)
        x = 1./(1.+np.exp(-logit_x))
        if self.rescaled:
            x = self.inverse_rescale(x, self.lower_bounds, self.upper_bounds)
        return x

    def log_jacobian(self, x):
        lj = np.zeros(x.shape[0])

        if self.rescaled:
            p = self.rescale(x, self.lower_bounds, self.upper_bounds)
            lj -= np.ones(x.shape[0])*np.sum(np.log(self.upper_bounds - self.lower_bounds))
        
        lj -= np.sum(np.log(p) + np.log1p(-p), axis=1)

        return lj