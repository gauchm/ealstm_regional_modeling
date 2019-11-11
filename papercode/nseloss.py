"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import torch
import numpy as np


class NSELoss(torch.nn.Module):
    """Calculate (batch-wise) NSE Loss.

    Each sample i is weighted by 1 / (std_i + eps)^2, where std_i is the standard deviation of the 
    discharge from the basin, to which the sample belongs.

    Parameters:
    -----------
    eps : float
        Constant, added to the weight for numerical stability and smoothing, default to 0.1
    """

    def __init__(self, eps: float = 0.1):
        super(NSELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, q_stds: torch.Tensor):
        """Calculate the batch-wise NSE Loss function.

        Parameters
        ----------
        y_pred : torch.Tensor
            Tensor containing the network prediction.
        y_true : torch.Tensor
            Tensor containing the true discharge values
        q_stds : torch.Tensor
            Tensor containing the discharge std (calculate over training period) of each sample

        Returns
        -------
        torch.Tenor
            The (batch-wise) NSE Loss
        """
        squared_error = (y_pred - y_true)**2
        weights = 1 / (q_stds + self.eps)**2
        scaled_loss = weights * squared_error

        return torch.mean(scaled_loss)


class NSEObjective:
    """Custom NSE XGBoost objective.
    
    This is a bit of a hack: We use a unique dummy target value for each sample, allowing us to look up
    the q_std that corresponds to the sample's station.
    When calculating the loss, we replace the dummy with the actual target, so the model learns the right thing.
    """
    def __init__(self, dummy_target, actual_target, q_stds, eps: float = 0.1):
        self.dummy_target = dummy_target.reshape(-1)
        self.actual_target = actual_target.reshape(-1)
        self.q_stds = q_stds.reshape(-1)
        self.eps = eps
        
    def nse_objective(self, y_true, y_pred):
        """ NSE objective for XGBoost (sklearn API) """
        indices = np.searchsorted(self.dummy_target, y_true)
        normalization = ((self.q_stds[indices] + self.eps)**2)
        grad = 2 * (y_pred - self.actual_target[indices]) / normalization
        hess = 2.0 / normalization
        return grad, hess
    
    def nse_objective_non_sklearn(self, y_pred, dtrain):
        """ NSE objective for XGBoost (non-sklearn API) """
        y_true = dtrain.get_label()
        indices = np.searchsorted(self.dummy_target, y_true)
        normalization = ((self.q_stds[indices] + self.eps)**2)
        grad = 2 * (y_pred - self.actual_target[indices]) / normalization
        hess = 2.0 / normalization
        return grad, hess
    
    def nse(self, y_pred, y_true, q_stds):
        squared_error = (y_pred - y_true)**2
        weights = 1 / (q_stds + self.eps)**2
        return np.mean(weights * squared_error)
    
    def nse_metric(self, y_pred, y_true):
        """ NSE metric for XGBoost """
        indices = np.searchsorted(self.dummy_target, y_true.get_label())
        nse = self.nse(y_pred, self.actual_target[indices], self.q_stds[indices])
        
        return 'nse', nse
    
    def neg_nse_metric_sklearn(self, estimator, X, y_true):
        """ -NSE metric for sklearn """
        y_pred = estimator.predict(X)
        indices = np.searchsorted(self.dummy_target, y_true)
        return -self.nse(y_pred, self.actual_target[indices], self.q_stds[indices])
