import numpy as np


class Scaler:
    """Scaler for feature arrays.

    Parameters
    ----------
    ndim : int
        The number of dimensions of the feature arrays.
    """
    def __init__(self, ndim):
        self._mean = None
        self._std = None

        self._axes = tuple(range(ndim - 1))

    @property
    def mean(self):
        mean = self._mean
        if mean is None:
            raise ValueError(f'{type(self).__name__} has not been fit')
        return mean

    @property
    def std(self):
        std = self._std
        if std is None:
            raise ValueError(f'{type(self).__name__} has not been fit')
        return std

    def fit(self, data):
        """Fit the scaler to the data.

        Parameters
        ----------
        data : np.ndarray
            The data to fit the scaler to.

        Returns
        -------
        transformed : np.ndarray
            The transformed input data.
        """
        self._mean = data.mean(axis=self._axes)
        self._std = data.std(axis=self._axes)
        self._std[self._std == 0] = 1  # when we demean we will just get 0
        return self.transform(data)

    def transform(self, data):
        """Scale data based on the fit data.

        Parameters
        ----------
        data : np.ndarray
            The data to transform.

        Returns
        -------
        transformed : np.ndarray
            The transformed input data.
        """
        return (data - self.mean) / self.std

    def invert(self, scaled_data):
        """Invert the scaling for some data.

        Parameters
        ----------
        scaled_data : np.ndarray
            The scaled data.

        Returns
        -------
        data : np.ndarray
            The pre-transformed data.
        """
        return scaled_data * self.std + self.mean

    def save_path(self, path):
        with open(path, 'wb') as f:
            np.savez(
                f,
                ndim=len(self._axes) + 1,
                mean=self.mean,
                std=self.std,
            )

    @classmethod
    def load_path(cls, path):
        with np.load(path) as f:
            self = cls(f['ndim'])
            self._mean = f['mean']
            self._std = f['std']

        return self
