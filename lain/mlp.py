import pathlib

import keras

from .model import OsuModel
from .manual_features import (
    extract_feature_array,
    extract_features_and_labels,
    features as _features,
)
from .scaler import Scaler
from .utils import summary


class MLPRegressor(OsuModel):
    """An osu! aware MLPRegressor.

    Parameters
    ----------
    l2_penalty : float
        L2 penalty (regularization term).
    hidden_layer_sizes : iterable[int]
        The number of neurons in each hidden layer.
    activation : str
        The activation function. Must be one of:
        'tanh', 'softplus, 'softsign', 'relu', 'sigmoid', 'hard_sigmoid', or
        'linear'.
    loss : {'mse', 'mae', 'mape', 'male', 'cosine'
        The loss function.
    optimizer : str
        The optimizer to use.
    """
    version = 0
    features = _features

    def __init__(self,
                 *,
                 l2_penalty=0.009,
                 hidden_layer_sizes=(54, 199, 66),
                 dropout=0.2,
                 activation='tanh',
                 loss='mse',
                 optimizer='rmsprop'):
        self._feature_scaler = Scaler(ndim=2)

        layer = input_ = keras.layers.Input(shape=(len(self.features),))

        # add all of the dense layers
        kernel_regularizer = keras.regularizers.l2(l2_penalty)
        for size in hidden_layer_sizes:
            layer = keras.layers.Dropout(dropout)(
                keras.layers.Dense(
                    size,
                    activation=activation,
                    kernel_regularizer=kernel_regularizer,
                )(layer),
            )

        accuracy = keras.layers.Dense(1, name='accuracy')(layer)
        self._model = model = keras.models.Model(
            inputs=input_,
            outputs=accuracy,
        )
        model.compile(
            loss=loss,
            optimizer=optimizer,
        )

    def save_path(self, path):
        path = pathlib.Path(path)
        path.mkdir(exist_ok=True)
        (path / 'version').write_text(str(self.version))
        self._model.save(path / 'model')
        self._feature_scaler.save_path(path / 'feature_scaler')

    @classmethod
    def load_path(cls, path):
        self = cls()
        version = int((path / 'version').read_text())
        if version != cls.version:
            raise ValueError(
                f'saved model is of version {version} but the code is on'
                f' version {cls.version}',
            )

        self._model = keras.models.load_model(path / 'model')
        self._feature_scaler = Scaler.load_path(path / 'feature_scaler')
        return self

    def fit(self, replays, *, epochs=10, verbose=False):
        features, labels = extract_features_and_labels(
            replays,
            verbose=verbose,
        )

        if verbose:
            # print some useful summary statistics; this helps quickly
            # identify data errors.
            print(summary(
                self.features,
                features,
                accuracy=labels,
            ))

        features = self._feature_scaler.fit(features)
        return self._model.fit(
            features,
            labels,
            epochs=epochs,
            verbose=verbose,
        )

    def predict(self, beatmaps_and_mods):
        features = extract_feature_array(beatmaps_and_mods)
        return self._model.predict(self._feature_scaler.transform(features))
