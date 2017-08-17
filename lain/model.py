from itertools import chain

from slider.abc import ABCMeta, abstractmethod


class OsuModel(metaclass=ABCMeta):
    """Abstract osu! prediction model.
    """
    @abstractmethod
    def save_path(self, path):
        """Save the state of the model to a file path.

        Parameters
        ----------
        path : path-like
            The path to save the model to.
        """
        raise NotImplementedError('save_path')

    @classmethod
    @abstractmethod
    def load_path(cls, path):
        """Load a model from a file path.

        Parameters
        ----------
        path : path-like
            The path to the saved model.

        Returns
        -------
        model : OsuModel
            The deserialized osu! model.
        """
    @abstractmethod
    def fit(self, replays, *, epochs=None, verbose=False):
        """Fit the model to some replays.

        Parameters
        ----------
        replays : iterable[Replay]
            The replays to fit the model to.
        verbose : bool, optional
            Print verbose information to ``sys.stdout``?
        **kwargs
            Model specific arguments, for example: ``epochs``.
        """
        raise NotImplementedError('fit')

    @abstractmethod
    def predict(self, beatmaps_and_mods):
        """Predict the user's accuracy for a sequence a beatmaps with the given
        mods.

        Parameters
        ----------
        beatmaps_and_mods : iterable[(Beatmap, dict)]
            The maps and mods to predict.

        Returns
        -------
        accuracy : np.ndarray[float]
            The user's expected accuracy in the range [0, 1] for each
            (beatmap, mods) pair.
        """
        raise NotImplementedError('predict')

    def predict_beatmap(self, beatmap, *mods, **mods_scalar):
        """Predict the user's accuracy for the given beatmap.

        Parameters
        ----------
        beatmap : Beatmap
            The map to predict the performance of.
        *mods
            A sequence of mod dictionaries to predict for.
        **mods_dict
            Mods to predict for.

        Returns
        -------
        accuracy : float
            The user's expected accuracy in the range [0, 1].
        """
        for mod_name in 'hidden', 'hard_rock', 'half_time', 'double_time':
            mods_scalar.setdefault(mod_name, False)

        return self.predict([
            (beatmap, ms) for ms in chain(mods, [mods_scalar])
        ])
