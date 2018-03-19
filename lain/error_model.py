import pathlib

import cytoolz as toolz
import keras
import numpy as np
import scipy.stats
import slider as sl

from .scaler import Scaler
from .utils import dichotomize, summary, rolling_window


class Prediction:
    """The model's predicted values.

    Attributes
    ----------
    predicted_aim_error : np.ndarray[float64]
        The predicted aim errors for each object.
    predicted_aim_distribution : scipy.stats.lognorm
        The predicted distribution of aim errors.
    predicted_accuracy_error : np.ndarray[float64]
        The predicted accuracy errors for each object.
    predicted_accuracy_distribution : scipy.stats.lognorm
        The predicted distribution of accuracy errors.
    accuracy_mean : float
        The mean predicted accuracy.
    accuracy_std : float
        The standard deviation of the predicted accuracy.
    pp_mean : float
        The mean predicted performance points.
    pp_std : float
        The standard deviation of the predicted performance points.
    miss_chance : float
        The chance to miss at least one circle in the beatmap.
    """
    def __init__(self,
                 *,
                 predicted_aim_error,
                 predicted_aim_distribution,
                 predicted_accuracy_error,
                 predicted_accuracy_distribution,
                 accuracy_mean,
                 accuracy_std,
                 pp_mean,
                 pp_std,
                 miss_chance):
        self.predicted_aim_error = predicted_aim_error
        self.predicted_aim_distribution = predicted_aim_distribution
        self.predicted_accuracy_error = predicted_accuracy_error
        self.predicted_accuracy_distribution = predicted_accuracy_distribution
        self.accuracy_mean = accuracy_mean
        self.accuracy_std = accuracy_std
        self.pp_mean = pp_mean
        self.pp_std = pp_std
        self.miss_chance = miss_chance

    @property
    def full_clear_chance(self):
        """The chance to full clear the beatmap.
        """
        return 1 - self.miss_chance


class _InnerErrorModel:
    """A model for osu! which trains on windows of time events and attempts to
    predict the user's aim and accuracy error for each circle.

    Parameters
    ----------
    hidden : bool
        Is this the hidden model?
    aim_pessimism_factor : float
        An exponential increase in aim error to account for the fact that most
        replays are heavily biased towards a user's best replays.
    accuracy_pessimism_factor : float
        An exponential increase in accuracy error to account for the fact that
        most replays are heavily biased towards a user's best replays.
    trailing_context : int
        The number of leading trailing hit objects or slider ticks to look at.
    forward_context : int
        The number of forward hit objects or slider ticks to look at.
    hidden_layer_sizes : tuple[int, int]
        The sizes of the hidden layers.
    dropout : float
        The droput ratio.
    activation : str
        The activation function. Must be one of:
        'tanh', 'softplus, 'softsign', 'relu', 'sigmoid', 'hard_sigmoid', or
        'linear'.
    loss : {'mse', 'mae', 'mape', 'male', 'cosine'
        The loss function.
    optimizer : str
        The optimizer to use.
    max_hit_objects : int
        The maximum number of hit objects allows in a beatmap. Data will be
        padded to fit this shape.

    Notes
    -----
    The inner model does not know about the difference between hidden and
    non-hidden plays. :class:`~slider.model.alpha.AlphaModel` internally
    holds two ``_InnerAlphaModel``\s which are fitted to the hidden and
    non-hidden plays respectively.
    """
    features = {
        k: n for n, k in enumerate(
            (
                # absolute position
                'absolute_x',
                'absolute_y',
                'absolute_time',

                # relative position
                'relative_x',
                'relative_y',
                'relative_time',

                # misc.
                'is_slider_tick',  # 1 if slider tick else 0
                'approach_rate',  # the map's approach rate

                # distances (magnitude of vector between two hit objects)
                'distance_from_previous',  # distance from the previous object
                'distance_to_next',  # distance to the next object

                # angles
                'pitch',
                'roll',
                'yaw',
            ),
        )
    }
    _absolule_features_columns = np.s_[0:3]
    _relative_features_columns = np.s_[3:6]
    _angle_features_columns = np.s_[10:13]

    def __init__(self,
                 hidden,
                 *,
                 aim_pessimism_factor=1.1,
                 accuracy_pessimism_factor=1.1,
                 trailing_context=10,
                 forward_context=3,
                 batch_size=32,
                 lstm_hidden_layer_sizes=(256, 128, 64),
                 dropout=0.1,
                 activation='linear',
                 loss='mae',
                 optimizer='rmsprop'):
        self.hidden = hidden

        self._aim_pessimism_factor = aim_pessimism_factor
        self._accuracy_pessimism_factor = accuracy_pessimism_factor

        self._trailing_context = trailing_context
        self._forward_context = forward_context
        self._window = window = trailing_context + forward_context + 1

        self._batch_size = batch_size

        if len(lstm_hidden_layer_sizes) == 0:
            raise ValueError('there must be at least one lstm hidden layer')

        input_ = keras.layers.Input(shape=(window, len(self.features)))
        lstm = keras.layers.LSTM(
            lstm_hidden_layer_sizes[0],
            dropout=dropout,
            return_sequences=True,
        )(input_)

        for size in lstm_hidden_layer_sizes[1:-1]:
            lstm = keras.layers.LSTM(
                size,
                dropout=dropout,
                return_sequences=True,
            )(lstm)

        lstm = keras.layers.LSTM(
            lstm_hidden_layer_sizes[-1],
            dropout=dropout,
        )(lstm)

        aim_error = keras.layers.Dense(
            1,
            activation=activation,
            name='aim_error',
        )(lstm)
        accuracy_error = keras.layers.Dense(
            1,
            activation=activation,
            name='accuracy_error',
        )(lstm)

        self._model = model = keras.models.Model(
            inputs=input_,
            outputs=[aim_error, accuracy_error],
        )
        model.compile(
            loss=loss,
            optimizer=optimizer,
        )

        self._feature_scaler = Scaler(ndim=3)

    _angle_axes_map = {
        'yaw': (0, 1),
        'roll': (0, 2),
        'pitch': (1, 2),
    }
    _raw_features = [
        features['absolute_x'],
        features['absolute_y'],
        features['absolute_time'],
        features['is_slider_tick'],
        features['approach_rate'],
    ]

    def _extract_features(self,
                          beatmap,
                          *,
                          double_time=False,
                          half_time=False,
                          hard_rock=False):
        """Extract the feature array from a beatmap.

        Parameters
        ----------
        beatmap : sl.Beatmap
            The beatmap to extract features for.
        double_time : bool, optional
            Extract features for double time?
        half_time : bool, optional
            Extract features for half time?
        hard_rock : bool, optional
            Extract features for hard rock?

        Returns
        -------
        windows : np.ndarray[float]
            An array of observations.
        mask : np.ndarray[bool]
            A mask of valid hit objects.
        """
        hit_objects = beatmap.hit_objects_no_spinners
        approach_rate = sl.mod.ar_to_ms(
            beatmap.ar(
                double_time=double_time,
                half_time=half_time,
                hard_rock=hard_rock,
            ),
        )

        # events holds an (x, y, time) tuple for every hit object and slider
        # tick; slider ticks are in ms
        events = []
        append_event = events.append
        extend_events = events.extend

        # hit_object_ixs holds the indices into ``events`` where hit objects
        # appear
        hit_object_ixs = []
        append_hit_object_ixs = hit_object_ixs.append
        for hit_object in hit_objects:
            # mark this index in events as the location of a hit object.
            append_hit_object_ixs(len(events))

            if double_time:
                hit_object = hit_object.double_time
            elif half_time:
                hit_object = hit_object.half_time

            if hard_rock:
                hit_object = hit_object.hard_rock

            position = hit_object.position

            append_event((
                position.x,
                position.y,
                hit_object.time.total_seconds() * 1000,
                0,  # is_slider_tick
                approach_rate,
            ))

            if isinstance(hit_object, sl.beatmap.Slider):
                # add all the slider ticks
                extend_events(
                    (
                        x,
                        y,
                        time.total_seconds() * 1000,
                        1,  # is_slider_tick
                        approach_rate,
                    )
                    for x, y, time in hit_object.tick_points
                )

        # allocate the empty output array
        out = np.empty((len(events), len(self.features)))

        # fill the output with the directly extracted features
        out[:, self._raw_features] = events
        out[:, self._relative_features_columns] = (
            out[:, self._absolule_features_columns]
        )

        # the baseline data to take windows out of
        baseline = np.vstack([
            np.full((self._trailing_context, len(self.features)), np.nan),
            out,
            np.full((self._forward_context, len(self.features)), np.nan),
        ])

        # pull out the x, y, time coordinates
        coords = baseline[:, self._absolule_features_columns]

        # draw triangles around each object with the leading and trailing hit
        # object
        triangles = rolling_window(coords, 3)

        # the squared value of the length of each side in the triangles around
        # each object across each axis
        diff_a_b_sq = np.square(triangles[:, 0] - triangles[:, 1])
        diff_a_c_sq = np.square(triangles[:, 0] - triangles[:, 2])
        diff_b_c_sq = np.square(triangles[:, 1] - triangles[:, 2])

        # for each hit object and axis (x, y, time), calculate the angle
        # between the previous hit object and the next hit object
        for angle_kind, (axis_0, axis_1) in self._angle_axes_map.items():
            a_b_sq = diff_a_b_sq[:, axis_0] + diff_a_b_sq[:, axis_1]
            b_c_sq = diff_b_c_sq[:, axis_0] + diff_b_c_sq[:, axis_1]

            numerator = (
                a_b_sq +
                b_c_sq -
                (diff_a_c_sq[:, axis_0] + diff_a_c_sq[:, axis_1])
            )
            denominator = (2 * np.sqrt(a_b_sq) * np.sqrt(b_c_sq))

            mask = np.isclose(denominator, 0)
            numerator[mask] = 1
            denominator[mask] = 1
            np.arccos(
                # clip the values because we sometimes get things like
                # 1.0000000000000002 which breaks ``np.arccos``
                np.clip(numerator / denominator, -1, 1),
                out=baseline[1:-1, self.features[angle_kind]],
            )

        # compute the distance from hit object to hit object in 3d space
        # and store the distance from the previous and the distance to the
        # next
        distances = np.sqrt(np.square(coords[1:] - coords[:-1]).sum(axis=1))
        baseline[:-1, self.features['distance_from_previous']] = distances
        baseline[1:, self.features['distance_to_next']] = distances

        # convert the hit object indices into a column vector; we add context
        # to account for the padding
        hit_object_ixs_array = (
            np.array(hit_object_ixs) + self._trailing_context
        )[:, np.newaxis]

        # get an array of offsets from some base ``ix`` which correspond to a
        # fully populated window; this is a row vector
        context_ix_offsets = np.arange(
            -self._trailing_context,
            self._forward_context + 1,
        )

        # broadcast the hit_object_ixs together with the context_ix_offsets
        # to get an indexer which produces a 3d array which is a sequence of
        # windows of events
        window_ixs = hit_object_ixs_array + context_ix_offsets

        # a sequence of windows of events where the time is still absolute
        windows = baseline[window_ixs]

        # slice out the window center's 'relative' values which are currently
        # absolute values
        center_values = windows[
            :,
            self._trailing_context,
            np.newaxis,
            self._relative_features_columns,
        ]

        # subtract the hit object features from the windows; this makes the
        # window relative to the object being predicted.
        windows[..., self._relative_features_columns] -= center_values

        # only accept complete windows
        mask = ~np.isnan(windows).any(axis=(1, 2))

        # remove the partial windows
        windows = windows[mask]

        return windows, mask

    def _extract_differences(self, replay):
        """Extract the time and position differences for each hit object.

        Parameters
        ----------
        replay : Replay
            The replay to get differences for.

        Returns
        -------
        differences : np.ndarray
            An array of shape (len(hit_objects), 2) where the first column
            is the time offset in milliseconds and the second column is the
            magnitude of (x, y) error in osu! pixels.
        """
        # get the x, y, and time of each click
        if replay.double_time:
            time_coefficient = 1000 * 2 / 3
        elif replay.half_time:
            time_coefficient = 1000 * 4 / 3
        else:
            time_coefficient = 1000

        clicks = np.array([
            (
                second.position.x,
                second.position.y,
                time_coefficient * second.offset.total_seconds(),
            )
            for first, second in toolz.sliding_window(2, replay.actions)
            if ((second.key1 and not first.key1 or
                 second.key2 and not first.key2) and
                # there are weird stray clicks at (0, 0)
                second.position != (0, 0))
        ])

        double_time = replay.double_time
        half_time = replay.half_time
        hard_rock = replay.hard_rock

        # accumulate the (x, y, start time) of each hit object
        hit_object_coords = []
        append_coord = hit_object_coords.append
        for hit_object in replay.beatmap.hit_objects_no_spinners:
            if double_time:
                hit_object = hit_object.double_time
            elif half_time:
                hit_object = hit_object.half_time

            if hard_rock:
                hit_object = hit_object.hard_rock

            position = hit_object.position
            append_coord((
                position.x,
                position.y,
                hit_object.time.total_seconds() * 1000,
            ))

        # convert the hit object coordinates into an array
        hit_object_coords = np.array(hit_object_coords)

        # get the time of each hit object as a row vector
        hit_object_times = hit_object_coords[:, 2]

        # get the time of each click as a column vector
        click_times = clicks[:, [2]]

        # find the nearest click by taking the absolute difference from
        # every hit object to every click (whose shape is:
        # (len(clicks), len(hit_objects))) and reducing with agrgmin to
        # get the index of the best match for each hit object
        nearest_click_ix = np.abs(
            hit_object_times - click_times,
        ).argmin(axis=0)

        # get the x, y, time of the matched clicks
        matched_clicks = clicks[nearest_click_ix]

        # get the squared distance for the x and y axes
        squared_distance = (
            hit_object_coords[:, :2] - matched_clicks[:, :2]
        ) ** 2

        aim_error = np.sqrt(squared_distance[:, 0] + squared_distance[:, 1])

        # clip the aim error to within 2 * circle radius; things farther
        # than this were probably just us skipping the circle entirely
        np.clip(
            aim_error,
            0,
            2 * sl.mod.circle_radius(replay.beatmap.cs(hard_rock=hard_rock)),
            out=aim_error,
        )

        accuracy_error = np.abs(hit_object_times - matched_clicks[:, 2])

        # clip the accuracy error to within 1.5 * 50 window; things farther
        # than this were probably just us skipping the circle entirely
        np.clip(
            accuracy_error,
            0,
            1.5 * sl.mod.od_to_ms(replay.beatmap.od(
                hard_rock=hard_rock,
                double_time=double_time,
                half_time=half_time,
            )).hit_50,
            out=accuracy_error,
        )

        return aim_error, accuracy_error

    def _sample_weights(self, aim_error, accuracy_error):
        """Sample weights based on the error.

        Parameters
        ----------
        aim_error : np.ndarray
            The aim errors for each sample.
        accuracy_error : np.ndarray
            The accuracy error errors for each sample.

        Returns
        -------
        weights : np.ndarray
            The weights for each sample.

        Notes
        -----
        This weighs samples based on their standard deviations above the mean
        with some clipping.
        """
        aim_zscore = (aim_error - aim_error.mean()) / aim_error.std()
        aim_weight = np.clip(aim_zscore, 1, 4) / 4

        accuracy_zscore = (
            accuracy_error - accuracy_error.mean()
        ) / accuracy_error.std()
        accuracy_weight = np.clip(accuracy_zscore, 1, 4) / 4

        return {
            'aim_error': aim_weight,
            'accuracy_error': accuracy_weight,
        }

    def fit(self, replays, *, verbose=False, epochs=10):
        extract_features = self._extract_features
        extract_differences = self._extract_differences
        model = self._model

        features = []
        append_features = features.append

        aim_error = []
        append_aim_error = aim_error.append

        accuracy_error = []
        append_accuracy_error = accuracy_error.append

        for n, replay in enumerate(replays):
            if verbose:
                print(f'{n:4}: {replay!r}')

            windows, mask = extract_features(
                replay.beatmap,
                double_time=replay.double_time,
                half_time=replay.half_time,
                hard_rock=replay.hard_rock,
            )
            append_features(windows)
            aim, accuracy = extract_differences(replay)
            append_aim_error(aim[mask])
            append_accuracy_error(accuracy[mask])

        pre_scaled_features = np.concatenate(features)
        features = self._feature_scaler.fit(pre_scaled_features)
        aim_error = np.concatenate(aim_error)
        accuracy_error = np.concatenate(accuracy_error)

        if verbose:
            # print some useful summary statistics; this helps quickly
            # identify data errors.
            print(summary(
                self.features,
                pre_scaled_features,
                aim_error=aim_error,
                accuracy_error=accuracy_error,
            ))

        return model.fit(
            features,
            {'aim_error': aim_error, 'accuracy_error': accuracy_error},
            verbose=int(bool(verbose)),
            batch_size=self._batch_size,
            epochs=epochs,
            sample_weight=self._sample_weights(aim_error, accuracy_error),
        )

    @staticmethod
    def _fit_lognorm_distribution(array):
        """Fit a probability distribution to the array.

        Parameters
        ----------
        array : np.ndarray
            The data to fit.

        Returns
        -------
        distribution : scipy.stats.lognorm
            A frozen distribution instance.
        """
        return scipy.stats.lognorm(*scipy.stats.lognorm.fit(array))

    def _predict_raw_error(self,
                           beatmap,
                           *,
                           pessimistic=True,
                           double_time=False,
                           half_time=False,
                           hard_rock=False):
        """Predict the time and position differences for each circle.

        Parameters
        ----------
        beatmap : sl.Beatmap
            The beatmap to predict.
        pessimistic : bool, optional
            Apply pessimistic error scaling?
        double_time : bool, optional
            Predict double time offsets.
        half_time : bool, optional
            Predict half time offsets.
        hard_rock : bool, optional
            Predict hard_rock offsets.

        Returns
        -------
        aim_error : np.ndarray
            The predicted magnitude of (x, y) error in osu! pixels.
        accuracy_error : np.ndarray
            The predicted magnitude of time error in milliseconds.
        """
        aim_error, accuracy_error = self._model.predict(
            self._feature_scaler.transform(
                self._extract_features(
                    beatmap,
                    double_time=double_time,
                    half_time=half_time,
                    hard_rock=hard_rock,
                )[0],
            ),
        )

        if pessimistic:
            aim_error **= self._aim_pessimism_factor
            accuracy_error **= self._accuracy_pessimism_factor

        return aim_error, accuracy_error

    def predict(self,
                beatmap,
                *,
                pessimistic=True,
                double_time=False,
                half_time=False,
                hard_rock=False,
                random_state=None,
                samples=1000):
        """Predict the user's accuracy on the beatmap with the given mods.

        Parameters
        ----------
        beatmap : sl.Beatmap
            The beatmap to predict.
        pessimistic : bool, optional
            Apply pessimistic error scaling?
        double_time : bool, optional
            Predict double time offsets.
        half_time : bool, optional
            Predict half time offsets.
        hard_rock : bool, optional
            Predict hard_rock offsets.
        random_state : np.random.RandomState, optional
            The numpy random state used to draw samples.
        samples : int, optional
            The number of plays to simulate.

        Returns
        -------
        prediction : Prediction
            A collection of predicted values for this play.
        """
        aim_error, accuracy_error = self._predict_raw_error(
            beatmap,
            double_time=double_time,
            half_time=half_time,
            hard_rock=hard_rock,
            pessimistic=pessimistic,
        )

        # fit the distributions to the predicted data
        aim_distribution = self._fit_lognorm_distribution(aim_error)
        accuracy_distribution = self._fit_lognorm_distribution(accuracy_error)

        predicted_object_count = len(beatmap.hit_objects_no_spinners)
        spinner_count = len(beatmap.hit_objects) - predicted_object_count

        aim_samples = aim_distribution.rvs(
            predicted_object_count * samples,
            random_state=random_state,
        )
        accuracy_samples = accuracy_distribution.rvs(
            predicted_object_count * samples,
            random_state=random_state,
        )

        hit_windows = np.array([
            sl.mod.od_to_ms(
                beatmap.od(
                    hard_rock=hard_rock,
                    double_time=double_time,
                    half_time=half_time,
                ),
            ),
        ]).T

        circle_radius = sl.mod.circle_radius(beatmap.cs(hard_rock=hard_rock))

        simulated_300, simulated_100, simulated_50, simulated_miss = (
            3 - (
                (aim_samples <= circle_radius) &
                (accuracy_samples <= hit_windows)
            ).sum(axis=0) ==
            np.array([[0, 1, 2, 3]]).T
        ).T.reshape(samples, -1, 4).sum(axis=1).T
        simulated_300 += spinner_count  # assume perfect spinners

        simulated_accuracies = sl.utils.accuracy(
            simulated_300,
            simulated_100,
            simulated_50,
            simulated_miss,
        )

        simulated_pp = beatmap.performance_points(
            count_300=simulated_300,
            count_100=simulated_100,
            count_50=simulated_50,
            count_miss=simulated_miss,
            hidden=self.hidden,
            hard_rock=hard_rock,
            double_time=double_time,
            half_time=half_time,
        )

        chance_to_miss_scalar = aim_distribution.cdf(circle_radius)
        miss_chance = chance_to_miss_scalar ** predicted_object_count

        return Prediction(
            predicted_aim_error=aim_error,
            predicted_aim_distribution=aim_distribution,
            predicted_accuracy_error=accuracy_error,
            predicted_accuracy_distribution=accuracy_distribution,
            accuracy_mean=simulated_accuracies.mean(),
            accuracy_std=simulated_accuracies.std(),
            pp_mean=simulated_pp.mean(),
            pp_std=simulated_pp.std(),
            miss_chance=miss_chance,
        )

    def save_path(self, path):
        self._model.save(path)
        self._feature_scaler.save_path(path.with_suffix('.feature_scaler'))

        with open(path.with_suffix('.pessimism'), 'wb') as f:
            np.savez(
                f,
                aim_pessimism_factor=self._aim_pessimism_factor,
                accuracy_pessimism_factor=self._accuracy_pessimism_factor,
            )

    @classmethod
    def load_path(cls, path, *, hidden):
        self = cls(hidden=hidden)
        self._model = keras.models.load_model(path)
        self._feature_scaler = Scaler.load_path(
            path.with_suffix('.feature_scaler'),
        )

        with np.load(str(path.with_suffix('.pessimism'))) as f:
            self._aim_pessimism_factor = f['aim_pessimism_factor']
            self._accuracy_pessimism_factor = f['accuracy_pessimism_factor']

        return self


class ErrorModel:
    """A model for osu! which trains on windows of time events and attempts to
    predict the user's aim and accuracy error for each circle.

    Parameters
    ----------
    context : int
        The number of leading and trailing hit objects or slider ticks to look
        at.
    aim_pessimism_factor : float
        An exponential increase in aim error to account for the fact that most
        replays are heavily biased towards a user's best replays.
    accuracy_pessimism_factor : float
        An exponential increase in accuracy error to account for the fact that
        most replays are heavily biased towards a user's best replays.
    trailing_context : int
        The number of leading trailing hit objects or slider ticks to look at.
    forward_context : int
        The number of forward hit objects or slider ticks to look at.
    hidden_layer_sizes : tuple[int, int]
        The sizes of the hidden layers.
    dropout : float
        The droput ratio.
    activation : str
        The activation function. Must be one of:
        'tanh', 'softplus, 'softsign', 'relu', 'sigmoid', 'hard_sigmoid', or
        'linear'.
    loss : {'mse', 'mae', 'mape', 'male', 'cosine'
        The loss function.
    optimizer : str
        The optimizer to use.
    max_hit_objects : int
        The maximum number of hit objects allows in a beatmap. Data will be
        padded to fit this shape.
    """
    version = 0

    def __init__(self, *args, **kwargs):
        self._hidden_model = _InnerErrorModel(True, *args, **kwargs)
        self._non_hidden_model = _InnerErrorModel(False, *args, **kwargs)

    def save_path(self, path):
        """Serialize the model as a directory.

        Parameters
        ----------
        path : path-like
            The path to the directory to serialize the model to.

        See Also
        --------
        lain.ErrorModel.load_path
        """
        path = pathlib.Path(path)
        path.mkdir(exist_ok=True)
        (path / 'version').write_text(str(self.version))
        self._hidden_model.save_path(path / 'hidden')
        self._non_hidden_model.save_path(path / 'non-hidden')

    @classmethod
    def load_path(cls, path):
        """Deserialize a model from a directory.

        Parameters
        ----------
        path : path-like
            The path to the directory to load.

        Returns
        -------
        self : lain.ErrorModel
            The loaded model.

        See Also
        --------
        lain.ErrorModel.save_path
        """
        path = pathlib.Path(path)
        version = int((path / 'version').read_text())
        if version != cls.version:
            raise ValueError(
                f'saved model is of version {version} but the code is on'
                f' version {cls.version}',
            )

        self = cls()
        self._hidden_model = _InnerErrorModel.load_path(
            path / 'hidden',
            hidden=True,
        )
        self._non_hidden_model = _InnerErrorModel.load_path(
            path / 'non-hidden',
            hidden=False,
        )
        return self

    def fit(self, replays, *, verbose=False, epochs=10):
        """Fit the model to data.

        Parameters
        ----------
        replays : iterable[sl.Replay]
            The replays to fit the model to.
        verbose : bool, optional
            Print verbose messages to stdout?
        epochs : int, optional
            The number of times to pass through the replays.

        Returns
        -------
        hidden_history : keras.History
            The history of training the keras model on the replays with hidden.
        non_hidden_history : keras.History
            The history of training the keras model on the replays without
            hidden.
        """
        hidden, non_hidden = dichotomize(lambda replay: replay.hidden, replays)

        if verbose:
            print('fitting the hidden replays')
        hidden_history = self._hidden_model.fit(
            hidden,
            verbose=verbose,
            epochs=epochs,
        )

        if verbose:
            print('fitting the non-hidden replays')
        non_hidden_history = self._non_hidden_model.fit(
            non_hidden,
            verbose=verbose,
            epochs=epochs,
        )

        return hidden_history, non_hidden_history

    def predict(self,
                beatmap,
                *,
                pessimistic=True,
                hidden=False,
                double_time=False,
                half_time=False,
                hard_rock=False,
                random_state=None,
                samples=1000):
        """Predict the user's accuracy on the beatmap with the given mods.

        Parameters
        ----------
        beatmap : sl.Beatmap
            The beatmap to predict.
        pessimistic : bool, optional
            Apply pessimistic error scaling?
        hidden : bool, optional
            Predict performance with hidden?
        double_time : bool, optional
            Predict performance with double time?
        half_time : bool, optional
            Predict performance with half time?
        hard_rock : bool, optional
            Predict performance with hard rock?
        random_state : np.random.RandomState, optional
            The numpy random state used to draw samples.
        samples : int, optional
            The number of plays to simulate.

        Returns
        -------
        prediction : Prediction
            A collection of predicted values for this play.
        """
        if hidden:
            predict = self._hidden_model.predict
        else:
            predict = self._non_hidden_model.predict

        return predict(
            beatmap,
            pessimistic=pessimistic,
            double_time=double_time,
            half_time=half_time,
            hard_rock=hard_rock,
            random_state=random_state,
            samples=samples,
        )
