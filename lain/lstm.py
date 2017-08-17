from itertools import chain
import pathlib

import cytoolz as toolz
import keras
import numpy as np
import scipy.stats
from slider.beatmap import Slider
from slider.mod import od_to_ms, circle_radius, ar_to_ms

from .model import OsuModel
from .scaler import Scaler
from .utils import dichotomize, summary


class _InnerLSTM:
    """An LSTM model for osu! which trains on windows of (x, y, time) events.

    Parameters
    ----------
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
    non-hidden plays. :class:`~slider.model.lstm.LSTM` internally
    holds two ``_InnerLSTM``\s which are fitted to the hidden and
    non-hidden plays respectively.
    """
    features = {
        k: n for n, k in enumerate(
            (
                # relative position
                'relative_x',
                'relative_y',
                'relative_time',
                # absolute position
                'absolute_x',
                'absolute_y',
                'absolute_time',
                # misc.
                'is_slider_tick',  # 1 if slider tick else 0
                'approach_rate',  # the map's approach rate
            ),
        )
    }
    relative_features_count = sum(
        1 for f in features if f.startswith('relative_')
    )

    def __init__(self,
                 *,
                 trailing_context=10,
                 forward_context=3,
                 batch_size=32,
                 lstm_hidden_layer_sizes=(64, 128, 256),
                 dropout=0.1,
                 activation='linear',
                 loss='mse',
                 optimizer='rmsprop'):

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

    def _extract_features(self,
                          beatmap,
                          *,
                          double_time=False,
                          half_time=False,
                          hard_rock=False):
        """Extract the feature array from a beatmap.

        Parameters
        ----------
        beatmap : Beatmap
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
        approach_rate = ar_to_ms(
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

            time = hit_object.time.total_seconds() * 1000
            x = position.x
            y = position.y
            append_event((
                x,
                y,
                time,
                x,
                y,
                time,
                0,  # is_slider_tick
                approach_rate,
            ))

            if isinstance(hit_object, Slider):
                # add all the slider ticks
                extend_events(
                    (
                        x,
                        y,
                        time.total_seconds() * 1000,
                        x,
                        y,
                        time.total_seconds() * 1000,
                        1,  # is_slider_tick
                        approach_rate,
                    )
                    for x, y, time in hit_object.tick_points
                )

        # convert the events into a 2d array of shape:
        # (len(events), len(self.features))
        event_array = np.array(events)

        # the baseline data to take windows out of
        baseline = np.vstack([
            np.full((self._trailing_context, len(self.features)), np.nan),
            event_array,
            np.full((self._forward_context, len(self.features)), np.nan),
        ])

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
            :self.relative_features_count,
        ]

        # subtract the hit object features from the windows; this makes the
        # window relative to the object being predicted.
        windows[..., :self.relative_features_count] -= center_values

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
            2 * circle_radius(replay.beatmap.cs(hard_rock=hard_rock)),
            out=aim_error,
        )

        accuracy_error = np.abs(hit_object_times - matched_clicks[:, 2])

        # clip the accuracy error to within 1.5 * 50 window; things farther
        # than this were probably just us skipping the circle entirely
        np.clip(
            accuracy_error,
            0,
            1.5 * od_to_ms(replay.beatmap.od(
                hard_rock=hard_rock,
                double_time=double_time,
                half_time=half_time,
            )).hit_50,
            out=accuracy_error,
        )

        return aim_error, accuracy_error

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
        )

    def predict_differences(self,
                            beatmap,
                            *,
                            double_time=False,
                            half_time=False,
                            hard_rock=False):
        """Predict the time and position differences for each circle.

        Parameters
        ----------
        beatmap : Beatmap
            The beatmap to predict.
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
        return self._model.predict(
            self._feature_scaler.transform(
                self._extract_features(
                    beatmap,
                    double_time=double_time,
                    half_time=half_time,
                    hard_rock=hard_rock,
                )[0],
            ),
        )

    def _fit_lognorm_distribution(self, array):
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

    def _predict_accuracy(self,
                          beatmap,
                          *,
                          double_time=False,
                          half_time=False,
                          hard_rock=False):
        """Predict the user's accuracy on the beatmap with the given mods.

        Parameters
        ----------
        beatmap : Beatmap
            The beatmap to predict.
        double_time : bool, optional
            Predict double time offsets.
        half_time : bool, optional
            Predict half time offsets.
        hard_rock : bool, optional
            Predict hard_rock offsets.

        Returns
        -------
        predicted_accuracy : float
            A scalar value for the user's predicted accuracy.
        """
        aim_error, accuracy_error = self.predict_differences(
            beatmap,
            double_time=double_time,
            half_time=half_time,
            hard_rock=hard_rock,
        )

        # fit the distributions to the predicted data
        aim_distribution = self._fit_lognorm_distribution(aim_error)
        accuracy_distribution = self._fit_lognorm_distribution(accuracy_error)

        # the expected aim is the probability that a user clicks within the
        # radius of a circle
        expected_aim = aim_distribution.cdf(
            circle_radius(beatmap.cs(hard_rock=hard_rock)),
        )

        hit_windows = od_to_ms(
            beatmap.od(
                hard_rock=hard_rock,
                double_time=double_time,
                half_time=half_time,
            ),
        )

        # To find the expected accuracy we need to calculate the probability of
        # clicking within the 300 threshold, between the 300 and 100
        # thresholds, and between the 100 and 50 thresholds. To find the
        # probability that a click falls between the given thresholds we need
        # to integrate the pdf from the starting threshold to the ending
        # threshold. Visually this looks like:
        #
        # Probability of clicking at a given time
        #
        #      |      ^
        #      |     / \
        #      |    /   -.
        # P(x) |   /    : \.
        #      |  /     :   \
        #      | |      :    ---
        #      | |      :      :\--..,..
        #      -------------------------
        #      :        :  ms  :     :
        #      :        :      :     :
        #      :  P(300):P(100):P(50):P(miss)
        #
        # We weigh the probabilities by the accuracy the givey, so we multiply
        # the probability of clicking within the 300 range by 1, we multiply
        # the probability of clicking within the 300-100 range by 1 / 3, we
        # multiply the probability of clicking within the 100-50 range by
        # 1 / 6, and finally we implicitly multiply the probability of missing
        # entirely by 0. The expected accuracy value is the sum of these
        # weighted probabilities.
        cdf_300 = accuracy_distribution.cdf(hit_windows.hit_300)
        cdf_100 = accuracy_distribution.cdf(hit_windows.hit_100)
        cdf_50 = accuracy_distribution.cdf(hit_windows.hit_50)
        expected_accuracy = (
            cdf_300 +
            (cdf_100 - cdf_300) / 3 +
            (cdf_50 - cdf_100) / 6
        )

        @accuracy_distribution.expect
        def expected_accuracy(error):
            if error <= hit_windows.hit_300:
                return 1
            elif error <= hit_windows.hit_100:
                return 1 / 3
            elif error <= hit_windows.hit_50:
                return 1 / 6
            else:
                return 0

        return expected_aim * expected_accuracy

    def predict_one(self, beatmap, mods):
        return self._predict_accuracy(
            beatmap,
            double_time=mods['double_time'],
            half_time=mods['half_time'],
            hard_rock=mods['hard_rock'],
        )

    def save_path(self, path):
        self._model.save(path)
        self._feature_scaler.save_path(path.with_suffix('.feature_scaler'))

    @classmethod
    def load_path(cls, path):
        self = cls()
        self._model = keras.models.load_model(path)
        self._feature_scaler = Scaler.load_path(
            path.with_suffix('.feature_scaler'),
        )
        return self


class LSTM(OsuModel):
    """An LSTM model for osu! which trains on windows of (x, y, time) events.

    Parameters
    ----------
    context : int
        The number of leading and trailing hit objects or slider ticks to look
        at.
    hidden_layer_sizes : tuple[int, int]
        The sizes of the hidden layers.
    dropout : float
        The droput ratio.
    activation : str
        The activation function.
    loss : str
        The loss function.
    optimizer : str
        The optimizer to use.
    max_hit_objects : int
        The maximum number of hit objects allows in a beatmap. Data will be
        padded to fit this shape.
    """
    version = 0

    def __init__(self, *args, **kwargs):
        self._hidden_model = _InnerLSTM(*args, **kwargs)
        self._non_hidden_model = _InnerLSTM(*args, **kwargs)

    def save_path(self, path):
        path = pathlib.Path(path)
        path.mkdir(exist_ok=True)
        (path / 'version').write_text(str(self.version))
        self._hidden_model.save_path(path / 'hidden')
        self._non_hidden_model.save_path(path / 'non-hidden')

    @classmethod
    def load_path(cls, path):
        path = pathlib.Path(path)
        version = int((path / 'version').read_text())
        if version != cls.version:
            raise ValueError(
                f'saved model is of version {version} but the code is on'
                f' version {cls.version}',
            )

        self = cls()
        self._hidden_model = _InnerLSTM.load_path(path / 'hidden')
        self._non_hidden_model = _InnerLSTM.load_path(path / 'non-hidden')
        return self

    def fit(self, replays, *, verbose=False, epochs=10):
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

    def predict(self, beatmaps_and_mods):
        hidden_model_predict = self._hidden_model.predict_one
        non_hidden_model_predict = self._non_hidden_model.predict_one

        return np.array([
            (
                hidden_model_predict
                if mods['hidden'] else
                non_hidden_model_predict
            )(beatmap, mods)
            for beatmap, mods in beatmaps_and_mods
        ])

    def predict_differences(self, beatmaps_and_mods):
        """Low level predictions for a sequence of beatmaps and mods.

        Parameters
        ----------
        beatmaps_and_mods : iterable[(Beatmap, dict)]
            The maps and mods to predict.

        Returns
        -------
        differences : list[np.ndarray[float]]
            For each beatmap and mod combination, an array of shape
            ``(len(hit_objects), 2)`` where the first column holds the
            predicted absolute click offset from the center of the circle and
            the second column holds the absolute time offset for the predicted
            click. The row indices match the order the beatmap and mod
            combinations are passed.

        Notes
        -----
        This method is specific to the :class:`~slider.model.lstm.LSTM` model,
        it may not be available on other model types.
        """
        hidden_model_predict = self._hidden_model.predict_differences
        non_hidden_model_predict = self._non_hidden_model.predict_differences

        return [
            (
                hidden_model_predict
                if mods['hidden'] else
                non_hidden_model_predict
            )(
                beatmap,
                double_time=mods['double_time'],
                half_time=mods['half_time'],
                hard_rock=mods['hard_rock'],
            )
            for beatmap, mods in beatmaps_and_mods
        ]

    def predict_beatmap_differences(self, beatmap, *mods, **mods_scalar):
        """Predict the user's accuracy for the given beatmap.

        Parameters
        ----------
        beatmap : Beatmap
            The map to predict per hit object differences of.
        *mods
            A sequence of mod dictionaries to predict per hit object
            differences for.
        **mods_dict
            Mods to predict per hit object differences for.

        Returns
        -------
        differences :list[np.ndarray[float]]
            For each mod combination, an array of shape
            ``(len(hit_objects), 2)`` where the first column is the user's
            expected absolute difference for the click position and the second
            column is the expected absolute difference click time.

        Notes
        -----
        This method is specific to the :class:`~slider.model.lstm.LSTM` model,
        it may not be available on other model types.
        """
        for mod_name in 'hidden', 'hard_rock', 'half_time', 'double_time':
            mods_scalar.setdefault(mod_name, False)

        return self.predict_differences([
            (beatmap, ms) for ms in chain(mods, [mods_scalar])
        ])
