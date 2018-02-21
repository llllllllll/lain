from collections import namedtuple
from enum import unique, IntEnum

import numpy as np
from toolz import first

from slider.beatmap import Circle, Slider


# the manually selected features in the order they appear in the extracted
# arrays
features = tuple(sorted((
    # basic attributes
    'OD',
    'CS',
    'AR',

    # mods
    'easy',
    'hidden',
    'hard_rock',
    'double_time',
    'half_time',
    'flashlight',

    # bpm
    'bpm-min',
    'bpm-max',

    # hit objects
    'circle-count',
    'slider-count',
    'spinner-count',

    # hit object angles
    'mean-pitch',
    'mean-roll',
    'mean-yaw',
    'median-pitch',
    'median-roll',
    'median-yaw',
    'max-pitch',
    'max-roll',
    'max-yaw',

    # stars
    'speed-stars',
    'aim-stars',
    'rhythm-awkwardness',

    # pp
    'PP-95%',
    'PP-96%',
    'PP-97%',
    'PP-98%',
    'PP-99%',
    'PP-100%',
)))


@unique
class Axis(IntEnum):
    """Axis indices.
    """
    x = 0
    y = 1
    z = 2


def hit_object_coordinates(hit_objects, *, double_time=False, half_time=False):
    """Return the coordinates of the hit objects as a (3, len(hit_objects))
    array.

    Parameters
    ----------
    hit_objects : iterable[HitObject]
        The hit objects to take the coordinates of.
    double_time : bool, optional
        Apply double time compression to the Z axis.
    half_time : bool, optional
        Apply half time expansion to the Z axis.

    Returns
    -------
    coordinates : np.ndarray[float64]
        A shape (3, len(hit_objects)) array where the rows are the x, y, z
        coordinates of the nth hit object.

    Notes
    -----
    The z coordinate is reported in microseconds to make the angles more
    reasonable.
    """
    xs = []
    ys = []
    zs = []

    x = xs.append
    y = ys.append
    z = zs.append

    for hit_object in hit_objects:
        position = hit_object.position

        x(position.x)
        y(position.y)
        z(hit_object.time.total_seconds() * 100)

    coords = np.array([xs, ys, zs], dtype=np.float64)

    if double_time:
        coords[Axis.z] *= 4 / 3
    elif half_time:
        coords[Axis.z] *= 2 / 3

    return coords


class Angle(IntEnum):
    """Angle indices.
    """
    pitch = 0
    roll = 1
    yaw = 2


def hit_object_angles(hit_objects, *, double_time=False, half_time=False):
    """Compute the angle from one hit object to the next in 3d space with time
    along the Z axis.

    Parameters
    ----------
    hit_objects : iterable[HitObject]
        The hit objects to compute the angles about.
    double_time : bool, optional
        Apply double time compression to the Z axis.
    half_time : bool, optional
        Apply half time expansion to the Z axis.

    Returns
    -------
    angles : ndarray[float]
        An array shape (3, len(hit_objects) - 1) of pitch, roll, and yaw
        between each hit object. All angles are measured in radians.
    """
    coords = hit_object_coordinates(
        hit_objects,
        double_time=double_time,
        half_time=half_time,
    )
    diff = np.diff(coords, axis=1)

    # (pitch, roll, yaw) x transitions
    out = np.empty((3, len(hit_objects) - 1), dtype=np.float64)
    np.arctan2(diff[Axis.y], diff[Axis.z], out=out[Angle.pitch])
    np.arctan2(diff[Axis.y], diff[Axis.x], out=out[Angle.roll])
    np.arctan2(diff[Axis.z], diff[Axis.x], out=out[Angle.yaw])

    return out


hit_object_count = namedtuple('hit_object_count', 'circles sliders spinners')


def count_hit_objects(hit_objects):
    """Count the different hit element types.

    Parameters
    ----------
    hit_objects : hit_objects
        The hit objects to count the types of.

    Returns
    -------
    circles : int
        The count of circles.
    sliders : int
        The count of sliders.
    spinners : int
        The count of spinners.
    """
    circles = 0
    sliders = 0
    spinners = 0

    for hit_object in hit_objects:
        if isinstance(hit_object, Circle):
            circles += 1
        elif isinstance(hit_object, Slider):
            sliders += 1
        else:
            spinners += 1

    return hit_object_count(circles, sliders, spinners)


def extract_features(beatmap,
                     *,
                     easy=False,
                     hidden=False,
                     hard_rock=False,
                     double_time=False,
                     relax=False,
                     half_time=False,
                     flashlight=False,
                     _cache=None):
    """Extract all features from a beatmap.

    Parameters
    ----------
    beatmap : Beatmap
        The beatmap to extract features from.
    easy : bool, optional
        Was the easy mod used?
    hidden : bool, optional
        Was the hidden mod used?
    hard_rock : bool, optional
        Was the hard rock mod used?
    double_time : bool, optional
        Was the double time mod used?
    hard : bool, optional
        Was the half time mod used?
    flashlight : bool, optional
        Was the flashlight mod used?

    Returns
    -------
    features : dict[str, np.float64]
        The features by name.
    """
    if _cache is not None:
        # we often see many of the same replay when training, this allows
        # us to locally cache the features of a beatmap
        cache_key = (
            beatmap,
            easy,
            hidden,
            hard_rock,
            double_time,
            relax,
            half_time,
            flashlight,
        )

        try:
            return _cache[cache_key]
        except KeyError:
            pass

    # ignore the direction of the angle, just take the magnitude
    angles = np.abs(hit_object_angles(
        beatmap.hit_objects_no_spinners,
        half_time=half_time,
        double_time=double_time,
    ))
    mean_angles = np.mean(angles, axis=1)
    median_angles = np.median(angles, axis=1)
    max_angles = np.max(angles, axis=1)

    circles, sliders, spinners = count_hit_objects(beatmap.hit_objects)

    pp_95, pp_96, pp_97, pp_98, pp_99, pp_100 = beatmap.performance_points(
        accuracy=[0.95, 0.96, 0.97, 0.98, 0.99, 1.00],
        easy=easy,
        hard_rock=hard_rock,
        half_time=half_time,
        double_time=double_time,
        hidden=hidden,
        flashlight=flashlight,
    )

    features = {
        # basic stats
        'OD': beatmap.od(easy=easy, hard_rock=hard_rock),
        'CS': beatmap.cs(easy=easy, hard_rock=hard_rock),
        'AR': beatmap.ar(
            easy=easy,
            hard_rock=hard_rock,
            half_time=half_time,
            double_time=double_time,
        ),

        # mods
        'easy': float(easy),
        'hidden': float(hidden),
        'hard_rock': float(hard_rock),
        'double_time': float(double_time),
        'half_time': float(half_time),
        'flashlight': float(flashlight),

        # bpm
        'bpm-min': beatmap.bpm_min(
            half_time=half_time,
            double_time=double_time,
        ),
        'bpm-max': beatmap.bpm_max(
            half_time=half_time,
            double_time=double_time,
        ),

        # hit objects
        'circle-count': circles,
        'slider-count': sliders,
        'spinner-count': spinners,

        # hit object angles
        'mean-pitch': mean_angles[Angle.pitch],
        'mean-roll': mean_angles[Angle.roll],
        'mean-yaw': mean_angles[Angle.yaw],
        'median-pitch': median_angles[Angle.pitch],
        'median-roll': median_angles[Angle.roll],
        'median-yaw': median_angles[Angle.yaw],
        'max-pitch': max_angles[Angle.pitch],
        'max-roll': max_angles[Angle.roll],
        'max-yaw': max_angles[Angle.yaw],

        # stars
        'speed-stars': beatmap.speed_stars(
            easy=easy,
            hard_rock=hard_rock,
            half_time=half_time,
            double_time=double_time,
        ),
        'aim-stars': beatmap.aim_stars(
            easy=easy,
            hard_rock=hard_rock,
            half_time=half_time,
            double_time=double_time,
        ),
        'rhythm-awkwardness': beatmap.rhythm_awkwardness(
            easy=easy,
            hard_rock=hard_rock,
            half_time=half_time,
            double_time=double_time,
        ),

        # pp
        'PP-95%': pp_95,
        'PP-96%': pp_96,
        'PP-97%': pp_97,
        'PP-98%': pp_98,
        'PP-99%': pp_99,
        'PP-100%': pp_100,

    }

    if _cache is not None:
        _cache[cache_key] = features

    return features


def extract_feature_array(beatmaps_and_mods):
    """Extract all features from a beatmap.

    Parameters
    ----------
    beatmaps_and_mods : list[Beatmap, dict[str, bool]]
        The beatmaps and mod information to extract features from.

    Returns
    -------
    features : np.ndarray[float64]
        The features as an array.
    """
    cache = {}
    return np.array(
        [
            [
                snd for
                fst, snd in sorted(
                    extract_features(
                        beatmap,
                        **mods,
                        _cache=cache,
                    ).items(),
                    key=first,
                )
            ]
            for beatmap, mods in beatmaps_and_mods
        ]
    )


_max_int64 = np.iinfo(np.int64).max


def extract_features_and_labels(replays, *, verbose=False):
    """Extract all features from the beatmap of a replay as well as the
    accuracies.

    Parameters
    ----------
    replays : iterable[Replay]
        The replays to extract training information from.
    verbose : bool, optional
        Print verbose output to stdout?

    Returns
    -------
    features : np.ndarray[float64]
        The features as an array.
    accuracies : np.ndarray[float64]
        The accuracies as an array.
    """
    beatmaps_and_mods = []
    append_beatmap_and_mods = beatmaps_and_mods.append

    accuracy = []
    append_accuracy = accuracy.append

    for n, replay in enumerate(replays):
        beatmap = replay.beatmap
        bpm_max = beatmap.bpm_max(
            half_time=replay.half_time,
            double_time=replay.double_time,
        )
        if bpm_max > _max_int64:
            # a replay has a messed up value here; skip it
            continue

        if verbose:
            print(f'{n:4}: {replay!r}')

        append_beatmap_and_mods((
            beatmap,
            {
                'hidden': replay.hidden,
                'flashlight': replay.flashlight,
                'double_time': replay.double_time,
                'half_time': replay.half_time,
                'hard_rock': replay.hard_rock,
                'easy': replay.easy,
                'relax': replay.relax,
            },
        ))
        append_accuracy(replay.accuracy)

    return (
        extract_feature_array(beatmaps_and_mods),
        np.array(accuracy),
    )
