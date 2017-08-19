from collections import deque
from textwrap import dedent

import numpy as np
from toolz import complement, concatv, first


def _predicate_iter(it, predicate, our_queue, other_queue):
    for element in our_queue:
        yield element
    our_queue.clear()

    for element in it:
        if predicate(element):
            yield element
        else:
            other_queue.append(element)

        for element in our_queue:
            yield element
        our_queue.clear()


def dichotomize(predicate, iterable):
    """Take a predicate and an iterable and return the pair of iterables of
    elements which do and do not satisfy the predicate.

    Parameters
    ----------
    predicate : callable[any, bool]
        The predicate function to partition with.
    iterable : iterable[any]
        The elements to partition.

    Returns
    -------
    trues : iterable[any]
        The sequence of values where the predicate evaluated to True.
    falses : iterable[any]
        The sequence of values where the predicate evaluated to False.

    Notes
    -----
    This is a lazy version of:

    .. code-block:: Python

       def partition(predicate, sequence):
           sequence = list(sequence)
           return (
               filter(predicate, sequence),
               filter(complement(predicate), sequence),
           )
    """
    true_queue = deque()
    false_queue = deque()
    it = iter(iterable)

    return (
        _predicate_iter(it, predicate, true_queue, false_queue),
        _predicate_iter(it, complement(predicate), false_queue, true_queue),
    )


def summary(feature_names, features, **labels):
    """Summarize the data we are about to train with.

    Parameters
    ----------
    feature_names : iterable[str]
        The names of the features in the ``features`` array.
    features : np.ndarray
        The 3d feature array.
    **labels
        The named label arrays.

    Returns
    -------
    summary : str
        A summary of the features and labels.
    """
    single_attribute_template = dedent(
        """\
        {name}:
          mean: {mean}
          std:  {std}
          min:  {min}
          max:  {max}""",
    )

    def format_attribute(name, values):
        return '    ' + '\n    '.join(
            single_attribute_template.format(
                name=name,
                mean=values.mean(),
                std=values.std(),
                min=values.min(),
                max=values.max(),
            ).splitlines(),
        )

    return '\n'.join(concatv(
        (
            'summary:',
            '  labels:',
        ),
        (
            format_attribute(name, value)
            for name, value in sorted(labels.items(), key=first)
        ),
        (
            'features:',
        ),
        (
            format_attribute(name, features[..., ix])
            for ix, name in enumerate(feature_names)
        )
    ))


def rolling_window(array, length):
    """Restride an array of shape (X_0, ... X_N) into an array of shape
    (length, X_0 - length + 1, ... X_N) where each slice at index i along the
    first axis is equivalent to result[i] = array[length * i:length * (i + 1)]

    Parameters
    ----------
    array : np.ndarray
        The base array.
    length : int
        Length of the synthetic first axis to generate.

    Returns
    -------
    out : np.ndarray

    Example
    -------
    >>> from numpy import arange
    >>> a = arange(25).reshape(5, 5)
    >>> a
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])

    >>> rolling_window(a, 2)
    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9]],
    <BLANKLINE>
           [[ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14]],
    <BLANKLINE>
           [[10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19]],
    <BLANKLINE>
           [[15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24]]])
    """
    orig_shape = array.shape
    if not orig_shape:
        raise IndexError("Can't restride a scalar.")
    elif orig_shape[0] <= length:
        raise IndexError(
            "Can't restride array of shape {shape} with"
            " a window length of {len}".format(
                shape=orig_shape,
                len=length,
            )
        )

    num_windows = (orig_shape[0] - length + 1)
    new_shape = (num_windows, length) + orig_shape[1:]

    new_strides = (array.strides[0],) + array.strides

    return np.lib.stride_tricks.as_strided(array, new_shape, new_strides)
