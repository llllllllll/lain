from datetime import datetime
import os
import sys
import traceback

from slider.game_mode import GameMode
from slider.replay import Replay


def load_replay_directory(path,
                          *,
                          library=None,
                          client=None,
                          age=None,
                          save=False,
                          verbose=False):
    """Load all eligible replays from a directory.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to the directory of ``.osr`` files.
    library : Library, optional
        The beatmap library to use when parsing the replays.
    client : Client, optional
        The client to use when parsing the replays.
    age : datetime.timedelta, optional
        Only count replays less than this age old.
    save: bool, optional
        If the beatmap does not exist, and a client is used to fetch it, should
        the beatmap be saved to disk?
    verbose : bool, optional
        Print error information to stderr?

    Yields
    ------
    replay : Replay
        The eligible replays in the directory.

    Notes
    -----
    The same beatmap may appear more than once if there are multiple replays
    for this beatmap.
    """
    def replays():
        for entry in os.scandir(path):
            if not entry.name.endswith('.osr'):
                continue

            try:
                yield Replay.from_path(
                    entry,
                    client=client,
                    library=library,
                    save=save,
                )
            except Exception:
                if verbose:
                    print(f'failed to read replay {entry}', file=sys.stderr)
                    traceback.print_exc(file=sys.stderr)

                # throw out any maps that fail to parse or download
                continue

    return eligible_replays(replays(), age=age)


def eligible_replays(replays, *, age=None):
    """Filter replays down to just the replays we want to train with.

    Parameters
    ----------
    replays : iterable[Replay]
        The replays to filter.
    age : datetime.timedelta, optional
        Only count replays less than this age old.

    Yields
    ------
    replay : Replay
        The eligible replays in the directory.

    Notes
    -----
    The same beatmap may appear more than once if there are multiple replays
    for this beatmap.
    """
    for replay in replays:
        if age is not None and datetime.utcnow() - replay.timestamp > age:
            continue

        if not (replay.mode != GameMode.standard or
                replay.failed or
                replay.autoplay or
                replay.auto_pilot or
                replay.cinema or
                replay.relax or
                len(replay.beatmap.hit_objects) < 2):
            # ignore plays with mods that are not representative of user skill
            yield replay
