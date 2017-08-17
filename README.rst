lain
====

Models for `osu! <https://osu.ppy.sh/>`_ that learn from user replays to predict
scores.

This project exists to build better recommendation systems for osu! as well as
explore how different users play osu!.

Lain is currently being used to power: `combine
<https://github.com/llllllllll/combine>`_

LSTM
----

The goal of the LSTM model is to learn the user's tendencies by looking at maps
on the detail individual hit objects. The LSTM model tries to predict the user's
expected accuracy by learning what their expected error will be on any given
object.

the LSTM model defines two kinds of error:

Aim Error
`````````

The distance between the (x, y) coordinate of the user's click and the center of
the given hit object. For sliders, this is the slider head.

Accuracy Error
``````````````

The absolute value of the difference between the time the user clicked and the
time when a hit object appears in the map.

Events
~~~~~~

The LSTM model works by breaking down a beatmap into a sequence of events. An
event is defined as the (x, y, time) position of each circle, slider, or slider
tick. Spinners are not included in the events because, for the users and maps we
are interested in working with, spinners always result in a score of 300 and can
be omitted. Each event is also tagged with a boolean value indicating whether it
is a slider tick or hit object as well as the effective approach rate for the
beatmap in milliseconds.

Note: effective approach rate is the approach rate accounting for double time
and half time which do not change the displayed AR value but change the distance
between a circle appearing and being clicked.

Windows
~~~~~~~

After extracting the sequence of events from a beatmap, the LSTM model takes a
sliding window over the events . Each window is designed to capture some leading
and trailing context for a given hit object. Windows are only ever "centered"
around hit objects, not slider ticks. By default, the leading context is much
longer than the trailing context because users cannot see more than a few events
in front of an object. More experimentation could be done to account for users
who remember the trailing context from past plays.

Each event in the window is augmented with the (x, y, time) relative to the
"center" of the window. For example, if my window x values were: [2, 4, 6], we
would also add a relative x feature with: [-4, -2, 0].

The window is labeled with the aim error and accuracy error for the "center" hit
object.

Network
~~~~~~~

The network consists of stacked long short-term memory layers feeding into one
densly connected layer for each error output.

Reducing Error to Accuracy
~~~~~~~~~~~~~~~~~~~~~~~~~~

The overall goal is to predict a user's accuracy. The LSTM model reduces the
predicted errors into a scalar accuracy by fitting one log-normal distribution
for each set of predicted errors.

To get the probability of missing the object from aim error we take the
cumulative distribution function of the aim error distribution at the radius of
the circles for the given map.

To find the expected accuracy we need to calculate the probability of clicking
within the 300 threshold, between the 300 and 100 thresholds, and between the
100 and 50 thresholds. To find the probability that a click falls between the
given thresholds we need to integrate the pdf from the starting threshold to the
ending threshold. We weigh the probabilities by the accuracy the givey, so we
multiply the probability of clicking within the 300 range by 1, we multiply the
probability of clicking within the 300-100 range by 1 / 3, we multiply the
probability of clicking within the 100-50 range by 1 / 6, and finally we
implicitly multiply the probability of missing entirely by 0. The expected
accuracy value is the sum of these weighted probabilities.

The final expected score is the expected aim multiplied by the expected
accuracy.

Multiayer Perceptron
--------------------

The goal of the multilayer perceptron model is to learn how a user performs by
looking at many weakly predictive features of beatmaps.

Below is an overview of the features of the model and the rationale for how they
are predictive of user performance.

Basic Attributes
~~~~~~~~~~~~~~~~

The first set of features are the basic attributes of the map's basic attributes
that you would see in the osu! client. These include:

- circle size (``CS``)
- overall difficulty (``OD``)
- approach rate (``AR``)

Rationale
`````````

These metrics affect how hard it is to make jumps, read the map, or accurately
hit elements. Health drain (``HP``) is not included because it does not affect
accuracy.

Mods
----

The model accounts for some mods that affect the difficulty of a song. The mods
included are:

- easy (``EZ``)
- hidden (``HD``)
- hard rock (``HR``)
- double time (``DT``) (or nightcore ``NC``)
- half time (``HT``)
- flashlight (``FL``)

.. note::

   If a mod is enabled that affects the basic attributes, those will be adjusted
   to account for this information. If a mod is enabled that affects the BPM,
   the ``bpm_min`` and ``bpm_max`` will be adjusted.

   The ``OD`` and ``AR`` are adjusted when using ``DT`` or ``HT`` to help the
   model make better predictions.

Rationale
`````````

These mods change the ability to read the map or play accurately.

BPM
---

The model accounts for the bpm with two values: ``bpm-min`` and
``bpm-max``. Songs with tempo changes will have different values here.

Rationale
`````````

The BPM affects how hard it is to accurately hit streams or single tap.

Hit Objects
-----------

The model has ``circle-count``, ``slider-count``, and ``spinner-count`` to count
the given element kinds.

Rationale
`````````

The number of each kind of hit element in combination with other metrics can
give a sense of the "kind" of map. For example, a high bpm song with many
circles and few sliders is probably very stream heavy.

Note Angles
~~~~~~~~~~~

Imagine an osu! standard beatmap in 3d space as ``(x, y, time)``, where the hit
elements form a path through this space. We can look at the angle from hit
object to hit object about each axis to get a sense of how "far" a jump is.

The model looks at the median, mean, and max angle magnitude about each axis.

Rationale
`````````

More extreme jumps are harder to hit and make maps harder to read. Maps with
small angles are likely stream maps where the element to element distance is
very small.

We look at median to get a sense of how much of a jump is any given note.

The max is to look at the most extreme jump in the map. Many maps have one or
two very hard jumps that cause misses. The hardest jump should account for that.

The mean shows how hard the jumps are across all of the notes. Comparing this to
median can give us a sense of how much more extreme the outliers are.

Osu! Difficulty Metrics
~~~~~~~~~~~~~~~~~~~~~~~

Osu! itself has a couple of metrics for measuring difficulty, these include:

- speed stars: a measure of how hard a song is from speed
- aim stars: a measure of how hard a song is to aim and accurately hit each
  object.
- rhythm awkwardness: how difficult is the rhythm of the beatmap

.. note::

   The speed and aim stars add up to the final value shown in the osu! client.

Rationale
`````````

The osu! team put a lot of work into these criteria, they are what I as a player
mainly use to know how hard a song is.

Performance Points Curve
~~~~~~~~~~~~~~~~~~~~~~~~

The model takes into account the performance points awarded for 95%-100%
accuracies at 1% steps.

Rationale
`````````

Like the raw difficulty metrics, the osu! team put a lot of work into defining
the performance points algorithm and I believe there is predictive power in it.
