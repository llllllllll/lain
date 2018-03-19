ErrorModel
----------

The goal of the :class:`~lain.ErrorModel` model is to learn the user's
tendencies by looking at maps on the detail individual hit objects. The error
model tries to predict the user's expected accuracy by learning what their
expected error will be on any given object.

the ErrorModel model defines two kinds of error:

Aim Error
~~~~~~~~~

The distance between the (x, y) coordinate of the user's click and the center of
the given hit object. For sliders, this is the slider head.

Accuracy Error
~~~~~~~~~~~~~~

The absolute value of the difference between the time the user clicked and the
time when a hit object appears in the map.

Events
``````

The ErrorModel model works by breaking down a beatmap into a sequence of events. An
event is defined as the (x, y, time) position of each circle, slider, or slider
tick. Spinners are not included in the events because, for the users and maps we
are interested in working with, spinners always result in a score of 300 and can
be omitted. Each event is also tagged with a boolean value indicating whether it
is a slider tick or hit object as well as the effective approach rate for the
beatmap in milliseconds.

.. note::

   Effective approach rate is the approach rate accounting for double time and
   half time which do not change the displayed AR value but change the distance
   between a circle appearing and being clicked.

Windows
```````

After extracting the sequence of events from a beatmap, the error model takes a
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
```````

The network consists of stacked long short-term memory layers feeding into one
densly connected layer for each error output.

To better learn what the user finds difficult, we weigh the samples based on
their zscore. Concretely we clip the zscore from 1 to 4 and weigh the samples
based on that value.

Reducing Error to Accuracy
``````````````````````````

The overall goal is to predict a user's accuracy. The error model reduces the
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

Pessimism Factors
`````````````````

Replays are only saved when a user completes a song. This heavily biases the
input samples to only be the user's best replays and makes it hard if not
impossible to learn what the user finds hard. We need to magnify the projected
error to actually see some misses or 50s so we apply a scaling factor.

Currently this is quite small at 1.05 for both aim and accuracy error.
