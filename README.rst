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

For more information see `the docs
<https://llllllllll.github.io/lain/lstm.html>`_.

Multiayer Perceptron
--------------------

The goal of the multilayer perceptron model is to learn how a user performs by
looking at many weakly predictive features of beatmaps.

For more information see `the docs
<https://llllllllll.github.io/lain/mlp.html>`_.
