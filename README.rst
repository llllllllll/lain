lain
====

Models for `osu! <https://osu.ppy.sh/>`_ that learn from user replays to predict
scores.

This project exists to build better recommendation systems for osu! as well as
explore how different users play osu!.

Lain is currently being used to power: `combine
<https://github.com/llllllllll/combine>`_

Error Model
-----------

The goal of the error model is to learn the user's tendencies by looking at maps
on the detail individual hit objects. The error model tries to predict the
user's expected accuracy by learning what their expected aim and accuracy error
will be on any given object.

For more information see `the docs
<https://llllllllll.github.io/lain/error-model.html>`_.
