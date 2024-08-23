import time

import keras


class TimeLimitCallback(keras.callbacks.Callback):
    """
    A callback that limits the training time of a model.
    """

    def __init__(self, max_minutes):
        """
        Args:
            max_minutes: The maximum training time in minutes.
        """
        super(TimeLimitCallback, self).__init__()
        self._max_minutes = max_minutes
        self._start_time = None

    def on_train_begin(self, logs=None):
        self._start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        elapsed_minutes = (time.time() - self._start_time) / 60
        if elapsed_minutes > self._max_minutes:
            self.model.stop_training = True
