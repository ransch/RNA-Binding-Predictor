import tensorflow as tf
from keras import layers


class AverageLast(layers.Layer):
    """
    A layer that averages the last indices of the previous one-dimensional layer.
    """

    def __init__(self, count):
        """
        Args:
            count: The number of indices to average.
        """
        super(AverageLast, self).__init__()
        self._count = count

    def call(self, input_layer):
        if self._count <= 1:
            return input_layer

        # Compute the average of the last self._count cells.
        average = tf.reduce_mean(input_layer[:, -self._count:], axis=1, keepdims=True)

        output = tf.concat(
            [input_layer[:, :-self._count],
             tf.repeat(average, repeats=self._count, axis=1)],
            axis=1)

        return output
