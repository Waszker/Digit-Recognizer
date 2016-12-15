import tensorflow as tf
import numpy as np


class SoftmaxNetwork:
    """
    TensorFlow neural network for classification tasks.
    Taken from: https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
    """

    def __init__(self, training_data, test_data, training_labels, test_labels=None):
        self.training_data = training_data
        self.test_data = test_data
        self.training_labels = _prepare_data_labels(training_labels)
        self.test_labels = _prepare_data_labels(test_labels)

        # Softmax variables
        w = tf.Variable(tf.zeros([self.training_data.shape[1], self.training_labels.shape[1]]))
        b = tf.Variable(tf.zeros(self.training_labels.shape[1]))

        self.session = None
        self.x = tf.placeholder(tf.float32, [None, self.training_data.shape[1]])
        self.y = tf.matmul(self.x, w) + b
        self.y_ = tf.placeholder(tf.float32, [None, self.training_labels.shape[1]])

    def train(self, iteration_count=1000):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        self.session = tf.InteractiveSession()
        tf.initialize_all_variables().run()
        for _ in range(iteration_count):
            self.session.run(train_step, feed_dict={self.x: self.training_data, self.y_: self.training_labels})

    def test(self):
        if self.session is None:
            raise ValueError("You must train network first!")

        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        predictions = tf.argmax(self.y, 1)
        return (self.session.run([accuracy, predictions], feed_dict={self.x: self.test_data,
                                                      self.y_: self.test_labels}))


def _prepare_data_labels(training_labels):
    max_class = int(np.amax(training_labels, axis=0))
    final_labels = np.zeros((len(training_labels), max_class + 1))

    for i in range(0, len(training_labels)):
        final_labels[i][int(training_labels[i])] = 1
    return final_labels
