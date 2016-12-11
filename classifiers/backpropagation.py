import tensorflow as tf
import neural_network as nn


class Backpropagation:
    """
    Back propagation neural network used for classification.
    Taken from: http://blog.aloni.org/posts/backprop-with-tensorflow/
    """

    def __init__(self, training_data, test_data, training_labels, test_labels=None):
        self.training_data = training_data
        self.test_data = test_data
        self.training_labels = nn._prepare_data_labels(training_labels)
        self.test_labels = nn._prepare_data_labels(test_labels)
        self.input_layer = tf.placeholder(tf.float32, [None, self.training_data.shape[1]])
        self.result_layer = tf.placeholder(tf.float32, [None, self.training_labels.shape[1]])
        self.network = self._get_network()
        self.session = None

    def train(self, iteration_count=1000):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.network, self.result_layer))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
        self.session = tf.InteractiveSession()
        tf.initialize_all_variables().run()
        for _ in range(iteration_count):
            self.session.run([optimizer, cost],
                             feed_dict={self.input_layer: self.training_data, self.result_layer: self.training_labels})

    def test(self):
        if self.session is None:
            raise ValueError("You must train network first!")

        correct_prediction = tf.equal(tf.argmax(self.network, 1), tf.argmax(self.result_layer, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        predictions = tf.argmax(self.network, 1)
        return (self.session.run([accuracy, predictions], feed_dict={self.input_layer: self.test_data,
                                                      self.result_layer: self.test_labels}))

    def _get_network(self):
        input_count = self.training_data.shape[1]
        classes_count = self.training_labels.shape[1]

        neuron_count = [input_count, input_count / 2, input_count / 4, classes_count]
        weights = Backpropagation._get_random_weights(neuron_count)
        biases = Backpropagation._get_random_biases(neuron_count[1:])

        layer = self.input_layer
        for i in range(0, len(weights) - 1):
            tmp = tf.add(tf.matmul(layer, weights[i]), biases[i])
            layer = tf.nn.sigmoid(tmp)
        out_layer = tf.matmul(layer, weights[len(weights) - 1]) + biases[len(biases) - 1]

        return out_layer

    @staticmethod
    def _get_random_weights(neuron_counts):
        weights = list()
        for i in range(0, len(neuron_counts) - 1):
            weights.append(tf.Variable(tf.random_normal([neuron_counts[i], neuron_counts[i + 1]])))
        return weights

    @staticmethod
    def _get_random_biases(neuron_counts):
        biases = list()
        for i in range(0, len(neuron_counts)):
            biases.append(tf.Variable(tf.random_normal([neuron_counts[i]])))
        return biases
