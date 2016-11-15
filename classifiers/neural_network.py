from pybrain.datasets import ClassificationDataSet
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer


class ClassificationNetwork:
    """
    Feed-forward neural network for classification tasks.
    """

    def __init__(self, training_data, test_data, training_labels, test_labels=None):
        self.training_data = training_data
        self.test_data = test_data
        self.training_labels = training_labels
        self.test_labels = test_labels
        self.dataset, self.network = self._prepare_data()
        self.trainer = BackpropTrainer(self.network, dataset=self.dataset, momentum=0.1, verbose=True,
                                       weightdecay=0.01)

    def train(self, iteration_count=10000, target_error=0.01):
        self.trainer.trainEpochs(iteration_count)
        return percentError(self.trainer.testOnClassData(), self.training_labels)

    def _prepare_data(self):
        dimensionality = self.training_data.shape[1]
        samples_count = self.training_data.shape[0]
        data = ClassificationDataSet(dimensionality, samples_count)
        index = 0
        for row in self.training_data:
            data.addSample(row, self.training_labels[index])
            index += 1
        data._convertToOneOfMany()
        network = buildNetwork(data.indim, 5, data.outdim, outclass=SoftmaxLayer)
        return data, network
