import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, Linear, ReLU, Sigmoid, Dropout, Sequential


class MLP(Block):
    """
    A simple multilayer perceptron model based on our custom Blocks.
    Architecture is (with ReLU activation):

        FC(in, h1) -> ReLU -> FC(h1,h2) -> ReLU -> ... -> FC(hn, num_classes)

    Where FC is a fully-connected layer and h1,...,hn are the hidden layer
    dimensions.
    If dropout is used, a dropout layer is added after every activation
    function.
    """

    def __init__(self, in_features, num_classes, hidden_features=(),
                 activation='relu', dropout=0, **kw):
        super().__init__()
        """
        Create an MLP model Block.
        :param in_features: Number of features of the input of the first layer.
        :param num_classes: Number of features of the output of the last layer.
        :param hidden_features: A sequence of hidden layer dimensions.
        :param activation: Either 'relu' or 'sigmoid', specifying which 
        activation function to use between linear layers.
        :param: Dropout probability. Zero means no dropout.
        """
        blocks = []

        # TODO: Build the MLP architecture as described.
        # ====== YOUR CODE: ======
        features = [in_features] + hidden_features + [num_classes]
        for f_in, f_out in zip(features[:-1], features[1:]):
            blocks.append(Linear(in_features=f_in, out_features=f_out))
            blocks.append(ReLU() if activation == 'relu' else Sigmoid())

        # remove last activation
        blocks = blocks[:-1]
        # ========================

        self.sequence = Sequential(*blocks)

    def forward(self, x, **kw):
        return self.sequence(x, **kw)

    def backward(self, dout):
        return self.sequence.backward(dout)

    def params(self):
        return self.sequence.params()

    def train(self, training_mode=True):
        self.sequence.train(training_mode)

    def __repr__(self):
        return f'MLP, {self.sequence}'


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(Conv -> ReLU)*P -> MaxPool]*(N/P) -> (Linear -> ReLU)*M -> Linear
    """

    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param filters: A list of of length N containing the number of
            filters in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        """
        super().__init__()
        self.in_size = in_size
        self.out_classes = out_classes
        self.filters = filters
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======
        N = len(self.filters)
        P = self.pool_every
        for pool in range(int(N / P)):
            offset = pool * P
            for conv_layer in range(P):
                if pool == 0 and conv_layer == 0:
                    layers.append(nn.Conv2d(len(self.in_size), self.filters[0], kernel_size=(3, 3), padding=(1, 1)))
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.Conv2d(self.filters[offset + conv_layer - 1], self.filters[offset + conv_layer],
                                            kernel_size=(3, 3), padding=(1, 1)))
                    layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=(2, 2)))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        N = len(self.filters)
        P = self.pool_every
        M = len(self.hidden_dims)
        curr_w = in_w
        curr_h = in_h
        for conv_num in range(1, N + 1):
            curr_w = curr_w
            curr_h = curr_h
            if conv_num % P == 0:
                curr_w /= 2
                curr_h /= 2
        num_of_features = int(curr_w * curr_h * self.filters[-1])
        layers.append(nn.Linear(num_of_features, self.hidden_dims[0]))
        layers.append(nn.ReLU())
        for linear_layer in range(1, M):
            layers.append(nn.Linear(self.hidden_dims[linear_layer - 1], self.hidden_dims[linear_layer]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        raise NotImplementedError()
        out = self.feature_extractor(x)
        out = self.classifier(out.flatten(1))
        # ========================
        return out


class YourCodeNet(ConvClassifier):
    def __init__(self, in_size, out_classes, filters, pool_every, hidden_dims):
        super().__init__(in_size, out_classes, filters, pool_every, hidden_dims)

    # TODO: Change whatever you want about the ConvClassifier to try to
    # improve it's results on CIFAR-10.
    # For example, add batchnorm, dropout, skip connections, change conv
    # filter sizes etc.
    # ====== YOUR CODE: ======
    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)
        layers = []

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        # [(Conv -> ReLU)*P -> MaxPool]*(N/P)
        # Use only dimension-preserving 3x3 convolutions. Apply 2x2 Max
        # Pooling to reduce dimensions.
        # ====== YOUR CODE: ======
        N = len(self.filters)
        P = self.pool_every
        for pool in range(int(N / P)):
            offset = pool * P
            for conv_layer in range(P):
                if pool == 0 and conv_layer == 0:
                    layers.append(nn.Conv2d(len(self.in_size), self.filters[0], kernel_size=(3, 3), padding=(1, 1)))
                    layers.append(nn.BatchNorm2d(self.filters[0]))
                    layers.append(nn.ReLU())
                else:
                    layers.append(nn.Conv2d(self.filters[offset + conv_layer - 1], self.filters[offset + conv_layer],
                                            kernel_size=(3, 3), padding=(1, 1)))
                    layers.append(nn.BatchNorm2d(self.filters[offset + conv_layer]))
                    layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=(2, 2)))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the classifier part of the model:
        # (Linear -> ReLU)*M -> Linear
        # You'll need to calculate the number of features first.
        # The last Linear layer should have an output dimension of out_classes.
        # ====== YOUR CODE: ======
        N = len(self.filters)
        P = self.pool_every
        M = len(self.hidden_dims)
        curr_w = in_w
        curr_h = in_h
        for conv_num in range(1, N + 1):
            curr_w = curr_w
            curr_h = curr_h
            if conv_num % P == 0:
                curr_w /= 2
                curr_h /= 2
        num_of_features = int(curr_w * curr_h * self.filters[-1])
        layers.append(nn.Linear(num_of_features, self.hidden_dims[0]))
        layers.append(nn.BatchNorm1d(self.hidden_dims[0]))
        layers.append(nn.ReLU())
        for linear_layer in range(1, M):
            layers.append(nn.Linear(self.hidden_dims[linear_layer - 1], self.hidden_dims[linear_layer]))
            layers.append(nn.BatchNorm1d(self.hidden_dims[linear_layer]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.hidden_dims[-1], self.out_classes))
        layers.append(nn.Softmax(-1))
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        # Extract features from the input, run the classifier on them and
        # return class scores.
        # ====== YOUR CODE: ======
        out = self.feature_extractor(x)
        out = self.classifier(out.flatten(1))
        # ========================
        return out
    # ========================
