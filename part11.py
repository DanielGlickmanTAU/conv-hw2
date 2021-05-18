import torch
import unittest

test = unittest.TestCase()

import hw2.blocks as blocks

from hw2.grad_compare import compare_block_to_torch


def test_block_grad(block: blocks.Block, x, y=None, delta=1e-2):
    diffs = compare_block_to_torch(block, x, y)

    # Assert diff values
    for diff in diffs:
        test.assertLess(diff, delta)


N = 100
in_features = 200
num_classes = 10


def linear():
    fc = blocks.Linear(in_features, 1000)
    x_test = torch.randn(N, in_features)
    # Test forward pass
    z = fc(x_test)
    test.assertSequenceEqual(z.shape, [N, 1000])
    # Test backward pass
    test_block_grad(fc, x_test)


def test_relu():
    relu = blocks.ReLU()
    x_test = torch.randn(N, in_features)
    # Test forward pass
    z = relu(x_test)
    test.assertSequenceEqual(z.shape, x_test.shape)
    # Test backward pass
    test_block_grad(relu, x_test)


def test_sigmoid():
    # Test Sigmoid
    sigmoid = blocks.Sigmoid()
    x_test = torch.randn(N, in_features)
    # Test forward pass
    z = sigmoid(x_test)
    test.assertSequenceEqual(z.shape, x_test.shape)
    # Test backward pass
    test_block_grad(sigmoid, x_test)


def test_cross_entropy():
    global cross_entropy
    # Test CrossEntropy
    cross_entropy = blocks.CrossEntropyLoss()
    scores = torch.randn(N, num_classes)
    labels = torch.randint(low=0, high=num_classes, size=(N,), dtype=torch.long)
    # Test forward pass
    loss = cross_entropy(scores, labels)
    expected_loss = torch.nn.functional.cross_entropy(scores, labels)
    test.assertLess(torch.abs(expected_loss - loss).item(), 1e-5)
    print('loss=', loss.item())
    # Test backward pass
    test_block_grad(cross_entropy, scores, y=labels)

# linear()
# test_relu()
# test_sigmoid()
# test_cross_entropy()


# Test Sequential
# Let's create a long sequence of blocks and see
# whether we can compute end-to-end gradients of the whole thing.

seq = blocks.Sequential(
    blocks.Linear(in_features, 100),
    blocks.Linear(100, 200),
    blocks.Linear(200, 100),
    blocks.ReLU(),
    blocks.Linear(100, 500),
    blocks.Linear(500, 200),
    blocks.ReLU(),
    blocks.Linear(200, 500),
    blocks.ReLU(),
    blocks.Linear(500, 1),
    blocks.Sigmoid(),
)
x_test = torch.randn(N, in_features)

# Test forward pass
z = seq(x_test)
test.assertSequenceEqual(z.shape, [N, 1])

# Test backward pass
test_block_grad(seq, x_test)