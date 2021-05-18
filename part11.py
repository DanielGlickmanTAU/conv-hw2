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

def test_sequential():
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

# linear()
# test_relu()
# test_sigmoid()
# test_cross_entropy()
# test_sequential()


import hw2.models as models

# Create MLP model
mlp = models.MLP(in_features, num_classes, hidden_features=[100, 50, 100])
print(mlp)

# Test MLP architecture
N = 100
in_features = 10
num_classes = 10
for activation in ('relu', 'sigmoid'):
    mlp = models.MLP(in_features, num_classes, hidden_features=[100, 50, 100], activation=activation)
    test.assertEqual(len(mlp.sequence), 7)

    num_linear = 0
    for b1, b2 in zip(mlp.sequence, mlp.sequence[1:]):
        if (str(b2).lower() == activation):
            test.assertTrue(str(b1).startswith('Linear'))
            num_linear += 1

    test.assertTrue(str(mlp.sequence[-1]).startswith('Linear'))
    test.assertEqual(num_linear, 3)

    # Test MLP gradients
    # Test forward pass
    x_test = torch.randn(N, in_features)
    labels = torch.randint(low=0, high=num_classes, size=(N,), dtype=torch.long)
    z = mlp(x_test)
    test.assertSequenceEqual(z.shape, [N, num_classes])

    # Create a sequence of MLPs and CE loss
    # Note: deliberately using the same MLP instance multiple times to create a recurrence.
    seq_mlp = blocks.Sequential(mlp, mlp, mlp, blocks.CrossEntropyLoss())
    loss = seq_mlp(x_test, y=labels)
    test.assertEqual(loss.dim(), 0)
    print(f'MLP loss={loss}, activation={activation}')

    # Test backward pass
    test_block_grad(seq_mlp, x_test, y=labels)
