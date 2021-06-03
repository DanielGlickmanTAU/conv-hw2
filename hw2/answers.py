r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0.1, 1.5e-2, 0.05
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp(opt_name):
    wstd, lr, reg = 0.1, 1e-2, 0.01

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different hyperparameters for each optimizer.
    if opt_name == 'vanilla':
        wstd, lr, reg = 0.1, 1e-2, 0.001
    if opt_name == 'momentum':
        wstd, lr, reg = 0.1, 5e-3, 0.001
    if opt_name == 'rmsprop':
        wstd, lr, reg = 0.1, 1e-4, 0.001
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    # raise NotImplementedError()
    wstd=0.01
    lr = 0.004

    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

1)The graphs matches what we expected. The purpose of dropout is to prevent overfitting. 
We can see that with no drop out, accuracy on the training set is higher, but drops significantly on the validation set,
while with dropout, the accuracy is much closer between validation and train set.

2)
 The accuracy of high dropout is lower on the train set, but the difference between train and validation accuracy is very small.
 While lower dropout achives higher accuracy on the train set, but suffers a drop in accuracy on the validation set.
 This is due to less overfitting when using a higher dropout.
 Finally, we can see both dropouts achieve about the same accuracy on the validation set, and outperform no dropout.    

"""

part2_q2 = r"""
**Your answer:**

Yes, it is possible. It could be for example, that the classification of only one example would change from false to true,
to the accuracy will increase. But the predictions of many, already missclassified examples, would get worse, which will also increase the loss.
 

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
