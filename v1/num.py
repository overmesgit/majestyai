import numpy as np

A = np.array([
    [0, 1, 0.8],
    [1, 1, 0.5],
    [1, 2, 1],
    [2, 1, 0.3],
    [2, 2, 0],
    [3, 1, 1],
    [3, 2, 1]]
)

Q = np.array([
    [1, 0, 0, 0, 0, 0, 0, 1, 0.8],
    [0, 1, 0, 0, 0, 0, 0, 1, 0.5],
    [0, 1, 0, 0, 0, 0, 1, 0, 0.5],
    [0, 0, 1, 0, 0, 0, 0, 1, 0.3],
    [0, 0, 1, 0, 0, 0, 1, 1, 0.6],
    [0, 0, 0, 1, 0, 0, 0, 1, 1],
])


def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract the maximum value to prevent overflow
    return e_x / e_x.sum(axis=0)  # Divide each element by the sum of exponential


# q1 * k1-n
result = Q[0] @ Q.transpose()
print(result)
# weights
weights = softmax(result)
print(weights)

# weights * values
print(weights @ Q)
