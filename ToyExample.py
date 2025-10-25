import numpy as np
import pandas as pd

X = np.array([
    [1, 0, 1, 0],
    [1, 1, 1, 0],
    [1, 0, 1, 1],
    [0, 1, 0, 1],
    [0, 1, 0, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [0, 0, 1, 1]
], dtype=float)

K = 2
N, D = X.shape

pi = np.array([0.5, 0.5])
theta = np.array([
    [0.6, 0.4, 0.6, 0.3],
    [0.3, 0.7, 0.2, 0.6]
])

def bernoulli_likelihood(x, theta_k):
    return np.prod(theta_k ** x * (1 - theta_k) ** (1 - x))

for step in range(2):
    
    # E-step: compute responsibilities
    likelihood = np.zeros((N, K))
    for i in range(N):
        for k in range(K):
            likelihood[i, k] = pi[k] * bernoulli_likelihood(X[i], theta[k])
    responsibilities = likelihood / likelihood.sum(axis=1, keepdims=True)
    
    # M-step: update parameters
    Nk = responsibilities.sum(axis=0)
    pi = Nk / N
    for k in range(K):
        theta[k] = (responsibilities[:, k].reshape(-1, 1) * X).sum(axis=0) / Nk[k]
    
    print(f"\nIteration {step + 1}")
    print("Responsibilities:\n", np.round(responsibilities, 4))
    print("π:", np.round(pi, 4))
    print("θ:\n", np.round(theta, 4))

# Final output as DataFrame
df = pd.DataFrame(theta, columns=[f"Feature_{i+1}" for i in range(D)])
df["pi"] = pi
print("\nFinal parameters:")
print(df)
