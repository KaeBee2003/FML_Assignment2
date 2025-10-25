import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("A2Q1.csv", header=None)
df2_test = pd.read_csv("A2Q2test.csv", header=None)
df2_train = pd.read_csv("A2Q2train.csv", header=None)
df2 = df2_train

# Functions

def bernoulli_log_pdf(x, mu):
    eps = 1e-10 
    log_prob = x[:, None, :] * np.log(mu[None, :, :] + eps) + \
               (1 - x[:, None, :]) * np.log(1 - mu[None, :, :] + eps)
    return np.sum(log_prob, axis=2)

def em_bernoulli_mixture(X, K=4, max_iter=100):
    N, D = X.shape
    log_likelihoods = []

    np.random.seed()
    pi = np.full(K, 1/K)
    mu = np.random.rand(K, D)

    for iteration in range(max_iter):
        log_prob = bernoulli_log_pdf(X, mu)
        log_prob += np.log(pi + 1e-10)
        log_sum = np.logaddexp.reduce(log_prob, axis=1, keepdims=True)
        log_resp = log_prob - log_sum
        resp = np.exp(log_resp)

        Nk = resp.sum(axis=0)
        pi = Nk / N
        mu = (resp.T @ X) / (Nk[:, None] + 1e-10)

        ll = np.sum(log_sum)
        log_likelihoods.append(ll)

    return pi, mu, log_likelihoods

def gaussian_pdf(x, mean, cov):
    D = len(x)
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    num = np.exp(-0.5 * (x - mean).T @ inv @ (x - mean))
    den = np.sqrt((2 * np.pi) ** D * det + 1e-10)
    return num / den

def run_em_gmm(X, K, max_iter):
    N, D = X.shape
    np.random.seed()
    means = X[np.random.choice(N, K, replace=False)]
    covs = np.array([np.eye(D) for _ in range(K)])
    pis = np.ones(K) / K

    log_likelihoods = []

    for it in range(max_iter):
        resp = np.zeros((N, K))
        for k in range(K):
            for i in range(N):
                resp[i, k] = pis[k] * gaussian_pdf(X[i], means[k], covs[k])
        resp /= resp.sum(axis=1, keepdims=True) + 1e-10

        Nk = resp.sum(axis=0)
        pis = Nk / N
        for k in range(K):
            means[k] = (resp[:, k].reshape(-1, 1) * X).sum(axis=0) / Nk[k]
            diff = X - means[k]
            covs[k] = (resp[:, k].reshape(-1, 1) * diff).T @ diff / Nk[k]
            covs[k] += np.eye(D) * 1e-6  # regularization

        ll = 0
        for i in range(N):
            tmp = 0
            for k in range(K):
                tmp += pis[k] * gaussian_pdf(X[i], means[k], covs[k])
            ll += np.log(tmp + 1e-10)
        log_likelihoods.append(ll / N)

    return np.array(log_likelihoods)

def euclidean_distance(row1, row2, feature_cols):
    return sum((row1[col] - row2[col]) ** 2 for col in feature_cols) ** 0.5

def K_means(df, K, plusplus=False):

    errors = []

    df_copy = df.copy()
    feature_cols = [col for col in df.columns]

    if not plusplus:
        centroids = df_copy.sample(n=K).copy()
        centroids['cluster'] = range(1, K + 1)
    else:
        pass

    while True:
        old_cluster_assignments = df_copy['cluster'].copy() if 'cluster' in df_copy else None

        # Assign points to nearest centroid
        clusters = []
        for _, row in df_copy.iterrows():
            min_dist = float('inf')
            assigned_cluster = None
            for _, centroid_row in centroids.iterrows():
                dist = euclidean_distance(row, centroid_row, feature_cols)
                if dist < min_dist:
                    min_dist = dist
                    assigned_cluster = centroid_row['cluster']
            clusters.append(assigned_cluster)

        df_copy['cluster'] = clusters

        # Compute new centroids
        new_centroids = df_copy.groupby('cluster')[feature_cols].mean().reset_index()

        # Compute current iteration error (objective)
        error = 0
        for _, row in df_copy.iterrows():
            centroid_row = new_centroids[new_centroids['cluster'] == row['cluster']].iloc[0]
            error += sum((row[col] - centroid_row[col]) ** 2 for col in feature_cols)
        errors.append(error)

        # Check for convergence
        if old_cluster_assignments is not None and df_copy['cluster'].equals(old_cluster_assignments):
            break

        centroids = new_centroids

    return df_copy, centroids, errors

class LinReg():
    def __init__(self):
        self.w = None
        self.w_history = []
        self.cost = None

    def Fit(self, Xo, Y, method=None, regularize=None, alpha=0.001, max_iter=10000, lambda_=1, capture_cost=False, tol=1e-6, batch_size=100, epoch=20):
        X = Xo.copy()
        X.insert(0, "bias", 1)
        X = X.to_numpy()
        Y = Y.to_numpy().reshape(-1, 1)
        m, n = X.shape

        self.w = np.zeros((n, 1))
        self.w_history = []

        rng = np.random.default_rng()

        def List_mean(a):
            return sum(a) / len(a)

        if method == 'Stoc_GD':
            for _ in range(max_iter):
                i = rng.integers(0, m)
                Xi = X[i:i+1]
                Yi = Y[i:i+1]

                grad = Xi.T @ (Xi @ self.w - Yi)

                if regularize == "Ridge":
                    reg_term = np.zeros_like(self.w)
                    reg_term[1:] = (lambda_ / m) * self.w[1:]
                    grad += reg_term

                self.w -= alpha * grad

                if capture_cost:
                    full_pred = X @ self.w
                    self.cost = np.sum((Y - full_pred) ** 2)
                    self.w_history.append(self.w.copy())

                if len(self.w_history) > 1 and np.linalg.norm(self.w_history[-1] - self.w_history[-2]) < tol:
                    break

            return (self.w, self.w_history) if capture_cost else self.w

        elif method == 'MiniBatch_GD':
            for epoch_idx in range(epoch):
                indices = rng.permutation(m)
                X_shuffled = X[indices]
                Y_shuffled = Y[indices]

                for start in range(0, m, batch_size):
                    end = start + batch_size
                    Xi = X_shuffled[start:end]
                    Yi = Y_shuffled[start:end]

                    grad = Xi.T @ (Xi @ self.w - Yi) / Xi.shape[0]

                    if regularize == "Ridge":
                        reg_term = np.zeros_like(self.w)
                        reg_term[1:] = (lambda_ / m) * self.w[1:]
                        grad += reg_term

                    self.w -= alpha * grad

                    if capture_cost:
                        full_pred = X @ self.w
                        self.cost = np.sum((Y - full_pred) ** 2)
                        self.w_history.append(self.w.copy())

                    if len(self.w_history) > 1 and np.linalg.norm(self.w_history[-1] - self.w_history[-2]) < tol:
                        break

            return (self.w, self.w_history) if capture_cost else self.w

        elif method == 'Batch_GD':
            for epoch_idx in range(epoch):
                grad = (X.T @ (X @ self.w - Y)) / m

                if regularize == "Ridge":
                    reg_term = np.zeros_like(self.w)
                    reg_term[1:] = (lambda_ / m) * self.w[1:]
                    grad += reg_term

                self.w -= alpha * grad

                if capture_cost:
                    full_pred = X @ self.w
                    self.cost = np.sum((Y - full_pred) ** 2)
                    self.w_history.append(self.w.copy())

                if len(self.w_history) > 1 and np.linalg.norm(self.w_history[-1] - self.w_history[-2]) < tol:
                    break

            return (self.w, self.w_history) if capture_cost else self.w

    
        else:
            tra = X.T
            try:
                self.w = np.linalg.inv(tra @ X) @ tra @ Y
            except np.linalg.LinAlgError:
                self.w = np.linalg.pinv(tra @ X) @ tra @ Y
            return self.w

    def Test(self, X_test):
        if self.w is None:
            raise Exception("Training Not yet Done")
        X_test = X_test.copy()
        X_test.insert(0, "bias", 1)
        X_test = X_test.to_numpy()
        y_pred = X_test @ self.w
        return y_pred

    def Eval(self, Y_Pred, Y_test):
        Y_Pred = np.array(Y_Pred).reshape(-1)
        Y_test = np.array(Y_test).reshape(-1)
        y_mean = np.mean(Y_test)
        ss_res = np.sum((Y_test - Y_Pred) ** 2)
        ss_tot = np.sum((Y_test - y_mean) ** 2)
        R = 1 - (ss_res / ss_tot)
        return R


def ridge_gd(X, y, alpha=0.001, lambda_=0.1, epochs=10000):
    N, d = X.shape
    w = np.zeros((d,1))
    for _ in range(epochs):
        y_pred = X @ w
        grad = (1/N) * (X.T @ (y_pred - y)) + lambda_ * w
        w -= alpha * grad
    return w

def cross_validate(X, y, lambdas, k=5, alpha=0.001, epochs=10000):
    N = X.shape[0]
    fold_size = N // k
    errors = []

    for lambda_ in lambdas:
        fold_errors = []
        for i in range(k):
            start, end = i*fold_size, (i+1)*fold_size
            X_val, y_val = X[start:end], y[start:end]
            X_train = np.vstack((X[:start], X[end:]))
            y_train = np.vstack((y[:start], y[end:]))

            w = ridge_gd(X_train, y_train, alpha=alpha, lambda_=lambda_, epochs=epochs)
            y_pred = X_val @ w
            fold_errors.append(mse(y_val, y_pred))
        errors.append(np.mean(fold_errors))
    return errors

def mse(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.mean((y_true - y_pred) ** 2)


def cross_validate_linreg(X_train_df, y_train_df, lambdas, k=5, alpha=0.001, max_iter=10000):
    N = len(X_train_df)
    fold_size = N // k
    errors = []

    for lambda_ in lambdas:
        fold_errors = []
        for i in range(k):
            start, end = i * fold_size, (i + 1) * fold_size

            X_val = X_train_df.iloc[start:end]
            y_val = y_train_df.iloc[start:end]
            X_tr = X_train_df.drop(X_train_df.index[start:end])
            y_tr = y_train_df.drop(y_train_df.index[start:end])

            model = LinReg()
            model.Fit(X_tr, y_tr, method='Batch_GD', regularize='Ridge', alpha=alpha,
                      max_iter=max_iter, lambda_=lambda_)
            y_pred = model.Test(X_val).reshape(-1)
            fold_errors.append(mse(y_val, y_pred))
        errors.append(np.mean(fold_errors))

    return errors


# Q1
# (i)

X = df1.values.astype(np.float64)

max_iter = 50
avg_ll = np.zeros(max_iter)

for run in range(100):
    _, _, ll = em_bernoulli_mixture(X, K=4, max_iter=50)
    avg_ll += np.array(ll)

avg_ll /= 100

plt.figure(figsize=(8, 5))
plt.plot(range(1, max_iter + 1), avg_ll, marker='o')
plt.title("Average Log-Likelihood over 100 EM Runs")
plt.xlabel("Iteration")
plt.ylabel("Average Log-Likelihood")
plt.grid(True)
plt.tight_layout()
plt.show()

# (ii)

data = df1.values
N, D = data.shape
K = 4
max_iter = 50
n_init = 100

all_logs = np.zeros((n_init, max_iter))
for r in range(n_init):
    all_logs[r] = run_em_gmm(data, K, max_iter)

avg_log = all_logs.mean(axis=0)

plt.figure(figsize=(7,5))
plt.plot(avg_log, marker='o')
plt.title("Gaussian Mixture Model (K=4): Average Log-Likelihood")
plt.xlabel("Iteration")
plt.ylabel("Average Log-Likelihood")
plt.grid(True)
plt.show()

# (iii)

_, _, errors = K_means(df1, K=4)

plt.plot(range(1, len(errors)+1), errors, marker='o')
plt.title('K-means Objective vs Iterations (K=4)')
plt.xlabel('Iteration')
plt.ylabel('Sum of Squared Errors (SSE)')
plt.grid(True)
plt.show()

# Q2
# (i)

X_train = df2.iloc[:, 0:100]
Y_train = df2.iloc[:, 100]
X_test = df2_test.iloc[:, 0:100]
Y_test = df2_test.iloc[:, 100]

ModelA = LinReg()
w_A = ModelA.Fit(X_train,Y_train)
y_hat_A = ModelA.Test(X_test)
Rsq_A = ModelA.Eval(y_hat_A, Y_test)
print(Rsq_A)

print(f"Analytical Solution of LinReg: {w_A}")

# (ii)

X_train_std = (X_train - X_train.mean()) / X_train.std()
X_test_std = (X_test - X_train.mean()) / X_train.std()

ModelB = LinReg()
w_B, res_B = ModelB.Fit(X_train, Y_train, alpha=0.001, epoch=100000, method = 'Batch_GD', capture_cost = True)
y_hat_B = ModelB.Test(X_test)
Rsq_B = ModelB.Eval(y_hat_B, Y_test)
print(Rsq_B)

normB = []
for wi in res_B:
    normB.append(np.linalg.norm(w_B - wi))


plt.plot(normB)
plt.xlabel("Iteration (t)")
plt.ylabel("Norm")
plt.title("Convergence of Gradient Descent towards Analytical Solution")
plt.grid(True)
plt.show()

# (iii)

ModelC = LinReg()
w_C, res_C = ModelC.Fit(X_train, Y_train, alpha=0.001, epoch=100000, method = 'MiniBatch_GD', capture_cost = True, batch_size=100)
y_hat_C = ModelC.Test(X_test)
Rsq_C = ModelC.Eval(y_hat_C, Y_test)
print(Rsq_C)

normC = []
for wi in res_C:
    normC.append(np.linalg.norm(w_C - wi))


plt.plot(normC)
plt.xlabel("Iteration (t)")
plt.ylabel("Norm")
plt.title("Convergence of Mini Batch Gradient Descent towards Analytical Solution")
plt.grid(True)
plt.show()


# (iv)

X_train = df2_train.iloc[:,:-1].to_numpy()
y_train = df2_train.iloc[:,-1].to_numpy().reshape(-1,1)
X_test = df2_test.iloc[:,:-1].to_numpy()
y_test = df2_test.iloc[:,-1].to_numpy().reshape(-1,1)


X_train = np.hstack((np.ones((X_train.shape[0],1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0],1)), X_test))

lambdas = [ 0, 0.001, 0.01, 0.1, 1, 10, 100]

errors = cross_validate(X_train, y_train, lambdas, k=5, alpha=0.001, epochs=5000)


best_lambda = lambdas[np.argmin(errors)]
print("Best λ:", best_lambda)

wR = ridge_gd(X_train, y_train, alpha=0.001, lambda_=best_lambda, epochs=20000)

wML = np.linalg.inv(X_train.T @ X_train + best_lambda * np.eye(X_train.shape[1])) @ X_train.T @ y_train

mse_R = mse(y_test, X_test @ wR)
mse_ML = mse(y_test, X_test @ wML)

print("Test MSE (Gradient Descent Ridge):", mse_R)
print("Test MSE (Closed-form Ridge):", mse_ML)

print(f"Ridge GD of LinReg: {wR}")

plt.plot(lambdas, errors, marker='o')
plt.xscale('log')
plt.xlabel('λ (log scale)')
plt.ylabel('Validation MSE')
plt.title('Cross-Validation Error vs λ')
plt.show()
