import numpy as np
from scipy.stats import f

class CustomOLS:
    def __init__(self):
        self.coef_ = None
        self.cov_matrix_ = None
        self.sigma2_ = None
        self.df_resid_ = None

    def fit(self, X, y):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n, k = X.shape
        Xt = X.T
        XtX = Xt @ X
        XtX_inv = np.linalg.inv(XtX)
        beta_hat = XtX_inv @ Xt @ y

        y_hat = X @ beta_hat
        resid = y - y_hat
        sse = np.sum(resid ** 2)
        df_resid = n - k
        sigma2 = sse / df_resid

        cov_matrix = sigma2 * XtX_inv

        self.coef_ = beta_hat
        self.sigma2_ = sigma2
        self.df_resid_ = df_resid
        self.cov_matrix_ = cov_matrix
        return self

    def predict(self, X):
        return X @ self.coef_

    def score(self, X, y):
        y_hat = self.predict(X)
        sse = np.sum((y - y_hat) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - sse / sst

    def f_test(self, C, d):
        Cb = C @ self.coef_
        diff = Cb - d
        CVCt = C @ self.cov_matrix_ @ C.T
        CVCt_inv = np.linalg.inv(CVCt)
        q = C.shape[0]
        f_stat = (diff.T @ CVCt_inv @ diff) / (q * self.sigma2_)
        f_stat = f_stat.item()
        p_value = 1 - f.cdf(f_stat, q, self.df_resid_)
        return {"f_stat": f_stat, "p_value": p_value}
