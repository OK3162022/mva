import numpy as np
import pandas as pd


def _standard_error(coef):
    pass

class LinearRegression:
    def __init__(self, fit_intercept = True):
        self.fit_intercept = fit_intercept



class MulitvariateRegression:

    def __init__(self, fit_intercept = True):
        self.fit_intercept = fit_intercept

    def fit(self, X, Y):
        """
        Parameters:
        -----------
        X: Data (nxq)
        Y: Response variables matrix (nxp)
        
        Returns:
        --------
        betas: coefficients or weights(pxq)
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must be of same length X: {X.shape[0]} Y: {Y.shape[0]}")

        if self.fit_intercept:
            X_copy = np.insert(X, 0, 1, axis = 1)
        else:
            X_copy =  X.copy()

        coef = np.matmul(np.linalg.inv(np.matmul(X_copy.T, X_copy)), np.matmul(X_copy.T, Y))    
        self.coef = coef
        return self

    def summary(self):
        """
        Returns:
        --------
        Summary table containing coefficients, their Standard error, test statistic, and pvalues
        """
        se = _standard_error(self.coef)
        pass

