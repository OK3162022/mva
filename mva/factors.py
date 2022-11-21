import numpy as np
import pandas as pd

def _eigen_decompostion(X, cor = True):
    """
    ..math:: A = Q.lambda.Q^-1
    lambda: eigenvalues in descending oreder
    Q: is normalized eigenvectors
    """
    X = np.asmatrix(X)

    if cor == True:
        R = np.corrcoef(X, rowvar=False)
        lmbda, Q = np.linalg.eigh(R)
        return lmbda, Q

    else:
        try:
            S = np.cov(X, rowvar=False)
            lmbda, Q = np.linalg.eigh(S)
            return lmbda, Q

        except:
            raise ValueError(f'data matrix must be all numeric')

#Principal components analysis
class PCA():

    def __init__(self, cor = True):
        self.cor = cor

    def fit(self, X):
        self.eigenvals, self.eigenvectors = _eigen_decompostion(X, self.cor)
        pass

    def summary(self):
        """eigenvalues table with % variance explained"""
        pass

    def scree_plot(self):
        pass

    def fit_transform(self, X, n_components):
        """
        Projects the data matrix onto the eigen vectors

        Parameters
        -----------
        X: Data Matrix
        n_components: number of princpal components

        Return
        ------
        C: principal components dataframe 
        """
        if n_components > X.shape[1]:
            raise ValueError(f'factors must be less than or equal number of variables')

        C = np.matmul(X, self.eigenvectors[:,- n_components :])
        return C



#################
#Factor analysis
#################
class PCF():

    def __init__(self, cor = True):
        self.cor = cor

    def fit(self, X):
        self.X = X
        self.eigenvals, self.eigenvectors = _eigen_decompostion(X, self.cor)
        self.loadings =  self._estimate_loadings(self.eigenvals, self.eigenvectors)
        return self

    def _estimate_loadings(self, eigenvals, eigenvectors):
        L = eigenvectors * np.sqrt(eigenvals)
        return L

    def summary(self):
        """
        loadings table
        """
        pass

    def _unique_factors(self):
        pass


    def estimate_factors(self, X, n_factors):
        if n_factors > self.X.shape[1]:
            raise ValueError(f'factors must be less or equal number of variables')

        factors = PCA.fit_transform(self, X, n_factors) / np.sqrt(self.eigenvals[:n_factors])        
        self.factors = factors        
        return self.factors

    def rotate_factors(self):
        pass
