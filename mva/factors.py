import numpy as np

def _eigen_decompostion(X, cor = True):
    """
    ..math:: A = Q.lambda.Q^-1
    lambda: eigenvalues in descending oreder
    Q: is normalized eigenvectors
    """
    if cor == True:
        R = X.corr()
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
        return self
    
    def summary(self):
        """eigenvalues table with % variance explained"""
        pass

    def scree_plot(self):
        pass

    def fit_transform(self, n_components):
        pass
            
            
#################
#Factor analysis
#################
class PCF():

    def __init__(self, cor = True):
        self.cor = cor

    def fit(self, X):
        self.eigenvals, self.eigenvectors = _eigen_decompostion(X, self.cor)
        self.loadings =  self._estimate_loadings(self.eigenvals, self.eigenvectors)
        return self
    
    def loadings_table(self):
        """
        loadings table
        """
        pass

    def _unique_factors(self):
        pass

    def fit_transform(self, eigenvals, eigenvectors, n_factors):
        
        #self.factors = _estimate_factors()
        pass

    def rotate_factors(self):
        pass

    def _estimate_loadings(eigenvals, eigenvectors):
        L = np.matmul(np.sqrt(eigenvals), eigenvectors)
        return L        