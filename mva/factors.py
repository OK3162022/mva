import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _eigen_decompostion(X, cor = True):
    """
    ..math:: A = P.lambda.P^-1 or P.lambda.P.T
    lambda: eigenvalues in descending order
    P: is normalized eigenvectors
    """
    X = np.asmatrix(X)

    if cor == True:
        R = np.corrcoef(X, rowvar=False)
        lmbda, P = np.linalg.eigh(R)
        lmbda = lmbda[::-1]
        P = np.flip(P, axis = 1)
        return lmbda, P

    else:
        try:
            S = np.cov(X, rowvar=False)
            lmbda, P = np.linalg.eigh(S)
            lmbda = lmbda[::-1]
            P = np.flip(P, axis = 1)
            return lmbda, P

        except:
            raise ValueError(f'data matrix must be all numeric')

def standardize(X):
    """
    standardizes variables
    
    Parameters
    ----------
    X: data matrix to standardize
    
    return
    ------
    Z: standardized data matrix
    """
    X = np.asmatrix(X)

    return (X - np.mean(X, axis = 0))/np.var(X, axis = 0)


#Principal components analysis
class PCA():

    def __init__(self, cor = True):
        self.cor = cor

    def fit(self, X):
        self.eigenvals, self.eigenvectors = _eigen_decompostion(X, self.cor)
        pass

    def summary(self):
        """
        eigenvalues table with % variance explained
        """
        
        pc_list = []
        for i in range(len(self.eigenvals)):                    ##indexing each principle component
            pc_list.append("PC"+str(i+1))
        

        self.explained_ratio = self._variance_explained_ratio(self.eigenvals)

        data= {'Principle components':pc_list,
            'Eigenvalues': self.eigenvals,
            'Proportion': self.explained_ratio
            }   
        df= pd.DataFrame(data)
        df['Cumulative'] = df['Proportion'].cumsum()    #cumulative proportion of variance explained

        return df

    def _variance_explained_ratio(self, lmbda):
        """
        proportion of variance explained by each principle component
        
        parameters
        ---------
        lmbda: list of eigenvalues

        returns
        -------
        list of proportions of variance explained by each eigenvalue
        """
        explained_ratio=[]
        for l in lmbda:                              
            explained_ratio.append(l/np.sum(lmbda))
        
        return explained_ratio
    
    def scree_plot(self):
        """
        plots scree plot of the eigenvalues
        dotted horizontal line indicates Kaiser-Gutmann rule. Select eigenvalues greater than 1
        """
        plt.plot(self.summary()['Principle components'], self.eigenvals, 'o-')
        plt.title('Scree Plot')
        plt.xlabel('Components')
        plt.ylabel('Eigenvalues')
        plt.axhline(1, linestyle = 'dotted')
        plt.show()


    def fit_transform(self, X, n_components):
        """
        Projects the data matrix onto the eigen vectors

        Parameters
        -----------
        X: Data Matrix
        n_components: number of princpal components

        Return
        ------
        C: principal components 
        """
        if n_components > X.shape[1]:
            raise ValueError(f'factors must be less than or equal number of variables')

        if self.cor == True:

            Z = standardize(X)
            C = np.matmul(Z, self.eigenvectors[:, :n_components])

        else:

            C = np.matmul(X, self.eigenvectors[:, :n_components])
        return C




#################
#Factor analysis
#################
class PCF():

    def __init__(self, n_factors, cor = True):
        """
        n_factors: desired number of factors.choose factors based on a previous hypothesis or using
        scree plot and Kaiser-Guttman rule in Principal component analysis
        
        cor: whether to use correlaion matrix or variance covariance-matrix
        """
        
        self.n_factors = n_factors
        self.cor = cor



    def fit(self, X):
        self.eigenvals, self.eigenvectors = _eigen_decompostion(X, self.cor)
        self.loadings =  self._estimate_loadings(self.eigenvals[:self.n_factors], self.eigenvectors[:, :self.n_factors])
    
        self.communalities = np.sum(np.power(self.loadings, 2), axis = 1)
        return self

    def _estimate_loadings(self, eigenvals, eigenvectors):
        L = np.sqrt(eigenvals) * eigenvectors 
        return L
    

    def summary(self):
        """
        loadings table
        """
        pass

    def _unique_factors(self):
        # 1 - communalities 
        pass


    def factor_scores(self, X):
        """
        Obtains factor scores for the data.
        
        Parameters
        ---------
        X: data matrix
        
        return
        ------
        factors: Factor scores
        """
        
        self.factors = PCA.fit_transform(self, X, self.n_factors) / np.sqrt(self.eigenvals[:self.n_factors])        
               
        return self.factors

    def rotate_factors(self):
        pass
