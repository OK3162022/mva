from scipy.stats.distributions import chi2
from scipy.stats.distributions import f
import numpy as np

class Inference():

    def __init__(self):
        
        pass
    
    def distance(self, data, MU):
        """
        ..math:: (Xbar-MU).T @ Sinv @ (Xbar-MU)
        """

        MU = np.asmatrix(MU).reshape(len(MU), 1)    #convert list to matrix of shape (p x 1)
        X = np.asmatrix(data)   #covert data to matrix 
        S = np.cov(data, rowvar = False)
        Sinv = np.linalg.inv(S) #inverse of S
        Xbar = np.mean(X, axis = 0).reshape(2,1)    #Mean vector

        d2 = np.matmul(np.matmul((Xbar - MU).T, Sinv), (Xbar - MU)) #contour
        
        return d2

    def hotteling1sample_test(self, data, MU):
        """
        Tests hypothesis about the mean vector. use when sample size is smaller than 20p and the varaince-covariance marix is unknown  
        
        Parameters
        ---------- 
        data: data matrix
        MU: mu vector
        
        return
        ------
        test-statistic, pvalue
        """

        if len(MU) != data.shape[1]:
            raise ValueError(f"MU and number of variables should have the same length. Mu shape({MU.shape[0]}), Number of variables ({data.shape[1]})") 

        if np.any(data.dtypes == object):
            raise ValueError(f'All columns must be numeric')

        n = data.shape[0]  #number of samples
        p = data.shape[1]  #number of variables
        df = n - p  #degrees of freedom
        T2 = n * self.distance(data, MU) #Hotelling's T
        F = ((n - p)/(n-1)*p) * T2  #test statistic
        pval = f.sf(F, p, df)   #pval

        return T2, pval

    def one_sample_chi2test(self, data, MU):
        """
        Tests hypothesis about the mean vector. use when sample size is larger than 20p 
        
        Parameters
        ----------
        data: data matrix
        MU: mu vector

        return
        ------
        test-statistic, pvalue


        Note
        -----
        Use when sample size is large (n>20p)
        """
        if len(MU) != data.shape[1]:
            raise ValueError(f"MU and number of variables should have the same length. Mu shape({MU.shape[0]}), Number of variables ({data.shape[1]})") 

        if np.any(data.dtypes == object):
            raise ValueError(f'All columns must be numeric')

        n = data.shape[0]  #number of samples
        p = data.shape[1]  #number of variables
        chi2_calc = n * self.distance(data, MU, n)   #test statistic 
        
        pval = chi2.sf(chi2_calc, p)   #pval

        return chi2_calc, pval
