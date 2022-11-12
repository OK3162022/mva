
from scipy.stats.distributions import chi2
from scipy.stats.distributions import f
import numpy as np

def distance(data, MU):
    """..math:: (Xbar-MU).T @ Sinv @ (Xbar-MU)"""

    MU = np.asmatrix(MU).reshape(len(MU), 1)    #convert list to matrix of shape (p x 1)
    X = np.asmatrix(data)   #convert data to matrix 
    S = np.cov(X, rowvar=False)     #Sample variance-covariance matrix
    Sinv = np.linalg.inv(S) #inverse of S
    Xbar = np.mean(X, axis = 0).reshape(2,1)    #Mean vector

    d2 = np.matmul(np.matmul((Xbar - MU).T, Sinv), (Xbar - MU))  #Mahalanobis distance
    
    return d2

def hotteling1sample_test(data, MU, test = "Hotteling"):
    """
    Tests hypothesis about the mean vector. use when sample size is smaller than 20p
    and the varaince-covariance matrix is unknown use the default (Hottelings)
    if sample size is larger than 20p Hottelings T2 follows a chi-squared distribution.
    
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

    if test == "Hotteling":
        n = data.shape[0]  #number of samples
        p = data.shape[1]  #number of variables
        df = n - p  #degrees of freedom
        T2 = n * distance(data, MU) #Hotelling's T
        f = ((n - p)/((n-1)*p)) * T2  #test statistic
        pval = f.sf(f, p, df)   #pval

        return f, pval

    if test == "chi":

        chi2_calc = n * distance(data, MU)   #test statistic 
        
        pval = chi2.sf(chi2_calc, p)   #pval

        return chi2_calc, pval
        
def s_pooled(data1, data2):
    n1= len(data1)
    n2= len(data2)
    X1 = np.asmatrix(data1)                                   # convert first sample to matrix
    S1 = np.cov(X1, rowvar=False)                             # variance covariance matrix of the first sample
    X2 = np.asmatrix(data2)                                   # convert second sample to matrix
    S2 = np.cov(X2, rowvar=False) 
    S_pooled= ((n1-1)/(n1+n2-2))*S1 +((n2-1)/(n1+n2-2))*S2 
    return  S_pooled, S1 , S2


def hotteling2sample_paired_test(data1, data2): 
    """
    Compares mean vectors of two dependent samples
    
    Parameters
    ---------- 
    data1: first sample data matrix
    data2: second sample data matrix
    
    return
    ------
    pvalue, test statistic"""    
    #function
    return None
    

def hotteling2sample_ind_ttest(data1, data2, equal_var = True):
    """
    Compares mean vector of two independent sample 
    
    Parameters
    ----------
    data1: first sample data matrix
    data2: second sample data matrix
    equal-var: assumes equal variance between two samples and uses pooled variance covariance matrix (default = True)
    
    return
    ------
    pvalue, test statistic"""   
    #function
    return None
    

def spherecity_test(data):
    """
    performs Mauchly's sphercity test.
    Checks if variables are independent and have the same variance
    
    Parameters
    ----------
    data: data matrix
    alpha: significance level (default = 0.05)

    return
    ------
    pvalue, test-statistic
    """
    if np.linalg.det(S) != 0 :
        n = data.shape[0]   #number of samples
        p = data.shape[1]   #number of variables
        X= np.asmatrix(data)    # convert data to matrix
        S = np.cov(X, rowvar=False)     # variance covariance matrix
        lamda = ((np.linalg.det(S) )/(np.trace(S)/p)**p)**(n/2)  # Spherecity test statistic
        calc_stat= -2*np.log(lamda)     # multiply by -2ln to follow Chi-square distribution
        df= (((p*(p+1))/2) -1 )     # degrees of freedom 
        pval = chi2.sf(calc_stat, df)   # P-value
    else :
        raise ValueError(f'determinant of variance covariance atrix = 0 / variables maybe perfectly correlated')
    return calc_stat , pval


def boxm_test(data1, data2):
    """
    compares covariance beween two samples using Box's M-test
    
    Parameters
    ----------
    data1: first sample data matrix
    data2: second sample data matrix
    
    returns
    -------
    pvalue, test statistic """
    S_pooled , S1 , S2 = s_pooled(data1, data2)
    n1= len(data1)
    n2= len(data2)
    num = (np.linalg.det(S1))**((n1-1)/2)*(np.linalg.det(S2))**((n2-1)/2) 
    dem= (np.linalg.det(S_pooled))**((n1+n2-2)/2)
    m= (num/dem)
    
    return None
