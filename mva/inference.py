from scipy.stats.distributions import chi2
from scipy.stats.distributions import f
import numpy as np 


def distance(data, MU):
    """..math:: (Xbar-MU).T @ Sinv @ (Xbar-MU)"""

    X = np.asmatrix(data)
    MU = np.asmatrix(MU).reshape(len(MU), 1)    #convert list to matrix of shape (p x 1)
    S = np.cov(X, rowvar=False)     #Sample variance-covariance matrix
    Sinv = np.linalg.inv(S) #inverse of S
    Xbar = np.mean(X, axis = 0).reshape(X.shape[1],1)    #Mean vector

    d2 = np.matmul(np.matmul((Xbar - MU).T, Sinv), (Xbar - MU))  #Mahalanobis distance
    
    return d2

def hotteling1sample_test(data, MU, test="hotteling"):
    """
    Tests hypothesis about the mean vector. use when sample size is smaller than 20p
    and the varaince-covariance matrix is unknown use the default (Hottelings)
    if sample size is larger than 20p Hottelings T2 follows a chi-squared distribution.
    
    Parameters
    ---------- 
    data: data matrix
    MU: mu vector
    test: test type. Choose between "hotteling" or "chi"
    
    return
    ------
    test-statistic, pvalue
    """

    if len(MU) != data.shape[1]:
        raise ValueError(f"MU and number of variables should have the same length. Mu shape({len(MU)}), Number of variables ({data.shape[1]})") 

    if np.any(data.dtypes == object):
        raise ValueError(f'All columns must be numeric')

    n = data.shape[0]  #number of samples
    p = data.shape[1]  #number of variables
    
    if test == 'hotteling':
        df = n - p  #degrees of freedom
        T2 = n * distance(data, MU) #Hotelling's T
        f_calc = ((n - p)/((n-1)*p)) * T2  #test statistic
        pval = f.sf(f_calc, p, df)   #pval

        return f_calc, pval

    elif test == 'chi':
        chi2_calc = n * distance(data, MU)   #test statistic 
        pval = chi2.sf(chi2_calc, p)   #pval

        return chi2_calc, pval
        
def s_pooled(data1, data2):

    n1= len(data1)
    n2= len(data2)
    S1 = np.cov(data1, rowvar=False)    # variance covariance matrix of the first sample
    S2 = np.cov(data2, rowvar=False) 
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
    if data1.shape[0] != data2.shape[0]:
    
        raise ValueError(f"two samples must be of same length. {data1.shape[0]} x {data2.shape[0]}")
    
    try:
        n = data1.shape[0]
        p = data1.shape[1]
        D = data1 - data1
        Dbar = np.asmatrix(np.mean(D, axis = 0)).reshape(p,1)    #Mean difference vector (2x1)
        S = np.cov(D, rowvar=False)
        Sinv = np.linalg.inv(S)
        T2 = n * np.matmul(np.matmult(Dbar.T, Sinv), Dbar)
        f_calc = (n-p)/((n-1)*p) * T2
        df = n - p
        pval = f.sf(f_calc, p, (df))
        
        return f, pval

    except:
        raise ValueError(f"varaince-covariance matrix is singular. Provide different samples! ")


def hotteling2sample_ind_test(data1, data2, equal_var = True):
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
    if data1.shape[0] != data2.shape[0]:
    
        raise ValueError(f"two samples must be of same length. {data1.shape[0]} x {data2.shape[0]}")

    X1 = np.asmatrix(data1)
    X2 = np.asmatrix(data2)     
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    p = X1.shape[1]
    Xbar1 = np.mean(X1, axis = 0).reshape(p,1)
    Xbar2 = np.mean(X2, axis = 0).reshape(p,1)

    if equal_var == True:
        
        Sinv = np.linalg.inv(s_pooled(X1, X2))    #inverse of S_pooled  
        T2 = ((n1*n2)/(n1 + n2)) * np.matmul(np.matmul((Xbar1 - Xbar2).T, Sinv), (Xbar1 - Xbar2))
        f_calc = (n1 + n2 - p - 1)/((n1+n2-2)*p) * T2
        df = n1+ n2 - p - 1
        pval = f.sf(f_calc, p, df)

        return f_calc, pval
    
    else:
        S1 = np.cov(X1, rowvar=False)
        S2 = np.cov(X2, rowvar = False)
        Sinv = S1/n1 + S2/n2 
        chi2_calc = np.matmul(np.matmul((Xbar1 - Xbar2).T, Sinv), (Xbar1 - Xbar2))
        pval = chi2.sf(chi2_calc, p)  
   
        return chi2_calc, pval

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
    if np.linalg.det(S) != 0:
        X= np.asmatrix(data)    # convert data to matrix     
        n = X.shape[0]   #number of samples
        p = X.shape[1]   #number of variables
        S = np.cov(X, rowvar=False)     # variance covariance matrix
        lamda = ((np.linalg.det(S) )/(np.trace(S)/p)**p)**(n/2)  # Spherecity test statistic
        chi2_calc= -2*np.log(lamda)     # multiply by -2ln to follow Chi-square distribution
        df= (((p*(p+1))/2) -1 )     # degrees of freedom 
        pval = chi2.sf(chi2_calc, df)   # P-value   
         
        return chi2_calc , pval

        
    else:
        raise ValueError(f'determinant of variance covariance matrix = 0 / variables maybe perfectly correlated')
    
    

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
