import numpy as np
from scipy.linalg import sqrtm
from scipy.interpolate import interp1d



#==============================================================================
############# Fast covariance matrix calculation ##############
### x: 2-d dataframe, n*p
#Reference for weighted covariance matrix
#https://link.springer.com/referenceworkentry/10.1007%2F978-3-642-04898-2_612
def fastCov(data, weight):
    data_weight = data * weight.reshape(-1, 1)
    data_mean = np.mean(data_weight, axis = 0)
    sdata = (data - data_mean)*np.sqrt(weight).reshape(-1, 1)
    data_cov = sdata.T.dot(sdata)/(data.shape[0]-1)
    return data_cov    
#==============================================================================



#==============================================================================
############### SAVE direction #################
### x, y: 2-d array
def saveDir(x_ori, y_ori, ws, wt): 

    pp = x_ori.shape[1]
    data_bind = np.concatenate((x_ori, y_ori))
    weight_bind = np.concatenate((ws, wt))
    data_cov = fastCov(data_bind, weight_bind)
    covinv = np.linalg.inv(data_cov)
    signrt = sqrtm(covinv)
    
    
    data_weight = data_bind * weight_bind.reshape(-1, 1)
    cm = data_weight.mean(axis = 0)
    #cm = data_bind.mean(axis = 0)
    v1 = fastCov((x_ori-cm)@signrt, ws)
    v2 = fastCov((y_ori-cm)@signrt, wt)
    
    diag = np.diag(np.repeat(1, pp))
    savemat = ((v1-diag)@(v1-diag) + (v2-diag)@(v2-diag))/2
    eigenValues, eigenVectors = np.linalg.eig(savemat)
    idx = eigenValues.argsort()[::-1] 
    vector = eigenVectors[:, idx[0]]
    dir_temp = signrt@vector
    return dir_temp/np.sqrt(dir_temp@dir_temp)
#==============================================================================

#==============================================================================
############### Directional regression (DR) #################
### x, y: 2-d array
def drDir(x_ori, y_ori, ws, wt):
    
    pp = x_ori.shape[1]
    data_bind = np.concatenate((x_ori, y_ori))
    weight_bind = np.concatenate((ws, wt))
    data_cov = fastCov(data_bind, weight_bind)
    covinv = np.linalg.inv(data_cov)
    signrt = sqrtm(covinv)
    
    data_weight = data_bind * weight_bind.reshape(-1, 1)
    cm = data_weight.mean(axis = 0)
    #cm = data_bind.mean(axis = 0)
    s1 = (x_ori-cm)@signrt
    s2 = (y_ori-cm)@signrt
    e1 = s1.mean(axis = 0)
    e2 = s2.mean(axis = 0)
    v1 = fastCov(s1, ws)
    v2 = fastCov(s2, wt)
    
    mat1 = ((v1 + np.outer(e1, e1))@(v1 + np.outer(e1, e1)) 
            + (v2 + np.outer(e2, e2))@(v2 + np.outer(e2, e2)))/2
    mat2 = (np.outer(e1, e1) + np.outer(e2, e2))/2
    
    diag = np.diag(np.repeat(1, pp))
    drmat = 2*mat1 + 2*mat2@mat2 + 2*sum(np.diag(mat2))*mat2 - 2*diag
    eigenValues, eigenVectors = np.linalg.eig(drmat)
    idx = eigenValues.argsort()[::-1] 
    vector = eigenVectors[:, idx[0]]
    dir_temp = signrt@vector
    #dir_temp = signrt@np.linalg.eig(drmat)[1][:,0]
    return dir_temp/np.sqrt(dir_temp@dir_temp)
#==============================================================================
    



#==============================================================================
############# uniform to sphere ##############
### vec: 1-d array
def uniform2sphere(vec):
    p = len(vec)
    vec_temp = 1.-2.*vec
    vec_temp[0] = 2.*np.pi*vec[0]
    x_temp = np.array([np.cos(vec_temp[0]), np.sin(vec_temp[0])])
    if p==1:
        return x_temp
    else:
        for i in range(1,p):
            xx_temp = np.append(np.sqrt(1-vec_temp[i]**2)*x_temp, vec_temp[i])
            x_temp = xx_temp
        return x_temp    
#==============================================================================
    
    
    
    

def Inv(x, weight):
# =============================================================================
#     rank_x = np.argsort(np.argsort(x))
#     res = np.array(range(len(x)))[rank_x]
# =============================================================================
    ww = weight[np.argsort(x)]
    rank_x = np.argsort(np.argsort(x)) #This works as the 'order' function in R
    res = ((np.cumsum(ww) - ww/2)/sum(ww)*len(x))[rank_x]
    return res







#Projected one-dimensional optimal transport using given direction     
def projOtmUtility(data_source, data_target, ws, wt, DIR):
    ori_proj = np.array(data_source@DIR)
    des_proj = np.array(data_target@DIR)    
    l = len(des_proj)
      
    #ori_proj = np.array([0.2, 0.5, 0.1, 0.4, 0.3])
    #des_proj = np.array([10,40,30,20,50])
    #weight = np.array([1,1,1,1,10])
    
    #x_lokup = Inv(np.array(range(1, l+1))-0.5, wt)
    #y_lokup = np.sort(des_proj)
    
    
    x_samples_nw = np.array(range(1, l+1))-0.5
    wt_sort = wt[np.argsort(des_proj)]
    x_samples = Inv(x_samples_nw, wt_sort)
    y_samples = np.sort(des_proj)
    
    
    lokup_interp = interp1d(x_samples, y_samples, kind='linear', fill_value="extrapolate")
    itr_temp = Inv(ori_proj, ws)
    ori_proj_new = lokup_interp(itr_temp)
    delta = ori_proj_new - ori_proj
    res = data_source + np.outer(delta, DIR)
    #return res, ori_proj, ori_proj_new
    return res





#Projected one-dimensional optimal transport        
def projOtm(data_source, data_target, weight_source= None, weight_target= None, method= "SAVE", nslice = 10):
        
    
    if weight_source is None:
        weight_source = np.repeat(1, data_source.shape[0])
    else:
        assert(len(weight_source)==data_source.shape[0]), print("The length of 'weight_source' and the number of source observations do not match!!!")
        weight_source = weight_source/sum(weight_source)*data_source.shape[0]

    if weight_target is None:
        weight_target = np.repeat(1, data_target.shape[0])
    else:        
        assert(len(weight_target)==data_target.shape[0]), print("The length of 'weight_target' and the number of target observations do not match!!!")
        weight_target = weight_target/sum(weight_target)*data_target.shape[0]
    
    
    if method == "SLICED":
        res_meta = np.empty((nslice, data_source.shape[0], data_source.shape[1]))
        for i in range(nslice):
            vec = np.random.uniform(size = np.shape(data_source)[1]-1)
            DIR = uniform2sphere(vec)
            res_meta[i,:,:] = projOtmUtility(data_source, data_target, weight_source, weight_target, DIR) 
        res = np.mean(res_meta, axis=0)
    else:
        if method == "SAVE":
            DIR = saveDir(data_source, data_target, weight_source, weight_target)
        if method == "DR":
            DIR = drDir(data_source, data_target, weight_source, weight_target)
        if method == "RANDOM":
            vec = np.random.uniform(size = np.shape(data_source)[1]-1)
            DIR = uniform2sphere(vec)
        res = projOtmUtility(data_source, data_target, weight_source, weight_target, DIR) 
    return res

        

#==============================================================================