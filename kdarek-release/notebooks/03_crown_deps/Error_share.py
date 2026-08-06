import math
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.special import softmax

def Error_Share(self, x, y, xt, yt, f1, fk, oint = None, oknot = None, eps = 1e-3, method = 'Equal'):
    """
    method = 'Equal' or 'Last' or 'Shap' or 'ApxShap'
    """
    if method == 'Equal':
        return Equal_Error_Share(self, x, y, xt, yt, f1, fk, oint = oint, oknot = oknot, eps = eps)
    elif method == 'Last':
        return LastLayer_Error_Share(self, x, y, xt, yt, f1, fk, oint = oint, oknot = oknot, eps = eps)    
    elif method == 'Shap':
        return SHAP_Error_Share(self, x, y, xt, yt, f1, fk, oint = oint, oknot = oknot, eps = eps)    
    elif method == 'ApxShap':
        return Apprx_SHAP_Error_Share(self, x, y, xt, yt, f1, fk, oint = oint, oknot = oknot, eps = eps)    
    else:
        raise f'Methos {method} is not defined.'
    
def LastLayer_Error_Share(self, x, y, xt, yt, f1, fk, oint = None, oknot = None, eps = 1e-3):
    depth = len(self.width) - 1    
    share = np.array([0.0 for l in range(depth)]    )
    share[-1] = 1.0
    yhat,u = self.predict(x, fk, f1, oint = oint, oknot = oknot, share=share)
    ub = yhat  + u + eps
    lb = yhat  - u - eps
    violations = (y < lb) | (y > ub)
    Eq_share_violation_count = violations.sum().item()
    Eq = {'pred':yhat, 
          'bound': u, 
          'up-bound': ub, 
          'low-bound': lb, 
          'violations': violations,
          'violation-rate': Eq_share_violation_count/ len(x), 
          'Share': share, 
          'fk' :fk,
          'f1' :f1}
    return Eq

def Equal_Error_Share(self, x, y, xt, yt, f1, fk, oint = None, oknot = None, eps = 1e-3):
    depth = len(self.width) - 1    
    s = 0
    for l in range(depth):    
        s += f1[l] * self.width_in[l] * self.width_out[l+1]
    share = 1 / s
    share = np.array([share for l in range(depth)]    )

    yhat,u = self.predict(x, fk, f1, oint = oint, oknot = oknot, share=share)
    ub = yhat  + u + eps
    lb = yhat  - u - eps
    violations = (y < lb) | (y > ub)
    Eq_share_violation_count = violations.sum().item()
    Eq = {'pred':yhat, 
          'bound': u, 
          'up-bound': ub, 
          'low-bound': lb, 
          'violations': violations,
          'violation-rate': Eq_share_violation_count/ len(x), 
          'Share': share, 
          'fk' :fk,
          'f1' :f1}
    return Eq

def power_set(SP):    
    SP = set(SP)
    List = list(SP)    
    PS = [[]] + [list(j) for i in range(len(List)) for j in combinations(List, i+1)]
    PS = [(set(ps),SP.difference(set(ps))) for ps in PS]
    return PS

def SHAP_Error_Share(self, x, y, xt, yt, f1, fk, oint = None, oknot = None, eps = 1e-3):
    # self = kan_model
    depth = len(self.width) - 1
    SP   = set()
    shap = dict()
    c = 0
    for l in range(depth):    
        for i in range(self.width_in[l]):
            for j in range(self.width_out[l+1]):
                SP.add(c)           
                # print(f'{l}-{j}-{i}')
                shap[c]={'indx'    : c,
                    'name'    : f'{l}-{j}-{i}', 
                    'addr'    : [l,j,i],                    
                    'shapval' : 0.0}
                c += 1
    # print(shap)
    C_SP = len(shap)
    F_SP = 1 / math.gamma(C_SP + 1)  # Avoid factorial overflow
    for cont,discont in power_set(SP):
        ### Zero out spline
        # print(cont, discont)    
        for c in cont:
            ## enable c in Network M
            l,j,i = shap[c]['addr']        
            self.act_fun[l].mask[i,j] = 1.
            pass
        for d in discont:
            ## disable d in Network M
            l,j,i = shap[d]['addr']        
            self.act_fun[l].mask[i,j] = 0.
            pass
        
        ## k in powerset
        V = (np.abs(self(xt).detach()-yt).cpu()).mean(dim=0)
        # print(V)
        Lcont = len(cont)
        
        C0 = math.gamma(Lcont) if Lcont > 0 else 1
        C1 = math.gamma(C_SP - len(cont) + 1)

        for c in cont:
            ## k + c
            shap[c]['shapval'] +=  F_SP*C0*C1*V ## 1/len(SP)! * (len(disc)-1)! * (len(SP)-len(disc))! V
        
        D0 = math.gamma(len(cont) + 1)
        D1 = math.gamma(C_SP - len(cont)) if Lcont < C_SP else 1

        for d in discont:
            ## k - d
            shap[d]['shapval'] +=  -F_SP*D0*D1*V ## 1/len(SP)! * (len(disc)-1)! * (len(SP)-len(disc))! (-m(discont))
        # print(cont,discont,F_SP*C0*C1*V,F_SP*D0*D1*V)
    self.shapley = shap
    # [sp['shapval'] for sp in shap.values()]
    
    share_layer = np.zeros(depth)    
    for sp in shap.values():
        share_layer[sp['addr'][0]] += sp['shapval'].item()
    
    # # print(share_layer)
    # # share_layer = np.exp(share_layer)
    # share_layer = np.abs(share_layer)
    # share_layer = np.exp(share_layer - np.max(share_layer))  # Numerical stability
    # share = softmax(share_layer)
    
    # mean, std, var = np.mean(share_layer), np.std(share_layer), np.var(share_layer) 
    # share = (share_layer  - mean) / std
    # share = np.exp(share)/np.exp(share).sum()
    # # print(share)
    share_layer = np.abs(share_layer)
    share = share_layer / share_layer.sum()
    
    shap['layer'] = share_layer
    shap['share'] = share

    yhat,u = self.predict(x, fk, f1, oint = oint, oknot = oknot, share=share)
    ub = yhat  + u + eps
    lb = yhat  - u - eps
    violations = (y < lb) | (y > ub)
    Sh_share_violation_count = violations.sum().item()
    Sh = {'pred':yhat, 
          'bound': u, 
          'up-bound': ub, 
          'low-bound': lb, 
          'violations': violations,
          'violation-rate': Sh_share_violation_count/ len(x), 
          'Share': share, 
          'fk' :fk,
          'f1' :f1}
    return Sh
    
def Apprx_SHAP_Error_Share(self, x, y, xt, yt, f1, fk, oint = None, oknot = None, eps = 1e-3, num_samples=500):
    """ Approximates Shapley values using Monte Carlo sampling """
    
    depth = len(self.width) - 1
    SP = list(range(sum(self.width_in[l] * self.width_out[l+1] for l in range(depth))))
    # print(SP)
    num_params = len(SP)
    
    shap_values = np.zeros(num_params)
    shap = dict()
    c = 0
    for l in range(depth):    
        for i in range(self.width_in[l]):
            for j in range(self.width_out[l+1]):
                # SP.add(c)           
                # print(f'{l}-{j}-{i}')
                shap[c]={'indx'    : c,
                    'name'    : f'{l}-{j}-{i}', 
                    'addr'    : [l,j,i],                    
                    'shapval' : 0.0}
                c += 1
    self.shapley = shap
    for _ in range(num_samples):
        # Generate a random permutation of indices
        perm = np.random.permutation(SP)
        # print(perm)
        active_set = set()
        prev_error = None
        
        for j in SP:
            l, j, i = self.shapley[j]['addr']
            self.act_fun[l].mask[i, j] = 1. if j in active_set else 0.

        for p in perm:
            # Add parameter `i` to active set
            active_set.add(p)
            # print(p)
            # Set mask for active parameters
            # for j in SP:
            l, j, i = self.shapley[p]['addr']
            self.act_fun[l].mask[i, j] = 1. #if j in active_set else 0.

            # Compute error
            error = (np.abs(self(xt).detach() - yt).cpu()).mean(dim=0)
            
            if prev_error is not None:
                shap_values[p] += (error - prev_error).item()
            
            prev_error = error

    # Normalize Shapley values
    shap_values /= num_samples
    
    # Assign computed Shapley values to `self.shapley`
    for i, key in enumerate(SP):
        self.shapley[key]['shapval'] = shap_values[i]
    
    # Convert Shapley values into importance scores
    share_layer = np.zeros(depth)
    for sp in self.shapley.values():
        share_layer[sp['addr'][0]] += sp['shapval']

    # Normalize using softmax
    # share_layer = np.exp(share_layer - np.max(share_layer))  # Numerical stability
    # share = share_layer / share_layer.sum()
    # mean, std, var = np.mean(share_layer), np.std(share_layer), np.var(share_layer) 
    # share = (share_layer  - mean) / std
    # share = softmax(share)
    share_layer = np.abs(share_layer)
    share = share_layer / share_layer.sum()
    
    shap['layer'] = share_layer
    shap['share'] = share

    # Predict using estimated Shapley values
    yhat, u = self.predict(x, fk, f1, oint = oint, oknot = oknot, share=share)
    ub = yhat + u + eps
    lb = yhat - u - eps

    violations = (y < lb) | (y > ub)
    violation_rate = violations.sum().item() / len(x)
    ApprxSH = {
        'pred': yhat,
        'bound': u,
        'up-bound': ub,
        'low-bound': lb,
        'violations': violations,
        'violation-rate': violation_rate,
        'Share': share,
        'fk': fk,
        'f1': f1
    }
    return ApprxSH

def get_val(dat, k):
    if k in dat:
        return dat[k]
    return None

def test_Lipschitz_1D(model, dataset, res):
    x, xindx = torch.sort(dataset['test_input'], dim = 0)
    y = dataset['test_label'][xindx[:,0]]
    # Lf1 = 1.0
    # Lfk = 1.0
    # depth = len(model.width) - 1
    # Eq = Equal_Lipschitz(model, x, y, Lf1, Lfk, eps= 1e-3)        
    yhat        = get_val(res, 'pred')
    u           = get_val(res, 'bound' )
    ub          = get_val(res, 'up-bound') 
    lb          = get_val(res, 'low-bound') 
    violations  = get_val(res, 'violations')         
    vio_rate    = get_val(res, 'violation-rate')       
    share       = get_val(res, 'Share') 
    fk          = get_val(res, 'fk' ) 
    f1          = get_val(res, 'f1' ) 
    plt.plot(x,y)
    plt.scatter(model.samples['x'], model.samples['y'])
    plt.plot(x,yhat)
    plt.fill_between(x.flatten(),(yhat-u).flatten(),(yhat+u).flatten(), alpha = 0.3, color = 'red')
    plt.scatter(x[violations],yhat[violations], color = 'red', alpha = 0.1)
    plt.ylim([-3,3])
    print('Lgk, Lhk =', fk)
    print('Lg1, Lh1 =', f1)    
    print('share =', share)    
    print('Err',vio_rate)

def test_Lipschitz_2D(model, dataset, res):
    x = dataset['test_input']
    y = dataset['test_label']
    # Lf1 = 1.0
    # Lfk = 1.0
    # depth = len(model.width) - 1
    # Eq = Equal_Lipschitz(model, x, y, Lf1, Lfk, eps= 1e-3)
    yhat        = get_val(res, 'pred')
    u           = get_val(res, 'bound' )
    ub          = get_val(res, 'up-bound') 
    lb          = get_val(res, 'low-bound') 
    violations  = get_val(res, 'violations')         
    vio_rate    = get_val(res, 'violation-rate')       
    share       = get_val(res, 'Share') 
    fk          = get_val(res, 'fk' ) 
    f1          = get_val(res, 'f1' ) 
    # plt.plot(x,y)
    # plt.scatter(model.samples['x'], model.samples['y'])
    # plt.plot(x,yhat)
    # plt.fill_between(x.flatten(),(yhat-u).flatten(),(yhat+u).flatten(), alpha = 0.3, color = 'red')
    # plt.scatter(x[violations],yhat[violations], color = 'red', alpha = 0.1)
    # plt.ylim([-3,3])
    fig,ax = plt.subplots(1,3, figsize =(15,4))
    ax[0].scatter(x[:,0],x[:,1], c = y[:,0])
    ax[1].scatter(x[:,0],x[:,1], c = yhat[:,0])
    ax[2].scatter(x[:,0],x[:,1], c = u[:,0])
    ax[2].scatter(x[violations[:,0]][:,0],x[violations[:,0]][:,1], c = 'red', s = 10, alpha = 0.2)

    print('Lgk, Lhk =', fk)
    print('Lg1, Lh1 =', f1)    
    print('share =', share)   
    print('Err',vio_rate)