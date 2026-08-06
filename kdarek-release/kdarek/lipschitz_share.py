import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.optimize import minimize

# def Lipschitz_Share(self, x, y, xt, yt, f1, fk, eps = 1e-3, method = 'Equal'):
#     """
#     method = 'Equal' or 'Last' or 'Shap' or 'ApxShap'
#     """
#     if method == 'Equal':
#         return Equal_Error_Share(self, x, y, xt, yt, f1, fk, eps = eps)
#     elif method == 'Last':
#         return LastLayer_Error_Share(self, x, y, xt, yt, f1, fk, eps = eps)    
#     elif method == 'Shap':
#         return SHAP_Error_Share(self, x, y, xt, yt, f1, fk, eps = eps)    
#     elif method == 'ApxShap':
#         return Apprx_SHAP_Error_Share(self, x, y, xt, yt, f1, fk, eps = eps)    
#     else:
#         raise f'Methos {method} is not defined.'
    
# kan_model.plot(plot_err= True)
def Equal_Lipschitz(self, x, y, Lf1, Lfk, oint = None, oknot = None, eps = 1e-3, share = 0):
    depth = len(self.width) - 1
    # NL = np.prod([w[0] for w in self.width])
    ws = np.array(self.width)[:,0]
    NL = np.sum(ws[:-1] * ws[1:])
    fk = [np.power(Lfk/ NL,1/depth) for l in range(depth)]    
    f1 = [np.power(Lf1/ NL,1/depth) for l in range(depth)]    
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
          'fk': fk, 
          'f1' :f1}
    return Eq


def Kth_Lipschitz(xt, yt, k = 1, func = 'max'):
    # Compute differences
    xt, xindex = torch.sort(xt, dim = 0)
    yt = yt[xindex]
    dx = xt[1:] - xt[:-1]
    dy = yt[1:] - yt[:-1]

    # Avoid division by zero by filtering out zero differences in dx
    nonzero_dx = dx != 0
    dx = dx[nonzero_dx]
    dy = dy[nonzero_dx]
    for _ in range(k):        
        derivative = dy / dx
        # Step 2: Second-order differences
        dx = dx[1:]  # Differences of x remain the same for higher-order
        dy = derivative[1:] - derivative[:-1]

    if func == 'max':
        Lipschitz_constant = derivative.abs().max().item()
    elif func == 'mean':
        Lipschitz_constant = derivative.abs().mean().item()

    return Lipschitz_constant

def Numerical_Lipschitz(self, x, y, k):
    yhat = self(x)
    Num_Lipschitz = []    
    depth = len(self.width) - 1
    Nl = []
    for l in range(depth):    
        Num_Lipschitz.append([])        
        for i in range(self.width_in[l]):
            for j in range(self.width_out[l+1]):
                xt = self.acts[l][:,i]
                yt = self.spline_postacts[l][:,j,i]
                L = Kth_Lipschitz(xt, yt, k = k, func = 'max')
                Num_Lipschitz[-1].append(L)
        Nl.append( len(Num_Lipschitz[-1]))
    return Num_Lipschitz, Nl

# kan_model.plot(plot_err= True)
def Heuristic_Lipschitz(self, x, y, xt, yt, Lf1, Lfk, oint = None, oknot = None, eps = 1e-3, share = 0):
    Num_Lipschitz1, Nl = Numerical_Lipschitz(self, xt, yt, 1)
    Num_Lipschitzk, Nl = Numerical_Lipschitz(self, xt, yt, self.k +1)
    
    depth = len(self.width) - 1    
    for l in range(depth):                    
        Num_Lipschitz1[l] = sum(Num_Lipschitz1[l])
        Num_Lipschitzk[l] = sum(Num_Lipschitzk[l])
    f1 = []
    fk = []
    for l in range(depth):
        f1.append( Num_Lipschitz1[l] / sum(Num_Lipschitz1) * Lf1 / Nl[l])
        fk.append( Num_Lipschitzk[l] / sum(Num_Lipschitzk) * Lfk / Nl[l])

    yhat,u = self.predict(x, fk, f1, oint = oint, oknot = oknot, share= share)
    ub = yhat  + u + eps
    lb = yhat  - u - eps
    violations = (y < lb) | (y > ub)
    He_share_violation_count = violations.sum().item()
    He = {'pred':yhat, 
          'bound': u, 
          'up-bound': ub, 
          'low-bound': lb, 
          'violations': violations,
          'violation-rate': He_share_violation_count/ len(x), 
          'fk': fk, 
          'f1' :f1}
    return He

def DataDriven_Lipschitz(self, x, y, xt, yt, Lf1, Lfk, oint = None, oknot = None, eps = 1e-3, share = 0):
    Num_Lipschitz1, Nl = Numerical_Lipschitz(self, xt, yt, 1)
    Num_Lipschitzk, Nl = Numerical_Lipschitz(self, xt, yt, self.k +1)
    
    depth = len(self.width) - 1    
    for l in range(depth):                    
        Num_Lipschitz1[l] = sum(Num_Lipschitz1[l])
        Num_Lipschitzk[l] = sum(Num_Lipschitzk[l])
    f1 = []
    fk = []
    for l in range(depth):
        f1.append( np.exp(np.log(Num_Lipschitz1[l]) / sum(np.log(Num_Lipschitz1)) * np.log(Lf1)) / Nl[l])
        fk.append( np.exp(np.log(Num_Lipschitzk[l]) / sum(np.log(Num_Lipschitzk)) * np.log(Lfk)) / Nl[l])

    yhat,u = self.predict(x, fk, f1, oint = oint, oknot = oknot, share= share)
    ub = yhat  + u + eps
    lb = yhat  - u - eps
    violations = (y < lb) | (y > ub)
    DD_share_violation_count = violations.sum().item()
    DD = {'pred':yhat, 
          'bound': u, 
          'up-bound': ub, 
          'low-bound': lb, 
          'violations': violations,
          'violation-rate': DD_share_violation_count/ len(x), 
          'fk': fk, 
          'f1' :f1}
    return DD

def NonOptimal_WorstCase_Lipschitz(self, x, y, Lf1, Lfk, oint = None, oknot = None, eps = 1e-3, share = 0):
    depth = len(self.width) - 1
    # NL = np.prod([w[0] for w in self.width])
    ws = np.array(self.width)[:,0]
    NL = np.sum(ws[:-1] * ws[1:])
    fk = [Lfk for l in range(depth)]    
    f1 = [Lf1 for l in range(depth)]    
    yhat,u = self.predict(x, fk, f1, oint = oint, oknot = oknot, share= share)
    ub = yhat  + u + eps
    lb = yhat  - u - eps
    violations = (y < lb) | (y > ub)
    NOL_share_violation_count = violations.sum().item()
    NOL = {'pred':yhat, 
          'bound': u, 
          'up-bound': ub, 
          'low-bound': lb, 
          'violations': violations,
          'violation-rate': NOL_share_violation_count/ len(x), 
          'fk': fk,
          'f1' :f1}
    return NOL

def Random_Lipschitz(self, x, y, Lf1, Lfk, oint = None, oknot = None, eps = 1e-3, share = 0, seed = None):
    """Randomly splits the Lf1/Lfk budget across layers, as a naive baseline to contrast
    against the heuristic/data-driven/Shapley allocation strategies. Layer weights are drawn
    from a Dirichlet distribution and rescaled so the aggregate bound (product of per-layer
    fk/f1 across depth) matches Equal_Lipschitz's Lfk/NL, Lf1/NL exactly -- only how that
    fixed total budget is distributed across layers is randomized, not the total itself."""
    depth = len(self.width) - 1
    ws = np.array(self.width)[:,0]
    NL = np.sum(ws[:-1] * ws[1:])
    rng = np.random.default_rng(seed)
    w = depth * rng.dirichlet(np.ones(depth))  # sums to depth, so aggregate budget is preserved
    fk = [np.power(Lfk / NL, w[l] / depth) for l in range(depth)]
    f1 = [np.power(Lf1 / NL, w[l] / depth) for l in range(depth)]
    yhat,u = self.predict(x, fk, f1, oint = oint, oknot = oknot, share= share)
    ub = yhat  + u + eps
    lb = yhat  - u - eps
    violations = (y < lb) | (y > ub)
    Rand_share_violation_count = violations.sum().item()
    Rand = {'pred':yhat,
          'bound': u,
          'up-bound': ub,
          'low-bound': lb,
          'violations': violations,
          'violation-rate': Rand_share_violation_count/ len(x),
          'fk': fk,
          'f1' :f1}
    return Rand

def Optimal_Lipschitz(self, x, y, xt, yt, Lf1, Lfk, oint = None, oknot = None, eps = 1e-3, share = 0):
    depth = len(self.width) - 1    
    
    Ng = self.width[0][0]
    Nh = self.width[1][0]    

    f1 = [1.0 for l in range(depth)]
    fk = [1.0 for l in range(depth)]
    yht, u = self.predict(x, fk, f1, oint = oint, oknot = oknot, share=share)
    
    Chat = []
    for l in range(depth):    
        Chat.append([])
        for i in range(self.width_in[l]):
            for j in range(self.width_out[l+1]):
                Chat[-1].append(self.splines[f'{l}-{i}-{j}']['Ubar_c'])

        Chat[-1] = np.mean( Chat[-1])
    Cg, Ch = Chat
    print('Chat', Chat)
    # Definei the function to maximize
    def objective(x):
        l1, l2, lh1, lh2, lh3, lh4, lg1, lg2, lg3, lg4  = x
        c1 = l1 * ((Ng * Nh * lh1 * lg1 - Lf1) )
        c2 = l2 * (((Nh * (Ng ** 4) * lh4 * lg1 * (lg2 ** 3) \
                    + 3 * Nh * (Ng ** 3) * lh3 * lg1 * lg2 * lg3 \
                    + Nh * (Ng ** 2) * lh2 * lg1 * lg4) - Lfk))
        o = -(lh4 * Ch + lh1 * lg4 * Cg ).sum()
        return -c1 - c2 + o  # Negate for maximization, as `minimize` minimizes by default

    # Bounds for a, b, c, and d (they must be greater than 0)
    bounds = [(-Lf1, Lf1), (-Lfk, Lfk), #lambda
              (1e-8, Lf1), (1e-8, Lfk), (1e-8, Lfk), (1e-8, Lfk), #Lhs
              (1e-8, Lf1), (1e-8, Lfk), (1e-8, Lfk), (1e-8, Lfk)  #Lgs
            ]

    # Initial guess for a, b, c, and d
    x0 = np.random.rand(10)

    # Run the optimization
    result = minimize(objective, x0, bounds=bounds, constraints=None, tol = 1e-2)


    # Print the optimized values of a, b, c, d and the maximum value
    if result.success:
        # Lfk_opt, Lf1_opt, Lgk1_opt, Lgk2_opt = result.x
        l1, l2, lh1, lh2, lh3, lh4, lg1, lg2, lg3, lg4 = result.x
        Lhk_opt, Lh1_opt, Lgk_opt = lh4, lh1, lg4
        max_value = -(result.fun)  # Negate to get the original maximum value
        print("Optimized values:")
        print(f"Lhk = {Lhk_opt:.2f}, Lh1 = {Lh1_opt:.2f}, Lgk = {Lgk_opt:.2f}")
        
        c1 = (Ng * Nh * lh1 * lg1 - Lf1)
        c2 = (Nh * (Ng ** 4) * lh4 * lg1 * (lg2 ** 3) + 3 * Nh * (Ng ** 3) * lh3 * lg1 * lg2 * lg3 + Nh * (Ng ** 2) * lh2 * lg1 * lg4) - Lfk
        print(f'l1={l1:.3f},l2={l2:.3f},c1={c1:.3f},c2={c2:.3f}')
        print(Lhk_opt, Lh1_opt, Lgk_opt)
        print('vars:',result.x)
        print(f"Maximum value of sum(err) = {max_value}")   
        
        fk = [lg4, lh4]
        f1 = [lg1, lh1]
        yhat,u = self.predict(x, fk, f1, oint = oint, oknot = oknot, share=share)
        ub = yhat  + u + eps
        lb = yhat  - u - eps
        violations = (y < lb) | (y > ub)
        Op_share_violation_count = violations.sum().item()
        # print(Nh * (Ng ** 4) * lh4 * lg1 * (lg2 ** 3),
        #       3 * Nh * (Ng ** 3) * lh3 * lg1 * lg2 * lg3,
        #        Nh * (Ng ** 2) * lh2 * lg1 * lg4,
        #          - Lfk)
        # Op = [yhat, u, ub, lb, Op_share_violation_count/ len(x), fk, f1]        
        opt = {'pred':yhat, 
          'bound': u, 
          'up-bound': ub, 
          'low-bound': lb, 
          'violations': violations,
          'violation-rate': Op_share_violation_count/ len(x), 
          'fk': fk, 
          'f1' :f1}
        return opt
    else:
        print("Optimization failed:", result.message)
        return None

# def Optimal_Lipschitz(self, x, y, xt, yt, Lf1, Lfk, eps = 1e-3):
#     depth = len(self.width) - 1    
    
#     Ng = self.width[0][0]
#     Nh = self.width[1][0]    

#     f1 = [1.0 for l in range(depth)]
#     fk = [1.0 for l in range(depth)]
#     yht, u = self.predict(x, fk, f1, share=0)
    
#     Chat = []
#     for l in range(depth):    
#         Chat.append([])
#         for i in range(self.width_in[l]):
#             for j in range(self.width_out[l+1]):
#                 Chat[-1].append(self.splines[f'{l}-{i}-{j}']['Ubar_c'])

#         Chat[-1] = np.sum( Chat[-1])
#     Cg, Ch = Chat
#     # Definei the function to maximize
#     def objective(x):
#         l1, l2, lh1, lh2, lh3, lh4, lg1, lg2, lg3, lg4  = x
#         c1 = 1.0 * ((Ng * Nh * lh1 * lg1 - Lf1) ** 2)
#         c2 = 1.0 * (((Nh * (Ng ** 4) * lh4 * lg1 * (lg2 ** 3) \
#                     + 3 * Nh * (Ng ** 3) * lh3 * lg1 * lg2 * lg3 \
#                     + Nh * (Ng ** 2) * lh2 * lg1 * lg4) - Lfk) ** 2)
#         o = -(lh4 * Ch + lh1 * lg4 * Cg ).sum()
#         return c1 + c2 - o  # Negate for maximization, as `minimize` minimizes by default
    
#     # Bounds for a, b, c, and d (they must be greater than 0)
#     bounds = [(1e-8, Lf1), (1e-8, Lfk), #lambda
#               (1e-8, Lf1), (1e-8, Lfk), (1e-8, Lfk), (1e-8, Lfk), #Lhs
#               (1e-8, Lf1), (1e-8, Lfk), (1e-8, Lfk), (1e-8, Lfk)  #Lgs
#             ]

#     # Initial guess for a, b, c, and d
#     x0 = np.random.rand(10)

#     # Run the optimization
#     result = minimize(objective, x0, bounds=bounds, constraints=None, tol=1e-5, method='L-BFGS-B')


#     # Print the optimized values of a, b, c, d and the maximum value
#     if result.success:
#         # Lfk_opt, Lf1_opt, Lgk1_opt, Lgk2_opt = result.x
#         l1, l2, lh1, lh2, lh3, lh4, lg1, lg2, lg3, lg4 = result.x
#         Lhk_opt, Lh1_opt, Lgk_opt = lh4, lh1, lg4
#         max_value = -(result.fun)  # Negate to get the original maximum value
#         print("Optimized values:")
#         print(f"Lhk = {Lhk_opt:.2f}, Lh1 = {Lh1_opt:.2f}, Lgk = {Lgk_opt:.2f}")
        
#         c1 = (Ng * Nh * lh1 * lg1 - Lf1)
#         c2 = (Nh * (Ng ** 4) * lh4 * lg1 * (lg2 ** 3) + 3 * Nh * (Ng ** 3) * lh3 * lg1 * lg2 * lg3 + Nh * (Ng ** 2) * lh2 * lg1 * lg4) - Lfk
#         print(f'l1={l1:.3f},l2={l2:.3f},c1={c1:.3f},c2={c2:.3f}')
#         print(Lhk_opt, Lh1_opt, Lgk_opt)
#         print('vars:',result.x)
#         print(f"Maximum value of sum(err) = {max_value}")   
        
#         fk = [lg4, lh4]
#         f1 = [lg1, lh1]
#         yhat,u = self.predict(x, fk, f1, share=0)
#         ub = yhat  + u + eps
#         lb = yhat  - u - eps
#         violations = (y < lb) | (y > ub)
#         Op_share_violation_count = violations.sum().item()
#         # Op = [yhat, u, ub, lb, Op_share_violation_count/ len(x), fk, f1]        
#         opt = {'pred':yhat, 
#           'bound': u, 
#           'up-bound': ub, 
#           'low-bound': lb, 
#           'violations': violations,
#           'violation-rate': Op_share_violation_count/ len(x), 
#           'fk': fk, 
#           'f1' :f1}
#         return opt
#     else:
#         print("Optimization failed:", result.message)
#         return None
    
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

def test_Lipschitz_2D(model, dataset, res, label = ''):
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
    ax[1].scatter(x[:,0],x[:,1], c = yhat[:,0].detach())
    ax[2].scatter(x[:,0],x[:,1], c = u[:,0].detach())
    ax[2].scatter(x[violations[:,0]][:,0],x[violations[:,0]][:,1], c = 'red', s = 10, alpha = 0.2)
    plt.title(label)
    # print('Lgk, Lhk =', fk)
    # print('Lg1, Lh1 =', f1)    
    # print('share =', share)   
    # print('Err',vio_rate)