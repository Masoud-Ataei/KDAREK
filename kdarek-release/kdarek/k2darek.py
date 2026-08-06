### KDAREK: Kurkova-DAREK: A Deep Learning Architecture for Regression with Error Control
### Version 1.0, March 2026
from .darek import DAREK
import torch
import torch.nn as nn
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from scipy.stats import qmc
from scipy.stats import gaussian_kde

# from sympy import *
# import sympy
# from kan import KANLayer
# import torch
# from kan.spline import *
# from kan.utils import sparse_mask
# import numpy as np
# from  scipy.interpolate import splrep, splev
# from sklearn.cluster import KMeans
# from scipy.spatial import cKDTree
# from scipy.stats import qmc
# import math
# from kan import MultKAN
# from kan.Symbolic_KANLayer import Symbolic_KANLayer
# from kan.utils import SYMBOLIC_LIB
# import random

class LipschitzLinear(nn.Module):
    def __init__(self, in_features, out_features, lipschitz_const = 1.0, coeff = 0.95, use_spectral_norm = True):
        super().__init__()
        self.lipschitz_const = lipschitz_const
        self.use_spectral_norm = use_spectral_norm
        if use_spectral_norm:
            self.linear = nn.utils.spectral_norm(nn.Linear(in_features, out_features))
            self. coeff = coeff
            self.linear.coeff = coeff
        else:
            self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.lipschitz_const * self.linear(x)  # Scale by L_l


class KDAREK(torch.nn.Module):

    def __init__(self, mlp_width = [1,1],
                 kan_width=[1,5,1], 
                 kan_grid=9, 
                 kan_k=3, 
                 kan_base_fun = 'identity', 
                 kan_seed=42, 
                 seed=42, 
                 device='cpu', 
                 L_l = 1.0, symbolic_enabled = False, auto_save = False,
                 kan_extend = False, use_spectral_norm = True):
        super(KDAREK, self).__init__()
        self.width_mlp            = mlp_width[:]
        self.width_kan            = kan_width[:]
        self.kan_grid             = kan_grid
        self.kan_k                = kan_k
        self.kan_base_fun         = kan_base_fun
        self.kan_seed             = kan_seed
        self.seed                 = seed
        self.device               = device
        self.L_l                  = L_l
        self.kan_symbolic_enabled = symbolic_enabled
        self.kan_auto_save        = auto_save
        self.kan_extend           = kan_extend
        self.use_spectral_norm    = use_spectral_norm

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        L = len(mlp_width) -1

        self.d = mlp_width[0]
        self.L = L
        if isinstance(L_l, (int, float)):
            L_l = [L_l for i in range(L)]
        else:
            assert isinstance(L_l, list), "L_l should be scalar or list size of MLP"
            assert len(L_l) == L , "L_l should be scalar or list size of MLP"
                
        mlps = []
        for i in range(mlp_width[0]):
            dinp, dout = 1, mlp_width[1]        
            # model = nn.Sequential()
            # for il in range(len(mlp_width)-1):
            #     model.append(LipschitzLinear(dinp, dout, L_l[il]))
            #     if il < L - 1:
            #         model.append(nn.ReLU())
            #         dinp, dout = mlp_width[il+1], mlp_width[il + 2]

            layers = []
            for il in range(len(mlp_width)-1):
                layers.append(LipschitzLinear(dinp, dout, L_l[il], use_spectral_norm=use_spectral_norm))
                if il < L - 1:
                    layers.append(nn.ReLU())
                    dinp, dout = mlp_width[il+1], mlp_width[il + 2]

            model = nn.Sequential(*layers)


            # model.append(LipschitzLinear(mlp_width[L - 1], mlp_width[L], L_l[L-1]))
            mlps.append(model.to(device=device))
        # model.append(torch.nn.Linear())
        self.MLPs = nn.ModuleList(mlps)
        if kan_width is not None:
            self.SNNs = DAREK(width=kan_width, grid=kan_grid, k=kan_k, base_fun = kan_base_fun, seed=kan_seed, device=device,
                                 symbolic_enabled = symbolic_enabled, auto_save = auto_save, extend=self.kan_extend)
        else:
            self.SNNs = nn.Identity()

    def predict(self, x0, L_mlp = 1.0, L_k = 1.0, L_1 = 1.0, share = None, noise = 0.0, oint = None, oknot = None):

        mlpw = np.array(self.width_mlp)
        kanw = np.array(self.width_kan)
        depth = len(mlpw[:-1]) + len(kanw[:-1])
        
        # L_k = np.power(L_k/ modelprod,1/depth)
        # L_1 = np.power(L_1/ modelprod,1/depth)
        L_1 = np.power(L_1/ self.d, 1/depth)
        # L_mlp  = (L_1 ** len(mlpw[:-1])) * mlpprod #/ self.d
        L_mlp2 = (L_1 ** len(mlpw[:-1])) #* mlpprod / self.d
        L_1darek = (L_1 ** len(kanw[:-1])) #* kanprod
        # L_kdarek = (L_1 ** len(kanw[:-1])) * kanprod
        L_kdarek = L_k
                
        with torch.no_grad():
            y0   = self.forward_mlps(x0)

            y1, err_sp = self.SNNs.predict(y0, fk = L_kdarek, f1= L_1darek, share  = share, noise = noise, oint = oint, oknot = oknot)



        xi0 = self.SNNs.samples['xi'].unsqueeze(0)
        xt0 = x0.unsqueeze(1)
        
        # Minimum distance from each test point to any knot point
        min_dist = (xi0 - xt0).abs().min(dim=1)[0]   # (n_test, d)
        err_mlp2 = (min_dist * L_mlp2).sum(dim=-1, keepdim=True)  # (n_test, 1)

        a = (xi0.max(dim=1)[0] - xi0.min(dim=1)[0]).max() / 6
        err_mlp2 = torch.tanh(err_mlp2 / a) * L_mlp

        # .expand() is a zero-copy view; original used .repeat() which copied
        err_mlp2 = err_mlp2.expand(-1, err_sp.shape[1])

        # self.darekk_results = (x0, y0, y1)
        return y1, err_mlp2 * L_1 + err_sp

    def forward_mlps(self, x):
        # return sum([self.MLPs[i](x[:,i].unsqueeze(1)) for i in range(self.d)])
        out = self.MLPs[0](x[:, 0:1])
        for i in range(1, self.d):
            out = out + self.MLPs[i](x[:, i:i+1])
        return out
        

    def forward(self, x, singularity_avoiding = False, y_th=1000.):
        x = self.forward_mlps(x)
        x = self.SNNs(x, singularity_avoiding=singularity_avoiding, y_th=y_th)
        return x
    
    def select_knots(self, x, g, seed = 0, method = 'random', index = None):    
        """
        method = 'random' or 'min_dist' or 'Kmean' or 'LHS' or 'custom'
        """
        # print('reindex')
        np.random.seed(seed)
        # x = dataset['train_input']
        # grid,g_indx = x.sort(dim=0)
        # g_indx = g_indx[:,0]            
        if method == 'random':                                
            indx = np.random.choice(x.shape[0], g+1, False)   
        elif method == 'min_dist':                
            # calculate distance from boundries                
            # x = np.array(x)
            d = np.inf
            d_min = np.inf
            tries = 100000
            x_min = x.min(dim = 0)[0]
            x_max = x.max(dim = 0)[0]
            while d > 1 and tries > 0:    
                tries -= 1
                indx = np.random.choice(x.shape[0], g+1, False)
                
                grid = x[indx].clone()                        
                g_min = grid.min(dim = 0)[0]
                g_max = grid.max(dim = 0)[0]
                # print(g_min, g_max)
                d = (x_min - g_min).abs().sum() + (x_max - g_max).abs().sum()
                
                if d < d_min:
                    indx_min = indx.copy()
                    d_min = d
            # print('min dist is: ',d_min)
            indx = indx_min.copy()

        elif method == 'Kmean':
            kmeans = KMeans(n_clusters=g+1, n_init=10, random_state=seed)
            kmeans.fit(x.detach())
            # Extract cluster centers as representative samples
            # samples = kmeans.cluster_centers_

            # Find the closest actual dataset points using KD-Tree
            tree = cKDTree(x.detach())
            _, indices = tree.query(kmeans.cluster_centers_, k=1)  # k=1 finds the nearest neighbor
            indx = indices.copy()
            # print(indx)
        
        elif method == 'LHS':
            sampler = qmc.LatinHypercube(x.shape[1])
            x_min = x.min(dim=0)[0].detach().numpy()
            x_max = x.max(dim=0)[0].detach().numpy()
            lhs_samples = qmc.scale(sampler.random(g+1), x_min, x_max)
            tree = cKDTree(x.detach())
            _, indices = tree.query(lhs_samples, k=1)  # Find closest actual data points
            indx = indices.copy()

        elif method == 'gw_kmean':

            density = gaussian_kde(x.detach().numpy().T)(x.detach().numpy().T)
            weights = density  # or 1/density for inverse

            w = np.ones(x.shape[0]) if weights is None else np.asarray(weights)
            if w.shape[0] != x.shape[0]:
                raise ValueError(f"weights length {w.shape[0]} != x.shape[0] {x.shape[0]}")
            kmeans = KMeans(n_clusters=g + 1, n_init=10, random_state=seed)
            kmeans.fit(x.detach(), sample_weight=w)
            tree = cKDTree(x.detach())
            _, indices = tree.query(kmeans.cluster_centers_, k=1)
            indx = indices.copy()

        elif method == 'igw_kmean':

            density = gaussian_kde(x.detach().numpy().T)(x.detach().numpy().T)
            weights = 1/ density  # or 1/density for inverse

            w = np.ones(x.shape[0]) if weights is None else np.asarray(weights)
            if w.shape[0] != x.shape[0]:
                raise ValueError(f"weights length {w.shape[0]} != x.shape[0] {x.shape[0]}")
            kmeans = KMeans(n_clusters=g + 1, n_init=10, random_state=seed)
            kmeans.fit(x.detach(), sample_weight=w)
            tree = cKDTree(x.detach())
            _, indices = tree.query(kmeans.cluster_centers_, k=1)
            indx = indices.copy()

        elif method == 'chebyshev':
            n = g + 1
            k = np.arange(n)
            # roots of the n-th Chebyshev polynomial, in [-1, 1]
            cheb_nodes = np.cos((2 * k + 1) / (2 * n) * np.pi)
            cheb_01 = np.sort((cheb_nodes + 1) / 2)  # rescale to [0, 1], ascending
            x_min = x.min(dim=0)[0].detach().numpy()
            x_max = x.max(dim=0)[0].detach().numpy()
            cheb_points = x_min + cheb_01[:, None] * (x_max - x_min)
            tree = cKDTree(x.detach())
            _, indices = tree.query(cheb_points, k=1)
            indx = indices.copy()

        elif method == 'custom':
            indx = index.copy()
        else:
            raise ValueError(f'Unknown method: {method}')
        
        return indx


    def fit(self, dataset, steps=100, 
            lamb=0.,  
            loss_fn=None, 
            lr=0.01, batch=-1,
            metrics=None, singularity_avoiding=False, 
            y_th=1000., 
            nonfixknot = True, seed_knots = 0, rand_method = 'random',
            reindex = False, verbose = True, custom_index = None,
            scheduler=None, gamma=0.95, step_sch = 100):
        """
        scheduler="exp", 'cos', 'dec'
        """

        
        kan = self.SNNs
        if lamb > 0. and not kan.save_act:
            print('setting lamb=0. If you want to set lamb > 0, set kan.save_act=True')
            
        old_save_act, old_symbolic_enabled = kan.disable_symbolic_in_fit(lamb)

        
        pbar = tqdm(range(steps), desc='description', ncols=100) if verbose  else range(steps)

        if loss_fn is None:
            loss_fn = lambda x, y: torch.mean((x - y) ** 2)
                
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)            

        if scheduler == "exp":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif scheduler == "cos":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_sch)
        elif scheduler == "dec":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_sch, gamma=gamma)
        # else:
        #     raise "scheduler is not defined"

        results = {'train_loss': [], 'test_loss': [], 'reg': []}

        if metrics is not None:
            for m in metrics:
                results[m.__name__] = []
        # breakpoint()
        n_train = dataset['train_input'].shape[0]
        n_test  = dataset['test_input'].shape[0]
        batch_size      = n_train if (batch == -1 or batch > n_train) else batch
        batch_size_test = n_test  if (batch == -1 or batch > n_test)  else batch

        if nonfixknot:        
            # if custom_index is None:    
            custom_index = self.select_knots(dataset['train_input'], kan.grid, 
                                                seed = seed_knots,
                                                method=rand_method, index=custom_index)
            with torch.no_grad():
                y = self.forward_mlps(dataset['train_input'])
            kan.forward_update_grid(y,dataset['train_label'], reindex = reindex, seed = seed_knots,
                                     method='custom', index=custom_index)
            if reindex or not 'xi' in kan.samples:                
                kan.samples['xi'] = dataset['train_input'][kan.samples['indx']]
            #     print('xi')
            # print('knots', kan.samples['indx'])
            # print(seed_knots)
            # breakpoint()
            self.knots = kan.knots
            self.samples = kan.samples

        for _ in pbar:
            self.train()
            
            
            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)            

            pred = self.forward(dataset['train_input'][train_id], singularity_avoiding=singularity_avoiding, y_th=y_th)
            train_loss = loss_fn(pred, dataset['train_label'][train_id])

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if scheduler:
                lr_scheduler.step()

            if nonfixknot:
                self.eval()
                gx = kan.samples['xi']
                with torch.no_grad():
                    gy = self.forward_mlps(gx)
                    # gy = selff.den_model(gx)                                
                    y = self.forward_mlps(dataset['train_input'])
                    kan.forward_update_grid(y, dataset['train_label'],seed = seed_knots, method=rand_method)
                kan.knots['x']   = gy
                kan.samples['x'] = gy                

        # revert back to original state        
        return results

    def saveckpt(self, path='model'):
        model = self
            
        dic = dict(
            width_mlp         = model.width_mlp,
            width_kan         = model.width_kan,
            kan_grid          = model.kan_grid,
            kan_k             = model.kan_k,
            kan_base_fun      = model.kan_base_fun,
            kan_seed          = model.kan_seed,
            seed              = model.seed,
            device            = model.device,
            L_l               = model.L_l,
            symbolic_enabled  = model.kan_symbolic_enabled,
            auto_save         = model.kan_auto_save,
            kan_extend        = model.kan_extend

        )
        model.SNNs.saveckpt(path + '_snn')
        torch.save(model.MLPs, f'{path}_state')
        with open(f'{path}_config.yml', 'w') as outfile:
            yaml.dump(dic, outfile, default_flow_style=False)

    @staticmethod
    def loadckpt(path='model'):
        with open(f'{path}_config.yml', 'r') as stream:
            config = yaml.safe_load(stream)
        state_mlp = torch.load(f'{path}_state')
        # state_kan = torch.load(f'{path}_kan_state')
        model = KDAREK(mlp_width= config['width_mlp'],
                    kan_width        = config['width_kan'],
                    kan_grid         = config['kan_grid'],
                    kan_k            = config['kan_k'],
                    kan_base_fun     = config['kan_base_fun'],
                    kan_seed         = config['kan_seed'],
                    seed             = config['seed'],
                    device           = config['device'],
                    L_l              = config['L_l'],
                    symbolic_enabled = config['symbolic_enabled'],
                    auto_save        = config['auto_save'],
                    kan_extend       = config['kan_extend']
                    )
        
        model.MLPs = state_mlp
        model.SNNs = model.SNNs.loadckpt(path + '_snn')
        if 'rand_index' in model.SNNs.__dir__():
            # dict_of_tensors = lambda dict_of_lists: {key: torch.tensor(value) for key, value in dict_of_lists.items()}
            model.samples = model.SNNs.samples
            model.knots   = model.SNNs.knots
            # model.rand_index = np.array(config['rand_index'])
        model.SNNs.eval()
        return model
    