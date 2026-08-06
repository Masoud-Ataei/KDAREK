import os
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from kan import KAN
from kan import KANLayer
from kan.spline import extend_grid, curve2coef, coef2curve
from kan.utils import sparse_mask
from kan.Symbolic_KANLayer import Symbolic_KANLayer
from kan.utils import SYMBOLIC_LIB
from kan.LBFGS import LBFGS

from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from scipy.stats import qmc

# Function to recursively convert NumPy arrays to lists
def convert_to_list(d):
    if isinstance(d, (np.ndarray, torch.Tensor)):
        return d.tolist()
    elif isinstance(d, dict):
        return {key: convert_to_list(value) for key, value in d.items()}
    elif isinstance(d, list):
        return [convert_to_list(item) for item in d]
    else:
        return d

class KANLayerD(KANLayer):
    def __init__(self, in_dim=3, out_dim=2, num=5, k=3, 
                 noise_scale=0.5, scale_base_mu=0.0, 
                 scale_base_sigma=1.0, scale_sp=1.0, 
                 base_fun=torch.nn.SiLU(), 
                 grid_eps=0.02, grid_range=[-1, 1], 
                 sp_trainable=True, sb_trainable=True, 
                 save_plot_data = True, device='cpu', sparse_init=False,
                 extend = False):
        ''''
        initialize a KANLayer
        
        Args:
        -----
            in_dim : int
                input dimension. Default: 2.
            out_dim : int
                output dimension. Default: 3.
            num : int
                the number of grid intervals = G. Default: 5.
            k : int
                the order of piecewise polynomial. Default: 3.
            noise_scale : float
                the scale of noise injected at initialization. Default: 0.1.
            scale_base_mu : float
                the scale of the residual function b(x) is intialized to be N(scale_base_mu, scale_base_sigma^2).
            scale_base_sigma : float
                the scale of the residual function b(x) is intialized to be N(scale_base_mu, scale_base_sigma^2).
            scale_sp : float
                the scale of the base function spline(x).
            base_fun : function
                residual function b(x). Default: torch.nn.SiLU()
            grid_eps : float
                When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            grid_range : list/np.array of shape (2,)
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable
            sb_trainable : bool
                If true, scale_base is trainable
            device : str
                device
            sparse_init : bool
                if sparse_init = True, sparse initialization is applied.
            
        Returns:
        --------
            self
            
        Example
        -------
        >>> from kan.KANLayer import *
        >>> model = KANLayer(in_dim=3, out_dim=5)
        >>> (model.in_dim, model.out_dim)
        '''
        super().__init__()
        # size 
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k
        self.extend = extend
        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None,:].expand(self.in_dim, num+1).clone()
        if self.extend:
            grid = extend_grid(grid, k_extend=k)
        self.grid = torch.nn.Parameter(grid).requires_grad_(False)
        noises = (torch.rand(self.num+1, self.in_dim, self.out_dim) - 1/2) * noise_scale / num

        if self.extend:
            self.coef = torch.nn.Parameter(curve2coef(self.grid[:,k:-k].permute(1,0), noises, self.grid, k))
        else:    
            self.coef = torch.nn.Parameter(curve2coef(self.grid[:,:].permute(1,0), noises, self.grid, k))
        
        if sparse_init:
            self.mask = torch.nn.Parameter(sparse_mask(in_dim, out_dim)).requires_grad_(False)
        else:
            self.mask = torch.nn.Parameter(torch.ones(in_dim, out_dim)).requires_grad_(False)
        
        self.scale_base = torch.nn.Parameter(scale_base_mu * 1 / np.sqrt(in_dim) + \
                         scale_base_sigma * (torch.rand(in_dim, out_dim)*2-1) * 1/np.sqrt(in_dim)).requires_grad_(sb_trainable)
        self.scale_sp = torch.nn.Parameter(torch.ones(in_dim, out_dim) * scale_sp * 1 / np.sqrt(in_dim) * self.mask).requires_grad_(sp_trainable)  # make scale trainable
        self.base_fun = base_fun

        
        self.grid_eps = grid_eps
        
        self.to(device)

        
    def update_grid_from_samples(self, x, mode='sample'):
        '''
        update grid from samples
        
        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
        
        Example
        -------
        >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(model.grid.data)
        >>> x = torch.linspace(-3,3,steps=100)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.grid.data)
        '''
        
        batch = x.shape[0]
        #x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch, self.size).permute(1, 0)
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        num_interval = self.grid.shape[1] - 1 - 2*self.k
        
        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1,0)
            margin = 0.00
            h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]] + 2 * margin)/num_interval
            grid_uniform = grid_adaptive[:,[0]] - margin + h * torch.arange(num_interval+1,)[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid
        
        grid = get_grid(num_interval)
        
        if mode == 'grid':
            sample_grid = get_grid(2*num_interval)
            x_pos = sample_grid.permute(1,0)
            y_eval = coef2curve(x_pos, self.grid, self.coef, self.k)
        if self.extend:
            grid = extend_grid(grid, k_extend=self.k)

        self.grid.data = grid
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

    def initialize_grid_from_parent(self, parent, x, mode='sample'):
        '''
        update grid from a parent KANLayer & samples
        
        Args:
        -----
            parent : KANLayer
                a parent KANLayer (whose grid is usually coarser than the current model)
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)
            
        Returns:
        --------
            None
          
        Example
        -------
        >>> batch = 100
        >>> parent_model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(parent_model.grid.data)
        >>> model = KANLayer(in_dim=1, out_dim=1, num=10, k=3)
        >>> x = torch.normal(0,1,size=(batch, 1))
        >>> model.initialize_grid_from_parent(parent_model, x)
        >>> print(model.grid.data)
        '''
        
        batch = x.shape[0]
        
        x_pos = torch.sort(x, dim=0)[0]
        y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)
        num_interval = self.grid.shape[1] - 1 - 2*self.k
        
        
        '''
        # based on samples
        def get_grid(num_interval):
            ids = [int(batch / num_interval * i) for i in range(num_interval)] + [-1]
            grid_adaptive = x_pos[ids, :].permute(1,0)
            h = (grid_adaptive[:,[-1]] - grid_adaptive[:,[0]])/num_interval
            grid_uniform = grid_adaptive[:,[0]] + h * torch.arange(num_interval+1,)[None, :].to(x.device)
            grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
            return grid'''
        
        #print('p', parent.grid)
        # based on interpolating parent grid
        def get_grid(num_interval):
            x_pos = parent.grid[:,parent.k:-parent.k]
            #print('x_pos', x_pos)
            sp2 = KANLayerD(in_dim=1, out_dim=self.in_dim,k=1,num=x_pos.shape[1]-1,scale_base_mu=0.0, scale_base_sigma=0.0).to(x.device)

            #print('sp2_grid', sp2.grid[:,sp2.k:-sp2.k].permute(1,0).expand(-1,self.in_dim))
            #print('sp2_coef_shape', sp2.coef.shape)
            sp2_coef = curve2coef(sp2.grid[:,sp2.k:-sp2.k].permute(1,0).expand(-1,self.in_dim), x_pos.permute(1,0).unsqueeze(dim=2), sp2.grid[:,:], k=1).permute(1,0,2)
            shp = sp2_coef.shape
            #sp2_coef = torch.cat([torch.zeros(shp[0], shp[1], 1), sp2_coef, torch.zeros(shp[0], shp[1], 1)], dim=2)
            #print('sp2_coef',sp2_coef)
            #print(sp2.coef.shape)
            sp2.coef.data = sp2_coef
            percentile = torch.linspace(-1,1,self.num+1).to(self.device)
            grid = sp2(percentile.unsqueeze(dim=1))[0].permute(1,0)
            #print('c', grid)
            return grid
        
        grid = get_grid(num_interval)
        
        if mode == 'grid':
            sample_grid = get_grid(2*num_interval)
            x_pos = sample_grid.permute(1,0)
            y_eval = coef2curve(x_pos, parent.grid, parent.coef, parent.k)
        
        if self.extend:
            grid = extend_grid(grid, k_extend=self.k)

        self.grid.data = grid
        self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k)

class KAND(KAN):
        
    def __init__(self, width=None, grid=3, k=3, mult_arity = 2, noise_scale=0.3, scale_base_mu=0.0, scale_base_sigma=1.0, 
                    base_fun='silu', 
                    symbolic_enabled=True, affine_trainable=False, 
                    grid_eps=0.02, grid_range=[-1, 1], 
                    sp_trainable=True, sb_trainable=True, 
                    seed=1, save_act=True, 
                    sparse_init=False, auto_save=True, 
                    first_init=True, ckpt_path='./model', 
                    state_id=0, round=0, device='cpu',
                    extend = False):
        '''
        initalize a KAN model
        
        Args:
        -----
            width : list of int
                Without multiplication nodes: :math:`[n_0, n_1, .., n_{L-1}]` specify the number of neurons in each layer (including inputs/outputs)
                With multiplication nodes: :math:`[[n_0,m_0=0], [n_1,m_1], .., [n_{L-1},m_{L-1}]]` specify the number of addition/multiplication nodes in each layer (including inputs/outputs)
            grid : int
                number of grid intervals. Default: 3.
            k : int
                order of piecewise polynomial. Default: 3.
            mult_arity : int, or list of int lists
                multiplication arity for each multiplication node (the number of numbers to be multiplied)
            noise_scale : float
                initial injected noise to spline.
            base_fun : str
                the residual function b(x). Default: 'silu'
            symbolic_enabled : bool
                compute (True) or skip (False) symbolic computations (for efficiency). By default: True. 
            affine_trainable : bool
                affine parameters are updated or not. Affine parameters include node_scale, node_bias, subnode_scale, subnode_bias
            grid_eps : float
                When grid_eps = 1, the grid is uniform; when grid_eps = 0, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
            grid_range : list/np.array of shape (2,))
                setting the range of grids. Default: [-1,1]. This argument is not important if fit(update_grid=True) (by default updata_grid=True)
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            device : str
                device
            seed : int
                random seed
            save_act : bool
                indicate whether intermediate activations are saved in forward pass
            sparse_init : bool
                sparse initialization (True) or normal dense initialization. Default: False.
            auto_save : bool
                indicate whether to automatically save a checkpoint once the model is modified
            state_id : int
                the state of the model (used to save checkpoint)
            ckpt_path : str
                the folder to store checkpoints. Default: './model'
            round : int
                the number of times rewind() has been called
            device : str
            
        Returns:
        --------
            self
            
        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        checkpoint directory created: ./model
        saving model version 0.0
        '''
        self.extend = extend
        super().__init__(width=width, auto_save=auto_save)

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        ### initializeing the numerical front ###

        self.act_fun = nn.ModuleList()
        self.depth = len(width) - 1
        self.extend = extend
        
        #print('haha1', width)
        for i in range(len(width)):
            #print(type(width[i]), type(width[i]) == int)
            if type(width[i]) == int or type(width[i]) == np.int64:
                width[i] = [width[i],0]
                
        #print('haha2', width)
            
        self.width = width
        
        # if mult_arity is just a scalar, we extend it to a list of lists
        # e.g, mult_arity = [[2,3],[4]] means that in the first hidden layer, 2 mult ops have arity 2 and 3, respectively;
        # in the second hidden layer, 1 mult op has arity 4.
        if isinstance(mult_arity, int):
            self.mult_homo = True # when homo is True, parallelization is possible
        else:
            self.mult_homo = False # when home if False, for loop is required. 
        self.mult_arity = mult_arity

        width_in = self.width_in
        width_out = self.width_out
        
        self.base_fun_name = base_fun
        if base_fun == 'silu':
            base_fun = torch.nn.SiLU()
        elif base_fun == 'identity':
            base_fun = torch.nn.Identity()
        elif base_fun == 'zero':
            base_fun = lambda x: x*0.
        else:
            assert False, 'wrong base_func'
            
        self.grid_eps = grid_eps
        self.grid_range = grid_range
            
        
        for l in range(self.depth):
            # splines
            if isinstance(grid, list):
                grid_l = grid[l]
            else:
                grid_l = grid
                
            if isinstance(k, list):
                k_l = k[l]
            else:
                k_l = k
                    
            
            sp_batch = KANLayerD(in_dim=width_in[l], out_dim=width_out[l+1], num=grid_l, k=k_l, 
                                noise_scale=noise_scale, scale_base_mu=scale_base_mu, 
                                scale_base_sigma=scale_base_sigma, scale_sp=1., 
                                base_fun=base_fun, grid_eps=grid_eps, grid_range=grid_range, 
                                sp_trainable=sp_trainable, sb_trainable=sb_trainable, sparse_init=sparse_init,
                                extend=self.extend)
            
            '''
            sp_batch = KANLayerD(in_dim=width_in[l], out_dim=width_out[l+1], num=grid, k=k, 
                                noise_scale=noise_scale, scale_base_mu=scale_base_mu, 
                                scale_base_sigma=scale_base_sigma, scale_sp=1.,
                                base_fun=base_fun, grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable, sb_trainable=sb_trainable, sparse_init=sparse_init,
                                extend=self.extend)
            '''
            self.act_fun.append(sp_batch)

        self.node_bias = []
        self.node_scale = []
        self.subnode_bias = []
        self.subnode_scale = []
        
        globals()['self.node_bias_0'] = torch.nn.Parameter(torch.zeros(3,1)).requires_grad_(False)
        exec('self.node_bias_0' + " = torch.nn.Parameter(torch.zeros(3,1)).requires_grad_(False)")
        
        for l in range(self.depth):
            exec(f'self.node_bias_{l} = torch.nn.Parameter(torch.zeros(width_in[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.node_scale_{l} = torch.nn.Parameter(torch.ones(width_in[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.subnode_bias_{l} = torch.nn.Parameter(torch.zeros(width_out[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.subnode_scale_{l} = torch.nn.Parameter(torch.ones(width_out[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.node_bias.append(self.node_bias_{l})')
            exec(f'self.node_scale.append(self.node_scale_{l})')
            exec(f'self.subnode_bias.append(self.subnode_bias_{l})')
            exec(f'self.subnode_scale.append(self.subnode_scale_{l})')
            
        
        self.act_fun = nn.ModuleList(self.act_fun)

        self.grid = grid
        self.k = k
        self.base_fun = base_fun

        ### initializing the symbolic front ###
        self.symbolic_fun = nn.ModuleList()
        for l in range(self.depth):
            sb_batch = Symbolic_KANLayer(in_dim=width_in[l], out_dim=width_out[l+1])
            self.symbolic_fun.append(sb_batch)

        self.symbolic_fun = nn.ModuleList(self.symbolic_fun)
        self.symbolic_enabled = symbolic_enabled
        self.affine_trainable = affine_trainable
        self.sp_trainable = sp_trainable
        self.sb_trainable = sb_trainable
        
        self.save_act = save_act
            
        self.node_scores = None
        self.edge_scores = None
        self.subnode_scores = None
        
        self.cache_data = None
        self.acts = None
        
        self.auto_save = auto_save
        self.state_id = 0
        self.ckpt_path = ckpt_path
        self.round = round
        
        self.device = device
        self.to(device)
        
        if auto_save:
            if first_init:
                if not os.path.exists(ckpt_path):
                    # Create the directory
                    os.makedirs(ckpt_path)
                print(f"checkpoint directory created: {ckpt_path}")
                print('saving model version 0.0')

                history_path = self.ckpt_path+'/history.txt'
                with open(history_path, 'w') as file:
                    file.write(f'### Round {self.round} ###' + '\n')
                    file.write('init => 0.0' + '\n')
                self.saveckpt(path=self.ckpt_path+'/'+'0.0')
            else:
                self.state_id = state_id
            
        self.input_id = torch.arange(self.width_in[0],)

    def log_history(self, method_name, verbose = True): 

        if self.auto_save:

            # save to log file
            #print(func.__name__)
            with open(self.ckpt_path+'/history.txt', 'a') as file:
                file.write(str(self.round)+'.'+str(self.state_id)+' => '+ method_name + ' => ' + str(self.round)+'.'+str(self.state_id+1) + '\n')

            # update state_id
            self.state_id += 1

            # save to ckpt
            self.saveckpt(path=self.ckpt_path+'/'+str(self.round)+'.'+str(self.state_id))
            if verbose:
                print('saving model version '+str(self.round)+'.'+str(self.state_id))

    def saveckpt(self, path='model'):
        '''
        save the current model to files (configuration file and state file)
        
        Args:
        -----
            path : str
                the path where checkpoints are saved

        Returns:
        --------
            None
            
        Example
        -------
        >>> from kan import *
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        >>> model.saveckpt('./mark')
        # There will be three files appearing in the current folder: mark_cache_data, mark_config.yml, mark_state
        '''
    
        model = self
        
        dic = dict(
            width = model.width,
            grid = model.grid,
            k = model.k,
            mult_arity = model.mult_arity,
            base_fun_name = model.base_fun_name,
            symbolic_enabled = model.symbolic_enabled,
            affine_trainable = model.affine_trainable,
            grid_eps = model.grid_eps,
            grid_range = model.grid_range,
            sp_trainable = model.sp_trainable,
            sb_trainable = model.sb_trainable,
            state_id = model.state_id,
            auto_save = model.auto_save,
            ckpt_path = model.ckpt_path,
            round = model.round,
            device = str(model.device),            
        )
        
        for i in range (model.depth):
            dic[f'symbolic.funs_name.{i}'] = model.symbolic_fun[i].funs_name

        ## add DAREK Parameters
        dic['extend' ] = model.extend            
        if 'rand_index' in model.__dir__():            
            
            dic['rand_index'] = convert_to_list(model.rand_index)
            dic['knots']      = convert_to_list(model.knots     )
            dic['samples']    = convert_to_list(model.samples   )            
        ###


        with open(f'{path}_config.yml', 'w') as outfile:
            yaml.dump(dic, outfile, default_flow_style=False)

        torch.save(model.state_dict(), f'{path}_state')
        torch.save(model.cache_data, f'{path}_cache_data')
    
    @staticmethod
    def loadckpt(path='model'):
        '''
        load checkpoint from path
        
        Args:
        -----
            path : str
                the path where checkpoints are saved

        Returns:
        --------
            MultKAN
            
        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, seed=0)
        >>> model.saveckpt('./mark')
        >>> KAN.loadckpt('./mark')
        '''
        with open(f'{path}_config.yml', 'r') as stream:
            config = yaml.safe_load(stream)

        state = torch.load(f'{path}_state')

        model_load = KAND(width=config['width'], 
                     grid=config['grid'], 
                     k=config['k'], 
                     mult_arity = config['mult_arity'], 
                     base_fun=config['base_fun_name'], 
                     symbolic_enabled=config['symbolic_enabled'], 
                     affine_trainable=config['affine_trainable'], 
                     grid_eps=config['grid_eps'], 
                     grid_range=config['grid_range'], 
                     sp_trainable=config['sp_trainable'],
                     sb_trainable=config['sb_trainable'],
                     state_id=config['state_id'],
                     auto_save=config['auto_save'],
                     first_init=False,
                     ckpt_path=config['ckpt_path'],
                     round = config['round']+1,
                     device = config['device'],
                     extend=config['extend'])

        model_load.load_state_dict(state)
        model_load.cache_data = torch.load(f'{path}_cache_data')
        
        depth = len(model_load.width) - 1
        for l in range(depth):
            out_dim = model_load.symbolic_fun[l].out_dim
            in_dim = model_load.symbolic_fun[l].in_dim
            funs_name = config[f'symbolic.funs_name.{l}']
            for j in range(out_dim):
                for i in range(in_dim):
                    fun_name = funs_name[j][i]
                    model_load.symbolic_fun[l].funs_name[j][i] = fun_name
                    model_load.symbolic_fun[l].funs[j][i] = SYMBOLIC_LIB[fun_name][0]
                    model_load.symbolic_fun[l].funs_sympy[j][i] = SYMBOLIC_LIB[fun_name][1]
                    model_load.symbolic_fun[l].funs_avoid_singularity[j][i] = SYMBOLIC_LIB[fun_name][3]
        
        ## add DAREK Parameters        
        # model_load.fk     = config['fk']
        # model_load.f1     = config['f1']
        model_load.extend = config['extend']
        if 'rand_index' in config:
            dict_of_tensors = lambda dict_of_lists: {key: torch.tensor(value) for key, value in dict_of_lists.items()}
            model_load.samples = dict_of_tensors(config['samples'])
            model_load.knots   = dict_of_tensors(config['knots'])
            model_load.rand_index = np.array(config['rand_index'])
        ###
        return model_load
    
    def forward_update_grid(self, x, y, singularity_avoiding=False, y_th=10., reindex =False, seed = 0, method = 'random', index = None):    
        """
        method = 'random' or 'min_dist' or 'Kmean' or 'LHS' or 'custom'
        """
        if (not 'rand_index' in self.__dir__()) or reindex:
            # print('reindex')
            self.update_grid_params =  {
                'singularity_avoiding':singularity_avoiding, 
                'y_th':y_th,
                'reindex':reindex,
                'seed' : seed, 
                'method': method, 
                'index' : None}
            np.random.seed(seed)
            # x = dataset['train_input']
            g = self.grid 
            grid,g_indx = x.sort(dim=0)
            g_indx = g_indx[:,0]            
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
                samples = kmeans.cluster_centers_

                # Find the closest actual dataset points using KD-Tree
                tree = cKDTree(x.detach())
                _, indices = tree.query(kmeans.cluster_centers_, k=1)  # k=1 finds the nearest neighbor
                indx = indices.copy()
                # print(indx)
            
            elif method == 'LHS':
                sampler = qmc.LatinHypercube(len(x.shape))
                lhs_samples = qmc.scale(sampler.random(g+1), x.min(axis = 0), x.max(axis = 0))
                tree = cKDTree(x.detach())
                _, indices = tree.query(lhs_samples, k=1)  # Find closest actual data points
                indx = indices.copy()

            elif method == 'custom':
                indx = index.copy()

            # indx = np.concatenate([indx,np.array([0,x.shape[0]-1])])
            # indx = g_indx[indx].numpy()
            # print(grid.shape, indx.shape, indx, g_indx)
            # print(x.shape, x[g_indx].shape)            
            # grid = torch.gather(x, 0, g_indx.unsqueeze(-1))[indx].clone()
            grid = x[indx].clone()
            y_grid = y[indx].clone()
            # if x.shape[1] == 1:
            #     y_grid = torch.gather(y, 0, g_indx)[indx].clone()
            # if self.extend:
            #     k = self.k
            #     gz = torch.zeros((grid.shape[0]+2*k,grid.shape[1]),dtype=torch.float32)
            #     grid0,indx0 =grid.sort(dim=0)  
            #     gz[k:-k,:] = grid0.clone()  
            #     for i in range(grid.shape[1]):
            #         dg = (grid0[-1,i]-grid0[0,i])/grid0.shape[1]
            #         gz[:k,i] = torch.arange(-k,0) * dg + grid0[0,i]
            #         gz[-k:,i] = torch.arange(1,k+1) * dg + grid0[-1,i]
            #     grid0 = gz
            # grid[0]  -= 1
            # grid[1]  -= 0.5
            # grid[-2] += 0.5
            # grid[-1] += 1
            self.rand_index = indx
            self.knots = {'x': grid.clone(),
                        #   'y_true': y_grid.clone()
                          }
            # if x.shape[1] == 1:
            self.knots['y_true'] = y_grid.clone()
            self.samples = {'x': grid.clone(),
                            'y': y_grid.clone(),
                            'indx': torch.tensor(indx)}
            
            # grid = grid0.clone()
        else:
            grid = self.knots['x'].clone()
            
        x = grid.clone()
        
        # x = x[:,self.input_id.long()]
        # assert x.shape[1] == self.width_in[0]
        
        for l in range(self.depth):
            self.knots[f'x{l}']    = grid.clone().cpu().detach()
            if self.extend:
                k = self.k
                gz = torch.zeros((grid.shape[0]+2*k,grid.shape[1]),dtype=torch.float32)
                grid0,indx0 =grid.sort(dim=0)  
                gz[k:-k,:] = grid0.clone()  
                for i in range(grid.shape[1]):
                    dg = (grid0[-1,i]-grid0[0,i])/grid0.shape[1]
                    gz[:k,i] = torch.arange(-k,0) * dg + grid0[0,i]
                    gz[-k:,i] = torch.arange(1,k+1) * dg + grid0[-1,i]
                grid0 = gz
                
            else:
                grid0,indx0 =grid.sort(dim=0)              
            self.knots[f'indx{l}'] = indx0.clone().cpu().detach()
            self.act_fun[l].grid.data = grid0.T        
            # # x.shape, g.shape
            # print('update-forward',l, x.shape)
            x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x)
            #print(preacts, postacts_numerical, postspline)
            
            if self.symbolic_enabled == True:
                x_symbolic, postacts_symbolic = self.symbolic_fun[l](x, singularity_avoiding=singularity_avoiding, y_th=y_th)
            else:
                x_symbolic = 0.
                postacts_symbolic = 0.

            x = x_numerical + x_symbolic
                    
            # subnode affine transform
            x = self.subnode_scale[l][None,:] * x + self.subnode_bias[l][None,:]
            
            # multiplication
            dim_sum = self.width[l+1][0]
            dim_mult = self.width[l+1][1]
            
            if self.mult_homo == True:
                for i in range(self.mult_arity-1):
                    if i == 0:
                        x_mult = x[:,dim_sum::self.mult_arity] * x[:,dim_sum+1::self.mult_arity]
                    else:
                        x_mult = x_mult * x[:,dim_sum+i+1::self.mult_arity]
                        
            else:
                for j in range(dim_mult):
                    acml_id = dim_sum + np.sum(self.mult_arity[l+1][:j])
                    for i in range(self.mult_arity[l+1][j]-1):
                        if i == 0:
                            x_mult_j = x[:,[acml_id]] * x[:,[acml_id+1]]
                        else:
                            x_mult_j = x_mult_j * x[:,[acml_id+i+1]]
                            
                    if j == 0:
                        x_mult = x_mult_j
                    else:
                        x_mult = torch.cat([x_mult, x_mult_j], dim=1)
                
            if self.width[l+1][1] > 0:
                x = torch.cat([x[:,:dim_sum], x_mult], dim=1)
            
            # x = x + self.biases[l].weight
            # node affine transform
            x = self.node_scale[l][None,:] * x + self.node_bias[l][None,:]
            grid = x.clone()
        self.knots[f'x{self.depth}'] = grid.clone().detach()
        self.knots[f'y_hat'] = grid.clone().detach()
        return x

    def fit(self, dataset, opt="LBFGS", steps=100, log=1, 
            lamb=0., lamb_l1=1., lamb_entropy=2., lamb_coef=0., lamb_coefdiff=0., 
            update_grid=False, grid_update_num=10, loss_fn=None, 
            lr=1., start_grid_update_step=-1, stop_grid_update_step=50, batch=-1,
            metrics=None, save_fig=False, in_vars=None, out_vars=None, beta=3, 
            save_fig_freq=1, img_folder='./video', singularity_avoiding=False, 
            y_th=1000., reg_metric='edge_forward_spline_n', display_metrics=None,
            nonfixknot = False, seed_knots = 0, rand_method = 'random',
            reindex = False, verbose = True, logsave = True, evaluate = True, custom_index = None,
            scheduler="exp", gamma=0.95, step_sch = 100):
        '''
        training

        Args:
        -----
            dataset : dic
                contains dataset['train_input'], dataset['train_label'], dataset['test_input'], dataset['test_label']
            opt : str
                "LBFGS" or "Adam"
            steps : int
                training steps
            log : int
                logging frequency
            lamb : float
                overall penalty strength
            lamb_l1 : float
                l1 penalty strength
            lamb_entropy : float
                entropy penalty strength
            lamb_coef : float
                coefficient magnitude penalty strength
            lamb_coefdiff : float
                difference of nearby coefficits (smoothness) penalty strength
            update_grid : bool
                If True, update grid regularly before stop_grid_update_step
            grid_update_num : int
                the number of grid updates before stop_grid_update_step
            start_grid_update_step : int
                no grid updates before this training step
            stop_grid_update_step : int
                no grid updates after this training step
            loss_fn : function
                loss function
            lr : float
                learning rate
            batch : int
                batch size, if -1 then full.
            save_fig_freq : int
                save figure every (save_fig_freq) steps
            singularity_avoiding : bool
                indicate whether to avoid singularity for the symbolic part
            y_th : float
                singularity threshold (anything above the threshold is considered singular and is softened in some ways)
            reg_metric : str
                regularization metric. Choose from {'edge_forward_spline_n', 'edge_forward_spline_u', 'edge_forward_sum', 'edge_backward', 'node_backward'}
            metrics : a list of metrics (as functions)
                the metrics to be computed in training
            display_metrics : a list of functions
                the metric to be displayed in tqdm progress bar
            scheduler="exp", 'cos', 'dec'
        Returns:
        --------
            results : dic
                results['train_loss'], 1D array of training losses (RMSE)
                results['test_loss'], 1D array of test losses (RMSE)
                results['reg'], 1D array of regularization
                other metrics specified in metrics

        Example
        -------
        >>> from kan import *
        >>> model = KAN(width=[2,5,1], grid=5, k=3, noise_scale=0.3, seed=2)
        >>> f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
        >>> dataset = create_dataset(f, n_var=2)
        >>> model.fit(dataset, opt='LBFGS', steps=20, lamb=0.001);
        >>> model.plot()
        # Most examples in toturals involve the fit() method. Please check them for useness.
        '''

        if lamb > 0. and not self.save_act:
            print('setting lamb=0. If you want to set lamb > 0, set self.save_act=True')
            
        old_save_act, old_symbolic_enabled = self.disable_symbolic_in_fit(lamb)

        if verbose:
            pbar = tqdm(range(steps), desc='description', ncols=100)
        else:
            pbar = range(steps)

        if loss_fn == None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        grid_update_freq = int(stop_grid_update_step / grid_update_num)

        if opt == "Adam":
            optimizer = torch.optim.Adam(self.get_params(), lr=lr)
        elif opt == "LBFGS":
            optimizer = LBFGS(self.get_params(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)

        if scheduler == "exp":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif scheduler == "cos":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_sch)
        elif scheduler == "dec":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_sch, gamma=gamma)
        # else:
        #     raise "scheduler is not defined"

        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        results['reg'] = []
        if metrics != None:
            for i in range(len(metrics)):
                results[metrics[i].__name__] = []

        if batch == -1 or batch > dataset['train_input'].shape[0]:
            batch_size = dataset['train_input'].shape[0]
            batch_size_test = dataset['test_input'].shape[0]
        else:
            batch_size = batch
            batch_size_test = batch

        global train_loss, reg_

        def closure():
            global train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(dataset['train_input'][train_id], singularity_avoiding=singularity_avoiding, y_th=y_th)
            train_loss = loss_fn(pred, dataset['train_label'][train_id])
            if self.save_act:
                if reg_metric == 'edge_backward':
                    self.attribute()
                if reg_metric == 'node_backward':
                    self.node_attribute()
                reg_ = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
            else:
                reg_ = torch.tensor(0.)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
        if nonfixknot:
            self.forward_update_grid(dataset['train_input'],dataset['train_label'],reindex = reindex, seed = seed_knots,
                                     method=rand_method, index=custom_index)

        for _ in pbar:
            
            if _ == steps-1 and old_save_act:
                self.save_act = True
                
            if save_fig and _ % save_fig_freq == 0:
                save_act = self.save_act
                self.save_act = True
            
            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

            if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid and _ >= start_grid_update_step:
                self.update_grid(dataset['train_input'][train_id])

            if opt == "LBFGS":
                optimizer.step(closure)

            if opt == "Adam":
                pred = self.forward(dataset['train_input'][train_id], singularity_avoiding=singularity_avoiding, y_th=y_th)
                train_loss = loss_fn(pred, dataset['train_label'][train_id])
                if self.save_act:
                    if reg_metric == 'edge_backward':
                        self.attribute()
                    if reg_metric == 'node_backward':
                        self.node_attribute()
                    reg_ = self.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
                else:
                    reg_ = torch.tensor(0.)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if scheduler:
                lr_scheduler.step()
                
            if evaluate:
                test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id]), dataset['test_label'][test_id])
            
            
                if metrics != None:
                    for i in range(len(metrics)):
                        results[metrics[i].__name__].append(metrics[i]().item())

                results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
                results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
                results['reg'].append(reg_.cpu().detach().numpy())

                if (_ % log == 0) and verbose:
                    if display_metrics == None:
                        pbar.set_description("| train_loss: %.2e | test_loss: %.2e | reg: %.2e | " % (torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(), reg_.cpu().detach().numpy()))
                    else:
                        string = ''
                        data = ()
                        for metric in display_metrics:
                            string += f' {metric}: %.2e |'
                            try:
                                results[metric]
                            except:
                                raise Exception(f'{metric} not recognized')
                            data += (results[metric][-1],)
                        pbar.set_description(string % data)
                        
                
                if save_fig and _ % save_fig_freq == 0:
                    self.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(_), beta=beta)
                    plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                    plt.close()
                    self.save_act = save_act
				
            if nonfixknot:
                self.forward_update_grid(dataset['train_input'],dataset['train_label'],seed = seed_knots,
                                         method=rand_method)

        if logsave:
            self.log_history('fit', verbose)
        # revert back to original state
        self.symbolic_enabled = old_symbolic_enabled
        return results
