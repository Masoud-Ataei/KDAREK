### Train a model
import GPy
import sys
import datetime
import yaml
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from .kand import KAND as KAN
class LipschitzLinear(nn.Module):
    def __init__(self, in_features, out_features, lipschitz_const):
        super().__init__()
        self.lipschitz_const = lipschitz_const
        self.linear = nn.utils.spectral_norm(nn.Linear(in_features, out_features))

    def forward(self, x):
        return self.lipschitz_const * self.linear(x)  # Scale by L_l
    
class KKAN(torch.nn.Module):

    def __init__(self, inp = [1,1],
                 kan_width=[1,5,1], 
                 kan_grid=9, 
                 kan_k=3, 
                 kan_base_fun = 'identity', 
                 kan_seed=42, 
                 device='cpu', 
                 L_l = 1.0, symbolic_enabled = False, auto_save = False):
        super(KKAN, self).__init__()
        self.width_mlp            = inp[:]
        self.width_kan            = kan_width[:]                
        self.kan_grid             = kan_grid 
        self.kan_k                = kan_k 
        self.kan_base_fun         = kan_base_fun 
        self.kan_seed             = kan_seed
        self.device               = device 
        self.L_l                  = L_l        
        self.kan_symbolic_enabled = symbolic_enabled
        self.kan_auto_save        = auto_save

        model = nn.Sequential()
        L = len(inp) -1
        if isinstance(L_l, (int, float)):
            L_l = [L_l for i in range(L)]
        else:
            assert isinstance(L_l, list), "L_l should be scalar or list size of MLP"
            assert len(L_l) == L , "L_l should be scalar or list size of MLP"
        
        for i in range(len(inp)-2):
            # model.append(torch.nn.Linear(inp[i], inp[i+1]))            
            # model.append(torch.nn.ReLU())

            # model.append(nn.utils.spectral_norm(nn.Linear(inp[i], inp[i + 1])))  # Add spectral norm
            model.append(LipschitzLinear(inp[i], inp[i + 1], L_l[i]))
            model.append(nn.ReLU())

        # model.append(nn.utils.spectral_norm(nn.Linear(inp[L - 1], inp[L])))  # Add spectral norm to last layer

        model.append(LipschitzLinear(inp[L - 1], inp[L], L_l[L-1]))
        # model.append(torch.nn.Linear())
        self.den_model = model.to(device=device)
        if kan_width is not None:
            self.kan_model = KAN(width=kan_width, grid=kan_grid, k=kan_k, base_fun = kan_base_fun, seed=kan_seed, device='cpu',
                                 symbolic_enabled = symbolic_enabled, auto_save = auto_save)
        else:
            self.kan_model = nn.Identity()

    def DAREKK(self, x0, L_mlp = 1.0, L_k = 1.0, L_1 = 1.0, share = None):
        y0   = self.den_model(x0)
        mlpw = np.array(self.width_mlp)
        kanw = np.array(self.width_kan)
        mlpprod = np.sum(mlpw[:-1] * mlpw[1:])
        kanprod = np.sum(kanw[:-1] * kanw[1:])
        depth = len(mlpw[:-1]) + len(kanw[:-1])
        modelprod = mlpprod + kanprod
        L_k = np.power(L_k/ modelprod,1/depth)
        L_1 = np.power(L_1/ modelprod,1/depth)
        L_mlp = (L_1 ** len(mlpw[:-1])) * mlpprod
        L_1kan = (L_1 ** len(kanw[:-1])) * kanprod
        L_kkan = (L_1 ** len(kanw[:-1])) * kanprod
        
        
        y1, err_sp = self.kan_model.DAREK_torch(y0, fk = L_kkan, f1= L_1kan, share  = share)
        xi0 = self.kan_model.samples['xi'][None,...]
        xt0 = x0[:, None, :]
        # L_mlp = torch.tensor(L_mlp, dtype=torch.float32)
        err_mlp = torch.linalg.norm(xi0 - xt0, axis=-1).min(axis = -1)[0].unsqueeze(1).repeat(1,err_sp.shape[1]) * L_mlp    
        self.darekk_results = (x0, y0, y1)
        return y1, (err_mlp + L_1 * err_sp)

    def forward(self, x, singularity_avoiding = False, y_th=1000.):
        x = self.den_model(x)
        x = self.kan_model(x, singularity_avoiding=singularity_avoiding, y_th=y_th)
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
        """
        scheduler="exp", 'cos', 'dec'
        """

        
        kan = self.kan_model
        if lamb > 0. and not kan.save_act:
            print('setting lamb=0. If you want to set lamb > 0, set kan.save_act=True')
            
        old_save_act, old_symbolic_enabled = kan.disable_symbolic_in_fit(lamb)

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
            # optimizer = torch.optim.Adam([selff.den_model.parameters(),kan.get_params()], lr=lr)
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)            
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
        # breakpoint()
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
            if kan.save_act:
                if reg_metric == 'edge_backward':
                    kan.attribute()
                if reg_metric == 'node_backward':
                    kan.node_attribute()
                reg_ = kan.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
            else:
                reg_ = torch.tensor(0.)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
        if nonfixknot:
            y = self.den_model(dataset['train_input']).detach()
            kan.forward_update_grid(y,dataset['train_label'],reindex = reindex, seed = seed_knots,
                                     method=rand_method, index=custom_index)
            if reindex or not 'xi' in kan.samples:                
                kan.samples['xi'] = dataset['train_input'][kan.samples['indx']]
            self.knots = kan.knots
            self.samples = kan.samples

        for _ in pbar:
            self.train()
            if _ == steps-1 and old_save_act:
                kan.save_act = True
                
            if save_fig and _ % save_fig_freq == 0:
                save_act = kan.save_act
                kan.save_act = True
            
            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

            if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid and _ >= start_grid_update_step:
                kan.update_grid(dataset['train_input'][train_id])

            if opt == "LBFGS":
                optimizer.step(closure)

            if opt == "Adam":
                pred = self.forward(dataset['train_input'][train_id], singularity_avoiding=singularity_avoiding, y_th=y_th)
                train_loss = loss_fn(pred, dataset['train_label'][train_id])
                if kan.save_act:
                    if reg_metric == 'edge_backward':
                        kan.attribute()
                    if reg_metric == 'node_backward':
                        kan.node_attribute()
                    reg_ = kan.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
                else:
                    reg_ = torch.tensor(0.)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if scheduler:
                lr_scheduler.step()

            if evaluate:
                self.eval()
                test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id]), dataset['test_label'][test_id])
                
                
                if metrics != None:
                    for i in range(len(metrics)):
                        results[metrics[i].__name__].append(metrics[i]().item())

                results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
                results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())
                results['reg'].append(reg_.cpu().detach().numpy())

                if (_ % log == 0) and verbose:
                    if display_metrics == None:
                        pbar.set_description("LR: %.2e | train_loss: %.2e | test_loss: %.2e | reg: %.2e | " % (optimizer.param_groups[0]['lr'],torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(), reg_.cpu().detach().numpy()))
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
                    kan.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(_), beta=beta)
                    plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                    plt.close()
                    kan.save_act = save_act
				
            if nonfixknot:
                self.eval()
                gx = kan.samples['xi']
                gy = self.den_model(gx).detach()
                kan.knots['x']   = gy.clone()
                kan.samples['x'] = gy.clone()
                # gy = selff.den_model(gx)                                
                y = self.den_model(dataset['train_input'])
                kan.forward_update_grid(y,dataset['train_label'],seed = seed_knots,
                                         method=rand_method)

        if logsave:
            kan.log_history('fit', verbose)
        # revert back to original state
        kan.symbolic_enabled = old_symbolic_enabled
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
            device            = model.device,
            L_l               = model.L_l,
            symbolic_enabled  = model.kan_symbolic_enabled,
            auto_save         = model.kan_auto_save
        )
        model.kan_model.saveckpt(path + '_kan')
        torch.save(model.den_model.state_dict(), f'{path}_state')
        with open(f'{path}_config.yml', 'w') as outfile:
            yaml.dump(dic, outfile, default_flow_style=False)

    @staticmethod
    def loadckpt(path='model'):
        with open(f'{path}_config.yml', 'r') as stream:
            config = yaml.safe_load(stream)
        state_mlp = torch.load(f'{path}_state')
        # state_kan = torch.load(f'{path}_kan_state')
        model = KKAN(inp = config['width_mlp'],
                    kan_width        = config['width_kan'],
                    kan_grid         = config['kan_grid'],
                    kan_k            = config['kan_k'],
                    kan_base_fun     = config['kan_base_fun'],
                    kan_seed         = config['kan_seed'],
                    device           = config['device'],
                    L_l              = config['L_l'],
                    symbolic_enabled = config['symbolic_enabled'],
                    auto_save        = config['auto_save']
                    )
        
        model.den_model.load_state_dict(state_mlp)
        model.kan_model = model.kan_model.loadckpt(path + '_kan')
        if 'rand_index' in model.kan_model.__dir__():
            # dict_of_tensors = lambda dict_of_lists: {key: torch.tensor(value) for key, value in dict_of_lists.items()}
            model.samples = model.kan_model.samples
            model.knots   = model.kan_model.knots
            # model.rand_index = np.array(config['rand_index'])
        model.den_model.eval()
        return model


class MLP(torch.nn.Module):

    def __init__(self, inp = [1,1],                  
                 device='cpu'):
        super(MLP, self).__init__()
        model = nn.Sequential()
        L = len(inp) -1
        for i in range(len(inp)-2):
            model.append(torch.nn.Linear(inp[i], inp[i+1]))
            model.append(torch.nn.ReLU())
        model.append(torch.nn.Linear(inp[L-1], inp[L]))
        # model.append(torch.nn.Linear())
        self.den_model = model.to(device=device)        

    def forward(self, x):
        x = self.den_model(x)        
        return x
    
    def fit(self, dataset, steps=100, loss_fn=None, 
            lr=0.01, opt = "Adam", scheduler="exp", gamma=0.95, step_sch = 100):        
        nlog = 10

        pbar = tqdm(range(steps), desc='description', ncols=100)

        if loss_fn == None:
            loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
        else:
            loss_fn = loss_fn_eval = loss_fn

        if opt == "Adam":
            # optimizer = torch.optim.Adam([selff.den_model.parameters(),self.get_params()], lr=lr)
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        if scheduler == "exp":
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        elif scheduler == "cos":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_sch)
        elif scheduler == "dec":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_sch, gamma=gamma)
        # else:
        #     raise "scheduler is not defined"

        for _ in pbar:
                        
            train_id = np.random.choice(dataset['train_input'].shape[0], dataset['train_input'].shape[0], replace=False)
            test_id = np.random.choice (dataset['test_input'].shape[0], dataset['test_input'].shape[0], replace=False)


            if opt == "Adam":
                pred = self.forward(dataset['train_input'][train_id])
                train_loss = loss_fn(pred, dataset['train_label'][train_id])
                loss = train_loss #+ lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if scheduler:
                lr_scheduler.step()

            test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id]), dataset['test_label'][test_id])
            if _ % nlog == 0:
                pbar.set_description("LR: %.2e | train_loss: %.2e | test_loss: %.2e " % (optimizer.param_groups[0]['lr'], torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy()))


class Ensemble_MLP(torch.nn.Module):
    def __init__(self, inp = [1,10,1], nens = 10, device = 'cpu'):
        super().__init__()
        self.inp = inp
        self.dev = device
        self.nens = nens
        self.Ens_mlp = [MLP(inp = inp, device = device) for i in range(nens)]        

    def forward(self, x, portion = 3.0):
        out = torch.stack([model(x) for model in self.Ens_mlp])        
        yhat = torch.mean(out, axis = 0)
        std  = torch.std(out, axis = 0)
        return yhat, std * portion

    def predict(self, x, portion = 3.0):
        out = torch.stack([model(x) for model in self.Ens_mlp])        
        yhat = torch.mean(out, axis = 0)
        std  = torch.std(out, axis = 0)
        return yhat, std * portion

    def fit(self, dataset, opt = "Adam", lr = 0.01, steps=100, scheduler="exp", gamma=0.95, step_sch = 100):
        for mlp in self.Ens_mlp:
            mlp.fit(dataset, opt=opt, lr = lr, steps=steps, scheduler=scheduler, gamma=gamma, step_sch = step_sch )        

    def saveckpt(self, path='model'):
        dic = dict(
            inp         = self.inp,
            nens        = self.nens,
            dev         = self.dev,            
        )
        torch.save([model.state_dict() for model in self.Ens_mlp], path + "_state")
        with open(f'{path}_config.yml', 'w') as outfile:
            yaml.dump(dic, outfile, default_flow_style=False)

    @staticmethod
    def loadckpt(path='model'):
        with open(f'{path}_config.yml', 'r') as stream:
            config = yaml.safe_load(stream)
        
        inp  = config['inp']
        nens = config['nens']
        dev  = config['dev']
        model = Ensemble_MLP(inp=inp, nens=nens, device = dev)
        state = torch.load(f'{path}_state')
        for i in range(nens):
            model.Ens_mlp[i].load_state_dict(state[i])
        return model
    
class GPs():
    def __init__(self, inp = 1, variance=1.0, lengthscale=1.0):
        super().__init__()
        self.inp    = inp
        self.var    = variance
        self.lscale = lengthscale
        self.kernel = GPy.kern.RBF(input_dim=inp, variance=variance, lengthscale=lengthscale,inv_l= True)
        
    def __call__(self, x, portion = 3.0):
        yhat, var = self.gpy.predict(x)
        yhat, var = yhat.flatten(),var.flatten()
        std = np.sqrt(var)
        return yhat, std * portion


    def train(self, xt, yt):
        self.gpy = GPy.models.GPRegression(xt, yt, self.kernel)
        self.gpy.optimize()

    def fit(self, dataset, ):
        xt,yt = dataset['train_input'].numpy(), dataset['train_label'].numpy()
        x ,y  = dataset['test_input' ].numpy(), dataset['test_label' ].numpy()
        self.gpy = GPy.models.GPRegression(xt, yt, self.kernel)
        self.gpy.optimize()

    def saveckpt(self, path='model'):
        dic = dict(
            inp         = self.inp,
            var         = self.var,
            lscale      = self.lscale,            
        )        
        self.gpy.save_model(path + "_state", compress=True, save_data=True)
        with open(f'{path}_config.yml', 'w') as outfile:
            yaml.dump(dic, outfile, default_flow_style=False)

    @staticmethod
    def loadckpt(path='model'):
        with open(f'{path}_config.yml', 'r') as stream:
            config = yaml.safe_load(stream)
        
        inp    = config['inp']
        var    = config['var']
        lscale = config['lscale']
        model = GPs(inp=inp, variance=var, lengthscale=lscale)
        model.gpy = GPy.models.GPRegression.load_model(path + "_state.zip")
        return model

class Ensemble_KAN(torch.nn.Module):
    def __init__(self, width=[1,5,1], nens = 10, grid=7, k=3, base_fun = 'identity', seed=None, device='cpu',
                symbolic_enabled = False, auto_save = False, extend = False):
        super().__init__()
        self.width = width
        self.nens  = nens 
        self.grid  = grid
        self.k     = k
        self.base_fun = base_fun
        self.seed     = seed
        self.device   = device
        self.symbolic_enabled = symbolic_enabled
        self.auto_save        = auto_save
        self.extend = extend
        
        if seed is None:
            seeds = np.random.randint(0, 1_000_000, size=nens)
            self.modelkan = [KAN(width=width, grid=grid, k=k, base_fun = base_fun, seed=seeds[i], device=device,
                symbolic_enabled = symbolic_enabled, auto_save = auto_save, extend=extend) for i in range(nens)]
        else:
            self.modelkan = [KAN(width=width, grid=grid, k=k, base_fun = base_fun, seed=seed, device=device,
                symbolic_enabled = symbolic_enabled, auto_save = auto_save, extend=extend) for i in range(nens)]

    def __call__(self, x, portion = 3.0):
        out = torch.stack([model(x) for model in self.modelkan])        
        yhat = torch.mean(out, axis = 0)
        std  = torch.std(out, axis = 0)
        return yhat, std * portion

    def predict(self, x, portion = 3.0):
        with torch.no_grad():
            out = torch.stack([model(x) for model in self.modelkan])        
        yhat = torch.mean(out, axis = 0)
        std  = torch.std(out, axis = 0)
        return yhat, std * portion

    def fit(self, dataset, opt = "Adam", lr = 0.01, steps=100, scheduler="exp", gamma=0.95, step_sch = 100,
            nonfixknot=True, verbose = False, logsave = False,
            evaluate=False, lamb=0.1,lamb_coef=0.1,seed_knots=0,
            rand_method = 'random',reindex = False, custom_index = None,
            loss_fn=None):
            loss = [self.modelkan[i].fit(dataset, opt = opt, steps = steps, 
                    nonfixknot=nonfixknot, verbose = verbose, logsave = logsave ,
                    evaluate=evaluate, lamb=lamb,lamb_coef=lamb_coef,
                    seed_knots=seed_knots, scheduler=scheduler, gamma=gamma, step_sch = step_sch,
                    rand_method=rand_method, reindex=reindex, custom_index=custom_index, 
                    loss_fn=loss_fn, lr=lr )  for i in range(self.nens)]
            return loss               
    def saveckpt(self, path='model'):
        dic = dict(
            width            = self.width           , 
            nens             = self.nens            ,
            grid             = self.grid            , 
            k                = self.k               , 
            base_fun         = self.base_fun        , 
            seed             = self.seed            , 
            device           = self.device          , 
            symbolic_enabled = self.symbolic_enabled, 
            auto_save        = self.auto_save       ,            
            extend           = self.extend          ,
        )
        with open(f'{path}_config.yml', 'w') as outfile:
            yaml.dump(dic, outfile, default_flow_style=False)
        
        for i in range(self.nens):
            self.modelkan[i].saveckpt(path + f"_{i}_")

    @staticmethod
    def loadckpt(path='model'):
        with open(f'{path}_config.yml', 'r') as stream:
            config = yaml.safe_load(stream)
        
        width            = config['width']
        nens             = config['nens']
        grid             = config['grid']
        k                = config['k']
        base_fun         = config['base_fun']
        seed             = config['seed']
        device           = config['device']
        symbolic_enabled = config['symbolic_enabled']
        auto_save        = config['auto_save']
        extend           = config['extend']

        model = Ensemble_KAN(width=width, nens=nens, grid=grid,k=k,
                             base_fun=base_fun, seed = seed, device=device,
                             symbolic_enabled=symbolic_enabled, auto_save=auto_save,
                             extend=extend)
        
        for i in range(nens):
            model.modelkan[i] = model.modelkan[i].loadckpt(path + f"_{i}_")
        return model

def Dataset(fx, n = 4, fix = True, a  = -2*np.pi, b  =  2*np.pi, seed = None, noise = 0.0):
    if not (seed is None):
        np.random.seed(seed)
    # Ti = np.linspace(-2*np.pi,2*np.pi,n)
    # a  = -2*np.pi
    # b  =  2*np.pi
    # Ti = a + b - (-a+b) * np.cos((2*j - 1) * np.pi / 2 / n) / 2
    if fix:
        Ti = np.linspace(a,b,n)
        Ttest = np.linspace(a,b,1000)
    else:
        Ti    = np.random.uniform(a,b,n)
        Ttest = np.random.uniform(a,b,1000)
    # Ti.sort()
    Yi    = fx(Ti)
    Ytest = fx(Ttest)
    Yi    = Yi    + noise * np.random.randn(*Yi.shape)
    Ytest = Ytest + noise * np.random.randn(*Ytest.shape)

    # train the model
    device='cpu' # cpu or cuda
    dataset = {}
    dataset['train_input'] = torch.tensor(Ti    , dtype=torch.float32).reshape((-1,1)).to(device)
    dataset['test_input']  = torch.tensor(Ttest , dtype=torch.float32).reshape((-1,1)).to(device)

    dataset['train_label'] = torch.tensor(Yi    , dtype=torch.float32).reshape((-1,1)).to(device)
    dataset['test_label']  = torch.tensor(Ytest , dtype=torch.float32).reshape((-1,1)).to(device)
    return dataset
