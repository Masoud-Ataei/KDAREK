from DAREK import DAREK
import torch
import torch.nn as nn
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm
from kan.LBFGS import *

class LipschitzLinear(nn.Module):
    def __init__(self, in_features, out_features, lipschitz_const = 1.0):
        super().__init__()
        self.lipschitz_const = lipschitz_const
        self.linear = nn.utils.spectral_norm(nn.Linear(in_features, out_features))

    def forward(self, x):
        return self.lipschitz_const * self.linear(x)  # Scale by L_l

class KDAREK(torch.nn.Module):

    def __init__(self, mlp_width = [1,1],
                 kan_width=[1,5,1], 
                 kan_grid=9, 
                 kan_k=3, 
                 kan_base_fun = 'identity', 
                 kan_seed=42, 
                 device='cpu', 
                 L_l = 1.0, symbolic_enabled = False, auto_save = False):
        super(KDAREK, self).__init__()
        self.width_mlp            = mlp_width[:]
        self.width_kan            = kan_width[:]                
        self.kan_grid             = kan_grid 
        self.kan_k                = kan_k 
        self.kan_base_fun         = kan_base_fun 
        self.kan_seed             = kan_seed
        self.device               = device 
        self.L_l                  = L_l        
        self.kan_symbolic_enabled = symbolic_enabled
        self.kan_auto_save        = auto_save

        
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
            model = nn.Sequential()
            for il in range(len(mlp_width)-1):
                model.append(LipschitzLinear(dinp, dout, L_l[il]))
                if il < L - 1:
                    model.append(nn.ReLU())
                    dinp, dout = mlp_width[il+1], mlp_width[il + 2]

            # model.append(LipschitzLinear(mlp_width[L - 1], mlp_width[L], L_l[L-1]))
            mlps.append(model.to(device=device))
        # model.append(torch.nn.Linear())
        self.MLPs = mlps
        if kan_width is not None:
            self.SNNs = DAREK(width=kan_width, grid=kan_grid, k=kan_k, base_fun = kan_base_fun, seed=kan_seed, device='cpu',
                                 symbolic_enabled = symbolic_enabled, auto_save = auto_save)
        else:
            self.SNNs = nn.Identity()

    def predict(self, x0, L_mlp = 1.0, L_k = 1.0, L_1 = 1.0, share = None, noise = 0.0):
        y0   = self.forward_mlps(x0)
        mlpw = np.array(self.width_mlp)
        kanw = np.array(self.width_kan)
        mlpprod = np.sum(mlpw[:-1] * mlpw[1:])
        kanprod = np.sum(kanw[:-1] * kanw[1:])
        depth = len(mlpw[:-1]) + len(kanw[:-1])
        modelprod = mlpprod + kanprod
        # L_k = np.power(L_k/ modelprod,1/depth)
        # L_1 = np.power(L_1/ modelprod,1/depth)
        L_1 = np.power(L_1/ self.d, 1/depth)
        # L_mlp  = (L_1 ** len(mlpw[:-1])) * mlpprod #/ self.d
        L_mlp2 = (L_1 ** len(mlpw[:-1])) #* mlpprod / self.d
        L_1darek = (L_1 ** len(kanw[:-1])) #* kanprod
        # L_kdarek = (L_1 ** len(kanw[:-1])) * kanprod
        L_kdarek = L_k
        
        y1, err_sp = self.SNNs.predict(y0, fk = L_kdarek, f1= L_1darek, share  = share, noise = noise)
        xi0 = self.SNNs.samples['xi'][None,...]
        xt0 = x0[:, None, :]
        ### Error for one mlp per dimension
        err_mlp2 = ((xi0 - xt0).abs().min(axis=1)[0] * L_mlp2).sum(axis = -1).unsqueeze(1).repeat(1,err_sp.shape[1])
        

        # L_mlp = torch.tensor(L_mlp, dtype=torch.float32)
        ### Error for one mlp
        # err_mlp = torch.linalg.norm(xi0 - xt0, axis=-1).min(axis = -1)[0].unsqueeze(1).repeat(1,err_sp.shape[1]) * L_mlp   
        # print('err1:', err_mlp)
        # print('err2:', err_mlp2)

        # L_mlp = torch.tensor(L_mlp, dtype=torch.float32)
        # err_mlp = torch.linalg.norm(xi0 - xt0, axis=-1).min(axis = -1)[0].unsqueeze(1).repeat(1,err_sp.shape[1]) * L_mlp    
        self.darekk_results = (x0, y0, y1)
        return y1, (err_mlp2 * L_1 + err_sp).detach()

    def forward_mlps(self, x):
        return sum([self.MLPs[i](x[:,i].unsqueeze(1)) for i in range(self.d)])
        

    def forward(self, x, singularity_avoiding = False, y_th=1000.):
        x = self.forward_mlps(x)
        x = self.SNNs(x, singularity_avoiding=singularity_avoiding, y_th=y_th)
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

        
        snn = self.SNNs
        if lamb > 0. and not snn.save_act:
            print('setting lamb=0. If you want to set lamb > 0, set snn.save_act=True')
            
        old_save_act, old_symbolic_enabled = snn.disable_symbolic_in_fit(lamb)

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
            if snn.save_act:
                if reg_metric == 'edge_backward':
                    snn.attribute()
                if reg_metric == 'node_backward':
                    snn.node_attribute()
                reg_ = snn.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
            else:
                reg_ = torch.tensor(0.)
            objective = train_loss + lamb * reg_
            objective.backward()
            return objective

        if save_fig:
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)
        if nonfixknot:
            y = self.forward_mlps(dataset['train_input']).detach()
            snn.forward_update_grid(y,dataset['train_label'],reindex = reindex, seed = seed_knots,
                                     method=rand_method, index=custom_index)
            if reindex or not 'xi' in snn.samples:                
                snn.samples['xi'] = dataset['train_input'][snn.samples['indx']]
            self.knots = snn.knots
            self.samples = snn.samples

        for _ in pbar:
            self.train()
            if _ == steps-1 and old_save_act:
                snn.save_act = True
                
            if save_fig and _ % save_fig_freq == 0:
                save_act = snn.save_act
                snn.save_act = True
            
            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

            if _ % grid_update_freq == 0 and _ < stop_grid_update_step and update_grid and _ >= start_grid_update_step:
                snn.update_grid(dataset['train_input'][train_id])

            if opt == "LBFGS":
                optimizer.step(closure)

            if opt == "Adam":
                pred = self.forward(dataset['train_input'][train_id], singularity_avoiding=singularity_avoiding, y_th=y_th)
                train_loss = loss_fn(pred, dataset['train_label'][train_id])
                if snn.save_act:
                    if reg_metric == 'edge_backward':
                        snn.attribute()
                    if reg_metric == 'node_backward':
                        snn.node_attribute()
                    reg_ = snn.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
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
                    snn.plot(folder=img_folder, in_vars=in_vars, out_vars=out_vars, title="Step {}".format(_), beta=beta)
                    plt.savefig(img_folder + '/' + str(_) + '.jpg', bbox_inches='tight', dpi=200)
                    plt.close()
                    snn.save_act = save_act
				
            if nonfixknot:
                self.eval()
                gx = snn.samples['xi']
                gy = self.forward_mlps(gx).detach()
                snn.knots['x']   = gy.clone()
                snn.samples['x'] = gy.clone()
                # gy = selff.den_model(gx)                                
                y = self.forward_mlps(dataset['train_input'])
                snn.forward_update_grid(y,dataset['train_label'],seed = seed_knots,
                                         method=rand_method)

        if logsave:
            snn.log_history('fit', verbose)
        # revert back to original state
        snn.symbolic_enabled = old_symbolic_enabled
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
                    device           = config['device'],
                    L_l              = config['L_l'],
                    symbolic_enabled = config['symbolic_enabled'],
                    auto_save        = config['auto_save']
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