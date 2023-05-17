import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import imageio 

import matplotlib.animation as animation
plt.rcParams["animation.html"] = "jshtml"

# Own helper function
import dcgan_sup as sup

import copy

import torch
import torch.nn as nn
import torch.optim as optim


from matpy import *

# Main reconstruction
def msep(**par_in):


    ########################################################################
    # Set network parameters
    ########################################################################
    par = parameter({})

    mse = nn.MSELoss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #print('Using {} device'.format(device))

    #Architecture pars
    par.n_small_z = 2      # Dimensions of z_1
    par.n_cont = 1           # Dimension of each part of z_1, must be less or equal to par.n_small_z
    par.nframes = 80      # Number of frames
    par.ld_conv = 0 #1e-02


    #Define mode for weight initialization
    par.weight_init = 'ortho' # can be 'ortho', 'normal', 'xavier' or '' for no particular init
    par.dpath = '' # Leave empy for phantom data
    
    par.usecounts = ['breathing'] # define counters to be used: 'breathing', 'heart'    
    
    # Data parameters for real data
    par.freq=4
    par.nhb=3

    par.countnoise = 0 #Optional, add noise on the counter
    par.count_relax = 0 # Optinal, use l2 penaltiy to given count instead of exact count. Only for breathing count
    #Algorithmic pars
    par.nepochs = [9000] #[6000,6000,2000]
    par.show_every = 100
    par.lrs = [0.01]
    par.nit_opt = [5000] # choose optimal loss from last nit_opt iterations, setting this to nepoch chooses the optimal loss over all iterations but requires more computations

    par.show_every = False # false means dont't show anything

    par.seed = False #False means do not set seed
    
    par.opt_static = False
    
    #Parameters for phantom data
    par.ramp = False
    par.breath_acc = 0
    par.freqh = 6
    par.freqb = 4
    

    #Set parameters according to par_in
    par_parse(par_in,[par])
    
    # Set seed
    if par.seed:
        torch.manual_seed(par.seed)
        np.random.seed(par.seed)

    
    # Prepare data
    if not par.dpath:
        print('Using phantom data')
        par.dtype = 'phantom'
    else:
        print('Loading ' + par.dpath)
        par.dtype = 'real'
        data = pload(par.dpath) # expecting dict with ['data'] and ['dims']


    #Define network
    if par.dtype == 'phantom':
        par.ngf = 16           # Internal parameter of the net related to number of internal channels
        par.nz = 64 #50           # Total number of latent space dimensions
        class DCGAN_pytorchsource(nn.Module):
          def __init__(self):
            super(DCGAN_pytorchsource, self).__init__()
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                #torch.nn.Linear(par.nz,par.ngf*8,bias=False),
                nn.ConvTranspose2d( par.nz, par.ngf * 8, 4, 1, 0, bias=False),
                #nn.BatchNorm2d(par.ngf * 8),
                nn.Tanh(),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(par.ngf * 8, par.ngf * 4, 4, 2, 1, bias=False),
                #nn.BatchNorm2d(par.ngf * 4),
                nn.LeakyReLU(),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d( par.ngf * 4, par.ngf * 2, 4, 2, 1, bias=False),
                #nn.BatchNorm2d(par.ngf * 2),
                nn.Tanh(),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d( par.ngf * 2, par.ngf, 4, 2, 1, bias=False),
                #nn.BatchNorm2d(par.ngf),
                nn.LeakyReLU(),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d( par.ngf, 1, 4, 2, 1, bias=False),
                # nn.Sigmoid()
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

          def forward(self, input):
            return self.main(input)

        # Get image data
        img,countb,counth = sup.get_toy_example(nframes=par.nframes,freqh=par.freqh,freqb=par.freqb,ramp=par.ramp,breath_acc=par.breath_acc)
        
            
    elif par.dtype == 'real':
        
        # Define network according to data set
        if ('case01' in par.dpath) or ('case02' in par.dpath):
        
            par.ngf = 20 #64          # Internal parameter of the net related to number of internal channels
            par.nz = 100 #50           # Total number of latent space dimensions
            class DCGAN_pytorchsource(nn.Module):
              def __init__(self):
                super(DCGAN_pytorchsource, self).__init__()
                self.main = nn.Sequential(
                    # input is Z, going into a convolution
                    #torch.nn.Linear(par.nz,par.ngf*8,bias=False),
                    nn.ConvTranspose2d( par.nz, par.ngf * 32, 4, 2, 0, bias=False),
                    #nn.BatchNorm2d(par.ngf * 8),
                    nn.ReLU(True),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(par.ngf * 32, par.ngf * 16, 4, 2, 2, bias=False),
                    #nn.BatchNorm2d(par.ngf * 4),
                    nn.LeakyReLU(),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d( par.ngf * 16, par.ngf * 8, 4, 2, 0, bias=False),
                    #nn.BatchNorm2d(par.ngf * 2),
                    nn.LeakyReLU(),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d( par.ngf * 8, par.ngf * 4, 4, 2, 2, bias=False),
                    #nn.BatchNorm2d(par.ngf),
                    nn.LeakyReLU(),        
                    nn.ConvTranspose2d( par.ngf * 4, par.ngf * 2, 4, 2, 2, bias=False),
                    nn.LeakyReLU(),
                    # state size. (ngf) x 32 x 32
                    nn.ConvTranspose2d( par.ngf * 2, par.ngf, 4, 2, 1, bias=False),
                    # nn.Sigmoid()
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d( par.ngf, 1, 3, 1, 1, bias=False),
                    # nn.Sigmoid()
                    nn.ReLU(True)
                )

              def forward(self, input):
                return self.main(input)

        # Define network according to data set
        elif ('case09' in par.dpath) or ('case07' in par.dpath):
        
            par.ngf = 20 #64          # Internal parameter of the net related to number of internal channels
            par.nz = 100 #50           # Total number of latent space dimensions
            class DCGAN_pytorchsource(nn.Module):
              def __init__(self):
                super(DCGAN_pytorchsource, self).__init__()
                self.main = nn.Sequential(
                    # input is Z, going into a convolution
                    #torch.nn.Linear(par.nz,par.ngf*8,bias=False),
                    nn.ConvTranspose2d( par.nz, par.ngf * 32, 4, 2, 0, bias=False),
                    #nn.BatchNorm2d(par.ngf * 8),
                    nn.ReLU(True),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(par.ngf * 32, par.ngf * 16, 4, 2, 2, bias=False),
                    #nn.BatchNorm2d(par.ngf * 4),
                    nn.LeakyReLU(),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d( par.ngf * 16, par.ngf * 8, 4, 2, 2, bias=False),
                    #nn.BatchNorm2d(par.ngf * 2),
                    nn.LeakyReLU(),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d( par.ngf * 8, par.ngf * 4, 4, 2, 2, bias=False),
                    #nn.BatchNorm2d(par.ngf),
                    nn.LeakyReLU(),        
                    nn.ConvTranspose2d( par.ngf * 4, par.ngf * 2, 4, 2, 2, bias=False),
                    nn.LeakyReLU(),
                    # state size. (ngf) x 32 x 32
                    nn.ConvTranspose2d( par.ngf * 2, par.ngf, 5, 2, 1, bias=False),
                    # nn.Sigmoid()
                    nn.LeakyReLU(),
                    nn.ConvTranspose2d( par.ngf, 1, 4, 1, 1, bias=False),
                    # nn.Sigmoid()
                    nn.ReLU(True)
                )

              def forward(self, input):
                return self.main(input)

        else:
            raise Warning('No network defined for this file')    

        # Get image data
        img,countb = sup.get_breathing(data,freq=par.freq,nhb=par.nhb,countnoise=par.countnoise,breath_acc=par.breath_acc)
        par.nframes = img.shape[0]


    else:
        raise Warning('Unknown data type')



    #Define target
    target = torch.from_numpy(img[:,np.newaxis,:,:]).float().to(device)        

    net = DCGAN_pytorchsource().to(device)

    ########################################################################
    # Employ weight initialization
    ########################################################################



    if par.weight_init == 'normal':
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
        net.apply(weights_init)
        print('Using standard normal distribution for weight initialization')
    elif par.weight_init == 'xavier':
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_normal_(m.weight.data)
        net.apply(weights_init)
        print('Using xavier for weight initialization')
    elif par.weight_init == 'ortho': #Orthogonal initialization works best for phantom data
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.orthogonal_(m.weight.data)
        print('starting orthogonal init')
        net.apply(weights_init)
        print('Using orthogonal for weight initialization')
    else :
        print('Using no particular weight initialization')



    ## Define variables
    #Static
    
    #static_init = np.linspace(0,1,par.nz-par.n_small_z)    
    #net_input_static = torch.from_numpy(static_init[np.newaxis,:,np.newaxis,np.newaxis]).float().to(device)
    #net_input_static.requires_grad = True
    
    net_input_static = torch.rand([1,  par.nz-par.n_small_z, 1, 1],device=device, requires_grad=False)

    
    #Dynamic

    if ('breathing' in par.usecounts) and (par.count_relax == 0):
        net_count = np.expand_dims(countb,(1,2,3))
        net_input_dynamic_1 = torch.from_numpy(net_count).float().to(device)

    elif ('breathing' in par.usecounts) and (par.count_relax >0 ):        
        net_count = np.expand_dims(countb,(1,2,3))
        net_input_dynamic_1 = torch.from_numpy(net_count).float().to(device)        
        net_input_dynamic_1.requires_grad = True
        
        net_input_dynamic_1_data = torch.from_numpy(net_count).float().to(device)
        
    else:
        net_input_dynamic_1 = torch.rand([par.nframes, par.n_cont, 1, 1],device=device, requires_grad=True)

    if 'heart' in par.usecounts:
        net_count = np.expand_dims(counth,(1,2,3))
        net_input_dynamic_2 = torch.from_numpy(net_count).float().to(device)
    else:
        net_input_dynamic_2 = torch.rand([par.nframes, par.n_cont, 1, 1],device=device, requires_grad=True)






    # Define loss
    if not (('breathing' in par.usecounts) and (par.count_relax > 0 )): #standard version
        def mse_loss(net_input_static_a,net_input_dynamic_1,net_input_dynamic_2):

            #Static
            net_input_static_full = net_input_static_a.repeat(par.nframes,1,1,1)
            #Joint
            net_input = torch.cat((net_input_dynamic_1,net_input_dynamic_2,net_input_static_full),1)

            loss = mse(net(net_input),target)
            # + (1e-05)*mse(net_input_dynamic_2,torch.zeros(net_input_dynamic_2.shape,device=device))


            return loss
    else:
        def mse_loss(net_input_static_a,net_input_dynamic_1,net_input_dynamic_2):

            #Static
            net_input_static_full = net_input_static_a.repeat(par.nframes,1,1,1)
            #Joint
            net_input = torch.cat((net_input_dynamic_1,net_input_dynamic_2,net_input_static_full),1)

            loss = mse(net(net_input),target) + par.count_relax*mse(net_input_dynamic_1,net_input_dynamic_1_data)
            # + (1e-05)*mse(net_input_dynamic_2,torch.zeros(net_input_dynamic_2.shape,device=device))


            return loss


    ########################################################################
    # Compute solution
    ########################################################################
    ob_val = np.zeros(sum(par.nepochs))

    epoch_counter = 0 # only necessary to store the obval

    for ii in range(len(par.nepochs)):

        nepoch = par.nepochs[ii]
        lr = par.lrs[ii]
        nit_opt = par.nit_opt[ii]
        
        pars_opt = [x for x in net.parameters()] # parameters to optimize over

        
        if ('breathing' not in par.usecounts) or (par.count_relax):
            print('Optimizing breathing counter')
            pars_opt.append(net_input_dynamic_1)
        if 'heart' not in par.usecounts:            
            print('Optimizing heartbeat counter')
            pars_opt.append(net_input_dynamic_2)

        if par.opt_static:
            pars_opt.append(net_input_static)
            print('Optimizing static counter')


        
        #print('static: ' + str(net_input_static.requires_grad))
        #print('dyn1: ' + str(net_input_dynamic_1.requires_grad))
        #print('dyn2: ' + str(net_input_dynamic_2.requires_grad))
            
        optimizer = optim.Adam(pars_opt, lr=lr,weight_decay = par.ld_conv )
        

        opt_loss = 1e10

        print('Starting optimization...')
        for epoch in range(nepoch):


            # zero the parameter gradients
            optimizer.zero_grad()

            loss = mse_loss(net_input_static,net_input_dynamic_1,net_input_dynamic_2)

            loss.backward()
            optimizer.step()

            # print statistics
            ob_val[epoch_counter + epoch] = loss.item()

            # If we have obtained a better loss, copy the state
            if (loss.item() < opt_loss) and (epoch >= nepoch - nit_opt):
                net_input_static_save = copy.deepcopy(net_input_static)
                net_input_dynamic_1_save = copy.deepcopy(net_input_dynamic_1)
                net_input_dynamic_2_save = copy.deepcopy(net_input_dynamic_2)
                net_save = copy.deepcopy(net.state_dict())

                opt_loss = loss.item()

            if par.show_every and np.remainder(epoch_counter + epoch,par.show_every)==0:
                print('Epoch: ' + str(epoch_counter + epoch) + ', Loss: ' + str(ob_val[epoch_counter + epoch]))

                #Static
                net_input_static_full_tmp = net_input_static.repeat(par.nframes,1,1,1)

                #Joint
                net_input_tmp = torch.cat((net_input_dynamic_1,net_input_dynamic_2,net_input_static_full_tmp),1)

                out_np = net(net_input_tmp).clone().detach().cpu().numpy()
                plt.imshow(out_np[int(0.4*par.nframes),0,:,:])
                plt.show()


        epoch_counter += nepoch


        loss1 = mse_loss(net_input_static,net_input_dynamic_1,net_input_dynamic_2)
        print('Loss of last iterate: ' + str(loss1.item()))
        # Now we set the optimal parameters   
        net_input_static = copy.deepcopy(net_input_static_save)
        net_input_dynamic_1 = copy.deepcopy(net_input_dynamic_1_save)
        net_input_dynamic_2 = copy.deepcopy(net_input_dynamic_2_save)
        net.load_state_dict(copy.deepcopy(net_save))

        loss2 = mse_loss(net_input_static,net_input_dynamic_1,net_input_dynamic_2)
        print('Loss of selected variable: ' + str(loss2.item()))



    ########################################################################
    # Postprocessing
    ########################################################################

    res = output({})
    
    res.par = par
    # System state
    res.net_input_static = net_input_static
    res.net_input_dynamic_1 = net_input_dynamic_1
    res.net_input_dynamic_2 = net_input_dynamic_2    
    res.net = net
    res.net_dict = net.state_dict()

    res.countb = countb
    if par.dtype == 'phantom':
        res.counth = counth

    # Get dynamic network input
    res.net_input_dynamic = torch.cat((net_input_dynamic_1,net_input_dynamic_2),1)
    # Static full static imput
    res.net_input_static_full = net_input_static.repeat(par.nframes,1,1,1)
    # Full input
    res.net_input = torch.cat((res.net_input_dynamic,res.net_input_static_full),1)

    #Reconstructed image sequence
    res.u = res.net(res.net_input).clone().detach().cpu().numpy()[:,0,...]

    #Ground truth data
    res.u0 = img
    res.ob_val = ob_val

    if res.par.dtype == 'real':
        res.data = data

    return res

