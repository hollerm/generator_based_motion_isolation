import numpy as np
import copy

# Own helper function
import dcgan_sup as sup
import matpy as mp
import dcgan_main as dcgan





# Container for output
res_script = mp.output({})
par = mp.parameter({})

# Loop over different seeds
par.n_ex = 20 # number of experiments
par.nepochs = [4000 for i in range(5)]
par.lrs=[0.01,0.008,0.005,0.003,0.001]
par.nit_opt=[4000 for i in range(5)]

par.dpath = 'data/case01D'

par.countnoise = 0.5
par.count_relax = 500.0

err_tot = np.zeros(par.n_ex)
err_breath = np.zeros(par.n_ex)
err_heart = np.zeros(par.n_ex)


res_script.par = par

res_script.results = []

for cc in range(par.n_ex):

    print('########################################')
    #res = dcgan.msep(seed=220,dpath='data/case01D',nepochs=[6000,6000,6000,6000],lrs=[0.01,0.005,0.003,0.001],nit_opt=[6000,3000,3000,6000],nhb=3)

    res = dcgan.msep(seed=cc+1,dpath=par.dpath,weight_init='',nepochs=par.nepochs,lrs=par.lrs,nit_opt=par.nit_opt,nhb=3,countnoise=par.countnoise,count_relax=par.count_relax)

    ## Evaluate errors and store it in res
    sup.eval_error(res)

    print('----------------------------------------')
    print('Seed: ' + str(cc+1))
    print('Total error: ' + str(res.total_error))
    print('Optimal/ mean breathing error: ' + str(res.breathing_error.min()) + ' / ' + str(res.breathing_error.sum()/res.par.nframes))
    print('Optimal/ mean heart error: ' + str(res.heart_error.min()) + ' / ' + str(res.heart_error.sum()/res.par.nframes))
    print('########################################')
    
    # Export result
    res.net = 0 #cannot store network, but network dict will be saved
    res_script.results.append(res)
        
        
    # Store error    
    err_tot[cc] = res.total_error
    err_breath[cc] = res.breathing_error.min()
    err_heart[cc] = res.heart_error.min()

    # Save result (also in between iterations in case the script fails)
    res_script.err_tot = err_tot
    res_script.err_breath = err_breath
    res_script.err_heart = err_heart

    res_script.save(folder='results',fname='real_seed_test_countnoise',outpars=['n_ex','nepochs'])
