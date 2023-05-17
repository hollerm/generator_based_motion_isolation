


import numpy as np
import copy

# Own helper function
import dcgan_sup as sup
import matpy as mp
import dcgan_main as dcgan
import os

# Container for output
res_script = mp.output({})
par = mp.parameter({})

par.nepochs = [4000,4000,4000,4000]
par.lrs=[0.01,0.005,0.001,0.0005]
par.nit_opt=[4000,4000,4000,4000]

       
print('########################################')
res = dcgan.msep(seed=13,nepochs=par.nepochs,lrs=par.lrs,nit_opt=par.nit_opt,ramp=True,breath_acc=0.1,weight_init='') 

## Evaluate errors and store it in res
sup.eval_error(res)

print('----------------------------------------')
print('Total error: ' + str(res.total_error))
print('Optimal/ mean breathing error: ' + str(res.breathing_error.min()) + ' / ' + str(res.breathing_error.sum()/res.par.nframes))
print('Optimal/ mean heart error: ' + str(res.heart_error.min()) + ' / ' + str(res.heart_error.sum()/res.par.nframes))

# Store result
res.net = 0 #cannot store network, but network dict will be saved
res.save(folder='demo_results',fname='phantom3_seed_test_',outpars=['nepochs'])

print('########################################')
## Creating images from the result    
###################################

#Get selected frame results
sup.get_selected_frames(res,bfixf=0,hfixf=0,mode='opt')

#Export paper results
expname = 'phantom'
sup.get_paper_results_phantom(folder='demo_results/images/',fname=expname,res=res)


###### Create video

# Create video frames
out_total = np.concatenate([res.u0,res.u,2.0*np.abs(res.u0-res.u)],axis=2)
out_heart = np.concatenate([res.u0_heart_selected,res.u_heart_selected,2.0*np.abs(res.u0_heart_selected-res.u_heart_selected)],axis=2)
out_breathing = np.concatenate([res.u0_breathing_selected,res.u_breathing_selected,2.0*np.abs(res.u0_breathing_selected-res.u_breathing_selected)],axis=2)

frames = np.concatenate([out_total,out_heart,out_breathing],axis=1)

frames  = (255.0*np.clip(frames,0,1)).astype('uint8')

from PIL import Image


imgs = [Image.fromarray(frame) for frame in frames]
# duration is the number of milliseconds between frames; this is 40 frames per second
if not os.path.isdir('demo_results/videos'):
        os.mkdir('demo_results/videos')
imgs[0].save('demo_results/videos/' + expname + '.gif', save_all=True, append_images=imgs[1:], duration=70, loop=0)




