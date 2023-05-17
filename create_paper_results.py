

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import imageio 

import matplotlib.animation as animation
plt.rcParams["animation.html"] = "jshtml"

import importlib

# Own helper function
import dcgan_sup as sup
import dcgan_main as dcgan
import matpy as mp


## Select files from which to compute the images and videos

#experiments = ['phantom','case01','case09','case07','case02']
experiments = ['phantom']


resfilenames = [
'results/phantom3_seed_test___n_ex_20__nepochs_[4000, 4000, 4000, 4000]',
#'results/real_seed_test_countnoise__n_ex_20__nepochs_[4000, 4000, 4000, 4000, 4000]',
#'results/real_case09_seed_test_countnoise__n_ex_20__nepochs_[4000, 4000, 4000, 4000, 4000]',
#'results/real_case07_seed_test_countnoise__n_ex_20__nepochs_[4000, 4000, 4000, 4000, 4000]',
#'results/real_case02_seed_test_countnoise__n_ex_20__nepochs_[4000, 4000, 4000, 4000, 4000]',
]

for expname,fname in zip(experiments,resfilenames):

    print('#################')
    print('Working on ' + expname)

    # Load result    
    res_script = mp.pload(fname)


    # Evaluate errors
    mad = np.median(np.abs(np.median(res_script.err_heart) - res_script.err_heart))
    print('Cardiac error: Median & MAD & Mean & Std-dev')
    print("{:.2e}".format(np.median(res_script.err_heart)) + ' & ' + "{:.2e}".format(mad) + ' & ' +  "{:.2e}".format(res_script.err_heart.mean()) + ' & ' + "{:.2e}".format(res_script.err_heart.std()))

    mad = np.median(np.abs(np.median(res_script.err_breath) - res_script.err_breath))
    print('Breathing error: Median & MAD & Mean & Std-dev')
    print("{:.2e}".format(np.median(res_script.err_breath)) + ' & ' + "{:.2e}".format(mad) + ' & ' +  "{:.2e}".format(res_script.err_breath.mean()) + ' & ' + "{:.2e}".format(res_script.err_breath.std()))

    # Select result with error closest to median of heart error
    med = np.median(res_script.err_heart)
    idx = np.argmin(np.abs(res_script.err_heart - med))

    print('Selected result: ' + str(idx))
    
    res = res_script.results[idx]

    print('Seed: ' + str(res.par.seed))
    
    # Automatic selection of frame (deactivated)
    #bfixf,hfixf= sup.select_frames(res,show=True)
    
    #Get selected frame results
    sup.get_selected_frames(res,bfixf=0,hfixf=0,mode='opt') #mode = 'opt' or 'auto' (experimental)
    
    #Export paper results
    if 'phantom' in expname:
        sup.get_paper_results_phantom(folder='paper/paper_results/' + expname + '/',fname=expname,res=res)
    else:
        sup.get_paper_results(folder='paper/paper_results/' + expname + '/',fname=expname,res=res)


    ###### Create video

    # Create video frames
    out_total = np.concatenate([res.u0,res.u,2.0*np.abs(res.u0-res.u)],axis=2)
    out_heart = np.concatenate([res.u0_heart_selected,res.u_heart_selected,2.0*np.abs(res.u0_heart_selected-res.u_heart_selected)],axis=2)
    out_breathing = np.concatenate([res.u0_breathing_selected,res.u_breathing_selected,2.0*np.abs(res.u0_breathing_selected-res.u_breathing_selected)],axis=2)

    frames = np.concatenate([out_total,out_heart,out_breathing],axis=1)

    if 'phantom' not in expname:
        frames -= frames .min()
        #frames  /= frames .max()*0.5
        frames  /= np.quantile(frames,0.9998)*0.5

    frames  = (255.0*np.clip(frames,0,1)).astype('uint8')
    
    
    from PIL import Image


    imgs = [Image.fromarray(frame) for frame in frames]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save('videos/' + expname + '.gif', save_all=True, append_images=imgs[1:], duration=70, loop=0)

    ####### Create video for optimal seed
    
    res = res_script.results[res_script.err_heart.argmin()]


    #Get selected frame results
    sup.get_selected_frames(res,bfixf=0,hfixf=0,mode='opt') #mode = 'opt' or 'auto' (experimental)


    # Create video frames
    out_total = np.concatenate([res.u0,res.u,2.0*np.abs(res.u0-res.u)],axis=2)
    out_heart = np.concatenate([res.u0_heart_selected,res.u_heart_selected,2.0*np.abs(res.u0_heart_selected-res.u_heart_selected)],axis=2)
    out_breathing = np.concatenate([res.u0_breathing_selected,res.u_breathing_selected,2.0*np.abs(res.u0_breathing_selected-res.u_breathing_selected)],axis=2)

    frames = np.concatenate([out_total,out_heart,out_breathing],axis=1)

    if 'phantom' not in expname:
        frames -= frames .min()
        #frames  /= frames .max()*0.5
        frames  /= np.quantile(frames,0.9998)*0.5

    frames  = (255.0*np.clip(frames,0,1)).astype('uint8')
    
    
    from PIL import Image


    imgs = [Image.fromarray(frame) for frame in frames]
    # duration is the number of milliseconds between frames; this is 40 frames per second
    imgs[0].save('videos/' + expname + '_opt.gif', save_all=True, append_images=imgs[1:], duration=70, loop=0)
        
    
    
