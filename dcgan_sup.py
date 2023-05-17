
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

import matplotlib.animation as animation
plt.rcParams["animation.html"] = "jshtml"

from scipy.ndimage import gaussian_filter


import os


from matpy import *
import torch.nn as nn


# Function to define toy example
# -1 means to not fix any value of the motion, otherwise it will be fixed to the respective frame
def get_toy_example(nframes=80,fixed_heart_frame = -1,fixed_breathing_frame = -1,freqh=6,freqb=4,ramp=False,breath_acc = 0):

    img = np.zeros((nframes,64,64))
    
    #Make base sheared circle
    def make_circle(a_size,cx,cy,radius,sigma=0.5,ramp=False):
      a = np.zeros((a_size, a_size)).astype('uint8')
      # cx, cy = 32, 32 # The center of circle
      # radius = 16
      y, x = np.ogrid[-radius: radius+1, -radius: radius+1]
      index = x**2 + y**2 <= radius**2
      vals = np.ones(a.shape)*255
      if ramp:
          for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    vals[i,j] = 170*(i/vals.shape[0])*(j/vals.shape[1]) + 120

      a[cy-radius:(cy+radius+1), cx-radius:cx+radius+1][index] = vals[cy-radius:(cy+radius+1), cx-radius:cx+radius+1][index]
      base_circle = gaussian_filter(a, sigma)/255.0
      return base_circle
    
    background = make_circle(64,32,32,29,sigma=0.5,ramp=ramp)

    countb = np.zeros(nframes)
    counth = np.zeros(nframes)

    #Set motion
    for frame in range(nframes):

        # Set heart counter        
        if fixed_heart_frame == -1:
            t = frame/float(nframes-1)
        else: 
            t = fixed_heart_frame/float(nframes-1)
        counth[frame] = (np.abs(np.sin(freqh*np.pi*0.5*t)))             
        
        # Set breathing counter
        if fixed_breathing_frame == -1:
            t = frame/float(nframes-1)
        else: 
            t = fixed_breathing_frame/float(nframes-1)


        countb[frame] = np.abs(np.sin(freqb*np.pi*0.5*t*np.power(1+t,breath_acc)))           
            
        # beating heart
        heart0 = make_circle(64,24,32,6,sigma=0.5,ramp=False)
        heart1 = make_circle(64,24,32,7,sigma=0.5,ramp=False)    
        heart2 = make_circle(64,24,32,8,sigma=0.5,ramp=False)

        heart = heart0+(min(counth[frame],0.5)/0.5)*(heart1-heart0)+((max(counth[frame]-0.5,0))/0.5)*(heart2-heart1)

        #Shearing transform    
        pts1 = np.float32([[8+16,8+16],[8+16,16+16],[16+16,8+16],[16+16,16+16]])
        pts2 = np.float32([[8+16,8+1.5*countb[frame]+16],[8+16,16+1.5*countb[frame]+16],[16+16,8+16],[16+16,16+16]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        warped = cv2.warpPerspective(background+heart,M,(64,64))
        
        img[frame,:,:] = warped
    
    img = img/img.max()  # normalize data

    return img,countb,counth

# Function to define toy example
# -1 means to not fix any value of the motion, otherwise it will be fixed to the respective frame
def get_perfusion_example(nframes=80,fixed_heart_frame = -1,fixed_breathing_frame = -1,freqh=6,freqb=4,ramp=False,breath_acc = 0):

    img = np.zeros((nframes,64,64))
    
    #Make base sheared circle
    def make_circle(a_size,cx,cy,radius,sigma=0.5,ramp=False):
      a = np.zeros((a_size, a_size)).astype('uint8')
      # cx, cy = 32, 32 # The center of circle
      # radius = 16
      y, x = np.ogrid[-radius: radius+1, -radius: radius+1]
      index = x**2 + y**2 <= radius**2
      vals = np.ones(a.shape)*255
      if ramp:
          for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    vals[i,j] = 170*(i/vals.shape[0])*(j/vals.shape[1]) + 120

      a[cy-radius:(cy+radius+1), cx-radius:cx+radius+1][index] = vals[cy-radius:(cy+radius+1), cx-radius:cx+radius+1][index]
      base_circle = gaussian_filter(a, sigma)/255.0
      return base_circle
    
    background = make_circle(64,32,32,29,sigma=0.5,ramp=ramp)

    countb = np.zeros(nframes)
    counth = np.zeros(nframes)

    #Set motion
    for frame in range(nframes):

        # Set heart counter        
        if fixed_heart_frame == -1:
            t = frame/float(nframes-1)
        else: 
            t = fixed_heart_frame/float(nframes-1)
        counth[frame] = (np.abs(np.sin(freqh*np.pi*0.5*t)))             
        
        # Set breathing counter
        if fixed_breathing_frame == -1:
            t = frame/float(nframes-1)
        else: 
            t = fixed_breathing_frame/float(nframes-1)


        countb[frame] = np.abs(np.sin(freqb*np.pi*0.5*t*np.power(1+t,breath_acc)))           
            
        # beating heart
        heart0 = make_circle(64,24,38,6,sigma=0.5,ramp=False)
        heart1 = make_circle(64,36,22,4,sigma=0.5,ramp=False)    




        heart = 0.7*counth[frame]*heart0 + 0.4*counth[frame]*heart1

        #Shearing transform    
        pts1 = np.float32([[8+16,8+16],[8+16,16+16],[16+16,8+16],[16+16,16+16]])
        pts2 = np.float32([[8+16,8+1.5*countb[frame]+16],[8+16,16+1.5*countb[frame]+16],[16+16,8+16],[16+16,16+16]])
        M = cv2.getPerspectiveTransform(pts1,pts2)
        warped = cv2.warpPerspective(background+heart,M,(64,64))
        
        img[frame,:,:] = warped
    
    img = img/img.max()  # normalize data

    return img,countb,counth




# Get breathing motion. fixed_frame =-1 means do not fix the frame,
# otherwise it should be the number of the frame to be fixed
# freq is frequency of sine
# nhb is the nuber of heartbeats

#Note: Different parameters for the perspecitve transform can also be entered as vairables!
#Note countnoise adds noise on the position where the counter is sampled. value is the noise in percentage of one timestep
def get_breathing(data,fixed_heart_frame = -1,fixed_breathing_frame = -1,freq=4,nhb=3,countnoise=0,fulldims=False,breath_acc = 0):

            (N,M,nframes_orig) = data['data'].shape
            
            nframes = nhb*nframes_orig
            
            img = np.zeros((nframes,N,M))

            bcount = np.zeros(nframes)  # breathing count
            
            #Load shear parameters if available
            if 'shear' in data.keys():
                shear = data['shear']
            else:
                shear = (0,0.4) #default value for case 01
            
            #Set motion
            for frame in range(nframes):

                #Global counter
                t = frame/float(nframes-1)
                    

                if fixed_breathing_frame == -1:
                    bcount[frame] = np.abs(np.sin(freq*np.pi*0.5*t*np.power(1+t,breath_acc)))   
                else:
                    t0 = fixed_breathing_frame/float(nframes-1)
                    bcount[frame] = np.abs(np.sin(freq*np.pi*0.5*t0*np.power(1+t0,breath_acc)))   
            
                #Shearing transform    
                pts1 = np.float32([[8+16,8+16],[8+16,16+16],[16+16,8+16],[16+16,16+16]])
                pts2 = np.float32([[8+shear[0]*bcount[frame]+16,8+shear[1]*bcount[frame]+16],[8+shear[0]*bcount[frame]+16,16+shear[1]*bcount[frame]+16],[16+16,8+16],[16+16,16+16]])

                trans = cv2.getPerspectiveTransform(pts1,pts2)
                
                if fixed_heart_frame == -1:
                    warped = cv2.warpPerspective(data['data'][:,:,frame%nframes_orig],trans,(M,N))
                else: 
                    warped = cv2.warpPerspective(data['data'][:,:,fixed_heart_frame%nframes_orig],trans,(M,N))
            
                img[frame,:,:] = warped

            if not fulldims:
                dims = data['dims']
                img = img[:,dims[0]:dims[1],dims[2]:dims[3]]

            if countnoise:
                tmpcount = np.zeros(nframes)

                #Set motion
                for frame in range(nframes):

                    #Global counter
                    t = frame/float(nframes-1) + np.random.normal(scale=countnoise/float(nframes-1.0))
                    
                    tmpcount[frame] = np.abs(np.sin(freq*np.pi*0.5*t*np.power(1+t,breath_acc)))   


                bcount = tmpcount
                
            return img,bcount

        
        
        
def get_breathing_osci(data,fixed_heart_frame = -1,fixed_breathing_frame = -1,freq=4,nhb=3,countnoise=0,fulldims=False):

            (N,M,nframes_orig) = data['data'].shape
            
            nframes = nhb*nframes_orig
            
            img = np.zeros((nframes,N,M))

            bcount = np.zeros(nframes)  # breathing count
            
            #Load shear parameters if available
            if 'shear' in data.keys():
                shear = data['shear']
            else:
                shear = (0,0.4) #default value for case 01
            
            #Set motion
            for frame in range(nframes):

                #Global counter
                t = frame/float(nframes-1)
                    

                if fixed_breathing_frame == -1:
                    bcount[frame] = np.random.normal(scale=0.1)#np.abs(np.sin(freq*np.pi*0.5*t))   
                else:
                    t0 = fixed_breathing_frame/float(nframes-1)
                    bcount[frame] = np.abs(np.sin(freq*np.pi*0.5*t0))   
            
                #Shearing transform    
                pts1 = np.float32([[8+16,8+16],[8+16,16+16],[16+16,8+16],[16+16,16+16]])
                pts2 = np.float32([[8+shear[0]*bcount[frame]+16,8+shear[1]*bcount[frame]+16],[8+shear[0]*bcount[frame]+16,16+shear[1]*bcount[frame]+16],[16+16,8+16],[16+16,16+16]])

                trans = cv2.getPerspectiveTransform(pts1,pts2)
                
                if fixed_heart_frame == -1:
                    warped = cv2.warpPerspective(data['data'][:,:,frame%nframes_orig],trans,(M,N))
                else: 
                    warped = cv2.warpPerspective(data['data'][:,:,fixed_heart_frame%nframes_orig],trans,(M,N))
            
                img[frame,:,:] = warped

            if not fulldims:
                dims = data['dims']
                img = img[:,dims[0]:dims[1],dims[2]:dims[3]]

            if countnoise:
                tmpcount = np.zeros(nframes)

                #Set motion
                for frame in range(nframes):

                    #Global counter
                    t = frame/float(nframes-1) + np.random.normal(scale=countnoise/float(nframes-1.0))
                    
                    tmpcount[frame] = np.abs(np.sin(freq*np.pi*0.5*t))   


                bcount = tmpcount
                
            return img,bcount
        
        
        
# comps provides components NOT to be fixed
def get_fixed_frame_result(net,net_input_static_full,net_input_dynamic,comps = [0],fixed_frame = 0):

    nframes = net_input_dynamic.shape[0]
    n_small_z = net_input_dynamic.shape[1]


    #Dynamic partially fixed
    #Components to be fixed
    ncomps = [i for i in range(n_small_z) if i not in comps]


    net_input_dynamic_fixed = net_input_dynamic[fixed_frame,ncomps,...].repeat(nframes,1,1,1)
    net_input_dynamic_rem = net_input_dynamic[:,comps,...]

    net_input_combined = torch.zeros(net_input_dynamic.shape,device=net_input_dynamic.device)
    
    net_input_combined[:,ncomps,...] = net_input_dynamic_fixed
    net_input_combined[:,comps,...] = net_input_dynamic_rem


    #Joint
    net_input_tmp = torch.cat((net_input_combined,net_input_static_full),1)
    
    
    return net(net_input_tmp).clone().detach().cpu().numpy()[:,0,...]
    

# Evaluate error and store it in result
def eval_error(res):


    nframes = res.par.nframes
    par = res.par
    
    # Total number of pixels for error normalization
    NN = np.prod(res.u.shape)
    
    if res.par.dtype == 'phantom':
        u0 = get_toy_example(nframes=par.nframes,freqh=par.freqh,freqb=par.freqb,ramp=par.ramp,breath_acc=par.breath_acc)[0]
    else:
        u0 = get_breathing(res.data,freq=res.par.freq,nhb=res.par.nhb,breath_acc=res.par.breath_acc)[0]

        
    # Total error
    res.total_error = np.sqrt(np.square(res.u - u0).sum()/np.square(u0).sum())
    
    
    # Error with heart motion fixed
    res.breathing_error = np.zeros(nframes)
    for f in range(nframes):
        
        if res.par.dtype == 'phantom':
            b_img = get_toy_example(nframes=par.nframes,freqh=par.freqh,freqb=par.freqb,ramp=par.ramp,breath_acc=par.breath_acc,fixed_heart_frame = f)[0]
        else:
            b_img = get_breathing(res.data,freq=res.par.freq,nhb=res.par.nhb,fixed_heart_frame=f,breath_acc=res.par.breath_acc)[0]
                
        rec_b_img = get_fixed_frame_result(res.net,res.net_input_static_full,res.net_input_dynamic,comps = [0],fixed_frame = f)
        
        res.breathing_error[f] = np.sqrt(np.square(rec_b_img-b_img).sum()/np.square(b_img).sum())

    # Error with breathing fixed
    res.heart_error = np.zeros(nframes)
    for f in range(nframes):

        if res.par.dtype == 'phantom':
            h_img = get_toy_example(nframes=par.nframes,freqh=par.freqh,freqb=par.freqb,ramp=par.ramp,breath_acc=par.breath_acc,fixed_breathing_frame = f)[0]
        else:
            h_img = get_breathing(res.data,freq=res.par.freq,nhb=res.par.nhb,fixed_breathing_frame=f,breath_acc=res.par.breath_acc)[0]
        
        rec_h_img = get_fixed_frame_result(res.net,res.net_input_static_full,res.net_input_dynamic,comps = [1],fixed_frame = f)
        
        res.heart_error[f] = np.sqrt(np.square(rec_h_img-h_img).sum()/np.square(h_img).sum())


    # Store results for optimal frame

    res.u_breathing = get_fixed_frame_result(res.net,res.net_input_static_full,res.net_input_dynamic,comps = [0],fixed_frame = res.breathing_error.argmin())

    res.u_heart = get_fixed_frame_result(res.net,res.net_input_static_full,res.net_input_dynamic,comps = [1],fixed_frame = res.heart_error.argmin())


    if res.par.dtype == 'phantom':
        res.u0_breathing = get_toy_example(nframes=par.nframes,freqh=par.freqh,freqb=par.freqb,ramp=par.ramp,breath_acc=par.breath_acc,fixed_heart_frame = res.breathing_error.argmin())[0]
        res.u0_heart = get_toy_example(nframes=par.nframes,freqh=par.freqh,freqb=par.freqb,ramp=par.ramp,breath_acc=par.breath_acc,fixed_breathing_frame = res.heart_error.argmin())[0]
    elif res.par.dtype == 'real':
        res.u0_breathing = get_breathing(res.data,freq=res.par.freq,nhb=res.par.nhb,fixed_heart_frame=res.breathing_error.argmin(),breath_acc=res.par.breath_acc)[0]
        res.u0_heart = get_breathing(res.data,freq=res.par.freq,nhb=res.par.nhb,fixed_breathing_frame=res.heart_error.argmin(),breath_acc=res.par.breath_acc)[0]
        

def animate_result(img,figsize=(10,10)):

    fig = plt.figure(figsize=figsize)
    plot_out = plt.imshow(img[0,:,:])
    def update_rec(i):
        plot_out.set_data(img[i,:,:]) 
        return plot_out
    ani_rec = animation.FuncAnimation(fig, update_rec, frames=img.shape[0])     

    
    return ani_rec




def select_frames(res,eps=1e-04,show=False,avg_window=3):

    #avg_window=3 # window for taking the local variance

    # Get counters
    var = res.net_input_dynamic.detach().cpu().numpy()[...,0,0]
    bc = var[:,0]
    hc = var[:,1]
    
    # Normalize
    if 1:
        bc -= bc.min()
        bc /= bc.max()
        hc -= hc.min()
        hc /= hc.max()


    # Get local variance
    
    bc_var = np.zeros(bc.shape)
    hc_var = np.zeros(hc.shape)
    
    for pos in range(avg_window,bc.shape[0]-avg_window):
        bc_mean = bc[pos-avg_window:pos+avg_window+1].mean()
        hc_mean = hc[pos-avg_window:pos+avg_window+1].mean()
    
        bc_var[pos] = np.square(bc_mean - bc[pos-avg_window:pos+avg_window+1]).sum()
        hc_var[pos] = np.square(hc_mean - hc[pos-avg_window:pos+avg_window+1]).sum()
        
    bfix_err_tmp = bc_var - hc_var
    hfix_err_tmp = hc_var - bc_var
    
    bfix_err = np.zeros(bfix_err_tmp.shape)
    hfix_err = np.zeros(hfix_err_tmp.shape)
    
    for pos in range(avg_window,bc.shape[0]-avg_window):
    
        # Get index of similar values
        bs_vals = np.where(np.abs(bc - bc[pos])<eps)[0].tolist()
        hs_vals = np.where(np.abs(hc - hc[pos])<eps)[0].tolist()
        
        bfix_err[pos] = bfix_err_tmp[bs_vals].sum()
        hfix_err[pos] = hfix_err_tmp[hs_vals].sum()

        
    bfixf = bfix_err[avg_window:-avg_window].argmin() + avg_window
    hfixf = hfix_err[avg_window:-avg_window].argmin() + avg_window


    # Scale variance to range of errors for plotting
    
    bfix_err -= bfix_err[avg_window:-avg_window].min()
    bfix_err = (res.heart_error.max() - res.heart_error.min())*bfix_err/bfix_err[avg_window:-avg_window].max() + res.heart_error.min()
    
    hfix_err -= hfix_err[avg_window:-avg_window].min()
    hfix_err = (res.breathing_error.max() - res.breathing_error.min())*hfix_err/hfix_err[avg_window:-avg_window].max() + res.breathing_error.min()



    #Show result
    if show:
        fig = plt.figure()    
        plt.plot(bfix_err,'-D',markevery=[bfixf])
        plt.plot(res.heart_error)
        plt.legend(['estimated error','true error'],loc='upper right')
        plt.title('Error for fixing the breathing frame')


    #Print error
    print(' #### Estimation of optimal breathing frame to fix: ######')
    print('Optimal error: ' + str(res.heart_error.min()))
    print('Acieved error: ' + str(res.heart_error[bfixf]))

    print('Fixed frame: ' + str(bfixf))


    #Show result    
    if show:
        fig = plt.figure()
        plt.plot(hfix_err,'-D',markevery=[hfixf])
        plt.plot(res.breathing_error)
        plt.legend(['estimated error','true error'],loc='upper right')
        plt.title('Error for fixing the heart frame')
    

    #Print error
    print(' #### Estimation of optimal breathing frame to fix: ######')
    print('Optimal error: ' + str(res.breathing_error.min()))
    print('Acieved error: ' + str(res.breathing_error[hfixf]))


    print('Fixed frame: ' + str(hfixf))
    

    if show:
        fig = plt.figure()
        
        bc -= bc.min()
        bc /= bc.max()
        
        hc -= hc.min()
        hc /= hc.max()

        
        plt.plot(bc,'-D',markevery=[bfixf,hfixf])
        plt.plot(hc,'-D',markevery=[bfixf,hfixf])

        plt.legend(['breathing counter','heart counter'],loc='upper right')
        plt.title('Counters')

    
    return bfixf,hfixf

def select_frames_working(res,eps=1e-04,show=False):

    avg_window=3 # window for taking the local variance

    # Get counters
    var = res.net_input_dynamic.detach().cpu().numpy()[...,0,0]
    bc = var[:,0]
    hc = var[:,1]

    # Get local variance
    
    bc_var = np.zeros(bc.shape)
    hc_var = np.zeros(hc.shape)
    
    for pos in range(avg_window,bc.shape[0]-avg_window):
    
        bc_var[pos] = np.square(bc[pos] - bc[pos-avg_window:pos+avg_window+1]).sum()
        hc_var[pos] = np.square(hc[pos] - hc[pos-avg_window:pos+avg_window+1]).sum()
        
    bfix_err_tmp = bc_var - hc_var
    hfix_err_tmp = hc_var - bc_var
    
    bfix_err = np.zeros(bfix_err_tmp.shape)
    hfix_err = np.zeros(hfix_err_tmp.shape)
    
    for pos in range(avg_window,bc.shape[0]-avg_window):
    
        # Get index of similar values
        bs_vals = np.where(np.abs(bc - bc[pos])<eps)[0].tolist()
        hs_vals = np.where(np.abs(hc - hc[pos])<eps)[0].tolist()
        
        bfix_err[pos] = bfix_err_tmp[bs_vals].sum()
        hfix_err[pos] = hfix_err_tmp[hs_vals].sum()

        
    bfixf = bfix_err[avg_window:-avg_window].argmin() + avg_window
    hfixf = hfix_err[avg_window:-avg_window].argmin() + avg_window


    # Scale variance to range of errors for plotting
    
    bfix_err -= bfix_err[avg_window:-avg_window].min()
    bfix_err = (res.heart_error.max() - res.heart_error.min())*bfix_err/bfix_err[avg_window:-avg_window].max() + res.heart_error.min()
    
    hfix_err -= hfix_err[avg_window:-avg_window].min()
    hfix_err = (res.breathing_error.max() - res.breathing_error.min())*hfix_err/hfix_err[avg_window:-avg_window].max() + res.breathing_error.min()



    #Show result
    if show:
        fig = plt.figure()    
        plt.plot(bfix_err,'-D',markevery=[bfixf])
        plt.plot(res.heart_error)
        plt.legend(['estimated error','true error'],loc='upper right')
        plt.title('Error for fixing the breathing frame')


    #Print error
    print(' #### Estimation of optimal breathing frame to fix: ######')
    print('Optimal error: ' + str(res.heart_error.min()))
    print('Acieved error: ' + str(res.heart_error[bfixf]))

    print('Fixed frame: ' + str(bfixf))


    #Show result    
    if show:
        fig = plt.figure()
        plt.plot(hfix_err,'-D',markevery=[hfixf])
        plt.plot(res.breathing_error)
        plt.legend(['estimated error','true error'],loc='upper right')
        plt.title('Error for fixing the heart frame')
    

    #Print error
    print(' #### Estimation of optimal breathing frame to fix: ######')
    print('Optimal error: ' + str(res.breathing_error.min()))
    print('Acieved error: ' + str(res.breathing_error[hfixf]))


    print('Fixed frame: ' + str(hfixf))
    

    if show:
        fig = plt.figure()
        
        bc -= bc.min()
        bc /= bc.max()
        
        hc -= hc.min()
        hc /= hc.max()

        
        plt.plot(bc,'-D',markevery=[bfixf,hfixf])
        plt.plot(hc,'-D',markevery=[bfixf,hfixf])

        plt.legend(['breathing counter','heart counter'],loc='upper right')
        plt.title('Counters')

    
    return bfixf,hfixf


def select_frames_tmp(res,eps=1e-04,show=False):

    avg_window=5 # window for averaging the gradient

    # Get counters
    var = res.net_input_dynamic.detach().cpu().numpy()[...,0,0]
    bc = var[:,0]
    hc = var[:,1]

    #Gradient of counters
    bc_grad = np.gradient(bc)
    hc_grad = np.gradient(hc)
    #Max of gradient
    mbcgrad = np.abs(bc_grad).max()
    mhcgrad = np.abs(hc_grad).max()

    #Get breathing frame to fix
    opt_pos = -1
    mxs = np.zeros(res.heart_error.shape)

    for pos in range(avg_window,bc.shape[0]-avg_window):

        svals = np.where(np.abs(bc - bc[pos])<eps)[0].tolist()
        

        svals = [ val for val in svals if (val < bc.shape[0]-avg_window) and (val >= avg_window)]
        #svals = [ val for val in svals]

        
        mxs[pos] = (np.abs(bc_grad[svals]/mbcgrad) - np.abs(hc_grad[svals]/mhcgrad)).sum()
        for i in range(avg_window):
            
            vp = [val+i for val in svals]
            vm = [val-i for val in svals]
            
            mxs[pos] += (np.abs(bc_grad[vp]/mbcgrad) - np.abs(hc_grad[vp]/mhcgrad)).sum()
            mxs[pos] += (np.abs(bc_grad[vm]/mbcgrad) - np.abs(hc_grad[vm]/mhcgrad)).sum()

        
    #Get frame to fix
    bfixf = mxs[avg_window:-avg_window].argmin() + avg_window

    # Scale mxs to range of errors for plotting
    mxs = mxs - mxs[avg_window:-avg_window].min()
    mxs = (res.heart_error.max() - res.heart_error.min())*mxs/mxs[avg_window:-avg_window].max() + res.heart_error.min()



    #Show result
    if show:
        fig = plt.figure()    
        plt.plot(mxs,'-D',markevery=[bfixf])
        plt.plot(res.heart_error)
        plt.legend(['estimated error','true error'],loc='upper right')
        plt.title('Error for fixing the breathing frame')


    #Print error
    print(' #### Estimation of optimal breathing frame to fix: ######')
    print('Optimal error: ' + str(res.heart_error.min()))
    print('Acieved error: ' + str(res.heart_error[bfixf]))

    print('Fixed frame: ' + str(bfixf))


    #Get heart frame to fix
    opt_pos = -1
    mxs = np.zeros(res.breathing_error.shape)

    for pos in range(bc.shape[0]):

        svals = np.where(np.abs(hc - hc[pos])<eps)
        mxs[pos] = (np.abs(hc_grad[svals]/mhcgrad) - np.abs(bc_grad[svals]/mbcgrad)).sum()


    # Scale mxs to range of errors
    mxs = mxs - mxs.min()
    mxs = (res.breathing_error.max() - res.breathing_error.min())*mxs/mxs.max() + res.breathing_error.min()
        
    #Get frame to fix
    hfixf = mxs[1:-1].argmin() + 1


    #Show result    
    if show:
        fig = plt.figure()
        plt.plot(mxs,'-D',markevery=[hfixf])
        plt.plot(res.breathing_error)
        plt.legend(['estimated error','true error'],loc='upper right')
        plt.title('Error for fixing the heart frame')
    

    #Print error
    print(' #### Estimation of optimal breathing frame to fix: ######')
    print('Optimal error: ' + str(res.breathing_error.min()))
    print('Acieved error: ' + str(res.breathing_error[hfixf]))


    print('Fixed frame: ' + str(hfixf))
    

    if show:
        fig = plt.figure()
        
        bc -= bc.min()
        bc /= bc.max()
        
        hc -= hc.min()
        hc /= hc.max()

        
        plt.plot(bc,'-D',markevery=[bfixf,hfixf])
        plt.plot(hc,'-D',markevery=[bfixf,hfixf])

        plt.legend(['breathing counter','heart counter'],loc='upper right')
        plt.title('Counters')

    
    return bfixf,hfixf


def get_selected_frames(res,bfixf,hfixf,mode='opt'):

    #Possible modes: 'opt': choose optimal frame, 'auto': select frame based on couters (experimental)


    if mode=='auto':
        # Redefine model to evaluate on selected frame
        par = parameter({})
        device = 'cuda' if torch.cuda.is_available() else 'cpu'    
        if res.par.dtype == 'phantom':
            
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
        else:
            if ('case01' in res.par.dpath) or ('case02' in res.par.dpath):
            
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

            elif ('case09' in res.par.dpath) or ('case07' in res.par.dpath):
            
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

        # Load network parameters from result
        net = DCGAN_pytorchsource().to(device)
        net.load_state_dict(res.net_dict)
        res.net = net


        # comps provides components NOT to be fixed
        res.u_heart_selected = get_fixed_frame_result(res.net,res.net_input_static_full,res.net_input_dynamic,comps = [1],fixed_frame = bfixf)

        res.u_breathing_selected = get_fixed_frame_result(res.net,res.net_input_static_full,res.net_input_dynamic,comps = [0],fixed_frame = hfixf)
        
        



        if res.par.dtype == 'phantom':
            nframes = res.u0.shape[0]
            res.u0_breathing_selected = get_toy_example(nframes=nframes,fixed_heart_frame = hfixf)[0]
            res.u0_heart_selected = get_toy_example(nframes=nframes,fixed_breathing_frame = bfixf)[0]
        elif res.par.dtype == 'real':
            res.u0_heart_selected = get_breathing(res.data,freq=res.par.freq,nhb=res.par.nhb,fixed_breathing_frame=bfixf)[0]
            
            res.u0_breathing_selected = get_breathing(res.data,freq=res.par.freq,nhb=res.par.nhb,fixed_heart_frame=hfixf)[0]


    elif mode=='opt':
    
        print('Fixing the optimal frame....')
        res.u_heart_selected = res.u_heart
        res.u_breathing_selected = res.u_breathing
        
        res.u0_heart_selected = res.u0_heart
        res.u0_breathing_selected = res.u0_breathing


        
def get_paper_results(fname,res,folder='images'):


    outfname = folder + fname

    # Select indices for gridding
    grid = np.zeros(res.u0[0,...].shape).astype('bool')
    if ('case01' in res.par.dpath) or ('case02' in res.par.dpath):

        grid[13:80,11] = 1
        grid[13:80,32] = 1
        grid[13,11:85] = 1
        grid[50,11:85] = 1
        grid[80,11:85] = 1        
        grid[13:80,85] = 1
        
        slc = np.s_[:,:,38]
        
        swap = True
        
        if 'case01' in fname:
            scalefac = 0.5
        else:
            scalefac = 0.3
            
    elif ('case07' in res.par.dpath) or ('case09' in res.par.dpath):

        grid[5,17:52] = 1
        grid[25,17:52] = 1
        grid[60,17:52] = 1
        grid[5:60,17]= 1
        grid[5:60,34] = 1        
        grid[5:61,52] = 1
        
        slc = np.s_[:,22,:]
        
        swap = False
        if 'case09' in fname:
            scalefac = 0.3
        else:
            scalefac = 0.4
    
    else:
        raise Warning('Unknown case of real data')

    #Generate folder if necessary
    if folder:
        if not os.path.exists(folder):
            os.makedirs(folder)


    ##############################################
    #Store video frames
    u0 = copy.deepcopy(res.u0)
    u_heart = copy.deepcopy(res.u_heart_selected)
    err = np.abs(u_heart - res.u0_heart_selected)
    #Scaling    
    mn = u0.min()
    rfac = scalefac*(u0.max()-u0.min())

    u0 = (u0 - mn)/rfac
    u_heart = (u_heart - mn)/rfac
    
    err /= rfac*scalefac
    
    for frame in range(u0.shape[0]):
        
        # Save ground truth
        img = np.stack([u0[frame,...],u0[frame,...],u0[frame,...]],axis=2)
        img[grid,0] = 1
        img[grid,1] = 0
        img[grid,2] = 0
        
        imsave(outfname + '_u0_f' + str(frame) + '.png',img)
        

        # Save only hear motion
        img = np.stack([u_heart[frame,...],u_heart[frame,...],u_heart[frame,...]],axis=2)
        img[grid,0] = 1
        img[grid,1] = 0
        img[grid,2] = 0
        
        imsave(outfname + '_u_heart_f' + str(frame) + '.png',img)
    
        # Save error images
        img = np.stack([err[frame,...],err[frame,...],err[frame,...]],axis=2)
        img[grid,0] = 1
        img[grid,1] = 0
        img[grid,2] = 0
        
        imsave(outfname + '_err_f' + str(frame) + '.png',img)
    
    
    
    ##############################################
    #Store slices
    u0_hslc = (res.u0_heart_selected[slc] - mn)/rfac
    u_hslc = (res.u_heart_selected[slc] - mn)/rfac
    
    u0_bslc = (res.u0_breathing_selected[slc] - mn)/rfac
    u_bslc = (res.u_breathing_selected[slc] - mn)/rfac
    
    err_hslc = 2.0*np.abs(u0_hslc-u_hslc)
    err_bslc = 2.0*np.abs(u0_bslc-u_bslc)
    
    
    u0_slc = u0[slc]
    
    u0 = np.stack([u0[0,...],u0[0,...],u0[0,...]],axis=2)
    u0[slc[1:] + (0,)] = 1
    u0[slc[1:] + (1,)] = 0
    u0[slc[1:] + (2,)] = 0

    #Swap axis for visualization if necessary
    if swap:
    
        u0_hslc = np.swapaxes(u0_hslc,0,1)
        u0_bslc = np.swapaxes(u0_bslc,0,1)
        u0_slc = np.swapaxes(u0_slc,0,1)
        u_hslc = np.swapaxes(u_hslc,0,1)
        u_bslc = np.swapaxes(u_bslc,0,1)
        err_hslc = np.swapaxes(err_hslc,0,1)
        err_bslc = np.swapaxes(err_bslc,0,1)

    imsave(outfname + '_u0_hslc.png',u0_hslc)
    imsave(outfname + '_u_hslc.png',u_hslc)
    
    imsave(outfname + '_u0_bslc.png',u0_bslc)
    imsave(outfname + '_u_bslc.png',u_bslc)
    
    imsave(outfname + '_u0_slc.png',u0_slc)
    
    imsave(outfname + '_err_hslc.png',err_hslc)
    imsave(outfname + '_err_bslc.png',err_bslc)
    
    imsave(outfname + '_u0_slmarked.png',u0)
    

    
    ##############################################
    #Store plots

    #Counters
    var = res.net_input_dynamic.detach().cpu().numpy()[...,0,0]
       
    fig = plt.figure(figsize=(6,4))

    plt.rcParams['font.size'] = '16'

    plt.title('Respiratory motion trigger')
    plt.plot(var[:,0])

    print(fname)
    if res.par.countnoise:
    
        img_tmp,countnoise_tmp = get_breathing(res.data,freq=res.par.freq,nhb=res.par.nhb,countnoise=res.par.countnoise)
        
        plt.plot(countnoise_tmp)
        
        plt.legend(['Reconstructed','Data'],loc='lower right')

    plt.savefig(outfname + '_fig_b_trigger' + '.pdf',format='pdf')    

    fig = plt.figure(figsize=(6,4))

    plt.title('Cardiac motion trigger')
    plt.plot(var[:,1])

    plt.savefig(outfname + '_fig_h_trigger' + '.pdf',format='pdf')    


    fig = plt.figure(figsize=(6,4))



    plt.plot(res.ob_val)

    #ax.set_yscale('log')    
    
    plt.yscale('log')

    plt.title('Loss over epochs (logarithmic scale)')


    plt.savefig(outfname + '_loss' + '.pdf',format='pdf')    





def get_paper_results_phantom(fname,res,folder='images'):


    outfname = folder + fname

        
    slc = np.s_[:,:,20]
    scalefac = 1.0

    #Generate folder if necessary
    if folder:
        if not os.path.exists(folder):
            os.makedirs(folder)


    ##############################################
    # Prepare color map
    cmap = plt.cm.viridis
    #norm = plt.Normalize(vmin=vid.min(), vmax=vid.max())
    

    ##############################################
    #Store video frames
    u0 = copy.deepcopy(res.u0)
    u = copy.deepcopy(res.u)

    for frame in range(u0.shape[0]):
        
        # Save ground truth
        imsave(outfname + '_u0_f' + str(frame) + '.png',cmap(u0[frame,...]))
        
    
    
    ##############################################
    #Store slices of reconstruction
    u0_hslc = res.u0_heart_selected[slc]
    u_hslc = res.u_heart_selected[slc]
    
    u0_bslc = res.u0_breathing_selected[slc]
    u_bslc = res.u_breathing_selected[slc]
    
    err_hslc = 10.0*np.abs(u0_hslc-u_hslc)
    err_bslc = 10.0*np.abs(u0_bslc-u_bslc)
    
    u0_slc = res.u0[slc]
    u_slc = res.u[slc]


    u0_hslc = np.swapaxes(u0_hslc,0,1)
    u0_bslc = np.swapaxes(u0_bslc,0,1)
    u_hslc = np.swapaxes(u_hslc,0,1)
    u_bslc = np.swapaxes(u_bslc,0,1)
    err_hslc = np.swapaxes(err_hslc,0,1)
    err_bslc = np.swapaxes(err_bslc,0,1)
    
    u0_slc = np.swapaxes(u0_slc,0,1)
    u_slc = np.swapaxes(u_slc,0,1)

    
    u0 = cmap(u0[0,...])
    u = cmap(u[0,...])

    u0[slc[1:] + (0,)] = 1
    u0[slc[1:] + (1,)] = 0
    u0[slc[1:] + (2,)] = 0

    u[slc[1:] + (0,)] = 1
    u[slc[1:] + (1,)] = 0
    u[slc[1:] + (2,)] = 0


    imsave(outfname + '_u0_hslc.png',cmap(u0_hslc))
    imsave(outfname + '_u_hslc.png',cmap(u_hslc))
    
    imsave(outfname + '_u0_bslc.png',cmap(u0_bslc))
    imsave(outfname + '_u_bslc.png',cmap(u_bslc))
    
    imsave(outfname + '_err_hslc.png',cmap(err_hslc))
    imsave(outfname + '_err_bslc.png',cmap(err_bslc))
    
    imsave(outfname + '_u0_sl.png',cmap(u0_slc))
    imsave(outfname + '_u_sl.png',cmap(u_slc))
    
    imsave(outfname + '_u0_slmarked.png',u0)
    imsave(outfname + '_u_slmarked.png',u)
    

    
    ##############################################
    #Store plots

    #Get true motion triggers
    img_tmp,countb_tmp,counth_tmp = get_toy_example(nframes=res.par.nframes,freqh=res.par.freqh,freqb=res.par.freqb,ramp=res.par.ramp,breath_acc=res.par.breath_acc)
    
    
    #Plot true motion triggers
    fig = plt.figure(figsize=(12,5.5))
    
    plt.rcParams['font.size'] = '15'
    
    
    (ax1, ax2) = fig.subplots(1, 2)

    ax1.set_title('Respiratory motion trigger')
    ax1.plot(countb_tmp)

    ax2.set_title('Cardiac motion trigger')
    ax2.plot(counth_tmp)

    plt.savefig(outfname + '_fig_true_triggers' + '.pdf',format='pdf')    
    
    
    #Plot reconstructed heart motion trigger
    var = res.net_input_dynamic.detach().cpu().numpy()[...,0,0]
       
    fig = plt.figure(figsize=(6,4))

    plt.title('Cardiac motion trigger')
    
    tmp = var[:,1]
    #Rescale to [0,1]
    tmp -= tmp.min()
    if np.abs(tmp.max())>1e-08:
        tmp /= tmp.max()
    
    plt.plot(tmp)

        
    plt.plot(counth_tmp)
        
    plt.legend(['Reconstructed','True'],loc='lower right')

    plt.savefig(outfname + '_fig_b_trigger' + '.pdf',format='pdf')    

    fig = plt.figure(figsize=(6,4))



    #Plot Loss function

    plt.plot(res.ob_val)

    #ax.set_yscale('log')    
    
    plt.yscale('log')

    plt.title('Loss over epochs (logarithmic scale)')


    plt.savefig(outfname + '_loss' + '.pdf',format='pdf')    
    
    
    
    
    # breathing and heart error
    fig = plt.figure(figsize=(12,5.5))
    (ax1, ax2) = fig.subplots(1, 2)

    ax1.set_title('Error with cardiac motion fixed' )
    ax1.plot(res.breathing_error)

    ax2.set_title('Error with respiratory motion fixed' )
    ax2.plot(res.heart_error)

    plt.savefig(outfname + '_fig_errors_fixedframe' + '.pdf',format='pdf')    


















        
        
        
