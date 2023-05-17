import numpy as np
import scipy.misc
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d #For surface plot
import scipy.signal
import sys #For error handling
#For saving data
import copyreg
import types
#import cPickle as pickle
import pickle
import imageio
from scipy import linalg

import matplotlib.colors as clr
from scipy.fftpack import dct, idct
import matplotlib.image as mpimg
import random

import scipy.ndimage.interpolation as intp

from IPython import get_ipython

import copy

import os
#For iterating dicts
from itertools import product

#Initialization
if __name__ == "__main__":
    
    #Set autoreload
    ipython = get_ipython()
    ipython.magic('load_ext autoreload')
    ipython.magic('autoreload 2')
    
    #Load main libraries (this requires to set the PYTHONPATH, see info.txt)
    import matpy as mp



### Helper functions ####################################################################
#########################################################################################


### Data I/O ############################################################################

#Class for output variables
#Note that res.__dict__ converts to dict
class parout(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


class parameter(object):
    
    def __init__(self,adict):
        self.__dict__.update(adict)
    
    #Define print function
    def __str__(self):
        #self.__dict__: Class members as dict
        #.__str__() is theprint function of a class
        return self.__dict__.__str__()
        
class data_input(object):
    
    def __init__(self,adict):
        self.__dict__.update(adict)
    
    #Define print function
    def __str__(self):
        #self.__dict__: Class members as dict
        #.__str__() is theprint function of a class
        return self.__dict__.__str__()




class output(object):
    
    def __init__(self,par=parameter({})):
        
        self.par = par

    def output_name(self,outpars=[],fname='',folder='results'):
    
        #Try to get filename
        if not fname:
            if hasattr(self.par,'imname'):
                fname = self.par.imname[self.par.imname.rfind('/')+1:]
            else:
                raise NameError('No filename given')

        #Generate folder if necessary
        if folder:
            if not os.path.exists(folder):
                os.makedirs(folder)

                
        #Remove ending if necessary
        pos = fname.find('.')
        if pos>-1:
            fname = fname[:pos]
        
        #Concatenate folder and filename
        outname = fname
        if folder:
            outname = folder + '/' +  outname
            #Remove double //
            outname = outname.replace('//','/')
        
        #Check for keyword DOT
        if outname.find('DOT')>-1:
            raise NameError('Keyword "DOT" not allowd')
        
        #If outpars are not given, try to generate them from par_in
        if 0:# not outpars: OPTION DEACTIVATED
            if hasattr(self,'par_in'):
                for key,val in self.par_in.items():
                    if isinstance(val, (int, float)): #Only including numbers
                        outpars.append(key)
            else:
                print('No parameters for filename given')
                
        #Add outpars to filename
        for par in outpars:
            if hasattr(self.par,par):
                val = self.par.__dict__[par]
                #exec('val = self.par.'+par)
                outname += '__' + par + '_' + num2str(val)
            else:
                raise NameError('Non-existent parameter: ' + par)

        
        return outname        

    def save(self,outpars=[],fname='',folder=''):
    
        #Get name
        outname = self.output_name(outpars,fname,folder)
        #Save
        psave(outname,self) 

    def show(self):
    
        print('Function "show" not initialized.')

#Class for parameter testing
class partest(object):

    def __init__(self,method,fixpars={},testpars={},namepars=[],folder=''):
    
        
        self.method = method
        self.fixpars = fixpars
        self.testpars = testpars
        self.namepars = namepars
        self.folder = folder
        
    def run_test(self):
    
        #Check for conflicts
        for key in self.testpars.keys():
            if key in self.fixpars:
                raise NameError('Double assignement of ' + key)
                
        #Get keys
        testkeys = self.testpars.keys()
        #Iterate over all possible combinations
        for valtuple in list(product(*self.testpars.values())):
            
            #Set test values
            for key,val in zip(testkeys,valtuple):
                self.fixpars[key] = val
                
                
            #Print parameter setup
            print('Testing: ')
            print(self.fixpars)
            #Get result
            res = self.method(**self.fixpars)
            #Save
            res.save(outpars=self.namepars,folder=self.folder)
                
def read_file(basename,pars={},folder='.',flist=[]):

    if not flist:
        flist = os.listdir(folder)
    
    flist = [ fl for fl in flist if basename in fl ]
    for key,val in pars.items():
        flist = [fl for fl in flist if '_' + key + '_' + num2str(val) in fl]
    
    if len(flist)>1:
        print('Warning: non-unique file specification. Reading first occurence')
        flist = [flist[0]]
   
    fname = folder + '/' + flist[0]
    #Remove double //
    fname = fname.replace('//','/')    
    return pload(fname)
    
    
#Return all file names with .pkl extension matching a parameter combination
def get_file_list(basename,pars = {},folder = '.'):


    flist = os.listdir(folder)


    
       
    #Remove non-matching filenames
    for fname in flist[:]:
        if (basename not in fname) or ('.pkl' not in fname): #Basename
            flist.remove(fname)
        else:
            for par in pars.keys():
                #Check parameter name
                if '_' + par + '_' not in fname:
                    flist.remove(fname)
                    break
                else:
                    #Check parameter values
                    valcount = len(pars[par])
                    if valcount>0:
                        for val in pars[par]:
                            if '_' + par + '_' + num2str(val) not in fname: #Parameter value pairs
                                valcount -= 1
                        if valcount == 0: #If no parameter is present
                            flist.remove(fname)
                            break


    return flist
                

#Return a list of file names with .pkl extension matching a parameter combination together with the parameters
def get_file_par_list(basename,pars = {},folder = '.'):

    #Get list of files matching pattern
    flist = get_file_list(basename,pars = pars,folder = folder)
    
    parnames = list(pars)
    parvals = []
    for fname in flist:
        parval = []
        for parname in parnames:
            parval.append(read_parval(fname,parname))
        parvals.append(parval[:])
    
    return flist,parnames,parvals

#Get data with best psnr in "folder" mathing a given pattern. Assuming "orig" and "u" to be available
def get_best_psnr(basename,pars={},folder='.',rescaled=True):

    #Get sortet list of filenames, parnames and values
    flist = get_file_list(basename,pars=pars,folder=folder)

    

    opt_psnr = 0.0
    for fname in flist:
    
        fullname = folder + '/' + fname
        fullname = fullname.replace('//','/') 
        
        res = pload(fullname)
        
        c_psnr = psnr(res.u,res.orig,smax = np.abs(res.orig.max()-res.orig.min()),rescaled=rescaled)
        
        if c_psnr > opt_psnr:
            opt_psnr = c_psnr
            opt_fname = fullname
            
    res = pload(opt_fname) 
    
    print('Best psnr: ' + str(np.round(opt_psnr,decimals=2)))
    
    return res

        
#Read value of parameter from file        
def read_parval(fname,parname):

    #Set position of value    
    star = fname.find('_'+parname+'_')+len('_'+parname+'_')
    #Set end position of value
    end = fname[star:].find('__')
    if end == -1:
        end = fname[star:].find('.')
    end += star 
    
    return str2num(fname[star:end])
            

#Convert number to string and reverse
def num2str(x):
    return str(x).replace('.','DOT')


def str2num(s):
    return float(s.replace('DOT','.'))


#Function to parse the arguments from par_in        
#Take a par_in dict and a list of parameter classes as input
#Sets the class members all elements of parlist according to par_in
#Raises an error when trying to set a non-existing parameter
def par_parse(par_in,parlist):

    for key,val in par_in.items():
        foundkey = False
        for par in parlist:
            if key in par.__dict__:
                par.__dict__[key] = val
                foundkey = True
        if not foundkey:
            raise NameError('Unknown parameter: ' + key)

#Data storage
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
        
    return func.__get__(obj, cls)

#Currently not used, only relevant as noted in psave
def pickle_get_current_class(obj):
    name = obj.__class__.__name__
    module_name = getattr(obj, '__module__', None)
    obj2 = sys.modules[module_name]
    for subpath in name.split('.'): obj2 = getattr(obj2, subpath)
    return obj2

def psave(name,data):
    
    #This might potentially fix erros with pickle and autoreload...try it next time the error ocurs
    if getattr(data, '__module__', None):
        data.__class__ = pickle_get_current_class(data)

    
    copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)
    
    if not name[-4:] == '.pkl':
        name = name + '.pkl'
        
    output = open(name,'wb')
    # Pickle the list using the highest protocol available.
    pickle.dump(data, output, -1)
    output.close()
    
def pload(name):

    copyreg.pickle(types.MethodType, _pickle_method, _unpickle_method)
    
    if not name[-4:] == '.pkl':
        name = name + '.pkl'
        
    
    try:
        pkl_file = open(name, 'rb')
        data = pickle.load(pkl_file)
    except:
        pkl_file.close()


        #print('Standard loading failed, resorting to python2 compatibility...')
        pkl_file = open(name, 'rb')
        data = pickle.load(pkl_file,encoding='latin1')

    pkl_file.close()
    return data


def server_transfer(*args,**kwargs):


    try:
        from server_transfer import server_transfer as st
        
        st(*args,**kwargs)
        
    except:
        print('Error: Sever transfer function not available')

        
    



### Plotting ############################################################################

def imshow(x,stack=1,fig=0,title=0,colorbar=1,cmap='gray',vrange=[]):


    try:

        if x.ndim>2 and stack:
            x = imshowstack(x)

        if not fig:
            fig = plt.figure()
            
        plt.figure(fig.number)
        if not vrange:
            plt.imshow(x,cmap=cmap,interpolation='none')
        else:
            plt.imshow(x,cmap=cmap,vmin=vrange[0],vmax=vrange[1],interpolation='none')
        if colorbar:
            plt.colorbar()
        if title:
            plt.title(title)
        fig.show()
        
    except:
        print('Display error. I assume that no display is available and continue...')
        fig = 0
    
    return fig


def plot(x,y=0,fig=0,title=0,label=0):

        

    try:
        if not fig:
            fig = plt.figure()
        plt.figure(fig.number)
        
        if not np.any(y):
            plt.plot(x,label=label)
        else:
            plt.plot(x,y,label=label)
            
        if title:
            plt.title(title)
            
        if label:
            plt.legend()


        fig.show()
        
    except:
        print('Display error. I assume that no display is available and continue...')
        fig = 0
    
    return fig
    
    
def surf(x,y=0,z=0,fig=0,title=0,label=0):

    try:
        if not fig:
            fig = plt.figure()
        plt.figure(fig.number)
        
        ax = fig.gca(projection='3d')
        
        if not np.any(y):
            ax.plot_surface(x,label=label)
        else:
            ax.plot_surface(x,y,z,label=label)
            
        if title:
            plt.title(title)
            
        if label:
            plt.legend()


        fig.show()
   
    except:
        print('Display error. I assume that no display is available and continue...')
        fig = 0
    
    return fig

#Stack a 3D array of images to produce a 2D image
#Optinal input: nimg = (n,m). Take n*m images and arrange them as n x m
def imshowstack(k,nimg = ()):

    N,M = k.shape[0:2]
    nk = k.shape[-1]

    if nimg:
        nx = nimg[1]
        ny = nimg[0]
    else:

        nx = np.ceil(np.sqrt(np.copy(nk).astype('float')))
        ny = np.ceil(nk/nx)

        nx = int(nx)
        ny = int(ny)

    if k.ndim == 3:
        kimg = np.zeros([N*ny,M*nx])
        for jj in range(ny):
            for ii in range(nx):
                    if ii + nx*jj < nk:
                        kimg[jj*N:(jj+1)*N,M*ii:M*(ii+1)] = k[...,ii + nx*jj]
    else:
        kimg = np.zeros([N*ny,M*nx,k.shape[2]])
        for ll in range(k.shape[2]):
            for jj in range(ny):
                for ii in range(nx):
                        if ii + nx*jj < nk:
                            kimg[jj*N:(jj+1)*N,M*ii:M*(ii+1),ll] = k[...,ll,ii + nx*jj]
    
    
    return kimg


def vecshow(z,step=1):

    #Optional argument: Take only every step'th entry

    fig = plt.figure()
    plt.quiver(z[::step,::step,0],z[::step,::step,1])
    fig.show()
    return fig

def veccolor(z,fig=0,title=0):

    if z.ndim>3:
        z = imshowstack(z)
    
    n = z.shape[0]
    m = z.shape[1]

    
    p = np.zeros([z.shape[0],z.shape[1],3])
    p[...,0] = (np.arctan2(z[...,1],z[...,0])/(2.0*np.pi)) + 0.5
    nz = np.sqrt(np.square(z).sum(axis=2))
    p[...,1] = nz/np.maximum(nz.max(),0.00001)
    p[...,2] = 1.0


    psz = 4
    l1 = np.linspace(-1,1,n+2*psz)
    l2 = np.linspace(-1,1,m+2*psz)
    a1,a2 = np.meshgrid(l2,l1)
    
    c = np.zeros( (n+2*psz,m+2*psz,3))
    
    c[...,0] = (np.arctan2(a1,a2)/(2.0*np.pi)) + 0.5
    c[...,1] = 1.0
    c[...,2] = 1.0
    
    c[psz:-psz,psz:-psz,:] = p
    
   
    fig = imshow(clr.hsv_to_rgb(c),stack=0,fig=fig,title=title,colorbar=0)
    return fig



def closefig():
    plt.close('all')

def rgb2gray(rgb):

    return 0.2989 * rgb[:,:,0] + 0.5870 * rgb[:,:,1] + 0.1140 * rgb[:,:,2]


def imread(imname):
    
    return imageio.imread(imname).astype('float')/255.0  


#Function to scale image to [0,1]. Range defines current image range (default: [img.min(),img.max()], values above and below will be cliped
def imnormalize(img,rg=[]):
    
    if not rg:
        rg = [img.min(),img.max()]
        

    #Clip range boundaries
    img = np.clip(np.copy(img.astype('float')),rg[0],rg[1])
    
    #Convert rage to [0,1]
    img = img - rg[0]
    if (rg[1]-rg[0])>0:
        img = img/(rg[1]-rg[0])
    elif np.any(img):
        raise ValueError('Function requires rg[0]<rg[1]')        
    else:
        print('Warning: empty image, ignoring range argument, no normalization carried out')



    return img
    
def imsave(fname,img,format=None,rg=[]): #rg defines grayscale boundary values.

    #img = imnormalize(img,rg=rg)
    
    if not rg:
        rg = [0,1]
    img = np.clip(np.copy(img.astype('float')),rg[0],rg[1])
    
    imageio.imwrite(fname,(255.0*img).astype('uint8'),format=format)



### Numerical ###########################################################################
def mse(u,u0,rescaled=False):

    c = 1.0
    if rescaled:
        c = (u*u0).sum()/np.square(u).sum()

    return np.square(c*u-u0).sum() / np.square(u0).sum()

def psnr(u,u0,smax=1.0,rescaled=False):

    c = 1.0
    if rescaled:
        c = (u*u0).sum()/np.square(u).sum()


    N = np.prod(u.shape).astype('float')
    err = np.square(c*u-u0).sum()/N
    
    return 20.0*np.log10( smax/ np.sqrt(err) )

   
def fgauss(sz,mu,sig):

    l1 = np.linspace(-1,1,sz)
    l2 = np.linspace(-1,1,sz)
    a1,a2 = np.meshgrid(l1,l2)

    return ( 1.0/np.sqrt(2.0*np.pi*sig*sig) )*np.exp( -(np.square(a1-mu) + np.square(a2-mu))/(2.0*sig*sig))

#All imput array must be odd
def f_sinc(x):
    sz = x.shape
    l1 = np.linspace(-2,2,sz[0])
    l2 = np.linspace(-2,2,sz[1])
    a1,a2 = np.meshgrid(l1,l2)
    
    z = np.sinc(a1)*np.sinc(a2)
    return z  - (z.sum()/(sz[0]*sz[1]))



def get_circle(sz=128,r=0.8,sharp=0):
    
    if not sharp:
        sharp = sz*0.5

    l1 = np.linspace(-1,1,sz)
    l2 = np.linspace(-1,1,sz)
    a1,a2 = np.meshgrid(l1,l2)

    rad = np.sqrt( np.square(a1) + np.square(a2))

    z = np.maximum(0.0,np.minimum(1.0,sharp*(r-rad)))
#    z = np.zeros([sz,sz])
#    z[rad<=r] = 1.0
    
    return z
    


### Algorithmic functions ###############################################################
#########################################################################################



