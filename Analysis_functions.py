import numpy as np

def fit_MSD(t,p_0,p_1):
    return p_0 * (t**(p_1))

def dif_dis(x,y):
    c = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    return c

def dist(x,y,c1,c2):
    '''Distance(s) x,y away from a point c1,c2 in 2D'''
    tx=np.abs(c1-x)
    ty=np.abs(c2-y)

    temp=np.sqrt((tx)**2 + (ty)**2)
    return temp

def MSD_tavg1(x,y,f,f_inc = False):
    if f_inc == True:
        return np.mean((np.diff(dist(np.array(x)[1:],np.array(y)[1:],np.array(x)[0],np.array(y)[0])/np.diff(f)))**2)/4.
    else:
        return np.mean(np.diff(dist(np.array(x)[1:],np.array(y)[1:],np.array(x)[0],np.array(y)[0]))**2)/4.


def MSD_tavg(x,y,f,f_inc = False):
    
    dists = np.zeros(len(x)-1)
    for i in range(len(x)-1):
        dists[i] = dist(x[i],y[i],x[i+1],y[i+1])
    if f_inc == True:
        return np.mean((np.diff(dists/np.diff(f)))**2)/4.
    else:
        return np.mean((np.diff(dists))**2)/4.
    
def MSD_tavg_single(x,f,f_inc = False):
    if f_inc == True:
        return np.mean((np.diff(x/f))**2)/4.
    else:
        return np.mean((np.diff(x))**2)/4.
    
    
    

def gaussian_fit(x,p0,p1,p2):
    return ((np.sqrt(2.*np.pi*p0))**-1)*np.exp(-((x-p1)**2)/(2*p0)) + p2



##################################################################################################################################
#implimenting MLE method for detecting diffusion coeff/ velocity change in single tracks as outlined in: Detection of Velocity and Diffusion Coefficient Change Points in Single-Particle Trajectories, Yin et al. 2018

def prop_vel(x,frame_rate,n):
    
    return (1./(n*frame_rate))*np.sum(np.diff(x))

def prop_diff_c(x,frame_rate,n):
    
    return (1./(2*n*frame_rate))*np.sum((np.diff(x) - prop_vel(x,frame_rate,n)*frame_rate)**2)

def ll_0(x,n,frame_rate):
    
    return 0.5*(n*np.log10(prop_diff_c(x,frame_rate,n)))
10
def log_likelihood_k(x,n,k,frame_rate):
    
    return ll_0(x,n,frame_rate) - ll_0(x[:k],k,frame_rate) - ll_0(x[k+1:n],n-k,frame_rate)
    
    
    
def MLE_decomp(x,y,frame_rate):
    N = len(x)
    pros_k = range(1,N)
    hold_prop_kx = np.zeros(len(pros_k)+1)
    hold_prop_ky = np.zeros(len(pros_k)+1)
    
    #log-likelihood linear in x,y
    for i in range(len(pros_k)):
        hold_prop_kx[i+1] = 2.*log_likelihood_k(x,N,pros_k[i],frame_rate)
        hold_prop_ky[i+1] = 2.*log_likelihood_k(y,N,pros_k[i],frame_rate)
    
    max_x = np.sqrt(np.max(hold_prop_kx))
    max_y = np.sqrt(np.max(hold_prop_ky))
    
    return 










def track_decomp(x,y,f,max_track_decomp):
    #takes tracks and finds MSD for various timestep conditions.
    
    #return array-like: 
    #msd = msd values at all tau values considered
    #popt = fitted parameters on MSD equation
    #pcov = covariance matrix of fit
    
    max_decomp = np.floor(len(x)/max_track_decomp)
    tau = range(1,int(max_decomp+1.0))
    msd = []
    for i in tau:
        if i < len(x):
            n_x = np.array(x)[::i]
            n_y = np.array(y)[::i]
            n_f = np.array(f)[::i]
            msd.append(MSD_tavg(n_x,n_y,n_f))
        
    #popt , pcov = curve_fit(fit_MSD,tau,np.array(msd),p0=[1,1],maxfev=10000)
    
    
    return np.array(msd)

def track_decomp_single(x,f,max_track_decomp):
    #takes tracks and finds MSD for various timestep conditions.
    
    #return array-like: 
    #msd = msd values at all tau values considered
    #popt = fitted parameters on MSD equation
    #pcov = covariance matrix of fit
    
    max_decomp = np.floor(len(x)/max_track_decomp)
    tau = range(1,int(max_decomp+1.0))
    msd = []
    for i in tau:
        if i < len(x):
            n_x = np.array(x)[::i]
            n_f = np.array(f)[::i]
            msd.append(MSD_tavg_single(n_x,n_f))
        
    #popt , pcov = curve_fit(fit_MSD,tau,np.array(msd),p0=[1,1],maxfev=10000)
    
    
    return np.array(msd)

def cumsum(x,y):
    dx = np.diff(np.array(x))
    dy = np.diff(np.array(y))
    
    dr = np.sqrt(dx**2 + dy**2)
    
    return dr