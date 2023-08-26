import numpy as np
import pylab as py
from scipy.optimize import curve_fit
import scipy.io

def fit_MSD(t,p_0,p_1):
    return p_0 * (t**(p_1))
def fit_linear_msd(t,p1,p2):
	return p1*(p2+t)
mat = scipy.io.loadmat('Stream01.tif_tracksMSDs.mat')
var = mat.get('tracks_MSDs')
D_arr = []
a_arr = []
for i in range(20):
	t = np.array(var[i][0][:,0])
	msd = np.array(var[i][0][:,1])
	#popt, pcov = curve_fit(fit_MSD,t,msd,[0.0,0.0], method = 'dogbox')
	popt, pcov = curve_fit(fit_linear_msd,np.log(t[0:3]),np.log(msd[0:3]),[0.0,0.0],maxfev = 100000)
	D_arr.append(popt[1])
	a_arr.append(popt[0])
	py.plot(np.log(t),np.log(msd),'o')
	#py.plot(fit_MSD(t,popt[0],popt[1]))
	py.plot(np.log(t),fit_linear_msd(np.log(t),popt[0],popt[1]))


py.show()

py.hist(np.exp(np.array(D_arr))/4.0)
py.show()
py.hist(a_arr)
py.show()


