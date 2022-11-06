import numpy as np
import matplotlib.pylab as plt
from sklearn import mixture


def create_box_plot(box_data,tick_list,y_label = "",x_label = "",y_lim = (),title = ""):
    ticks = tick_list
    plt.boxplot(box_data,positions = range(1,len(tick_list)+1), notch = True, showfliers = False)
    for i in range(1,len(tick_list)+1):
        y = box_data[i-1]
        x = np.random.normal(i, 0.04, size=len(y))
        plt.plot(x, y, 'r.', alpha=0.2)
    try:
        plt.ylim(y_lim) 
    except:
        print "Warning: y_lim not valid"
    plt.xticks(xrange(1, len(ticks) * 1 + 1, 1), ticks)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.show()
        
    return


def GMM_utility(data, n, biners=50, inclusion_thresh = [0,100], verbose=True, title_1d="", title_2d="", x_label="", y_label_2d="", log=True, x_limit = ()):
    
    data = np.array(data)
    
    p_thresh = np.percentile(data,inclusion_thresh)
    inds = ((data>=p_thresh[0]) & (data<=p_thresh[1]))
    data = data[inds]
    
    gmix = mixture.GMM(n_components=n, covariance_type='diag')
    if log:
        (results,bins) = np.histogram(np.log10(data),density='true',bins=biners)
    else:
        (results,bins) = np.histogram(data,density='true',bins=biners)


    data_arr = np.zeros((len(data),2))
    data_arr[:,0] = np.random.normal(1, 0.04, size=len(data))
    if log:
        data_arr[:,1] = np.log10(data)
    else: 
        data_arr[:,1] = data
    if verbose:
        plt.plot(data_arr[:,1],data_arr[:,0],'r.')
        plt.ylim((0,2))
        plt.title(title_1d)
        plt.xlabel(x_label)
        plt.show()
    gmix.fit(data_arr)
    
    if log:
        print "Fitted Mean: {0} +/- {1}".format(gmix.means_[:,1],np.sqrt(gmix.covars_[:,1]))
        print "Fitted Mean(normal): {0} +/- {1}".format(np.exp(gmix.means_[:,1]),np.exp(gmix.means_[:,1])*np.sqrt(gmix.covars_[:,1]))
    else:
        print "Fitted Mean: {0} +/- {1}".format(gmix.means_[:,1],np.sqrt(gmix.covars_[:,1]))
    max_r = np.max(results)
    plt.plot(np.diff(bins)+bins[:len(bins)-1],results)
    
    for i in gmix.means_:
        plt.axvline(x=i[1],color='red')
    plt.title(title_2d)
    plt.xlabel(x_label)
    plt.ylabel(y_label_2d)
    try:
        plt.xlim(x_limit)
    except:
        print "Warning: x_limit is invalid"
    plt.show()
    
    return 


def BGMM_utility(data, n, biners=50, inclusion_thresh = [0,100], verbose=True, title_1d="", title_2d="", x_label="", y_label_2d="", log=True, x_limit = ()):
    
    data = np.array(data)
    
    p_thresh = np.percentile(data,inclusion_thresh)
    inds = ((data>=p_thresh[0]) & (data<=p_thresh[1]))
    data = data[inds]
    
    bgmix = mixture.BayesianGaussianMixture(n_components=n, covariance_type='diagonal')
    if log:
        (results,bins) = np.histogram(np.log10(data),density='true',bins=biners)
    else:
        (results,bins) = np.histogram(data,density='true',bins=biners)


    data_arr = np.zeros((len(data),2))
    data_arr[:,0] = np.random.normal(1, 0.04, size=len(data))
    if log:
        data_arr[:,1] = np.log10(data)
    else: 
        data_arr[:,1] = data
    if verbose:
        plt.plot(data_arr[:,1],data_arr[:,0],'r.')
        plt.ylim((0,2))
        plt.title(title_1d)
        plt.xlabel(x_label)
        plt.show()
    bgmix.fit(data_arr)
    
    if log:
        print "Fitted Mean: {0} +/- {1}".format(gmix.means_[:,1],np.sqrt(gmix.covars_[:,1]))
        print "Fitted Mean(normal): {0} +/- {1}".format(np.exp(gmix.means_[:,1]),np.exp(gmix.means_[:,1])*np.sqrt(gmix.covars_[:,1]))
    else:
        print "Fitted Mean: {0} +/- {1}".format(gmix.means_[:,1],np.sqrt(gmix.covars_[:,1]))
    max_r = np.max(results)
    plt.plot(np.diff(bins)+bins[:len(bins)-1],results)
    
    for i in gmix.means_:
        plt.axvline(x=i[1],color='red')
    plt.title(title_2d)
    plt.xlabel(x_label)
    plt.ylabel(y_label_2d)
    try:
        plt.xlim(x_limit)
    except:
        print "Warning: x_limit is invalid"
    plt.show()
    
    return 
