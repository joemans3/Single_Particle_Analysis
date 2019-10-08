import numpy as np



def read_data(path, delimiter = ',',skiprow = 1):

	data = np.loadtxt(path,delimiter = delimiter,skiprows = skiprow)

	return data