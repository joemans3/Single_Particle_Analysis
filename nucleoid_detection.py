import math
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from skimage import feature, future
from skimage.measure import label, regionprops
from sklearn.ensemble import RandomForestClassifier


def test(img,regions = True,**kwargs):
    connectivity = kwargs.get('connectivity', 2)
    sigma_max = kwargs.get('sigma_max', 10)
    sigma_min = kwargs.get('sigma_min', 1)
    training_labels = np.zeros(img.shape[:2], dtype=np.uint8) + 3

    #better way to do this
    index_nuc = np.where(img > 9200)
    index_rest = np.where(img < 7000)
    index_back = np.where(img < 2000)
    training_labels[index_rest] = 2
    training_labels[index_nuc] = 1

    features_func = partial(feature.multiscale_basic_features,
                            intensity=True, edges=False, texture=True,
                            sigma_min=sigma_min, sigma_max=sigma_max)
    features = features_func(img)
    clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                                max_depth=10, max_samples=0.05)
    clf = future.fit_segmenter(training_labels, features, clf)
    result = future.predict_segmenter(features, clf)
    if regions:

        copy_result = np.copy(result)
        copy_result[np.invert(copy_result == 1)] = 0
        nuc_result = result*copy_result

        region_result = get_region(nuc_result,type = regionprops,connectivity=connectivity)
        return [result, region_result]
    else:
        return [result, 0]

def get_training_set(img):
    return

def get_region(image,type = regionprops, connectivity = 2):
    '''
    Parameters
    ----------
    image : 2D array-like
        binary image (0,1) where 1 indicates the region to fit 
    type : functional, default = regionprops
        the function type used to fit the image. Default assumes elliptical shapes
    
    Returns
    -------
    propertieslist of RegionProperties
        Each item describes one labeled region, and can be accessed using the attributes listed below.
    
    Notes
    -----
    The following properties can be accessed as attributes or keys:

    areaint
    -------
    Number of pixels of the region.

    area_bboxint
    ------------
    Number of pixels of bounding box.

    area_convexint
    --------------
    Number of pixels of convex hull image, which is the smallest convex polygon that encloses the region.

    area_filledint
    --------------
    Number of pixels of the region will all the holes filled in. Describes the area of the image_filled.

    axis_major_lengthfloat
    ----------------------
    The length of the major axis of the ellipse that has the same normalized second central moments as the region.

    axis_minor_lengthfloat
    ----------------------
    The length of the minor axis of the ellipse that has the same normalized second central moments as the region.

    bboxtuple
    ---------
    Bounding box (min_row, min_col, max_row, max_col). Pixels belonging to the bounding box are in the half-open interval [min_row; max_row) and [min_col; max_col).

    centroidarray
    -------------
    Centroid coordinate tuple (row, col).

    centroid_localarray
    -------------------
    Centroid coordinate tuple (row, col), relative to region bounding box.

    centroid_weightedarray
    ----------------------
    Centroid coordinate tuple (row, col) weighted with intensity image.

    centroid_weighted_localarray
    ----------------------------
    Centroid coordinate tuple (row, col), relative to region bounding box, weighted with intensity image.

    coords(N, 2) ndarray
    Coordinate list (row, col) of the region.

    eccentricityfloat
    -----------------
    Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.

    equivalent_diameter_areafloat
    -----------------------------
    The diameter of a circle with the same area as the region.

    euler_numberint
    ---------------
    Euler characteristic of the set of non-zero pixels. Computed as number of connected components subtracted by number of holes (input.ndim connectivity). In 3D, number of connected components plus number of holes subtracted by number of tunnels.

    extentfloat
    -----------
    Ratio of pixels in the region to pixels in the total bounding box. Computed as area / (rows * cols)

    feret_diameter_maxfloat
    -----------------------
    Maximum Feret’s diameter computed as the longest distance between points around a region’s convex hull contour as determined by find_contours. [5]

    image(H, J) ndarray
    Sliced binary region image which has the same size as bounding box.

    image_convex(H, J) ndarray
    Binary convex hull image which has the same size as bounding box.

    image_filled(H, J) ndarray
    Binary region image with filled holes which has the same size as bounding box.

    image_intensityndarray
    ----------------------
    Image inside region bounding box.

    inertia_tensorndarray
    ---------------------
    Inertia tensor of the region for the rotation around its mass.

    inertia_tensor_eigvalstuple
    ---------------------------
    The eigenvalues of the inertia tensor in decreasing order.

    intensity_maxfloat
    ------------------
    Value with the greatest intensity in the region.

    intensity_meanfloat
    -------------------
    Value with the mean intensity in the region.

    intensity_minfloat
    ------------------
    Value with the least intensity in the region.

    labelint
    --------
    The label in the labeled input image.

    moments(3, 3) ndarray
    Spatial moments up to 3rd order:

    m_ij = sum{ array(row, col) * row^i * col^j }
    where the sum is over the row, col coordinates of the region.

    moments_central(3, 3) ndarray
    Central moments (translation invariant) up to 3rd order:

    mu_ij = sum{ array(row, col) * (row - row_c)^i * (col - col_c)^j }

    where the sum is over the row, col coordinates of the region, and row_c and col_c are the coordinates of the region’s centroid.

    moments_hutuple
    ---------------
    Hu moments (translation, scale and rotation invariant).

    moments_normalized(3, 3) ndarray
    Normalized moments (translation and scale invariant) up to 3rd order:

    nu_ij = mu_ij / m_00^[(i+j)/2 + 1]

    where m_00 is the zeroth spatial moment.

    moments_weighted(3, 3) ndarray
    Spatial moments of intensity image up to 3rd order:

    wm_ij = sum{ array(row, col) * row^i * col^j }
     
    where the sum is over the row, col coordinates of the region.

    moments_weighted_central(3, 3) ndarray
    Central moments (translation invariant) of intensity image up to 3rd order:

    wmu_ij = sum{ array(row, col) * (row - row_c)^i * (col - col_c)^j }
     
    where the sum is over the row, col coordinates of the region, and row_c and col_c are the coordinates of the region’s weighted centroid.

    moments_weighted_hutuple
    ------------------------
    Hu moments (translation, scale and rotation invariant) of intensity image.

    moments_weighted_normalized(3, 3) ndarray
    Normalized moments (translation and scale invariant) of intensity image up to 3rd order:

    wnu_ij = wmu_ij / wm_00^[(i+j)/2 + 1]
     
    where wm_00 is the zeroth spatial moment (intensity-weighted area).

    orientationfloat
    ----------------
    Angle between the 0th axis (rows) and the major axis of the ellipse that has the same second moments as the region, ranging from -pi/2 to pi/2 counter-clockwise.

    perimeterfloat
    --------------
    Perimeter of object which approximates the contour as a line through the centers of border pixels using a 4-connectivity.

    perimeter_croftonfloat
    ----------------------
    Perimeter of object approximated by the Crofton formula in 4 directions.

    slicetuple of slices
    --------------------
    A slice to extract the object from the source image.

    solidityfloat
    -------------
    Ratio of pixels in the region to pixels of the convex hull image.

    Each region also supports iteration, so that you can do:

    for prop in region:
        print(prop, region[prop])
    '''
    regions = regionprops(label(image,connectivity=connectivity))
    return regions

def plot_regions(regions,fig,ax,colorbar_mappable):
    '''
    Parameters
    ----------
    regions : list, output from regionprops
        takes the output of the regionprops from sklearn
    fig : plt.figure object
        figure object to plot onto
    ax : axis object
        axis object on which to plot to
    colorbar_mappable : colobar mappable
        colorbar_mappable opbject 

    Returns
    -------

    '''
    for props in regions:
        y0, x0 = props.centroid
        orientation = props.orientation
        x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
        y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
        x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
        y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

        ax.plot((x0, x1), (y0, y1), '-r', linewidth=2.5)
        ax.plot((x0, x2), (y0, y2), '-r', linewidth=2.5)
        ax.plot(x0, y0, '.g', markersize=15)

        minr, minc, maxr, maxc = props.bbox
        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)
        ax.plot(bx, by, '-b', linewidth=2.5)
    ax.set_xlim((minc - 5,maxc + 5))
    ax.set_ylim((minr - 5,maxr + 5))
    plt.colorbar(colorbar_mappable)
    plt.show()
    