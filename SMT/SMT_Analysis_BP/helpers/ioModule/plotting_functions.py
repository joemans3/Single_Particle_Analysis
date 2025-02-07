"""
This script contains all the plotting functions used in the trajectory_analysis_script.py

Author: Baljyot Singh Parmar

"""

import matplotlib.pylab as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from skimage import io
from sklearn import mixture

from SMT.SMT_Analysis_BP.helpers.analysisFunctions.Analysis_functions import *
from SMT.SMT_Analysis_BP.helpers.ioModule.import_functions import *


class run_analysis_plotting:
    def __init__(self) -> None:
        pass

    @staticmethod
    def draw_item(
        which_object,
        plots=1,
        all_tracks=False,
        movie_ID="0",
        cell_ID=["0", "1"],
        movie_frame_index=3,
    ):
        """
        Plot the projected frame (movie_frame_index) of a movie (movie_ID) with specific cells (cell_ID, can be
        marray of cells).
        If plots == 1, plot only one subfigure, else plot a hard coded identicle set of subplots (2,3)
        If all_tracks == True use the raw_tracks variable for each cell (all possible localization)


        Parameters
        ----------
        which_object : run_analysis object from trajectory_analysis_script.py
            eg. rp_ez,nusa_ez,ll_ez
        plots : int, tuple (n,m)
            number of total subplots
            if 1 plot only one subplot
            else plot (n,m)
        all_tracks : bool
            if true use all raw_tracks localizations
            else use only "viable" tracks
        movie_ID : str
            key identifier of the movie in this dataset
        cell_ID : str, array-like of str
            key identifier of the cell in this movie
            if array plot multiple cell attributes
        movie_frame_index : int
            the subframe of the movie (usually 0-4 for 5 total subframes)

        RETURNS
        -------

        Array-like
            [x,y,fig,ax]
            x : float
                x coordinates of all tracks used
            y : float
                y coordinates of all tracks used
            fig : figure object
                the figure object which defines the plotting
            ax :  Axes object
                All the sub_plot ax (if plots == 1, this is a single ax, else it is of shape (n,m))
        """
        viable_drop_circles = []
        fig, ax = run_analysis_plotting.plot_img(
            which_object, plots, movie_ID, cell_ID, movie_frame_index
        )
        x = []
        y = []
        for k in cell_ID:
            for i in range(
                len(
                    which_object.Movie[movie_ID]
                    .Cells[k]
                    .sorted_tracks_frame[0][movie_frame_index]
                )
            ):
                x += (
                    which_object.Movie[movie_ID]
                    .Cells[k]
                    .sorted_tracks_frame[1][movie_frame_index][i]
                )
                y += (
                    which_object.Movie[movie_ID]
                    .Cells[k]
                    .sorted_tracks_frame[2][movie_frame_index][i]
                )

        if all_tracks:
            x = []
            y = []
            for k in cell_ID:
                arr_1 = np.array(which_object.Movie[movie_ID].Cells[k].raw_tracks)
                x += list(arr_1[:, 2])
                y += list(arr_1[:, 3])
        for k in cell_ID:
            for i, j in which_object.Movie[movie_ID].Cells[k].Drop_Collection.items():
                if i[0] == str(movie_frame_index + 1):
                    viable_drop_circles.append(j)
                    if plots == 1:
                        Drawing_uncolored_circle = create_circle_obj(j, fill=False)
                        ax.add_artist(Drawing_uncolored_circle)
                    else:
                        for art in range(len(ax)):
                            for art2 in range(len(ax[art])):
                                Drawing_uncolored_circle = create_circle_obj(
                                    j, fill=False
                                )
                                ax[art, art2].add_artist(Drawing_uncolored_circle)
        return [x, y, fig, ax]

    @staticmethod
    def plot_img(which_object, plots=1, movie_ID="0", cell_ID="0", movie_frame_index=3):
        """
        Given the location of the image in our dataset, plot it given the plotting rules and return the figure/axis objects

        Parameters
        ----------
        which_object : run_analysis object from trajectory_analysis_script.py
            eg. rp_ez,nusa_ez,ll_ez
        plots : int, tuple (n,m)
            number of total subplots
            if 1 plot only one subplot
            else plot (n,m)
        movie_ID : str
            key identifier of the movie in this dataset
        cell_ID : str, array-like of str
            key identifier of the cell in this movie
            if array plot multiple cell attributes
        movie_frame_index : int
            the subframe of the movie (usually 0-4 for 5 total subframes)

        Returns
        -------
        array-like of figure,ax objects
            returns the object of the figure and axis created by the plotting of the image

        Notes
        -----
        squeeze is used in plt.subplots(... ,squeeze) to change the shape of the ax objects created


        squeeze : bool, default: True
            - If True, extra dimensions are squeezed out from the returned
            array of `~matplotlib.axes.Axes`:
            - if only one subplot is constructed (nrows=ncols=1), the
                resulting single Axes object is returned as a scalar.
            - for Nx1 or 1xM subplots, the returned object is a 1D numpy
                object array of Axes objects.
            - for NxM, subplots with N>1 and M>1 are returned as a 2D array.
            - If False, no squeezing at all is done: the returned Axes object is
            always a 2D array containing Axes instances, even if it ends up
            being 1x1.
        """

        img = read_file(which_object._get_movie_path(movie_ID, movie_frame_index))

        if plots == 1:
            fig, ax = plt.subplots(plots, squeeze=True)
            ax.imshow(img, cmap="gray")
        else:
            fig, ax = plt.subplots(*plots, squeeze=False)
            for i in range(plots[0]):
                for j in range(plots[1]):
                    ax[i, j].imshow(img, cmap="gray")
        return [fig, ax]


def circles(x, y, s, c="b", vmin=None, vmax=None, **kwargs):
    """
    Make a scatter of circles plot of x vs y, where x and y are sequence
    like objects of the same lengths. The size of circles are in data scale.

    Parameters
    ----------
    x,y : scalar or array_like, shape (n, )
        Input data
    s : scalar or array_like, shape (n, )
        Radius of circle in data unit.
    c : color or sequence of color, optional, default : 'b'
        `c` can be a single color format string, or a sequence of color
        specifications of length `N`, or a sequence of `N` numbers to be
        mapped to colors using the `cmap` and `norm` specified via kwargs.
        Note that `c` should not be a single numeric RGB or RGBA sequence
        because that is indistinguishable from an array of values
        to be colormapped. (If you insist, use `color` instead.)
        `c` can be a 2-D array in which the rows are RGB or RGBA, however.
    vmin, vmax : scalar, optional, default: None
        `vmin` and `vmax` are used in conjunction with `norm` to normalize
        luminance data.  If either are `None`, the min and max of the
        color array is used.
    kwargs : `~matplotlib.collections.Collection` properties
        Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
        norm, cmap, transform, etc.

    Returns
    -------
    paths : `~matplotlib.collections.PathCollection`

    Examples
    --------
    a = np.arange(11)
    circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
    plt.colorbar()

    License
    --------
    This code is under [The BSD 3-Clause License]
    (http://opensource.org/licenses/BSD-3-Clause)
    """

    if np.isscalar(c):
        kwargs.setdefault("color", c)
        c = None
    if "fc" in kwargs:
        kwargs.setdefault("facecolor", kwargs.pop("fc"))
    if "ec" in kwargs:
        kwargs.setdefault("edgecolor", kwargs.pop("ec"))
    if "ls" in kwargs:
        kwargs.setdefault("linestyle", kwargs.pop("ls"))
    if "lw" in kwargs:
        kwargs.setdefault("linewidth", kwargs.pop("lw"))

    patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
    collection = PatchCollection(patches, **kwargs)
    if c is not None:
        collection.set_array(np.asarray(c))
        collection.set_clim(vmin, vmax)

    ax = plt.gca()
    ax.add_collection(collection)
    ax.autoscale_view()
    if c is not None:
        plt.sci(collection)
    return collection


def create_circle_obj(dims, fill=False):
    """
    INPUTS:
    dims = array-like
        [x,y,r]
        x = x coordinate of center of circle
        y = y coordinate of center of circle
        r = radius of circle

    RETURNS:
    Circle object to be used in ax.add_artist
    """

    cir_object = plt.Circle((dims[0], dims[1]), dims[2], fill=fill)
    return cir_object


def create_box_plot(
    box_data, tick_list, y_label="", x_label="", y_lim=(), title="", show=False
):
    ticks = tick_list
    f, ax = plt.subplots()
    ax.boxplot(
        box_data,
        positions=list(range(1, len(tick_list) + 1)),
        notch=True,
        showfliers=False,
    )
    for i in range(1, len(tick_list) + 1):
        y = box_data[i - 1]
        x = np.random.normal(i, 0.04, size=len(y))
        ax.plot(x, y, "r.", alpha=0.2)
    try:
        ax.ylim(y_lim)
    except:
        print("Warning: y_lim not valid")
    ax.set_xticks(range(1, len(ticks) * 1 + 1, 1), ticks)
    ax.set_xticklabels(ticks)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    if show:
        plt.show()
    return ax

    # def BGMM_utility(data, n, biners=50, inclusion_thresh = [0,100], verbose=True, title_1d="", title_2d="", x_label="", y_label_2d="", log=True, x_limit = ()):

    data = np.array(data)

    p_thresh = np.percentile(data, inclusion_thresh)
    inds = (data >= p_thresh[0]) & (data <= p_thresh[1])
    data = data[inds]

    bgmix = mixture.BayesianGaussianMixture(n_components=n, covariance_type="diagonal")
    if log:
        (results, bins) = np.histogram(np.log10(data), density="true", bins=biners)
    else:
        (results, bins) = np.histogram(data, density="true", bins=biners)

    data_arr = np.zeros((len(data), 2))
    data_arr[:, 0] = np.random.normal(1, 0.04, size=len(data))
    if log:
        data_arr[:, 1] = np.log10(data)
    else:
        data_arr[:, 1] = data
    if verbose:
        plt.plot(data_arr[:, 1], data_arr[:, 0], "r.")
        plt.ylim((0, 2))
        plt.title(title_1d)
        plt.xlabel(x_label)
        plt.show()
    bgmix.fit(data_arr)

    if log:
        print(
            "Fitted Mean: {0} +/- {1}".format(
                gmix.means_[:, 1], np.sqrt(gmix.covars_[:, 1])
            )
        )
        print(
            "Fitted Mean(normal): {0} +/- {1}".format(
                np.exp(gmix.means_[:, 1]),
                np.exp(gmix.means_[:, 1]) * np.sqrt(gmix.covars_[:, 1]),
            )
        )
    else:
        print(
            "Fitted Mean: {0} +/- {1}".format(
                gmix.means_[:, 1], np.sqrt(gmix.covars_[:, 1])
            )
        )
    max_r = np.max(results)
    plt.plot(np.diff(bins) + bins[: len(bins) - 1], results)

    for i in gmix.means_:
        plt.axvline(x=i[1], color="red")
    plt.title(title_2d)
    plt.xlabel(x_label)
    plt.ylabel(y_label_2d)
    try:
        plt.xlim(x_limit)
    except:
        print("Warning: x_limit is invalid")
    plt.show()

    return


def read_imag(path, fig=False, ax=False, show=True):
    ori_img = io.imread(path)
    if fig == False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.imshow(ori_img)

    if show:
        plt.show()
    return ori_img


def contour_intens(img, fig=False, ax=False, show=True, seg=True, perc=95):
    grey_img = rgb_to_grey(img)
    normed_grey_img = grey_img * (grey_img > np.percentile(grey_img, perc))
    if fig == False:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    image_plot = ax.imshow(grey_img)
    contour_img_plot = ax.contour(normed_grey_img)
    fig.colorbar(image_plot)

    if show:
        plt.show()

    return


def spacialplot_msd(op, fig=False, ax=False, show=True):
    x = op.all_tracks_x
    y = op.all_tracks_y
    z = op.all_msd
    if fig == False:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    ax.scatter(x, y, z, marker=".", alpha=0.3)

    if show:
        plt.show()

    return


def other_plot(op, fig=False, ax=False, show=True):
    fraction_tick = [i for i in range(1, int(op.frame_total / op.frame_step) + 1)]
    create_box_plot(
        op.tmframe_occ,
        fraction_tick,
        y_label="Fraction inside the drop",
        x_label="Frame number",
        y_lim=(),
        title="Percent Occupation of Track in Drop per Frame Over All Experiments",
    )

    for i in op.tmframe_occ:
        w_i = np.ones_like(i) / float(len(i))
        if fig == False:
            plt.hist(i, histtype="step", weights=w_i)
            plt.xlabel("Fraction inside the drop")
            plt.ylabel("Probability")
            plt.title("Percent Occupation of Track in Drop per Frame")
        else:
            ax.hist(i, histtype="step", weights=w_i)
            ax.xlabel("Fraction inside the drop")
            ax.ylabel("Probability")
            ax.title("Percent Occupation of Track in Drop per Frame")
    if show:
        plt.show()
    return


def animate(i, ax):
    # azimuth angle : 0 deg to 360 deg
    ax.view_init(elev=i, azim=i * 4)

    return


# #def overall_plot3D(op,which = "all", fig = False, ax = False,cmap = 'warm', save = True):
#     is_fig = fig

#     b = op.viable_drop_total
#     d = op.segmented_drop_files
#     c = op.in_track_total
#     c1 = op.io_track_total
#     c2 = op.ot_track_total
#     cp = op.in_msd_all
#     cp1 = op.io_msd_all
#     cp2 = op.ot_msd_all


#     for i in range(len(b)):
#         if fig == False:
#             is_fig = plt.figure()
#             ax = fig.add_subplot(111,projection = '3d')
#         if len(d[i]) != 0:
#             img = mpimg.imread(d[i][0])
#             #timg = ax2.imshow(img, cmap=plt.get_cmap('gray'))

#         for j in range(len(b[i])):
#             if which == "all" or which == "in":
#                 for l in range(len(c[i][j])):
#                     if len(c[i][j][l])!=0:
#                         temp = np.array(c[i][j][l])
#                         #plt.plot(temp[0],temp[1],'b-')
#                         im = ax.scatter(temp[0],temp[1],(np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),cmap=cmap, norm = norm)
#             if which == "all" or which == "io":
#                 for l in range(len(c1[i][j])):
#                     if len(c1[i][j][l])!=0:
#                         temp = np.array(c1[i][j][l])
#                         #plt.plot(temp[0],temp[1],'g-')
#                         im = ax.scatter(temp[0],temp[1],(np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),s = 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),cmap=cmap, norm = norm)
#             if which == "all" or which == "out":
#                 for l in range(len(c2[i][j])):
#                     if len(c2[i][j][l])!=0:
#                         temp = np.array(c2[i][j][l])
#                         #plt.plot(temp[0],temp[1],'r-')
#                         im = ax.scatter(temp[0],temp[1],(np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),cmap=cmap, norm = norm)

#             # if (len(b[i][j])>0):
#             #   for k in range(len(b[i][j])):
#             #       circles(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], c=drop_color[j], alpha = 0.35)
#         #fig.colorbar(im,ax=ax3)
#         #plt.savefig("Frame_{0}".format(i))
#         #fig.show()
#         if save:
#             ani = animation.FuncAnimation(fig, animate,fargs = [ax],frames=180, interval=50)
#             fn = op.wd + "{0}".format(i)
#             ani.save(fn+'.mp4',writer='ffmpeg',fps=1000/50)
#             ani.save(fn+'.gif',writer='imagemagick',fps=1000/50)
#     return


# #def overall_plot2D_contour(op, which = "all", scatter = 0, line = 0, fig = False, ax = False,cmap = 'warm', show = True, delay = False):

#     is_fig = fig

#     b = op.viable_drop_total
#     d = op.segmented_drop_files
#     c = op.in_track_total
#     c1 = op.io_track_total
#     c2 = op.ot_track_total
#     cp = op.in_msd_all
#     cp1 = op.io_msd_all
#     cp2 = op.ot_msd_all

#     for i in range(len(b)):
#         if is_fig == False:
#             fig = plt.figure()
#             ax = fig.add_subplot(111)
#         if len(d[i]) != 0:
#             img = mpimg.imread(d[i][0])
#             timg = plt.imshow(img,cmap=plt.get_cmap('gray'),origin = "lower")
#             copy_array_in = np.zeros(np.shape(img))
#             copy_array_io = np.zeros(np.shape(img))
#             copy_array_ot = np.zeros(np.shape(img))
#             copy_array_all = np.zeros(np.shape(img))


#         random_choose_c = [0,3]
#         random_choose_c1 = [0,3]
#         random_choose_c2 = [0,3]
#         choose_b = np.random.randint(0,len(b[i]),2)
#         for j in range(len(b[i])):


#             for l in range(len(c[i][j])):
#                 if len(c[i][j][l])!=0:
#                     temp = np.array(c[i][j][l])
#                     copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#                     copy_array_in[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#                     #plt.plot(temp[0],temp[1],'b-')
#                     if (which == "all" or which == "in") and scatter :
#                         ax.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),cmap=cmap, norm = norm)
#                     if (which == "all" or which == "in") and line and (l in random_choose_c) and (j in choose_b):
#                         ax.plot(temp[0],temp[1],c = 'r')

#             for l in range(len(c1[i][j])):
#                 if len(c1[i][j][l])!=0:
#                     temp = np.array(c1[i][j][l])
#                     copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#                     copy_array_io[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#                     #plt.plot(temp[0],temp[1],'g-')
#                     if (which == "all" or which == "io") and scatter:
#                         ax.scatter(temp[0],temp[1],s = 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),cmap=cmap, norm = norm)
#                     if (which == "all" or which == "io") and line and l in random_choose_c1:
#                         ax.plot(temp[0],temp[1],c = 'b')

#             for l in range(len(c2[i][j])):
#                 if len(c2[i][j][l])!=0:
#                     temp = np.array(c2[i][j][l])
#                     copy_array_all[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#                     copy_array_ot[np.array(temp[1],'int'),np.array(temp[0],'int')] += 1
#                     #plt.plot(temp[0],temp[1],'r-')
#                     if (which == "all" or which == "out") and scatter:
#                         ax.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),cmap=cmap, norm = norm)
#                     if (which == "all" or which == "out") and line and l in random_choose_c2:
#                         ax.plot(temp[0],temp[1],c = 'g')
#             # if (len(b[i][j])>0):
#             #   for k in range(len(b[i][j])):
#             #       circles(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], c=drop_color[j], alpha = 0.35)
#         cont = ax.contour(copy_array_all)
#         fig.colorbar(cont)
#         #plt.savefig("Frame_{0}".format(i))
#         if show:
#             plt.show()
#     return


# #def overall_plot2D(op, which = "all", fig = False, ax = False,cmap = 'warm', show = True):
#     drop_color = ["y","b","r","g","m"]
#     is_fig = fig

#     b = op.viable_drop_total
#     d = op.segmented_drop_files
#     c = op.in_track_total
#     c1 = op.io_track_total
#     c2 = op.ot_track_total
#     cp = op.in_msd_all
#     cp1 = op.io_msd_all
#     cp2 = op.ot_msd_all

#     for i in range(len(b)):
#         if is_fig == False:
#             fig = plt.figure()
#             ax = fig.add_subplot(111)
#         if len(d[i]) != 0:
#             img = mpimg.imread(d[i][0])
#             timg = plt.imshow(img,cmap=plt.get_cmap('gray'))

#         for j in range(len(b[i])):
#             if which == "all" or which == "in":
#                 for l in range(len(c[i][j])):
#                     if len(c[i][j][l])!=0:
#                         temp = np.array(c[i][j][l])
#                         #plt.plot(temp[0],temp[1],'b-')
#                         ax.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp[i][j][l]))),cmap=cmap, norm = norm)
#             if which == "all" or which == "io":
#                 for l in range(len(c1[i][j])):
#                     if len(c1[i][j][l])!=0:
#                         temp = np.array(c1[i][j][l])
#                         #plt.plot(temp[0],temp[1],'g-')
#                         ax.scatter(temp[0],temp[1],s = 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp1[i][j][l]))),cmap=cmap, norm = norm)
#             if which == "all" or which == "out":

#                 for l in range(len(c2[i][j])):
#                     if len(c2[i][j][l])!=0:
#                         temp = np.array(c2[i][j][l])
#                         #plt.plot(temp[0],temp[1],'r-')
#                         ax.scatter(temp[0],temp[1],s= 2,c = (np.zeros(len(temp[0]))+np.log10(np.array(cp2[i][j][l]))),cmap=cmap, norm = norm)

#             # if (len(b[i][j])>0):
#             #   for k in range(len(b[i][j])):
#             #       circles(b[i][j][k][0], b[i][j][k][1], b[i][j][k][2], c=drop_color[j], alpha = 0.35)
#         ax.colorbar()
#         #plt.savefig("Frame_{0}".format(i))
#         if show:
#             plt.show()
#     return


# https://stackoverflow.com/questions/22562364/circular-polar-histogram-in-python
def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True, **kwargs):
    """
    Produce a circular histogram of angles on ax.

    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').

    x : array
        Angles to plot, expected in units of radians.

    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.

    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.

    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.

    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    **kwargs : type
        Other arguments are passed directly to `matplotlib.axes.Axes.bar`.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.

    bins : array
        The edges of the bins.

    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x + np.pi) % (2 * np.pi) - np.pi

    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins + 1)

    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)

    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area / np.pi) ** 0.5
    # Otherwise plot frequency proportional to radius
    else:
        radius = n

    # Plot data on ax
    if "color" not in kwargs and "label" not in kwargs:
        patches = ax.bar(
            bins[:-1],
            radius,
            zorder=1,
            align="edge",
            width=widths,
            edgecolor="C0",
            fill=False,
            linewidth=1,
        )
    else:
        patches = ax.bar(
            bins[:-1],
            radius,
            zorder=1,
            align="edge",
            width=widths,
            edgecolor="C0",
            fill=True,
            linewidth=1,
            color=kwargs["color"],
            label=kwargs["label"],
            alpha=kwargs["alpha"],
        )

    # Set the direction of the zero angle
    ax.set_theta_offset(offset)

    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

    return n, bins, patches


def create_circular_mask(h, w, center=None, radius=None):
    """h,w are the dimensions of the image"""

    if center is None:  # use the middle of the image
        center = [int(w / 2), int(h / 2)]
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def plot_stacked_bar(
    data,
    series_labels,
    category_labels=None,
    show_values=False,
    value_format="{}",
    y_label=None,
    colors=None,
    grid=True,
    reverse=False,
):
    ##https://stackoverflow.com/questions/44309507/stacked-bar-plot-using-matplotlib
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    colors          -- List of color labels
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        axes.append(
            plt.bar(
                ind, row_data, bottom=cum_size, label=series_labels[i], color=colors[i]
            )
        )
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label)

    plt.legend()

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(
                    bar.get_x() + w / 2,
                    bar.get_y() + h / 2,
                    value_format.format(h),
                    ha="center",
                    va="center",
                )

    return
