import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import logging

import sklearn.cluster as clust

import sompy
from sompy.visualization.mapview import View2D
from sompy.visualization.umatrix import UMatrixView

from tfprop_sompy import tfprop_config as tfpinit

from . import tfprop_config as tfpinit

from .utils.strings import str_wrap_chem

HUGE = 10000000
labels_rand_seed = 555  # tweak random seed if you do not like labeling result


def kmeans_clust(som, n_clusters=8):
    print("Performing K-means clustering to SOM trained data...")
    cl_labels = clust.KMeans(n_clusters=n_clusters, random_state=tfpinit.km_seed).fit_predict(som.codebook.matrix)

    return cl_labels

# This returns our function used to "filter"
# our output in order to make differences more exaggerated
def create_knee_function(cutoff: float, middle: float, maxm: float, minm: float):
    def new_knee(x: float):
        # [-1,1] value determining closeness to "extremes"
        if x < middle:
            diff = 0.5*(x-middle)/(middle-minm)
        else:
            diff = 0.5*(x-middle)/(maxm-middle)
        # 0 if we're below our cutoff
        if np.abs(diff) < cutoff:
            return middle
        # Linearly interpolate between our middle and the boundary
        else:
            if diff < 0:
                interp_diff = (diff + cutoff)/(0.5 - cutoff)
                return middle + interp_diff*(middle - minm)
            else:
                interp_diff = (diff - cutoff)/(0.5 - cutoff)
                return middle + interp_diff*(maxm - middle)
    return new_knee
    
def render_cluster_borders_to_axes(ax, cl_labels: np.ndarray, msz: int):
    for i in range(len(cl_labels)):
        rect_x = [i // msz[1], i // msz[1],
                  i // msz[1] + 1, i // msz[1] + 1]
        rect_y = [i % msz[1], i % msz[1] + 1,
                  i % msz[1] + 1, i % msz[1]]

        if i % msz[1] + 1 < msz[1]:  # top border
            if cl_labels[i] != cl_labels[i+1]:
                ax.plot([rect_x[1], rect_x[2]], [rect_y[1], rect_y[2]], 'k-',
                        lw=1.5)

        if i + msz[1] < len(cl_labels):  # right border
            if cl_labels[i] != cl_labels[i+msz[1]]:
                ax.plot([rect_x[2], rect_x[3]], [rect_y[2], rect_y[3]], 'k-',
                        lw=1.5)
def dataframe_to_coords(som, target_dataframe):
    trained_columns = som._component_names[0]
    return som.bmu_ind_to_xy(som.project_data(target_dataframe[trained_columns].values))

def render_points_to_axes(ax, coord):
    ax.scatter(coord[:, 0]+0.5, coord[:, 1]+0.5, c='k', marker='o')                          

# This function prints labels on cluster map
def clusteringmap_category(ax,sm,n_clusters,dataset,colorcategory,labels, savepath):
    """
    Description:
    This function is used to output maps that prints colors on dots based
    on their properties
    """
    categories = dataset[colorcategory] #if colorcategory is one col of the dataset
    cmap = plt.get_cmap("tab20") #cmap for background
    n_palette = 20  # number of different colors in this color palette
    color_list = [cmap((i % n_palette)/n_palette) for i in range(n_clusters)]
    msz = sm.codebook.mapsize
    proj = sm.project_data(sm.data_raw)
    coord = sm.bmu_ind_to_xy(proj)

    fig, ax = plt.subplots(1, 1, figsize=(30,30))

    cl_labels = clust.KMeans(n_clusters=n_clusters,random_state=555).fit_predict(sm.codebook.matrix)
        
    # fill each rectangular unit area with cluster color
    #  and draw line segment to the border of cluster
    norm = mpl.colors.Normalize(vmin=0, vmax=n_palette, clip=True)
    ax.pcolormesh(cl_labels.reshape(msz[0], msz[1]).T % n_palette,
                cmap=cmap, norm=norm, edgecolors='face',
                lw=0.5, alpha=0.5)

    ax.scatter(coord[:, 0]+0.5, coord[:, 1]+0.5, c='k', marker='o')
    ax.axis('off')

    categoryname = list(dataset.groupby(colorcategory).count().index)
    categories_int = categories.apply(categoryname.index)

    N = len(categoryname)
    cmap_labels = plt.cm.gist_ncar
    # extract all colors from the .jet map
    cmaplist = [cmap_labels(i) for i in range(cmap_labels.N)]
    # create the new map
    cmap_labels = cmap_labels.from_list('Custom cmap', cmaplist, cmap_labels.N)
    # define the bins and normalize
    bounds = np.linspace(0,N,N+1)
    norm_labels = mpl.colors.BoundaryNorm(bounds, cmap_labels.N)

    scat = ax.scatter(coord[:, 0]+0.5, coord[:, 1]+0.5, c=categories_int,s=300,cmap=cmap_labels,norm=norm_labels)
    cbar = plt.colorbar(scat, spacing='proportional',ticks=bounds)
    cbar.ax.get_yaxis().set_ticks([])
    
    for j, lab in enumerate(categoryname):
        cbar.ax.text(1, (2 * j + 1) / (2*(len(categoryname))), lab, ha='left', va='center', fontsize=30)
    cbar.ax.get_yaxis().labelpad = 15
    # cbar.ax.set_ylabel('# of contacts', rotation=270)
    ax.axis('off')
    
    
    for label, x, y in zip(labels, coord[:, 0], coord[:, 1]):
        x += 0.2
        y += 0.2
        # "+ 0.1" means shift of label location to upperright direction

        # randomize the location of the label
        #   not to be overwrapped with each other
        x += 0.1 * np.random.randn()
        y += 0.3 * np.random.randn()

        # wrap of label for chemical compound
        #label = str_wrap(label)

        ax.text(x+0.4, y+0.4, label, horizontalalignment='left', verticalalignment='bottom',rotation=30, fontsize=12, weight='semibold')
    # cl_labels = som.cluster(n_clusters)
    cl_labels = clust.KMeans(n_clusters = n_clusters, random_state = 555).fit_predict(sm.codebook.matrix)

    for i in range(len(cl_labels)):
        rect_x = [i // msz[1], i // msz[1],
                i // msz[1] + 1, i // msz[1] + 1]
        rect_y = [i % msz[1], i % msz[1] + 1,
                i % msz[1] + 1, i % msz[1]]

        if i % msz[1] + 1 < msz[1]:  # top border
            if cl_labels[i] != cl_labels[i+1]:
                ax.plot([rect_x[1], rect_x[2]],
                        [rect_y[1], rect_y[2]], 'k-', lw=2.5)

        if i + msz[1] < len(cl_labels):  # right border
            if cl_labels[i] != cl_labels[i+msz[1]]:
                ax.plot([rect_x[2], rect_x[3]],
                        [rect_y[2], rect_y[3]], 'k-', lw=2.5)

    plt.savefig(savepath)
    return cl_labels

def render_posmap_to_axes(ax, som, placement_name_df, raw_name_df, n_clusters, cl_labels,
                show_data=True, labels=False):
    # user defined color list
    # color_list = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
    #               '#ffff33', '#a65628', '#f781bf', '#C71585', '#00FFFF',
    #               '#00FF00', '#F08080', '#DAA520', '#B0E0E6', '#FAEBD7']

    # predefined color map in matplotlib
    # see http://matplotlib.org/examples/color/colormaps_reference.html
    cmap = plt.get_cmap("tab20")
    n_palette = 20  # number of different colors in this color palette
    color_list = [cmap((i % n_palette)/n_palette) for i in range(n_clusters)]

    msz = som.codebook.mapsize
    proj = som.project_data(som.data_raw)
    coord = som.bmu_ind_to_xy(proj)

    ax.set_xlim([0, msz[0]])
    ax.set_ylim([0, msz[1]])

    # fill each rectangular unit area with cluster color
    #  and draw line segment to the border of cluster
    norm = mpl.colors.Normalize(vmin=0, vmax=n_palette, clip=True)
    ax.pcolormesh(cl_labels.reshape(msz[0], msz[1]).T % n_palette,
                  cmap=cmap, norm=norm, edgecolors='face',
                  lw=0.5, alpha=0.5)
    
    render_cluster_borders_to_axes(ax, cl_labels, msz)
    ax.axis('off')

    if show_data:
        ax.scatter(coord[:, 0]+0.5, coord[:, 1]+0.5, c='k', marker='o')

    # place label of each chemical substance
    if labels:
        labels = []
        for i in range(len(raw_name_df)):
            for t in range(len(placement_name_df)):
                if raw_name_df.iloc[i, 0] == placement_name_df.iloc[t, 0]:
                    labels.append(raw_name_df.iloc[i, 0])

        # tweak random seed if you do not like labeling result
        # np.random.seed(labels_rand_seed)
        for label, x, y in zip(labels, coord[:, 0], coord[:, 1]):
            x += 0.1
            y += 0.1
            # "+ 0.1" means shift of label location to upperright direction

            # randomize the location of the label
            #   not to be overwrapped with each other
            # x_text += 0.1 * np.random.randn()
            # y += 0.3 * np.random.randn()

            # wrap of label for chemical compound
            # label = str_wrap_chem(label)

            ax.text(x+0.5, y+0.5, label,
                    horizontalalignment='left', verticalalignment='bottom',
                    rotation=30, fontsize=14, weight='semibold')

            
def show_posmap(som, placement_name_df, raw_name_df, n_clusters, cl_labels,
                show_data=True, labels=False, isOutPosmap=False):

    fig, ax = plt.subplots(1, 1, figsize=tfpinit.posmap_size)
    ax.set_aspect('equal')
    ax.tick_params(labelbottom=False)  # turn off tick label
    ax.tick_params(labelleft=False)  # turn off tick label
    ax.grid()
    ax.axis('off')
    render_posmap_to_axes(ax, som, placement_name_df, raw_name_df, n_clusters, cl_labels,
                show_data, labels)

    fig.show()

    # save figure of SOM positioning map
    if isOutPosmap:
        print("Saving figure of SOM positioning map to {}...".
              format(tfpinit.fout_posmap))
        fig.savefig(tfpinit.fout_posmap)


class ViewTFP(View2D):
    """Map viewer override that allows for specifying the normalization method"""
    
    def __init__(self, *args, knee_value=0, stdev_colorscale_coeff=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.knee_value = knee_value
        self.stdev_colorscale_coeff = stdev_colorscale_coeff

    def _calculate_figure_params(self, som, which_dim, col_sz,
                                 width=None, height=None):
        """ Class method in MapView._calculate_figure_params() overrided """
        codebook = som._normalizer.denormalize_by(som.data_raw,
                                                  som.codebook.matrix)

        indtoshow, sV, sH = None, width, height

        if which_dim == 'all':
            dim = som._dim
            row_sz = np.ceil(float(dim) / col_sz)
            msz_row, msz_col = som.codebook.mapsize
            ratio_hitmap = msz_row / float(msz_col)
            ratio_fig = row_sz / float(col_sz)
            indtoshow = np.arange(0, dim).T
            sH, sV = (width, height) or (16, 16*ratio_fig*ratio_hitmap)

        elif type(which_dim) == int:
            dim = 1
            msz_row, msz_col = som.codebook.mapsize
            ratio_hitmap = msz_row / float(msz_col)
            indtoshow = np.zeros(1)
            indtoshow[0] = int(which_dim)
            sH, sV = (width, height) or (16, 16*ratio_hitmap)

        elif type(which_dim) == list:
            max_dim = codebook.shape[1]
            dim = len(which_dim)
            row_sz = np.ceil(float(dim) / col_sz)
            msz_row, msz_col = som.codebook.mapsize
            ratio_hitmap = msz_row / float(msz_col)
            ratio_fig = row_sz / float(col_sz)
            indtoshow = np.asarray(which_dim).T
            sH, sV = (width, height) or (16, 16*ratio_fig*ratio_hitmap)

        no_row_in_plot = dim / col_sz + 1  # 6 is arbitrarily selected
        if no_row_in_plot <= 1:
            no_col_in_plot = dim
        else:
            no_col_in_plot = col_sz

        axis_num = 0

        width = sH
        height = sV

        return (width, height, indtoshow, no_row_in_plot, no_col_in_plot,
                axis_num)

    def prepare(self, *args, **kwargs):
        self._close_fig()
        self._fig = plt.figure(figsize=(self.width, self.height))
        self._fig.suptitle(self.title)
        plt.rc('font', **{'size': self.text_size})

# NOTE: I'm using type rST syntax right now because I don't want to make
# excessive changes to make this use python's formal typing system.
# We may want to move to python's typing system later
    def show(self, som :sompy.sompy.SOM, cl_labels :list, what='codebook', which_dim='all',
             cmap=None, col_sz=None, desnormalize=False, col_norm=None, normalizer="linear", savepath="",
             isOutHtmap=True):
        """ Class method in View2D.show() overridden
        
        There's now an extra parameter, "col_norm", which is used to determine whether to normalize by
        the median or the mean

        :param som: The self-organizing map to visualize
        :type som: sompy.sompy.SOM
        :param cl_labels: Cluster labels (?)
        :type cl_labels: list
        :param what: unused
        :type what: str
        :param which_dim: Which dimensions to display
        :type which_dim: "all" or int or list
        :param cmap: The color map to use for the plot
        :type cmap: matplotlib.colors.Colormap
        :param col_sz: Number of columns
        :type col_sz: integer
        :param desnormalize: Whether or not to denormalize the codebook
        :type desnormalize: Boolean
        :param col_norm: Determines what "middle value" to use for normalization
        :type col_norm: "median" or "mean"
        :param normalizer: Which normalizer to use
        :type normalizer: "linear" or "log"
        """
        (self.width, self.height, indtoshow, no_row_in_plot, no_col_in_plot,
         axis_num) = \
            self._calculate_figure_params(som, which_dim, col_sz,
                                          width=self.width, height=self.height)
        self.prepare()
        # Mathtext font to sans-serif
        mpl.rcParams['mathtext.fontset'] = 'custom'
        mpl.rcParams['mathtext.rm'] = 'sans\-serif'
        mpl.rcParams['mathtext.cal'] = 'sans\-serif'

        cmap = cmap or plt.get_cmap('RdYlBu_r')

        if not desnormalize:
            codebook = som.codebook.matrix
        else:
            codebook = som._normalizer.denormalize_by(som.data_raw,
                                                      som.codebook.matrix)

        if which_dim == 'all':
            names = som._component_names[0]
        elif type(which_dim) == int:
            names = [som._component_names[0][which_dim]]
        elif type(which_dim) == list:
            names = som._component_names[0][which_dim]

        while axis_num < len(indtoshow):
            axis_num += 1
            ax = plt.subplot(no_row_in_plot, no_col_in_plot, axis_num)
            ind = int(indtoshow[axis_num-1])

            if col_norm == 'median':  # normalized by median        
                middle_point = np.median(codebook[:, ind].flatten())
            else: # normalized by mean
                middle_point = np.mean(codebook[:, ind].flatten())

            cb_min = np.min(codebook[:, ind].flatten())
            cb_max = np.max(codebook[:, ind].flatten())

            min_color_scale = middle_point - self.stdev_colorscale_coeff \
                * np.std(codebook[:, ind].flatten())
            max_color_scale = middle_point + self.stdev_colorscale_coeff \
                * np.std(codebook[:, ind].flatten())
            min_color_scale = min_color_scale if min_color_scale >= \
                cb_min else cb_min
            max_color_scale = max_color_scale if max_color_scale <= \
                cb_max else cb_max
            
            # FIXME: Break this out into less hacked-in code
            # "Middle color" should be mean or median?
            # The min value should be at the bottom of the range
            # The max value should be at the top of the range
            # The "min_color_scale" value should be the absolute bottom of the original color range
            # The "max_color_scale" value should be the absolute top of the original color range
            # .5 - .5 * min((middle_point - min_color_scale) / (middle_point - cb_min), 1)
            # .5 + .5 * min((middle_point - max_color_scale) / (middle_point - cb_max), 1)
            # 
            # I probably can reorder this to make more sense
            # FIXME: This code currently biases the colors in a not very useful manner
            #cmap_bottom = .5 * (1 - min((middle_point - min_color_scale) / (middle_point - cb_min), 1))
            #cmap_top = .5 * (1 + min((middle_point - max_color_scale) / (middle_point - cb_max), 1))
            #my_cmap = ListedColormap(cmap(np.linspace(cmap_bottom, cmap_top, num=512)))
            my_cmap = cmap
            if normalizer == 'log':
                norm = mpl.colors.SymLogNorm(linthresh=(max_color_scale-min_color_scale)/100, vmin=min_color_scale,
                                          vmax=max_color_scale,
                                          clip=True)
            else:
                norm = mpl.colors.Normalize(vmin=min_color_scale,
                                            vmax=max_color_scale,
                                            clip=True)

            mp = codebook[:, ind].reshape(som.codebook.mapsize[0],
                                          som.codebook.mapsize[1])
            # FIXME: Break this out into less hacked-in code
            # Insert the scaling function here
            # as-is this is likely very slow - an immediate improvement would come
            # from using something such as "numba" here
            scaler = np.vectorize(create_knee_function(self.knee_value, middle_point, np.max(codebook[:, ind].flatten()), np.min(codebook[:, ind].flatten())))
            pl = ax.pcolormesh(scaler(mp.T), norm=norm, cmap=my_cmap)
            ax.set_xlim([0, som.codebook.mapsize[0]])
            ax.set_ylim([0, som.codebook.mapsize[1]])
            ax.set_aspect('equal')
            ax.set_title(names[axis_num - 1])
            # Disable ticks and tick labels
            disable_ticks = ["labelbottom", "labelleft", "bottom", "left", "right", "top"]
            disable_ticks_dict = {a: False for a in disable_ticks}
            # This unpacks the dict into the keyword arguments,
            # setting them all as false without needing to write each one out
            ax.tick_params(**disable_ticks_dict)

            plt.colorbar(pl, shrink=0.7)

            # draw line segment to the border of cluster
            msz = som.codebook.mapsize

            for i in range(len(cl_labels)):
                rect_x = [i // msz[1], i // msz[1],
                          i // msz[1] + 1, i // msz[1] + 1]
                rect_y = [i % msz[1], i % msz[1] + 1,
                          i % msz[1] + 1, i % msz[1]]

                if i % msz[1] + 1 < msz[1]:  # top border
                    if cl_labels[i] != cl_labels[i+1]:
                        ax.plot([rect_x[1], rect_x[2]],
                                [rect_y[1], rect_y[2]], 'k-', lw=1.5)

                if i + msz[1] < len(cl_labels):  # right border
                    if cl_labels[i] != cl_labels[i+msz[1]]:
                        ax.plot([rect_x[2], rect_x[3]],
                                [rect_y[2], rect_y[3]], 'k-', lw=1.5)

        # plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.tight_layout()
        plt.show()

        # save figure of heat map
        if isOutHtmap:
            print("Saving figure of heat map for all thermofluid prop. to {}...".format(savepath))
            self._fig.savefig(savepath)


class UMatrixTFP(UMatrixView):
    def show(self, som, placement_name_df, raw_name_df, savepath,
             distance2=1, row_normalized=False, show_data=True,
             contooor=True, blob=False, labels=False, cmap=None,
             isOutUmat=False):
        """ Class method in UMatrixView.show() overrided """
        umat = self.build_u_matrix(som, distance=distance2,
                                   row_normalized=row_normalized)
        msz = som.codebook.mapsize
        proj = som.project_data(som.data_raw)
        coord = som.bmu_ind_to_xy(proj)

        cmap = cmap or plt.get_cmap('RdYlBu_r')  # set color map

        # colorbar normalization
        min_color_scale = np.mean(umat.flatten()) - 1 * np.std(umat.flatten())
        max_color_scale = np.mean(umat.flatten()) + 1 * np.std(umat.flatten())
        min_color_scale = min_color_scale if min_color_scale >= \
            np.min(umat.flatten()) else np.min(umat.flatten())
        max_color_scale = max_color_scale if max_color_scale <= \
            np.max(umat.flatten()) else np.max(umat.flatten())
        norm = mpl.colors.Normalize(vmin=min_color_scale,
                                    vmax=max_color_scale,
                                    clip=True)

        fig, ax = plt.subplots(1, 1, figsize=(tfpinit.umatrix_size))
        ax.imshow(umat.T, cmap=cmap, alpha=0.7, norm=norm,
                  interpolation='lanczos')

        if contooor:
            mn = np.min(umat.flatten())
            mx = np.max(umat.flatten())
            std = np.std(umat.flatten())
            md = np.median(umat.flatten())
            mx = md + 0*std
            ax.contour(umat.T, np.linspace(mn, mx, 15), linewidths=0.7,
                       cmap=plt.cm.get_cmap('Blues'))

        if show_data:
            ax.scatter(coord[:, 0], coord[:, 1], c='k', marker='o')
            ax.axis('off')

        if labels:
            labels = []
            for i in range(len(raw_name_df)):
                for t in range(len(placement_name_df)):
                    if raw_name_df.iloc[i, 0] == placement_name_df.iloc[t, 0]:
                        labels.append(raw_name_df.iloc[i, 0])

            # tweak random seed if you do not like labeling result
            np.random.seed(labels_rand_seed)
            for label, x, y in zip(labels, coord[:, 0], coord[:, 1]):
                x += 0.1
                y += 0.1
                # "+ 0.1" means shift of label location to upperright direction

                # randomize the location of the label
                #   not to be overwrapped with each other
                # x_text += 0.1 * np.random.randn()
                y += 0.3 * np.random.randn()

                # wrap of label for chemical compound
                label = str_wrap(label)

                ax.text(x, y, label,
                        horizontalalignment='left', verticalalignment='bottom',
                        rotation=30, fontsize=14, weight='semibold')

        ax.set_xlim([0 - 0.5, msz[0] - 0.5])  # -0.5 for the sake of imshow()
        ax.set_ylim([0 - 0.5, msz[1] - 0.5])
        ax.set_aspect('equal')
        ax.tick_params(labelbottom='off')  # turn off tick label
        ax.tick_params(labelleft='off')  # turn off tick label
        ax.tick_params(bottom='off', left='off',
                       top='off', right='off')  # turn off ticks

        # fig.tight_layout()
        # fig.subplots_adjust(hspace=.0, wspace=.0)
        sel_points = list()

        plt.show()

        # save figure of U-matrix
        if isOutUmat:
            print("Saving figure of U-matrix to {}..."
                  .format(savepath))
            fig.savefig(savepath)

        return sel_points, umat


def potential_func(som, placement_name_df, raw_name_df, gauss_alpha=None,
                   show_data=True, labels=False, cmap=None):
    # predefined color map in matplotlib
    # see http://matplotlib.org/examples/color/colormaps_reference.html
    cmap = cmap or plt.get_cmap("RdYlBu_r")  # colormap for image
    # color_list = [cmap((i % n_palette)/n_palette) for i in range(n_clusters)]

    # *** calculate square distance and potential values on each nodes
    nnodes = som.codebook.nnodes
    codebook = som.codebook.matrix
    msz = som.codebook.mapsize
    proj = som.project_data(som.data_raw)
    coord = som.bmu_ind_to_xy(proj)

    # calculate variance of distance between all nodes
    if not gauss_alpha:
        dist_nodes = np.zeros(nnodes * (nnodes-1) // 2)

        k = 0
        for i in range(nnodes - 1):
            for j in range(i+1, nnodes):
                dist_vec = codebook[i] - codebook[j]
                dist_nodes[k] = np.linalg.norm(dist_vec)  # distance
                k += 1
                # print("max. distance: {}, min. distance: {}"
                #       .format(np.max(dist_nodes), np.min(dist_nodes)))
        gauss_alpha = np.std(dist_nodes)
        print("STD of distance between nodes: {}".format(gauss_alpha))

    # calculate potential function
    pot_mean = np.zeros(nnodes)
    npot_sum = np.zeros(nnodes, dtype=int)

    for i in range(nnodes - 1):
        for j in range(i+1, nnodes):
            dist_vec = codebook[i] - codebook[j]
            dist2 = (dist_vec * dist_vec).sum()  # square distance
            gauss_fac = np.exp(-dist2 / (2 * gauss_alpha*gauss_alpha))
            pot_mean[i] += gauss_fac
            pot_mean[j] += gauss_fac
            npot_sum[i] += 1
            npot_sum[j] += 1

    pot_mean = pot_mean / (npot_sum * np.sqrt(2 * np.pi * gauss_alpha))

    # *** plot mean potential values
    fig, ax = plt.subplots(1, 1, figsize=(tfpinit.potfunc_size))
    pl = ax.imshow(pot_mean.reshape(msz[0], msz[1]).T, cmap=cmap,
                   alpha=0.7, interpolation='lanczos')

    # fig.colorbar(pl, shrink=0.7)

    if show_data:
        ax.scatter(coord[:, 0], coord[:, 1], c='k', marker='o')
        ax.axis(False)

    if labels:
        labels = []
        for i in range(len(raw_name_df)):
            for t in range(len(placement_name_df)):
                if raw_name_df.iloc[i, 0] == placement_name_df.iloc[t, 0]:
                    labels.append(raw_name_df.iloc[i, 0])

        # tweak random seed if you do not like labeling result
        np.random.seed(labels_rand_seed)
        for label, x, y in zip(labels, coord[:, 0], coord[:, 1]):
            x += 0.1
            y += 0.1
            # "+ 0.1" means shift of label location to upperright direction

            # randomize the location of the label
            #   not to be overwrapped with each other
            # x_text += 0.1 * np.random.randn()
            y += 0.3 * np.random.randn()

            # wrap of label for chemical compound
            label = str_wrap(label)

            ax.text(x, y, label,
                    horizontalalignment='left', verticalalignment='bottom',
                    rotation=30, fontsize=14, weight='semibold')

    ax.set_xlim([0 - 0.5, msz[0] - 0.5])  # -0.5 for the sake of imshow()
    ax.set_ylim([0 - 0.5, msz[1] - 0.5])
    ax.set_aspect('equal')
    ax.tick_params(labelbottom=False)  # turn off tick label
    ax.tick_params(labelleft=False)  # turn off tick label
    ax.tick_params(bottom=False, left=False,
                   top=False, right=False)  # turn off ticks

    plt.show()

    # save figure of potential surface
    if tfpinit.isOutPot:
        print("Saving figure of potential surface to {}..."
              .format(tfpinit.fout_pot))
        fig.savefig(tfpinit.fout_pot)

    # *** automated clustering based on potential function
    print("Performing potential func. based clustering to SOM trained data...")

    cl_labels = np.empty(nnodes, dtype=int)

    UD2 = som.calculate_map_dist()  # square distance
    distance = 1
    neighborbor_inds = []
    for i in range(nnodes):
        # pick nearest neighborbors
        neighborbor_inds.append([j for j in range(nnodes)
                                 if UD2[i][j] <= distance and j != i])

    n_assigned = 0
    is_assigned = np.zeros(nnodes, dtype=bool)
    is_assigned_tmp = np.zeros(nnodes, dtype=bool)  # temporary for copy
    # masked list of unassigned nodes
    pot_mean_nassign = np.ma.array(pot_mean, mask=is_assigned)
    n_clusters = 0

    while n_assigned < nnodes:
        # search index of max. potential which is not assigned to any cluster
        max_ind = np.ma.argmax(pot_mean_nassign)
        cl_labels[max_ind] = n_clusters
        is_assigned_tmp[max_ind] = True
        n_assigned += 1

        srch_inds = []  # search indices
        srch_inds.append(max_ind)
        # is_valley = np.zeros(nnodes, dtype=bool)  # flag if node is valley
        while len(srch_inds) > 0:
            for i in [j for j in neighborbor_inds[srch_inds[0]]
                      if not is_assigned_tmp[j]]:
                if pot_mean_nassign[srch_inds[0]] >= pot_mean_nassign[i]:
                    cl_labels[i] = n_clusters
                    is_assigned_tmp[i] = True
                    n_assigned += 1

                    srch_inds.append(i)

                else:
                    pass

            # delete searched node
            srch_inds.pop(0)

        n_clusters += 1
        # masked list updated
        is_assigned = np.copy(is_assigned_tmp)
        pot_mean_nassign = np.ma.array(pot_mean, mask=is_assigned)

    # eliminate small cluster
    for i in range(n_clusters):
        clust_list = np.where(cl_labels == i)[0]
        if len(clust_list) <= 3:  # tiny cluster
            n_clusters -= 1
            cl_labels[clust_list] = -1  # -1 assigned to tiny cluster

    print('Number of clusters= {}'.format(n_clusters))
    # print(cl_labels)

    return cl_labels


def main():
    from . import tfprop_som as tfpsom

    codemat_df = pd.read_hdf(tfpsom.fout_train, 'sm_codebook_matrix')
    tfpsom.sm.codebook.matrix = codemat_df.as_matrix()

    # apply matplotlib plotting style
    try:
        plt.style.use(tfpinit.plt_style)
    except OSError:
        print('Warning: cannot find matplotlib style: {}'
              .format(tfpinit.plt_style))
        print('Use default style...')

    # perform K-means clustering
    if tfpinit.isExeKmean:
        cl_labels = kmeans_clust(tfpsom.sm, n_clusters=tfpsom.km_cluster)

        # plot positioning map with clustered groups
        show_posmap(tfpsom.sm, tfpsom.fluid_name_df, tfpsom.fluid_name_df,
                    tfpsom.km_cluster, cl_labels,
                    show_data=True, labels=True)

    # plot potential method and clustering
    if tfpinit.isExePot:
        cmap = plt.get_cmap('RdYlBu_r')  # set color map
        cl_labels = potential_func(tfpsom.sm, tfpsom.fluid_name_df,
                                   tfpsom.fluid_name_df,
                                   gauss_alpha=tfpinit.gauss_alpha,
                                   show_data=True, labels=True,
                                   cmap=cmap)

        # plot positioning map with clustered groups
        show_posmap(tfpsom.sm, tfpsom.fluid_name_df, tfpsom.fluid_name_df,
                    tfpsom.km_cluster, cl_labels,
                    show_data=True, labels=True)

    # plot heat map for each thermofluid property using SOMPY View2D
    if tfpinit.isExeHtmap:
        cmap = plt.get_cmap('RdYlBu_r')  # set color map
        cl_labels = kmeans_clust(tfpsom.sm, n_clusters=tfpsom.km_cluster)
        htmap_x, htmap_y = tfpinit.heatmap_size
        viewTFP = ViewTFP(htmap_x, htmap_y, '', text_size=14)
        
        viewTFP.show(tfpsom.sm, cl_labels, col_sz=tfpinit.heatmap_col_sz,
                     which_dim='all', desnormalize=True, col_norm='median',
                     cmap=cmap)

    # plot U-matrix using SOMPY UMatrixView
    if tfpinit.isExeUmat:
        umatrixTFP = UMatrixTFP(0, 0, '', text_size=14)

        cmap = plt.get_cmap('RdYlBu_r')  # set color map
        umat = umatrixTFP.show(tfpsom.sm, tfpsom.fluid_name_df,
                               tfpsom.fluid_name_df,
                               show_data=True, labels=True, contooor=False,
                               cmap=cmap)

if __name__ == "__main__":
    main()
