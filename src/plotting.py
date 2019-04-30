import matplotlib.pyplot as plt

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

#import plotly.graph_objs as go

import matplotlib.cm as cm
import numpy as np

from src.clusterings_space import *


def plot_dendrogram(linkage_matrix):

    n_leaves = len(leaves_list(linkage_matrix))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params(
      axis='x',          # changes apply to the x-axis
      which='both',      # both major and minor ticks are affected
      bottom=False,      # ticks along the bottom edge are off
      top=False,         # ticks along the top edge are off
      labelbottom=False) # labels along the bottom edge are off    

    ddata = dendrogram(linkage_matrix, show_leaf_counts=False, orientation='left',
                       color_threshold=0, above_threshold_color = '#000000', no_labels=False)
    for i, d, c in zip(ddata['icoord'], ddata['dcoord'], 
                       ddata['color_list']):       
        x = 0.5 * sum(i[1:3])
        y = d[1]
        
        node_name = (y+n_leaves-1)

    plt.savefig('iris-dendrogram-nolabels.png', bbox_inches='tight')

# plot counts of columns
def plot_sizes(counts, how_many) :
    counts_ = counts.copy()
    counts_.reverse()
    counts_ = [counts_[i] for i in how_many]
    plt.bar(range(len(counts_)),counts_)
    plt.xticks(range(0,len(counts_)),map(lambda n: n+1, how_many))
    #plt.yticks([])
    plt.xlabel("nth most repeated column")
    plt.ylabel("multiplicity")


def plot_clusterings_grid(data, clusterings, figsize = (30,30)) :

    #colors for clusterings
    nc = 10000
    color_names = ['grey','red','deepskyblue','saddlebrown','darkblue','lightpink','olivedrab',
                   'palegreen','darkorchid','palevioletred','cyan','orange'] + (['black'] * nc)

    m_axis = len(set([key[0] for key in clusterings.keys()]))
    
    k_axis = len(set([key[1] for key in clusterings.keys()]))

    fig, axes = plt.subplots(nrows = m_axis, ncols = k_axis,figsize=figsize,
                                 subplot_kw={'xticks': [], 'yticks': []})

    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    
    for ax, kc in zip(axes.flat, clusterings.items()) :
        key, clustering = kc
        if num_clusters(clustering) > len(color_names) :
            print("more clusters than available colors!")
        ax.scatter(data.T[0], data.T[1], c=[color_names[i+1] for i in clustering], s = 30)
        ax.set_title(str(key[0]) + ", " + str(key[1]) + "; " + str(num_clusters(clustering)))

    plt.tight_layout()
    plt.show()


def plot_clustering(data, clustering, figsize=(15,10), dotsize=30) :
    #colors for clusterings
    nc = 10000
    color_names = ['grey','red','deepskyblue','saddlebrown','darkblue','lightpink','olivedrab',
                   'palegreen','darkorchid','palevioletred','cyan','orange'] + (['black'] * nc)

    plt.figure(figsize=figsize)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(data.T[0], data.T[1], c=[color_names[i+1] for i in clustering], s = dotsize)
    plt.show()
    
    
def plot_pairs(data, pairs, i = None, figsize=(15,10), dotsize = 30) :
    
    if i == None :
        points = list(sum(pairs, ()))
    else :
        points = pairs[i]
    
    colors = len(data) * ['grey'] + len(points) * ['red']
    sizes = len(data) * [dotsize] + len(points) * [int(dotsize * 3)]
    
    plt.figure(figsize=figsize)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(list(data.T[0]) + [data.T[0][i] for i in points],
                list(data.T[1]) + [data.T[1][i] for i in points], c=colors, s = sizes)
    plt.show()
    plt.show()


def plot_cl_space_grouping(keys, clustering, figsize=(10,10), rev=True, dotsize=100 ) :

    color_names = [i * 100 / len(clustering) for i in range(max(clustering)+2)]

    points = np.asarray([[key[0],key[1]] for key in keys])
    #noise_points = np.asarray([ points[i] for i in range(len(points)) if clustering[i] == -1 ])
    #points = np.asarray([ points[i] for i in range(len(points)) if clustering[i] != -1 ])
    
    plt.figure(figsize=figsize)
    plt.scatter(points.T[1], points.T[0], c=[color_names[i] for i in clustering if i!=-1], cmap=cm.jet, s=dotsize, marker="s")
    #if len(noise_points) > 0 :
    #    plt.scatter(noise_points.T[1], noise_points.T[0], c=['grey']*len(noise_points), s=dotsize, marker="s")
    if rev :
        plt.gca().invert_yaxis()
    plt.grid(False)
    plt.show()


def plot_cl_space_2d(cl_space, clustering_clspace = None, dotsize = 10, figsize = (7,7), alpha = 0.8) :

    color_names = [i * 100 / len(clustering_clspace) for i in range(max(clustering_clspace)+2)]

    plt.figure(figsize=figsize)
    plt.scatter(cl_space[0], cl_space[1], s=dotsize, c=[color_names[i] for i in clustering_clspace if i!=-1], cmap=cm.jet, alpha=alpha)
    plt.xticks([])
    plt.yticks([])
    plt.show()


