import matplotlib.pyplot as plt

import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools

import numpy as np

from src.clusterings_space import *



def plot_clusterings_grid(data, clusterings, figsize = (30,30)) :

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



def plot_cl_space_grouping(cl_space, clustering_cl_space, figsize=(10,10), rev=True, alpha=1 ) :

    keys = cl_space.index

    #be careful with orderings
    nc = 10000
    color_names = ['grey','red','deepskyblue','saddlebrown','darkblue','lightpink','olivedrab',
                   'palegreen','darkorchid','palevioletred','cyan','orange'] + (['black'] * nc)
    
    points = np.asarray([[key[0],key[1]] for key in keys])
    plt.figure(figsize=figsize)
    plt.scatter(points.T[1], points.T[0], c=[color_names[i+1] for i in clustering_cl_space], s=500, marker="s", alpha=alpha)
    if rev :
      plt.gca().invert_yaxis()
    plt.grid(False)
    plt.show()




def plot_cl_space_3d(cl_space, clustering_clspace = None) :

    #colors for clusterings
    nc = 10000
    color_names = ['grey','red','deepskyblue','saddlebrown','darkblue','lightpink','olivedrab',
                   'palegreen','darkorchid','palevioletred','cyan','orange'] + (['black'] * nc)


    keys = cl_space.index
    points = np.asarray(cl_space.values)

    trace2 = go.Scatter3d(
        x = points.T[0],
        y = points.T[1],
        z = points.T[2],
        text = list(map(str,keys)),
        mode='markers',
        marker=dict(
            color= 'grey' if clustering_clspace is None
                        else [color_names[i+1] for i in clustering_clspace],
            size=4,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.5
        )
    )
    data = [trace2]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='simple-3d-scatter')


def plot_cl_space_2d(cl_space, clustering_clspace = None) :

    nc = 10000
    color_names = ['grey','red','deepskyblue','saddlebrown','darkblue','lightpink','olivedrab',
                   'palegreen','darkorchid','palevioletred','cyan','orange'] + (['black'] * nc)


    keys = cl_space.index
    points = np.asarray(cl_space.values)


    trace2 = go.Scatter(
        x = points.T[0],
        y = points.T[1],
        text = list(map(str,keys)),
        mode='markers',
        marker=dict(
            color= 'grey' if clustering_clspace is None
                        else [color_names[i+1] for i in clustering_clspace],
            size=4,
            symbol='circle',
            line=dict(
                color='rgb(204, 204, 204)',
                width=1
            ),
            opacity=0.5
        )
    )
    data = [trace2]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig, filename='simple-2d-scatter')

