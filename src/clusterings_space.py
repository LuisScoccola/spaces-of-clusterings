import numpy as np
import math
import pandas as pd

from src.partition_tree import *

from scipy.stats import mode
from sklearn.decomposition import *
from scipy.cluster.hierarchy import *
from scipy.spatial.distance import pdist
import hdbscan


### Main Algorithms

def pref(cls_df, n_dim = 3, tol_var = 0.0):

    keys = cls_df.index
    cols = cls_df.columns
    cls_df.index = range(len(cls_df.index))
    cls_df.columns = range(len(cls_df.columns))

    cl_space, counts, reps = dim_red(cls_df, n_dim = n_dim, tol_var = tol_var)

    tol = 0.1

    clustering_cl_space = trivial_clustering_cl_space(cl_space,rtol=tol)

    cls_df.index = keys
    cls_df.columns = cols

    return(clustering_cl_space, keys, counts)


def hpref(cls_df, max_l = 5) :
    keys = cls_df.index
    cols = cls_df.columns
    cls_df.index = range(len(cls_df.index))
    cls_df.columns = range(len(cls_df.columns))

    pt = PartitionTree(cls_df)

    pt.split_until(score_hpref, max_l, 0)

    clustering_cl_space = partition_to_labels(pt.induced_partition(), cls_df)

    cls_df.index = keys
    cls_df.columns = cols

    return(clustering_cl_space, keys, pt.to_linkage_matrix(score_hpref))



#####

def partition_to_labels(p, dataset) :
    m = len(dataset)
    labels = [-1] * m
    for n,l in enumerate(p) :
        for x in l :
            labels[x] = n
            
    return(labels)


#count number of clusters
def num_clusters(cl_labels) :
    return(len(set(cl_labels).difference([-1])))


#compute sizes of clusters
def clusters_sizes(cl_labels) :

    sizes = { l:0 for l in list(set(cl_labels)) }

    for l in cl_labels :
        sizes[l] += 1

    return(sizes)

# return labeles ordered by the size of the cluster they represent
def clusters_by_size(cl_labels) :
    sizes = clusters_sizes(cl_labels)
    labels = list(set(cl_labels))

    labels.sort(key = lambda l : sizes[l], reverse = True)

    return(labels)

# given two labels (for two points), determine if the labeled points belong to the same cluster or not
def comparison(n,m) :
    if n == -1 or m == -1 :
        return(1)
    if n == m :
        return(0)
    else :
        return(1)


# given a labeling (of a set), return a vector of 0s and 1s encoding whether points belong to the same cluster or not
def clustering_as_vector_efficient2(cl_labels, pairs) :
    return([comparison(cl_labels[n],cl_labels[m]) for (n,m) in pairs])


# given a dictionary of clusterings (labelings) of a fixed set X return a data frame where the columns are pairs
# of points of X (pairs of natural numbers (n,m), n != m < |X|) and a row for each clustering
def clusterings_as_df(cls, n_dim = None, seed = None) :
    N = len(list(cls.values())[0])

    if n_dim != None :
        if seed != None :
            np.random.seed(seed)
        fst_components = np.random.randint(N,size=n_dim)
        snd_components = [ np.random.randint(low = i, high = N) for i in fst_components ]
        pairs = list(zip(fst_components,snd_components))
    else :
        pairs = [ (n,m) for n in range(0,N) for m in range(n, N) ]


    cls_as_vecs = { k:(clustering_as_vector_efficient2(cl, pairs)) for k,cl in cls.items() }

    cls_as_df = pd.DataFrame.from_dict(cls_as_vecs, orient='index', columns = pairs)
   
    return(cls_as_df)

# given a dictionary of clusterings (labelings) of a fixed set X return a data frame where the columns are
# *a random subsample* (of length |X|) of the pairs of points of X (pairs of natural numbers (n,m), n != m < |X|)
# and a row for each clustering
def clusterings_as_df_subsample(cls) :
    N = len(list(cls.values())[0])
    return(clusterings_as_df(cls, n_dim = N))



# dimension reduction of clusterings space by joining together equal columns,
# and keeping only the ones that appear the most
def dim_red(cls_df, n_dim = 3, tol_var = 0.0) :

    #N = len(list(clusterings.values())[0])

    if tol_var > 0 :
        cls_df = cls_df.loc[:, cls_df.var() >= tol_var ]

    # calculate how many times each column repeats
    counts = cls_df.T.groupby(cls_df.T.columns.tolist()).cumcount() + 1
    #print(counts)

    # delete repeated columns
    cls_df_ = cls_df.T.drop_duplicates(keep='last').T

    representatives = cls_df_.columns.tolist()
    
    # delete repeated columns from the counts dataframe and sort it
    counts = counts[cls_df_.columns.tolist()]
    counts_l = list(counts)
    counts_l.sort()

    # keep the first n_dim columns
    min_count = counts_l[-n_dim]

    cls_df_ = cls_df_.loc[:, counts >= min_count]

    return(cls_df_, counts_l, representatives)#, low_variance_pairs)



### standard dimension reductions

# reduce dimension using pca
def pca_reduction(cl_space, ncomponents) :

    keys = cl_space.index
    vals = cl_space.values

    print("pca with " + str(ncomponents) + " components")

    pca = PCA(n_components=ncomponents)
    pca_result = pca.fit_transform(vals)

    print("explained variance ratio: ", pca.explained_variance_ratio_)
    print("singular values: ", pca.singular_values_)

    return(pd.DataFrame.from_dict(dict(zip(keys,pca_result)), orient='index'))

# reduce dimension using FA
def fa_reduction(cl_space, ncomponents) :

    keys = cl_space.index
    vals = cl_space.values

    print("factor analysis with " + str(ncomponents) + " components")

    fa = FactorAnalysis(n_components=ncomponents)
    fa_result = fa.fit_transform(vals)

    return(pd.DataFrame.from_dict(dict(zip(keys,fa_result)), orient='index'))

# reduce dimension using MCA
def mca_reduction(cl_space, ncomponents) :

    keys = cl_space.index
    vals = cl_space.values

    X = pd.DataFrame(vals)

    print("MCA with " + str(ncomponents) + " components")

    mca = prince.MCA(n_components = ncomponents)
    mca = mca.fit(X)
    mca = mca.transform(X)

    return(pd.DataFrame.from_dict(dict(zip(keys,mca.values)), orient='index'))
    #return(mca)

# reduce dimension using ICA
def ica_reduction(cl_space, ncomponents) :

    keys = cl_space.index
    vals = cl_space.values

    print("ica with " + str(ncomponents) + " components")

    ica = FastICA(n_components=ncomponents)
    ica_result = ica.fit_transform(vals)

    return(pd.DataFrame.from_dict(dict(zip(keys,ica_result)), orient='index'))



####cluster clustering space

def trivial_clustering_cl_space(cls_df, rtol = 0.001):

    unique_vals = np.unique(np.asarray(cls_df.values),axis=0)

    clusters = [ [k for k,v in zip(cls_df.index,cls_df.values) if np.allclose(v,val, rtol = rtol)]
                 for val in unique_vals ]

    labels = [[i for i,c in enumerate(clusters) if p in c][0] for p in cls_df.index]

    return(labels)

# grouping equal points together

#def key_merge_dictionary(d, rtol, positions = None) : 
#
#    unique_vals = np.unique(np.asarray(list(d.values())),axis=0)
#
#    if positions == None :
#        res = [(tuple([k for k,v in d.items() if np.allclose(v,val, rtol = rtol)]), val) for val in unique_vals ]
#    else :
#        res = [(tuple([k for k,v in d.items() if np.allclose(v[:positions],val[:positions], rtol = rtol)]), val) for val in unique_vals ]
#
#    return(dict(res))
#
#def clustering_in_merged_to_clustering(points, d, clustering) :
#    return([clustering[ [i for i in range(len(d)) if p in d[i]][0] ] for p in points])


def group_cl_space(cl_space, eps = 0.1, method = "complete") :

    vals = list(cl_space.values())

    dims = len(vals[0])

    groups = fclusterdata(vals, eps * np.sqrt(dims), method = method)

    return(groups)


def key_merge_dictionary(d, rtol, positions = None) : 

    unique_vals = np.unique(np.asarray(list(d.values())),axis=0)

    if positions == None :
        res = [(tuple([k for k,v in d.items() if np.allclose(v,val, rtol = rtol)]), val) for val in unique_vals ]
    else :
        res = [(tuple([k for k,v in d.items() if np.allclose(v[:positions],val[:positions], rtol = rtol)]), val) for val in unique_vals ]

    return(dict(res))

def clustering_in_merged_to_clustering(points, d, clustering) :
    return([clustering[ [i for i in range(len(d)) if p in d[i]][0] ] for p in points])
