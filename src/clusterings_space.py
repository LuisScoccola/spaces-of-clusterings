import numpy as np
import math
import pandas as pd

from scipy.stats import mode
from sklearn.decomposition import *
from scipy.cluster.hierarchy import *
from scipy.spatial.distance import pdist
import hdbscan


SMALL = 1e-300
vectorized_len = np.vectorize(len)


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



# dimension reduction of clusterings space by randomly dropping columns
def dim_red_col_drop(cls_df, n_dim = None, approx = 1) :

    if n_dim != None and approx < 1:
        print("Either use n_dim or approx, but not both!")
        return(cls_df)

    if approx < 1 :
        remove_n = len(cls_df.columns) - int(approx * len(cls_df.columns))
        drop_cols = np.random.choice(cls_df.columns, remove_n, replace=False)
        cls_df = cls_df.drop(columns = drop_cols)

    if n_dim != None :
        remove_n = len(cls_df.columns) - n_dim
        drop_cols = np.random.choice(cls_df.columns, remove_n, replace=False)
        cls_df = cls_df.drop(columns = drop_cols)
    
    return(cls_df)

# dimension reduction of clusterings space by joining together equal columns,
# and keeping only the ones that appear the most
def dim_red(cls_df, n_dim = 3, tol_var = 0.0) :


    if tol_var > 0 :
        cls_df = cls_df.loc[:, cls_df.var() >= tol_var ]

    # calculate how many times each column repeats
    counts = cls_df.T.groupby(cls_df.T.columns.tolist()).cumcount() + 1

    # delete repeated columns
    cls_df_ = cls_df.T.drop_duplicates(keep='last').T

    representatives = cls_df_.columns.tolist()
    
    represented = [ list(cls_df.T.index[ (cls_df == list(cls_df_[col])).all() ])
                    for col in cls_df_.columns.tolist()]

    # delete repeated columns from the counts dataframe and sort it
    counts = counts[cls_df_.columns.tolist()]
    counts_l = list(counts)
    counts_l.sort()

    # keep the first n_dim columns
    min_count = counts_l[-n_dim]

    cls_df_ = cls_df_.loc[:, counts >= min_count]


    return(cls_df_, representatives, represented)


def clustering_to_modes(vecs,clustering,n_cls) :
    clusters = clusters_by_size(clustering)[:n_cls]
    res = [ mode([vecs[i] for i in range(len(vecs)) if clustering[i] == l])[0][0] for l in clusters]
    return(res)


def vectorwise_mode(vecs) :
    return(np.unique(vecs, axis = 0)[np.argmax(np.unique(vecs, return_counts = True, axis=0)[1])])

def medioid(vecs) :
    unique, counts = np.unique(vecs, axis=0, return_counts=True)
    dists = pdist(unique, metric = 'hamming')
    N = len(unique)

    scores = [ np.sum([ counts[j] * dists[square_to_condensed(i,j,N) ] for j in range(N) if i!=j]) for i in range(N) ]
    min_idx = scores.index(min(scores))

    return(unique[min_idx])


def square_to_condensed(i, j, n):
    assert i != j, "no diagonal elements in condensed matrix"
    if i < j:
        i, j = j, i
    return n*j - j*(j+1)//2 + i - 1 - j


def clustering_to_vectorwise_mode(vecs, clustering, n_cls) :
    clusters = clusters_by_size(clustering)[:n_cls]
    res = [ vectorwise_mode([vecs[i] for i in range(len(vecs)) if clustering[i] == l]) for l in clusters]
    return(res)

def clustering_to_medioids(vecs, clustering, n_cls) :
    clusters = clusters_by_size(clustering)[:n_cls]
    res = [ medioid([vecs[i] for i in range(len(vecs)) if clustering[i] == l]) for l in clusters]
    return(res)

def clustering_to_random_representative(vecs, clustering, n_cls) :
    # to do: implement seed
    clusters = clusters_by_size(clustering)[:n_cls]
    res_ = [ [vecs[i] for i in range(len(vecs)) if clustering[i] == l] for l in clusters]
    idx = [ np.random.choice(len(r)) for r in res_ ]
    res = [r[i] for i,r in zip(idx,res_)]
    return(res)



def columns_that_match(df, val) :
    return(df.columns[(df.values == np.asarray(val)[:,None]).all(0)])

def dim_red_eps(cls_df, n_dim = 3, tol_var = 0.0, stage = 0, representative = 'componentwise_mode', likelihood = 'standard') :

    cls_df = cls_df.loc[:, cls_df.var() > tol_var ]

    h_cl = linkage(cls_df.T.values, method='complete', metric='hamming')

    nontrivial_heights = []
    was_at_zeroes = False
    for j in range(len(h_cl)) :
        if was_at_zeroes and h_cl[j,2] == 0 :
            continue
        if not was_at_zeroes and h_cl[j,2] == 0 :
            was_at_zeroes = True
            continue
        if was_at_zeroes and h_cl[j,2] != 0 :
            was_at_zeroes = False
            nontrivial_heights.append(j-1)
            nontrivial_heights.append(j)
            continue
        if not was_at_zeroes and h_cl[j,2] != 0:
            nontrivial_heights.append(j)
            continue

    ct = cut_tree(h_cl).T
    relevant_clusterings = [ct[i+1] for i in nontrivial_heights]

    if representative == 'componentwise_mode' :
        reductions = [ pd.DataFrame(np.asarray(clustering_to_modes(cls_df.T.values, cl, n_dim)).T,
                                    index = cls_df.index) for cl in relevant_clusterings ]

    elif representative == 'vectorwise_mode' :
        reductions_ = [ clustering_to_vectorwise_mode(cls_df.T.values, cl, n_dim) for cl in relevant_clusterings ]

        reductions = [ pd.DataFrame(np.asarray(red).T, index = cls_df.index,
                                    columns = [columns_that_match(cls_df,r)[0] for r in red])
                       for red in reductions_ ]

    elif representative == 'medioid' :

        reductions_ = [ clustering_to_medioids(cls_df.T.values, cl, n_dim) for cl in relevant_clusterings ]

        reductions = [ pd.DataFrame(np.asarray(red).T, index = cls_df.index,
                                    columns = [columns_that_match(cls_df,r)[0] for r in red])
                       for red in reductions_ ]

    elif representative == 'random' :

        reductions_ = [ clustering_to_random_representative(cls_df.T.values, cl, n_dim) for cl in relevant_clusterings ]

        reductions = [ pd.DataFrame(np.asarray(red).T, index = cls_df.index,
                                    columns = [columns_that_match(cls_df,r)[0] for r in red])
                       for red in reductions_ ]


    else :
        print("Must specify a valid mode!")
        return()

    if stage == 'optimize' :
        print("There are " + str(len(nontrivial_heights)) + " stages")
        summarized_full_df, counts = summarized_df_for_fast_likelihood(cls_df)

        likelihoods = []
        for i,red_df in enumerate(reductions) :
            if len(red_df.columns) < n_dim :
                continue
            if likelihood == 'standard' :
                current_likelihood = likelihood_fast(cls_df, summarized_full_df, red_df, counts)
            elif likelihood == 'approx' :
                current_likelihood = likelihood_approx(cls_df, summarized_full_df, red_df, counts)
            else :
                print("Must specify a valid likelihood!")
                return()
            print("stage " + str(i) + " with likelihood " + str(current_likelihood))
            likelihoods.append(current_likelihood)


        max_index = likelihoods.index(max(likelihoods))
        print(max_index)
        cls_df_ = reductions[max_index]

    else :
        cls_df_ = reductions[stage]

    return(cls_df_)


def summarized_df_for_fast_likelihood(df) :
    # calculate how many times each column repeats
    counts = df.T.groupby(df.T.columns.tolist()).cumcount() + 1

    # delete repeated columns
    df_ = df.T.drop_duplicates(keep='last').T

    counts = counts[df_.columns.tolist()]

    return(df_, counts)


# TO DO: give a reasonable name
def aux(pi, xval) :
    if xval == 1 :
        return(pi)
    else :
        return(1-pi)


# Computes likelihood function from binary factor analysis.
# Take full_df as output of clusterings_as_df and
# take red_df as output of dim_red_eps
def likelihood_fast(full_df, summarized_full_df, red_df, counts) :
    
    P = len(list(summarized_full_df))

    N = len(red_df.index)
    classes_reps = red_df.drop_duplicates(inplace=False)
    K = len(classes_reps.index)

    classes = [[i for i in range(N) if red_df.iloc[i].tolist() == classes_reps.iloc[j].tolist()] for j in range(K)]

    pi_ = {j: [np.mean([np.longdouble(summarized_full_df.iloc[k,i]) for k in classes[j]]) for i in range(P)] for j in range(K)}
    pi = pd.DataFrame.from_dict(pi_)

    eta = {j : np.longdouble(len(classes[j]) / N) for j in range(K)}

    L = np.sum(
          np.log(SMALL +
            np.asarray(
            [np.sum(
              [eta[j] *
                np.prod(
                  [aux(pi.iloc[i,j],np.longdouble(summarized_full_df.iloc[h,i])) ** counts[i]
                   for i in range(P)]
                ) for j in range(K) ]) for h in range(N) ])))

    return(L)


def compute_alpha(pi, chi, sigma) :
    if pi == 1 :
        return(0)
    if pi == 0 :
        return(0)
    else :
        return(chi * np.log(pi) + (sigma - chi) * np.log(1 - pi))


def likelihood_approx(full_df, summarized_full_df, red_df, counts) :
    
    P = len(list(summarized_full_df))

    N = len(red_df.index)
    classes_reps = red_df.drop_duplicates(inplace=False)
    K = len(classes_reps.index)

    classes = [[i for i in range(N) if red_df.iloc[i].tolist() == classes_reps.iloc[j].tolist()]
                         for j in range(K)]

    sigma = list(map(len,classes))

    chi = np.asarray([np.asarray([np.sum(np.asarray([full_df.iloc[h,i] for h in classes[j]])) for j in range(K)]) for i in range(P)])

    pi = np.asarray([ np.asarray([ np.mean([summarized_full_df.iloc[k,i] for k in classes[j]]) for j in range(K)]) for i in range(P)])

    leta = np.log(np.asarray(sigma)/N)

    L = np.sum([sigma[j] * leta[j] +
                np.sum([counts[i] * compute_alpha(pi[i,j], chi[i,j], sigma[j]) for i in range(P)]) for j in range(K)])

    return(L)



### standard dimension reductions
#TO DO: dont use dictionaries

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
def key_merge_dictionary(d, rtol, positions = None) : 

    unique_vals = np.unique(np.asarray(list(d.values())),axis=0)

    if positions == None :
        res = [(tuple([k for k,v in d.items() if np.allclose(v,val, rtol = rtol)]), val) for val in unique_vals ]
    else :
        res = [(tuple([k for k,v in d.items() if np.allclose(v[:positions],val[:positions], rtol = rtol)]), val) for val in unique_vals ]

    return(dict(res))

def clustering_in_merged_to_clustering(points, d, clustering) :
    return([clustering[ [i for i in range(len(d)) if p in d[i]][0] ] for p in points])


def group_cl_space(cl_space, eps = 0.1, method = "complete") :

    vals = list(cl_space.values())

    dims = len(vals[0])

    groups = fclusterdata(vals, eps * np.sqrt(dims), method = method)

    return(groups)


# grouping equal points together
def key_merge_dictionary(d, rtol, positions = None) : 

    unique_vals = np.unique(np.asarray(list(d.values())),axis=0)

    if positions == None :
        res = [(tuple([k for k,v in d.items() if np.allclose(v,val, rtol = rtol)]), val) for val in unique_vals ]
    else :
        res = [(tuple([k for k,v in d.items() if np.allclose(v[:positions],val[:positions], rtol = rtol)]), val) for val in unique_vals ]

    return(dict(res))

def clustering_in_merged_to_clustering(points, d, clustering) :
    return([clustering[ [i for i in range(len(d)) if p in d[i]][0] ] for p in points])
