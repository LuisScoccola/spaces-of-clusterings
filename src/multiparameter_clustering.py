import hdbscan
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation


def multiparameter_hdbscan(data, range_min_cluster_sizes, range_min_samples) :
    
    clusterers = { (m,k) : hdbscan.HDBSCAN(min_cluster_size=m, min_samples = k, gen_min_span_tree=True)
                   for m in range_min_cluster_sizes for k in range_min_samples}

    clusterings = { key : cl.fit(data).labels_ for (key,cl) in clusterers.items() }

    return(clusterings)

def multiparameter_hdbscan_(data, range_min_cluster_sizes_min_samples) :
    
    clusterers = { (m,k) : hdbscan.HDBSCAN(min_cluster_size=m, min_samples = k, gen_min_span_tree=True)
                   for m,k in range_min_cluster_sizes_min_samples}

    clusterings = { key : cl.fit(data).labels_ for (key,cl) in clusterers.items() }

    return(clusterings)



def multiparameter_kmeans(data, range_k , range_random_seed ) :
    
    clusterers = { (k,s) : KMeans(n_clusters = k, init = 'random', random_state = s)
                   for k in range_k for s in range_random_seed }

    clusterings = { key : cl.fit(data).labels_ for (key,cl) in clusterers.items() }

    return(clusterings)


def multiparameter_affprop(data, range_damping) :
    
    clusterers = { (1,d) : AffinityPropagation(damping = d)
                   for d in range_damping }

    clusterings = { key : cl.fit(data).labels_ for (key,cl) in clusterers.items() }

    return(clusterings)
