import pandas as pd
import numpy as np



def rep_and_count2(df) :
    df_ = df.T
    df_ = df_.groupby(df_.columns.tolist()).apply(lambda x: list(x.index)).tolist()
    
    reps_counts = { c[0]:len(c) for c in df_ }
    return(reps_counts)



def rep_and_count(df, old_counts) :
    df_ = df.T
    df_ = df_.groupby(df_.columns.tolist()).apply(lambda x: list(x.index)).tolist()

    reps_counts = { c[0]:(sum([old_counts[e] for e in c])) for c in df_ }
    return(reps_counts)

def equal_at_indices(df,c1,c2,idx) :
    return(np.array_equal(df.loc[idx,[c1]].values,df.loc[idx,[c2]].values))

def score_hpref(P,P0,P1,N,M,fP,fP0,fP1,fN,fM) :
    return(N - P0 - P1 + M)


class PartitionTree:

    df = None
    n_cols = 0
    root = None
    leaves = []
    splitting_order = {}
    splitting_until = 0

    class Node:
        df = None
        father = None
        children = []
        S = []
        representatives_with_counts = {}
        m = None
        p0 = None
        p1 = None

        def __str__(self) :

            chs = []

            return( 'chd: ' + str(self.children) + '\n' +
                   'S  : ' + str(self.S) + '\n' +
                   'rc : ' + str(self.representatives_with_counts) + '\n' +
                   'm  : ' + str(self.m) + '\n' +
                   'p0 : ' + str(self.p0) + '\n' +
                   'p1 : ' + str(self.p1) + '\n' +
                   'chs:\n' + '\n'.join(chs) )


        def score_using(self,f) :

            if self.father == None :
                fP, fP0, fP1, fN, fM = self.get_stats()
            else :
                fP, fP0, fP1, fN, fM = self.father.get_stats()

            P, P0, P1, N, M = self.get_stats()

            return(f(P,P0,P1,N,M,fP,fP0,fP1,fN,fM))

        def class_size(self, obj) :
            if obj not in self.representatives_with_counts :
                return(0)
            else :
                return(self.representatives_with_counts[obj])

        def get_stats(self) :
            P = len(self.representatives_with_counts.keys())
            P0 = self.class_size(self.p0)
            P1 = self.class_size(self.p1)
            M = self.class_size(self.m)

            N = len(self.df.columns)

            return(P,P0,P1,N,M)

        def print_stats(self) :
            print("# elements in cluster: " + str(len(self.S)) )
            print("# distinct elements in cluster: " + str( len(np.unique(self.df.loc[ self.S, :].values, axis = 0))))
            print("# distinct columns: " + str(len(self.representatives_with_counts.keys())))
            print("# of constant 0 columns: " + str(self.class_size(self.p0)))
            print("# of constant 1 columns: " + str(self.class_size(self.p1)))
            print("# score: " + str(self.score_using(score_hpref)))


        def splittable(self) :
            if self.m == None :
                return(False)

            S0 = [s for s in self.S if (self.df.loc[s,self.m]== 0).all()]
            S1 = [s for s in self.S if (self.df.loc[s,self.m]== 1).all()]


            if len(S0) == 0 or len(S1) == 0 :
                return(False)

            return(True)


        def split(self) :

            if self.m == None :
                return([])
            else :
                S0 = [s for s in self.S if (self.df.loc[s,self.m]== 0).all()]
                S1 = [s for s in self.S if (self.df.loc[s,self.m]== 1).all()]

                if len(S0) == 0 or len(S1) == 0 :
                    return([])
                else :

                    ch0 = self.__class__(self.df, self, S0, self.representatives_with_counts, self.p0, self.p1)
                    ch1 = self.__class__(self.df, self, S1, self.representatives_with_counts, self.p0, self.p1)

                    self.children.extend([ch0,ch1])
                    return([ch0,ch1])

        def __init__(self, df, father_, S_, old_counts, p0_, p1_) :
            self.df = df
            self.father = father_
            self.children = []
            self.S = S_

            self.representatives_with_counts = rep_and_count2(df.loc[self.S,:])

            reps_sizes = list(self.representatives_with_counts.items()) #list(zip(list(self.PC.class_representatives()),map(self.PC.class_size, list(self.PC.class_representatives()))))
            reps = list(self.representatives_with_counts.keys())

            # set m
            if len(reps_sizes) >= 3 :
                reps_sizes.sort( key = lambda rs: rs[1] )

                first = reps_sizes[-1][0]
                second = reps_sizes[-2][0]
                third = reps_sizes[-3][0]

                if (df.loc[ self.S , first ].values == 0).all() or (df.loc[ self.S , first ].values == 1).all() :
                    if (df.loc[ self.S , second ].values == 0).all() or (df.loc[ self.S , second ].values == 1).all() :
                        self.m = third
                    else :
                        self.m = second
                else :
                    self.m = first


            # set p0
            for r in reps :
                if (df.loc[ self.S , r ].values == 0).all() :
                    self.p0 = r
                    break


            # set p1
            for r in reps :
                if (df.loc[ self.S , r ].values == 1).all() :
                    self.p1 = r
                    break




    def __str__(self) :
        return('df :\n' + str(self.df) + '\n' +
               'ncl: ' + str(self.n_cols) + '\n' +
               'les:' + str(self.leaves) + '\n' +
               'rot:\n' + str(self.root) )

    def split_leaf(self,leaf) :
        chs = leaf.split()
        if len(chs) == 0 :
            return([])
        else :
            self.leaves.remove(leaf)
            self.leaves.extend(chs)
            return(chs)


    def induced_partition(self, nl = None) :
        if nl == None :
            nl = self.leaves

        return([l.S for l in nl])

    def split_until(self, score_fun, max_leaves, score_th) :

        cur_depth = 0
        
        while True :

            if len(self.leaves) >= max_leaves :
                break

            spl = list(filter(self.Node.splittable,self.leaves))

            if len(spl) == 0 :
                break

            score = lambda n : n.score_using(score_fun)

            scores = list(map(score, spl))

            best_index = np.argmax(scores)
            best_leaf = spl[best_index]
            best_score = scores[best_index]

            if best_score < score_th :
                break
            else :
                self.split_leaf(best_leaf)
                self.splitting_order[best_leaf] = cur_depth
                cur_depth += 1

        self.splitting_until = cur_depth - 1
        nodes = self.nodes()
        for n in nodes :
            if n not in self.splitting_order :
                self.splitting_order[n] = cur_depth
                cur_depth += 1


    def nodes_by_depth(self) :
        nodes_depth = {self.root : 0}

        def add_successors(node) :
            for c in node.children :
                nodes_depth[c] = nodes_depth[node] + 1
                add_successors(c)

        add_successors(self.root)

        return(sorted(nodes_depth, key=nodes_depth.get, reverse=True))

    def nodes(self) :
        nodes = [self.root]

        def add_successors(node) :
            for c in node.children :
                nodes.append(c)
                add_successors(c)

        add_successors(self.root)

        return(nodes)




    def to_linkage_matrix(self, score_fun):

        score = lambda n : n.score_using(score_fun)

        by_splitting_order = sorted(self.splitting_order, key=self.splitting_order.get, reverse=True)

        scored = list(map(score,by_splitting_order[::-1]))
        eps = 0.1
        scored = list(map(lambda n : n+eps, scored))

        y_coords = [ sum(scored[i:]) for i in range(len(scored)) ]

        node_to_index = {}
        cur_index = 0
        internal_node_index = len(scored)-1

        score_acum = 0

        Z = []

        idx_to_node = {}

        for n in by_splitting_order :
            node_to_index[n] = cur_index
            cur_index += 1

            if len(n.children) == 0 :
                idx_to_node[n] = cur_index
                continue
            else :
                internal_node_index -= 1
                score_acum += score(n)
                c0 = n.children[0]
                c1 = n.children[1]

                Z.append([float(node_to_index[c0]), float(node_to_index[c1]), float(score_acum), float(len(n.S))])

        Z = np.asarray(Z)

        return(Z)


    def __init__(self,df_):
        self.df = df_
        self.n_cols = len(self.df.columns)
        self.splitting_order = {}

        reps = { c : 1 for c in self.df.columns }

        self.root = self.Node(self.df, None, self.df.index.values.copy(), reps, None, None)

        self.leaves = [self.root]

        self.splitting_order[self.root] = 0
