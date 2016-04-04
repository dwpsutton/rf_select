import numpy as np
import sys
import copy

from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score,auric
from collections import defaultdict


class rf_select():
    '''
        Here be documentation
    '''
    def __init__(self,X,y,clf=None,recursive=True,metric='OOB',importance='permutation'):
        # Initialise
        self.clf_=clf
        self.recursive_= recusive
        if metric in [None,'OOB','AUC']:
            self.metric_= metric
        else:
            raise ValueError('Error: metric not recognised: '+str(metric))
        if importance in ['gini','permutation','conditional']:
            self.importance_= importance
        else:
            raise ValueError('Error: importance not recognised: '+str(importance))
        self.nCV_= 5
        #
        # Run
        if self.recursive_:
            return self.recursiveFeatureElimination(X,y)
        else:
            return self.staticFeatureElimination(X,y)


    def _recursiveFeatureElimination(self,X,y):
        print 'Not implemented'
        return None

    def _staticFeatureElimination(self,X,y):
        #
        # Get initial importances
        Imp= importanceEstimator(clf=self.clf_,nCV=self.nCV_,metric=self.metric_,algorithm=self.importance_)
        importances,_= Imp.fit(X,y)
        #
        # Elminate features using static importance
        ordering= np.argsort( importances )
        resultDict= {}
        toCut= []
        for ind,i in enumerate(ordering[:-2]):
            toUse= filter(lambda x: True if x not in toCut else False,range(len(ordering)))
            #
            Imp_i = copy.deepcopy( Imp )
            _,metric= Imp_i.fit(X,y)
            resultDict[ind]= ( toUse, metric )
            #
            toCut.append( i )
        return resultDict


class importanceEstimator():
    '''
        Here be documentation.
    '''
    def __init__(clf=None,nCV=5,metric=None,algorithm='gini'):
        self.clf= clf
        self.nCV= nCV
        if metric in [None,'OOB','AUC']:
            self.metric= metric
        else:
            raise ValueError('Error: metric not recognised: '+str(metric))
        if algorithm in ['gini','permutation','conditional']:
            self.algorithm= algorithm
        else:
            raise ValueError('Error: algorithm not recognised: '+str(algorithm))


    def fit(X,y):
        '''
            This function actually calculates the importances and accuracy metric
            using cross validation.
            Usage:
              imp,acc = fit(X,y)
            Arguments:
              X: feature vector, numpy array
              y: label vector, numpy array
            Return values:
              imp: feature importance vector
              acc: estimator accuracy metric
        '''
        scores = defaultdict(list) # Any unknown element is automatically a list
        rf= copy.deepcopy(self.clf)
        #
        #crossvalidate the scores on a number of different random splits of the data
        outAcc= 0.
        for train_idx, test_idx in ShuffleSplit(len(X), self.nCV, .3):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            r = rf.fit(X_train, Y_train)
            # Get accuracy metric
            if metric is None:
                outAcc= None
            elif metric == 'OOB':
                outAcc += rf.oob_score
            elif metric == 'AUC':
                outAcc += sklearn.metrics.roc_auc_score(y_test, rf.predict_proba(X_test) )
            if self.algorithm == 'gini':
                scores[i].append( self.giniImportance(rf,X_test,y_test) )
            elif self.algorithm == 'permutation':
                scores[i].append( self.permutationImportance(rf,X_test,y_test) )
            elif self.algorithm == 'conditional':
                scores[i].append( self.conditionalPermutationImportance(rf,X_test,y_test) )
        #
        # Return mean importance and metric
        importances= np.array([np.mean(scores[i]) for i in range(X.shape[1])])
        return importances, outAcc / float(self.nCV)
            
            

    def giniImportance(self,rf,X,y):
        return rf.feature_importance_

    def permutationImportance(X,y,rf):
        # Get feature importances
        acc = r2_score(y, rf.predict(X))
        scores= defaultdict(list)
        for i in range(X.shape[1]):
            X_t = X.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = r2_score(Y, rf.predict(X_t))
            scores[i].append((acc-shuff_acc)/acc)
        return np.array([ np.mean(scores[i]) for i in range(X.shape[1]) ])


    def conditionalPermutationImportance(X,y):
        if False:
            rf= copy.deepcopy(self.clf)
            var= 1
            for decTree in rf.estimators_:
                tree= decTree.tree_
                binID= getDecisionBins(X,decTree,var)
        else:
            raise ValueError('Not implemented yet')




####################################
#                                  #
#  Functions not bound to a class  #
#                                  #
####################################



def fitElbow(x,y):
    '''
    Fits best elbow or trade-off by finding point on curve furthest
    from the straight line connecting the curves start and end points.
    Formally, maximises  d= |p - (p.c)c|, where c= b / |b| and:
        p is the vector from [x[0],y[0]] to [x[i],y[i]]
        b is the vector from [x[0],y[0]] to [x[n-1],y[n-1]]
    '''
    n= len(x)
    px= x-x[0]
    py= y-y[0]
    b= np.array( [x[n-1]-x[0], y[n-1]-y[0]] )
    c= b / np.linalg.norm(b)
    d= []
    for i in range(n):
        p= np.array( [px,py] )
        dvec=  p - np.dot(p,c)*c
        d.append( np.linalg.norm( dvec ) )
    return np.max( d )






def getDecisionBins(X,decTree,idx_ignore):
    '''
        return decision bins for each observation
    '''
    # Get the number of splits to safely consider
    K= np.log2(0.368 * n_nodes/2)
    ind= getFirstKSplits(decTree.tree_,K)
    #
    # propagate each sample through tree to get bin
    node_indicator = decTree.decision_path(X_test) # Only in 0.18 !!
    #
    decisionBin= np.zeros( np.shape(X)[0])
    for i in range(np.shape(X)[0]):
        js= np.where( node_indicator[i,:] > 0)[0] #The nodes this sample passes through
        binIDSum= 0
        for j in js:
            # is node in our first K splits?
            indicatorFunction= j in ind #Might be inefficient as searching a list
            #
            # is decision made on our test feature?
            indicatorFunction= indicatorFunction and (not decTree.tree_.feature[j] == idx_ignore)
            #
            # is the value beneath the threshold?
            indicatorFunction= indicatorFunction and (X[i,decTree.tree_.feature[j]] <= decTree.tree_.threshold[j])
            #
            # Add to bin ID sum
            if indicatorFunction:
                ii= np.where( ind == j )[0]
                binIDSum += 2**(ii-1)
        decisionBin[i]= binIDSum
    return decisionBin

def getFirstKSplits(tree,K):
    '''
    '''
    n_nodes = tree_.node_count
    children_left = tree_.children_left
    children_right = tree_.children_right
    feature = tree_.feature
    threshold = tree_.threshold

    # From sklearn itself:
    # The tree structure can be traversed to compute various properties such
    # as the depth of each node and whether or not it is a leaf.
    node_depth = np.zeros(shape=n_nodes)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1
        
        # If we have a test node
        if (children_left[node_id] != children_right[node_id]):
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
        else:
            is_leaves[node_id] = True

    n_older= 0
    generation= 0
    while n_older < K:
        generation += 1
        n_older += sum( node_depth == generation )
    ind= np.argsort( node_depth )
    ind= ind[ np.where( node_depth[ind] <= generation )[0] ]
    # return only first K examples argsort
    return ind[0:K]





# Useful example from http://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
def get_lineage(tree, feature_names):
    left      = tree.tree_.children_left
    right     = tree.tree_.children_right
    threshold = tree.tree_.threshold
    features  = [feature_names[i] for i in tree.tree_.feature]
                    
    # get ids of child nodes
    idx = np.argwhere(left == -1)[:,0]
                        
    def recurse(left, right, child, lineage=None):
        if lineage is None:
            lineage = [child]
        if child in left:
            parent = np.where(left == child)[0].item()
            split = 'l'
        else:
            parent = np.where(right == child)[0].item()
            split = 'r'
                                                            
        lineage.append((parent, split, threshold[parent], features[parent]))
                                                                
        if parent == 0:
            lineage.reverse()
            return lineage
        else:
            return recurse(left, right, parent, lineage)
                                                                                    
    for child in idx:
        for node in recurse(left, right, child):
            print node
