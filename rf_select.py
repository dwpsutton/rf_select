import numpy as np
import sys

from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from collections import defaultdict


class rf_select():
    '''
        Here be documentation
    '''
    def __init__(self,X,y,clf=None,recursive=True,metric='OOB',importance='permutation'):
        # Initialise
        self.clf_=clf
        self.recursive_= recusive
        self.metric_= metric
        self.importance_= importance
        #
        # Run
        if self.recursive_:
            return self.recursiveFeatureElimination(X,y)
        else:
            return self.staticFeatureElimination(X,y)


    def _recursiveFeatureElimination(X,y):
        print 'Not implemented'
        return None

    def _staticFeatureElimination(X,y):
        # TODO: this needs to be able to run on straight-up data, so importances/metric need to cross-validate.
        #
        clf= copy.deepcopy( self.clf_ )
        clf.fit(X,y)
        if self.importance_ == 'gini':
            importances= giniImportance(X,y,clf=clf)
        elif self.importance_ == 'permutation':
            importances= permutationImportance(X,y,clf=clf)
        elif self.importance_ == 'conditional':
            importances= conditionalPermutationImportance(X,y,clf=clf)
        else:
            print 'Error: importance not recognised: '+str(self.importance_)
        #
        ordering= np.argsort( importances )
        resultDict= {}
        toCut= ordering[0:2]
        for ind,i in enumerate(ordering[2:]):
            toUse= filter(lambda x: True if x not in toCut else False,range(len(ordering)))
            #
            clf= copy.deepcopy( self.clf_ )
            clf.fit(X[:,np.array(toUse),:],y)
            #
            if self.metic_ == 'OOB':
                 metric= clf.oob_score 
            elif self.metric=='AUC':
                 print 'Not implemented'
            else:
                print 'Error: metric not recognised: '+str(self.metric_)
            #
            resultDict[ind]= ( toUse, metric )
        return resultDict


def giniImportance(X,y,clf=None):
    return rf.feature_importance_

def permutationImportance(X,y,clf=None):
    scores = defaultdict(list) # Any unknown element is automatically a list
    #
    #crossvalidate the scores on a number of different random splits of the data
    for train_idx, test_idx in ShuffleSplit(len(X), 100, .3):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        r = rf.fit(X_train, Y_train)
        acc = r2_score(Y_test, rf.predict(X_test))
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = r2_score(Y_test, rf.predict(X_t))
            scores[i].append((acc-shuff_acc)/acc)
    for i in range(X.shape[1]):
        importance[i]= np.mean(scores[i])
    return importance


def conditionalPermutationImportance(X,y,clf=None):
    if False:
        var= 1
        for decTree in clf.estimators_:
            tree= decTree.tree_

    
    
    else:
        print 'Not implemented yet'
        return None


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
