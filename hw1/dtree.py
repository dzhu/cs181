#!/usr/bin/env python

"""
dtree.py -- CS181 Assignment 1: Decision Trees

Implements decision trees, decision stumps, decision tree pruning, and
adaptive boosting.
"""

import math

def log2(dbl):
    return math.log(dbl)/math.log(2.0) if dbl > 0.0 else 0.0

class Instance(object):
    """Describes a piece of data. The features are contained in listAttrs,
    the instance label in fLabel, and the instance weight (for use in boosting)
    in dblWeight."""
    def __init__(self, listAttrs, fLabel=None, dblWeight=1.0):
        self.listAttrs = listAttrs
        self.fLabel = fLabel
        self.dblWeight = dblWeight
    def copy(self):
        return Instance(list(self.listAttrs), self.fLabel, self.dblWeight)
    def __repr__(self):
        """This function is called when you 'print' an instance."""
        if self.dblWeight == 1.0:
            return "Instance(%r, %r)" % (self.listAttrs, self.fLabel)
        return ("Instance(%r, %r, %.2f)"
                % (self.listAttrs, self.fLabel, self.dblWeight))

def compute_entropy(dblWeightTrue,dblWeightFalse):
    """ Given the total weight of true instances and the total weight
    of false instances in a collection, return the entropy of this
    collection.
    >>> compute_entropy(0.0,1000.0)
    -0.0
    >>> compute_entropy(0.0001, 0.0)
    -0.0
    >>> compute_entropy(10.0,10.0)
    1.0"""
    p = dblWeightTrue / (dblWeightTrue + dblWeightFalse)
    return -(p * log2(p) + (1-p) * log2(1-p))

def separate_by_attribute(listInst, ixAttr):
    """Build a dictionary mapping attribute values to lists of instances.
    
    >>> separate_by_attribute([Instance([0,0],True),Instance([1,0],True)], 0)
    {0: [Instance([0, 0], True)], 1: [Instance([1, 0], True)]}
    """
    res = {}
    for inst in listInst:
        res.setdefault(inst.listAttrs[ixAttr], []).append(inst)
    return res

def compute_entropy_of_split(dictInst):
    """Compute the average entropy of a mapping of attribute values to lists
    of instances.
    The average should be weighted by the sum of the weight in each list of
    instances.
    >>> listInst0 = [Instance([],True,0.5), Instance([],False,0.5)]
    >>> listInst1 = [Instance([],False,3.0), Instance([],True,0.0)]
    >>> dictInst = {0: listInst0, 1: listInst1}
    >>> compute_entropy_of_split(dictInst)
    0.25
    """
    print >>open('/tmp/foo', 'w'), dictInst
    entropy = 0.
    weight = 0.
    for listInst in dictInst.itervalues():
        dblWeightTrue = sum(inst.dblWeight for inst in listInst if inst.fLabel)
        dblWeightFalse = sum(inst.dblWeight for inst in listInst
                             if not inst.fLabel)

        wt = dblWeightTrue + dblWeightFalse
        entropy += wt * compute_entropy(dblWeightTrue, dblWeightFalse)
        weight += wt
    return entropy / weight

def compute_list_entropy(listInst):
    return compute_entropy_of_split({None:listInst})

def choose_split_attribute(iterableIxAttr, listInst, dblMinGain=0.0):
    """Given an iterator over attributes, choose the attribute which
    maximimizes the information gain of separating a collection of
    instances based on that attribute.
    Returns a tuple of (the integer best attribute, a dictionary of the
    separated instances).
    If the best information gain is less than dblMinGain, then return the
    pair (None,None).

    >>> listInst = [Instance([0,0],False), Instance([0,1],True)]
    >>> choose_split_attribute([0,1], listInst)
    (1, {0: [Instance([0, 0], False)], 1: [Instance([0, 1], True)]})
    """
    initialEntropy = compute_list_entropy(listInst)
    bestGain = -1.
    bestAttr = None
    bestSplit = None

    for ixAttr in iterableIxAttr:
        split = separate_by_attribute(listInst, ixAttr)
        afterEntropy = compute_entropy_of_split(split)
        gain = initialEntropy - afterEntropy
        if gain > bestGain:
            bestGain = gain
            bestAttr = ixAttr
            bestSplit = split
    if bestGain >= dblMinGain:
        return bestAttr, bestSplit
    return None, None

def check_for_common_label(listInst):
    """Return the boolean label shared by all instances in the given list of
    instances, or None if no such label exists.

    >>> check_for_common_label([Instance([],True), Instance([],True)])
    True
    >>> check_for_common_label([Instance([],False), Instance([],False)])
    False
    >>> check_for_common_label([Instance([],True), Instance([],False)])
    """
    if len(listInst) == 0:
        return None
    labels = [inst.fLabel for inst in listInst]
    if all(labels):
        return True
    if not any(labels):
        return False
    return None

def majority_label(listInst):
    """Return the boolean label with the most weight in the given list of
    instances.
    
    >>> majority_label([Instance([],True,1.0),Instance([],False,0.75)])
    True
    >>> listInst =[Instance([],False),Instance([],True),Instance([],False)]
    >>> majority_label(listInst)
    False
    """
    trueWeight = sum(inst.dblWeight for inst in listInst if inst.fLabel)
    falseWeight = sum(inst.dblWeight for inst in listInst if not inst.fLabel)

    return trueWeight > falseWeight

class DTree(object):
    def __init__(self, fLabel=None, ixAttr=None, fDefaultLabel=None):
        if fLabel is None and ixAttr is None:
            raise TypeError("DTree must be given a label or an attribute,"
                            " but received neither.")
        self.fLabel = fLabel
        self.ixAttr = ixAttr
        self.dictChildren = {}
        self.fDefaultLabel = fDefaultLabel
        if self.is_node() and self.fDefaultLabel is None:
            raise TypeError("Nodes require a valid fDefaultLabel")
    def is_leaf(self):
        return self.fLabel is not None
    def is_node(self):
        return self.ixAttr is not None
    def add(self, dtChild, v):
        if not isinstance(dtChild,self.__class__):
            raise TypeError("dtChild was not a DTree")
        if v in self.dictChildren:
            raise ValueError("Attempted to add a child with"
                             " an existing attribute value.")
        self.dictChildren[v] = dtChild
    def convert_to_leaf(self):
        if self.is_leaf():
            return
        self.fLabel = self.fDefaultLabel
        self.ixAttr = None
        self.fDefaultLabel = None
        self.dictChildren = {}
    # the following methods are used in testing -- you should need
    # to worry about them
    def copy(self):
        if self.is_leaf():
            return DTree(fLabel=self.fLabel)
        dt = DTree(ixAttr=self.ixAttr, fDefaultLabel=self.fDefaultLabel)
        for ixValue,dtChild in self.dictChildren.iteritems():
            dt.add(dtChild.copy(),ixValue)
        return dt
    def _append_repr(self,listRepr):
        if self.is_leaf():
            listRepr.append("[%s]" % str(self.fLabel)[0])
        else:
            sDefaultLabel = str(self.fDefaultLabel)[0]
            listRepr.append("<%d,%s,{" % (self.ixAttr, sDefaultLabel))
            for dtChild in self.dictChildren.values():
                dtChild._append_repr(listRepr)
            listRepr.append("}>")
    def __repr__(self):
        listRepr = []
        self._append_repr(listRepr)
        return "".join(listRepr)

def build_tree_rec(setIxAttr, listInst, dblMinGain, cRemainingLevels):
    """Recursively build a decision tree.

    Given a set of integer attributes, a list of instances, a boolean default
    label, and a floating-point valued minimum information gain, create
    a decision tree leaf or node.

    If there is a common label across all instances in listInst, the function
    returns a leaf node with this common label.

    If setIxAttr is empty, the function returns a leaf with the majority label
    across listInst.

    If cRemainingLevels is zero, return the majority label. (If
    cRemainingLevels is less than zero, then we don't want to do anything
    special -- this is our mechanism for ignoring the tree depth limit).

    If no separation of the instances yields an information gain greater than
    dblMinGain, the function returns a leaf with the majority label across
    listInst.

    Otherwise, the function finds the attribute which maximizes information
    gain, splits on the attribute, and continues building the tree
    recursively.

    When building tree nodes, the function specifies the majority label across
    listInst as the node's default label (fDefaultLabel argument to DTree's
    __init__). This will be useful in pruning."""
    commonLabel = check_for_common_label(listInst)
    majLabel = majority_label(listInst)

    if commonLabel is not None:
        return DTree(commonLabel)

    ixAttr, split = choose_split_attribute(setIxAttr, listInst, dblMinGain)

    if len(setIxAttr) == 0 or cRemainingLevels == 0 or ixAttr is None:
        return DTree(majLabel)

    tree = DTree(ixAttr=ixAttr, fDefaultLabel=majLabel)
    subset = setIxAttr.difference(set([ixAttr]))
    for attrVal, subList in split.iteritems():
        child = build_tree_rec(subset, subList, dblMinGain, cRemainingLevels-1)
        tree.add(child, attrVal)
    return tree

def count_instance_attributes(listInst):
    """Return the number of attributes across all instances, or None if the
    instances differ in the number of attributes they contain.

    >>> listInst = [Instance([1,2,3],True), Instance([4,5,6],False)]
    >>> count_instance_attributes(listInst)
    3
    >>> count_instance_attributes([Instance([1,2],True),Instance([3],False)])
    """
    if not listInst: return None

    n = len(listInst[0].listAttrs)
    for inst in listInst[1:]:
        if n != len(inst.listAttrs):
            return None
    return n

def build_tree(listInst, dblMinGain=0.0, cMaxLevel=-1):
    """Build a decision tree with the ID3 algorithm from a list of
    instances."""
    cAttr = count_instance_attributes(listInst)
    if cAttr is None:
        raise TypeError("Instances provided have attribute lists of "
                        "varying lengths.")
    setIxAttr = set(xrange(cAttr))
    return build_tree_rec(setIxAttr, listInst, dblMinGain, cMaxLevel)

def classify(dt, inst):
    """Using decision tree dt, return the label for instance inst."""
    if dt.is_leaf():
        return dt.fLabel
    else:
        try:
            attrVal = inst.listAttrs[dt.ixAttr]
            child = dt.dictChildren[attrVal]
            return classify(child, inst)
        except KeyError: # attribute value that was not seen before
            return dt.fDefaultLabel

class EvaluationResult(object):
    def __init__(self, listInstCorrect, listInstIncorrect, oClassifier):
        self.listInstCorrect = listInstCorrect
        self.listInstIncorrect = listInstIncorrect
        self.oClassifier = oClassifier

def weight_correct_incorrect(rslt):
    """Return a pair of floating-point numbers denoting the weight of
    (correct, incorrect) instances in EvaluationResult rslt.

    >>> listInstCorrect = [Instance([],True,0.25)]
    >>> listInstIncorrect = [Instance([],False,0.50)]
    >>> rslt = EvaluationResult(listInstCorrect, listInstIncorrect, None)
    >>> weight_correct_incorrect(rslt)
    (0.25, 0.5)
    """
    correctSum = sum(inst.dblWeight for inst in rslt.listInstCorrect)
    incorrectSum = sum(inst.dblWeight for inst in rslt.listInstIncorrect)
    return correctSum, incorrectSum

class CrossValidationFold(object):
    def build(self):
        raise NotImplemented
    def classify(self, dt, inst):
        raise NotImplemented
    def check_insts(self, listInst):
        for inst in (listInst or []):
            if inst.fLabel is None:
                raise TypeError("missing instance label")
        return listInst

class TreeFold(CrossValidationFold):
    def __init__(self, listInstTraining, listInstTest, listInstValidate=None):
        super(TreeFold,self).__init__()
        self.listInstTraining = self.check_insts(listInstTraining)
        self.listInstTest = self.check_insts(listInstTest)
        self.listInstValidate = self.check_insts(listInstValidate)
        self.cMaxLevel = -1
    def build(self):
        return build_tree(self.listInstTraining, cMaxLevel=self.cMaxLevel)
    def classify(self, dt, inst):
        return classify(dt,inst)

def evaluate_classification(cvf):
    """Given a CrossValidationFold, build a classifier and build an
    EvaluationResult that correctly partitions test instances into a list of
    correctly and incorrectly classified instances.

    Classifiers can be built using cvf.build().

    Evaluation results are built with
    EvaluationResult(listInstCorrect,listInstIncorrect,dt)
    where dt is the classifier built with cvf.build().
    """
    classifier = cvf.build()
    listInstCorrect = []
    listInstIncorrect = []
    for inst in cvf.listInstTest:
        if cvf.classify(classifier, inst) == inst.fLabel:
            listInstCorrect.append(inst)
        else:
            listInstIncorrect.append(inst)

    return EvaluationResult(listInstCorrect, listInstIncorrect, classifier)

def check_folds(listInst, cFold, cMinFold):
    """Raise a ValueError if cFold is greater than the number of instances, or
    if cFold is less than the minimum number of folds.

    >>> check_folds([Instance([],True), Instance([],False)], 1, 2)
    >>> check_folds([Instance([],True)], 2, 1)
    Traceback (most recent call last):
    ...
    ValueError: Cannot have more folds than instances
    >>> check_folds([Instance([],False)], 1, 2)
    Traceback (most recent call last):
    ...
    ValueError: Need at least 2 folds.
    """
    if cFold > len(listInst):
        raise ValueError('Cannot have more folds than instances')
    if cFold < cMinFold:
        raise ValueError('Need at least 2 folds.')

def yield_cv_folds(listInst, cFold):
    """Yield a series of TreeFolds, which represent a partition of listInst
    into cFold folds.

    You may either return a list, or `yield` (http://goo.gl/gwOfM)
    TreeFolds one at a time.
    """
    n = len(listInst)
    for i in range(cFold):
        ind1 = n * i / cFold
        ind2 = n * (i+1) / cFold
        listInstTest = listInst[ind1:ind2]
        listInstTraining = listInst[:ind1] + listInst[ind2:]
        yield TreeFold(listInstTraining, listInstTest)

def cv_score(iterableFolds):
    """Determine the fraction (by weight) of correct instances across a number
    of cross-validation folds.
    """
    dblWeightCorrect = 0.
    dblWeightIncorrect = 0.
    for fold in iterableFolds:
        evalResult = evaluate_classification(fold)
        dblCorrect, dblIncorrect = weight_correct_incorrect(evalResult)
        dblWeightCorrect += dblCorrect
        dblWeightIncorrect += dblIncorrect
    return dblWeightCorrect / (dblWeightCorrect + dblWeightIncorrect)

def prune_tree(dt, listInst):
    """Recursively prune a decision tree.

    Given a subtree to prune and a list of instances,
    recursively prune the tree, then determine if the current node should
    become a leaf.

    The function does not return anything, and instead modifies the tree
    in-place.
    """
    if dt.is_leaf(): return

    separated = separate_by_attribute(listInst, dt.ixAttr)
    for val, subList in separated.iteritems():
        try:
            child = dt.dictChildren[val]
            prune_tree(child, subList)
        except KeyError: #TODO
            pass

    # The most common label found in the data set. Could use
    # dt.fDefaultLabel's original value? But it doesn't look like the
    # data here are the same as what dt was constructed with. TODO ask
    # about this
    #majLabel = majority_label(listInst)
    majLabel = dt.fDefaultLabel

    # the sum of weights of instances with the majority label
    majWeight = sum(inst.dblWeight for inst in listInst
                    if inst.fLabel == majLabel)

    # the sum of weights of correctly classified things
    correctWeight = sum(inst.dblWeight for inst in listInst
                        if inst.fLabel == classify(dt, inst))

    if majWeight > correctWeight:
        dt.fDefaultLabel = majLabel
        dt.convert_to_leaf()

def build_pruned_tree(listInstTrain, listInstValidate):
    """Build a pruned decision tree from a list of training instances, then
    prune the tree using a list of validation instances.

    Return the pruned decision tree.
    """
    dt = build_tree(listInstTrain)
    prune_tree(dt, listInstValidate)
    return dt

class PrunedFold(TreeFold):
    def __init__(self, *args, **kwargs):
        super(PrunedFold,self).__init__(*args,**kwargs)
        if self.listInstValidate is None:
            raise TypeError("PrunedCrossValidationFold requires "
                            "listInstValidate argument.")
    def build(self):
        return build_pruned_tree(self.listInstTraining,self.listInstValidate)

def yield_cv_folds_with_validation(listInst, cFold):
    """Yield a number cFold of PrunedFolds, which together form a partition of
    the list of instances listInst.

    You may either return a list or yield successive values."""
    n = len(listInst)
    for i in range(cFold - 1):
        ind1 = n * i / cFold
        ind2 = n * (i+1) / cFold
        ind3 = n * (i+2) / cFold
        listInstTest = listInst[ind1:ind2]
        listInstValidate = listInst[ind2:ind3]
        listInstTraining = listInst[:ind1] + listInst[ind3:]

        yield PrunedFold(listInstTraining, listInstTest, listInstValidate)

    ind1 = n * (cFold - 1) / cFold
    ind2 = n / cFold
    listInstTest = listInst[ind1:]
    listInstValidate = listInst[:ind2]
    listInstTraining = listInst[ind2:ind1]
    yield PrunedFold(listInstTraining, listInstTest, listInstValidate)

def normalize_weights(listInst):
    """Normalize the weights of all the instances in listInst so that the sum
    of their weights totals to 1.0.
    
    The function modifies the weights of the instances in-place and does
    not return anything.

    >>> listInst = [Instance([],True,0.1), Instance([],False,0.3)]
    >>> normalize_weights(listInst)
    >>> print listInst
    [Instance([], True, 0.25), Instance([], False, 0.75)]
    """
    total = sum(inst.dblWeight for inst in listInst)
    for inst in listInst:
        inst.dblWeight /= total

def init_weights(listInst):
    """Initialize the weights of the instances in listInst so that each
    instance has weight 1/(number of instances). This function modifies
    the weights in place and does not return anything.

    >>> listInst = [Instance([],True,0.5), Instance([],True,0.25)]
    >>> init_weights(listInst)
    >>> print listInst
    [Instance([], True, 0.50), Instance([], True, 0.50)]
    """
    v = 1. / len(listInst)
    for inst in listInst:
        inst.dblWeight = v

def classifier_error(rslt):
    """Given an evaluation result, return the (floating-point) fraction
    of incorrect instances by weight.

    >>> listInstCorrect = [Instance([],True,0.15)]
    >>> listInstIncorrect = [Instance([],True,0.45)]
    >>> rslt = EvaluationResult(listInstCorrect,listInstIncorrect,None)
    >>> classifier_error(rslt)
    0.75
    """
    corr, inc = weight_correct_incorrect(rslt)
    return inc / (corr + inc)

def classifier_weight(dblError):
    """Return the classifier weight alpha from the classifier's training
    error."""
    return .5 * math.log((1 - dblError) / dblError)

def update_weight_unnormalized(inst, dblClassifierWeight, fClassifiedLabel):
    """Re-weight an instance given the classifier weight, and the label
    assigned to the instance by the classifier. This function acts in place
    and does not return anything."""
    expt = dblClassifierWeight * (-1 if fClassifiedLabel == inst.fLabel else 1)
    inst.dblWeight *= math.exp(expt)

class StumpFold(TreeFold):
    def __init__(self, listInstTraining, cMaxLevel=1):
        self.listInstTraining = listInstTraining
        self.listInstTest = listInstTraining
        self.cMaxLevel = cMaxLevel
    def build(self):
        return build_tree(self.listInstTraining, cMaxLevel=self.cMaxLevel)

def one_round_boost(listInst, cMaxLevel):
    """Conduct a single round of boosting on a list of instances. Returns a
    triple (classifier, error, classifier weight).

    Implementation suggestion:
    - build a StumpFold from the list of instances and the given
      cMaxLevel (it's obnoxious that cMaxLevel has to be passed around
      like this -- just pass it into Stumpfold() as the second argument
      and you should be fine).
    - using the StumpFold, build an EvaluationResult using
      evaluate_classification
    - get the error rate of the EvaluationResult using classifier_error
    - obtain the classifier weight from the classifier error
    - update the weight of all instances in the evaluation results
    - normalize all weights
    - return the EvaluationResult's oClassifier member, the classifier error,
      and the classifier weight in a 3-tuple
    - remember to return early if the error is zero.
    """
    print '-'*50
    print cMaxLevel
    print '\n'.join(map(str, listInst))

    fold = StumpFold(listInst, cMaxLevel)
    evalRslt = evaluate_classification(fold)
    error = classifier_error(evalRslt)

    if error == 0.: return None, 0., 0.

    weight = classifier_weight(error)

    # maybe we're supposed to use update_weight_unnormalized, but I
    # couldn't bring myself to do that, thereby having to classify all
    # of the things over again; we've already determined which
    # instances are correct and which are not!
    mult = math.exp(weight)
    for inst in evalRslt.listInstCorrect:
        inst.dblWeight /= mult
    for inst in evalRslt.listInstIncorrect:
        inst.dblWeight *= mult
    normalize_weights(listInst)

    return evalRslt.oClassifier, error, weight

class BoostResult(object):
    def __init__(self, listDblCferWeight, listCfer):
        self.listDblCferWeight = listDblCferWeight
        self.listCfer = listCfer

def boost(listInst, cMaxRounds=50, cMaxLevel=1):
    """Conduct up to cMaxRounds of boosting on training instances listInst
    and return a BoostResult containing the classifiers and their weights."""
    raise NotImplementedError

def classify_boosted(br,inst):
    """Given a BoostResult and an instance, return the (boolean) label
    predicted for the instance by the boosted classifier."""
    raise NotImplementedError

class BoostedFold(TreeFold):
    def __init__(self, *args, **kwargs):
        super(BoostedFold,self).__init__(*args, **kwargs)
        self.cMaxLevel = 1
        self.cMaxRounds = 50        
    def build(self):
        listInst = [inst.copy() for inst in self.listInstTraining]
        return boost(listInst, self.cMaxRounds, self.cMaxLevel)
    def classify(self, br, inst):
        return classify_boosted(br, inst)

def yield_boosted_folds(listInst, cFold):
    """Yield a number cFold of BoostedFolds, constituting a partition of
    listInst.

    Implementation suggestion: Generate TreeFolds, and yield BoostedFolds
    built from your TreeFolds."""
    raise NotImplementedError

def read_csv_dataset(infile):
    listInst = []
    for sRow in infile:
        listRow = map(int, sRow.strip().split())
        inst = Instance(map(int,listRow[:-1]), bool(listRow[-1]))
        listInst.append(inst)
    return listInst

def load_csv_dataset(oFile):
    if isinstance(oFile,basestring):
        with open(oFile) as infile: return read_csv_dataset(infile)
    return read_csv_dataset(infile)

def main(argv):
    import doctest
    doctest.testmod()
    listInst = load_csv_dataset("data.csv")
    cFold = 10
    iterableFolds = yield_cv_folds_with_validation(listInst,cFold)
    #iterableFolds = yield_cv_folds(listInst,cFold)
    #iterableFolds = yield_boosted_folds(listInst,cFold)
    print "%.2f%% correct" % (100.0*cv_score(iterableFolds))
    return 0



if __name__ == "__main__":
    import doctest
    doctest.testmod()
