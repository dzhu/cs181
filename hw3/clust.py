#!/usr/bin/env python
import heapq
import math
import random
from itertools import imap, combinations

def dist(v1, v2):
    """Returns the Euclidean distance between instance 1 and instance 2."""
    return math.sqrt(sum(imap(lambda x,y: (x-y) * (x-y), v1, v2)))

def centroid(pts):
    """Given a list of lists, representing points, determine their centroid."""
    n = float(len(pts))
    return list(imap((lambda *x: sum(x)/n), *pts))

def kmeans(dataset, num_clusters, initial_means=None):
    """Runs the kmeans algorithm.

   dataset: A list of instances.
   num_clusters: The number of clusters to maintain.
   initial_means: (optional) If specified, gives the indices of the data points
     which will be the initial centers of each cluster.  Must have length
     num_clusters if specified.  If not specified, then kmeans should randomly
     initial the means to random data points.

  Returns (means, error), where means is the list of mean vectors, and error is
  the mean squared distance from a datapoint to its cluster.
  """
    assert(initial_means is None or len(initial_means) == num_clusters)
    print 'clusters:', num_clusters#, 'data:', dataset

    means = [dataset[i] for i in initial_means] if initial_means else random.sample(dataset, num_clusters)

    while True:
        #print 'means:', means
        assts = [[] for _ in xrange(num_clusters)]
        num_assts = [0] * num_clusters

        for dat in dataset:
            dists = [dist(mean, dat) for mean in means]
            mean_ind = dists.index(min(dists))
            assts[mean_ind].append(dat)
        #print 'assts:', '\n'.join(map(str, assts))

        new_means = [centroid(pts) for pts in assts]
        max_dist = max(dist(m, m2) for m, m2 in zip(means, new_means))
        means = new_means
        print 'max_dist:', max_dist
        if max_dist < .0001:
            break

    total_error = sum(sum(dist(mean, dat)**2 for dat in dats) for dats, mean in zip(assts, means))
    print 'returning:', means, total_error / len(dataset)
    return means, total_error / len(dataset)

def parse_input(datafile, num_examples):
    data = []
    ct = 0
    for line in datafile:
        instance = line.split(",")
        instance = instance[:-1]
        data.append(map(lambda x:float(x),instance))
        ct += 1
        if not num_examples is None and ct >= num_examples:
          break
    return data

class Cluster(object):
    def __init__(self, arg1, arg2):
        """Create a new cluster, either from a list of list of floats
        and a list of indices, or from two clusters (then this is the
        merge of those two)."""

        self.valid = True
        if isinstance(arg1, list):
            self.pts = arg1
            self.inds = arg2
        else:
            self.pts = arg1.pts #eh, they'll all be the same
            self.inds = list(heapq.merge(arg1.inds, arg2.inds))
            arg1.valid = arg2.valid = False

    def __str__(self):
        return str([tuple(self.pts[i]) for i in self.inds])
    def __repr__(self):
        return str([tuple(self.pts[i]) for i in self.inds])


def run_hac(func, dataset, num_clusters):
    """Runs the hac algorithm for an arbitrary evaluation
    function. func should take two Clusters and return the "distance"
    between them."""

    def make_tuple(clust1, clust2):
        return func(clust1, clust2), clust1, clust2

    clusters = set(Cluster(dataset, [i]) for i in range(len(dataset)))
    heap = [make_tuple(c1, c2) for c1, c2 in combinations(clusters, 2)]
    heapq.heapify(heap)
    # start with len(dataset) clusters, end with num_clusters -- must
    # do this many merges
    for i in xrange(len(dataset) - num_clusters):
        print i
        # find first pair of clusters that haven't already been merged
        pair = heapq.heappop(heap)
        while not (pair[1].valid  and pair[2].valid):
            pair = heapq.heappop(heap)

        new_clust = Cluster(pair[1], pair[2])
        clusters.discard(pair[1])
        clusters.discard(pair[2])
        #clusters.remove(pair[1])
        #clusters.remove(pair[2])

        for clust in clusters:
            heapq.heappush(heap, make_tuple(new_clust, clust))

        clusters.add(new_clust)
        #clusters.append(new_clust)
        print 'all clusters:', clusters

    print 'returning:', [c.inds for c in clusters]
    return [c.inds for c in clusters]

def clust_func(func, c1, c2):
    return func(dist(c1.pts[i1], c2.pts[i2]) for i1 in c1.inds for i2 in c2.inds)

def clust_min(c1, c2): return clust_func(min, c1, c2)
def clust_max(c1, c2): return clust_func(max, c1, c2)

def min_hac(dataset, num_clusters):
    """Runs the min hac algorithm in dataset.  Returns a list of the clusters
  formed.
  """
    return run_hac(clust_min, dataset, num_clusters)

def max_hac(dataset, num_clusters):
    """Runs the max hac algorithm in dataset.  Returns a list of the clusters
  formed.
  """
    return run_hac(clust_max, dataset, num_clusters)


def mean_hac(dataset, num_clusters):
    """Runs the mean hac algorithm in dataset.  Returns a list of the clusters
  formed.
  """
    raise NotImplementedError

def centroid_hac(dataset, num_clusters):
    """Runs the centroid hac algorithm in dataset.  Returns a list of the clusters
  formed.
  """
    raise NotImplementedError

def main(argv):
    import optparse
    parser = optparse.OptionParser()
    parser.add_option("--num_clusters", action="store", type=int,
                      default=4,
                      dest="num_clusters",help="number of clusters")
    parser.add_option("--num_examples", action="store", type=int,
                      default=1000,
                      dest="num_examples",help="number of examples to read in. defaults to None, which means read in all examples.")
    parser.add_option("--datafile", action="store",
                      default="adults.txt",
                      dest="datafile",help="data file")
    parser.add_option("--random_seed", action="store", type=int,
                      default=None,
                      dest="random_seed",help="the random seed to use")
    parser.add_option("--run_hac", action="store_true",
                      default=False,
                      dest="run_hac",help="if true, then run hac.  otherwise, run kmeans.")
    parser.add_option("--hac_alg", action="store",
                      default="min",
                      dest="hac_alg",
                      help="the hac algorithm to use. { min, max, mean, centroid }")
    opts, args = parser.parse_args(argv)

    if opts.run_hac:
      opts.datafile = "adults-small.txt"
      if opts.hac_alg == 'min' or opts.hac_alg == 'max':
        opts.num_examples = 100
      elif opts.hac_alg == 'mean' or opts.hac_alg == 'centroid':
        opts.num_examples = 200

    #Initialize the data
    dataset = parse_input(open(opts.datafile, "r"), opts.num_examples)
    if opts.num_examples:
      assert len(dataset) == opts.num_examples
    if opts.run_hac:
      if opts.hac_alg == 'min':
        clusters = min_hac(dataset, opts.num_clusters)
      elif opts.hac_alg == 'max': 
        clusters = max_hac(dataset, opts.num_clusters)
      elif opts.hac_alg == 'mean':
        clusters = mean_hac(dataset, opts.num_clusters)
      elif opts.hac_alg == 'centroid':
        clusters = centroid_hac(dataset, opts.num_clusters)
      # Print out the lengths of the different clusters
      # Write the results to files.
      print 'Cluster Lengths:'
      index = 0
      for c in clusters:
        print '%d ' % len(c),
        outfile = open('%s-%d.dat' % (opts.hac_alg, index), 'w')
        for pt in c:
          d = dataset[pt]
          print >> outfile, '%f %f %f' % (d[0], d[1], d[2])
        index += 1
        outfile.close()
      print ''
    else:
      print 'Running K-means for %d clusters' % opts.num_clusters
      (means, error) = kmeans(dataset, opts.num_clusters)
      print 'Total mean squared error: %f' % error

if __name__ == "__main__":
  import sys
  main(sys.argv)
