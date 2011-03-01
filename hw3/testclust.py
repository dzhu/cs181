#!/usr/bin/env python

"""
testnn.py -- unit tests for artificial neural nets implemented in nn.py
"""

import clust
import functools
import math
import unittest

DEFAULT_REPEAT = 20

def repeated(fxn):
    @functools.wraps(fxn)
    def wrapper(obj, *args, **kwargs):
        cRepeat = obj.REPEAT if hasattr(obj, "REPEAT") else DEFAULT_REPEAT
        for _ in xrange(cRepeat):
            fxn(obj, *args, **kwargs)
    return wrapper

class UtilsTest(unittest.TestCase):
  def test_dist(self):
    # [ 1, 1 ], [ 2, 2 ]
    self.assertAlmostEqual(math.sqrt(2), clust.dist([1, 1], [2, 2]))

class KMeansTest(unittest.TestCase):
  def setUp(self):
    self.dataset = clust.parse_input(open('adults.txt', 'r'), 100)

  @repeated
  def test_kmeans(self):
    for i in xrange(2, 5):
      # Test kmeans with many different initial assignments.
      (means, error) = clust.kmeans(self.dataset, i)
      self.assertEqual(i, len(means))

  def test_simple_kmeans(self):
    # Test out a simple kmeans example.
    # Suppose we have two-dimensional data points
    # (1,1), (1,2), (2,2), (3,3), (5,5)
    # 3 clusters
    # initial means: [0, 1, 4]
    dataset = [ [1,1], [1,2], [2,2], [3,3], [5,5] ]

    # initial means:
    # [ 1, 1 ], [ 1, 2 ], [ 5, 5 ]
    # new assignment:
    # [ 0, 1, 1, 1, 2 ]
    # next means:
    # [ 1, 1 ], [ 2, 7 / 3.0 ], [ 5, 5 ]
    # new assignment:
    # [ 0, 0, 1, 1, 2 ]
    # next means:
    # [ 1, 1.5 ], [ 2.5, 2.5 ], [ 5, 5 ]
    # total error:
    # (0.5^2 + 0.5^2 + 2 * 0.5^2 + 2 * 0.5^2) / 5.0
    (means, error) = clust.kmeans(dataset, 3, [ 0, 1, 4 ])
    print means
    self.assertEqual(1, means.count([5, 5]))
    self.assertEqual(1, means.count([1, 1.5]))
    self.assertEqual(1, means.count([2.5, 2.5]))
    self.assertAlmostEqual(6 * math.pow(0.5, 2) / 5.0, error)

class HACTest(unittest.TestCase):
  def setUp(self):
    self.dataset = [ [1], [1], [2], [4], [5], [7] ]

  def sort_clusters(self, clusters):
    return map(lambda x: sorted(x), clusters)

  # Test out some simple hac examples
  def test_min_hac(self):
    # Merge steps
    # (1, 1), (2), (4), (5), (8)
    # (1, 1, 2), (4, 5), (8)
    # (1, 1, 2, 4, 5), (8)
    clusters = self.sort_clusters(clust.min_hac(self.dataset, 2))
    self.assertEqual(2, len(clusters))
    self.assertEqual(1, clusters.count([0,1,2,3,4]))
    self.assertEqual(1, clusters.count([5]))

  def test_max_hac(self):
    # Merge steps
    # (1, 1), (2), (4, 5), (8)
    # (1, 1, 2), (4, 5), (8)
    # (1, 1, 2), (4, 5, 8)
    clusters = self.sort_clusters(clust.max_hac(self.dataset, 2))
    self.assertEqual(2, len(clusters))
    self.assertEqual(1, clusters.count([0,1,2]))
    self.assertEqual(1, clusters.count([3,4,5]))

if __name__ == '__main__':
  unittest.main()
