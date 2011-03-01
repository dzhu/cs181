#!/usr/bin/env python

"""
tasknn.py -- Visualizations for neural networks.
"""

from os import path
import random

from tfutils import tftask

import clust

class KMeans(tftask.ChartTask):
    SAMPLES = 3
    CLUSTERS = xrange(2, 11)
    def get_name(self):
        return "KMeans"

    def get_description(self):
        return ("Implement the K-means algorithm and run it on the adults.txt"
                 " dataset for 2 to 10 clusters.  Plot the mean squared error for"
                 " the different numbers of clusters.")

    def get_priority(self):
        return 1

    def task(self):
      dataset = clust.parse_input(open('adults.txt', 'r'), 1000)
      all_errors = []
      # Generate a random initial assignment
      for clusters in self.CLUSTERS:
        # get the lowest error over three runs
        errors = [ clust.kmeans(dataset, clusters)[1] for i in xrange(self.SAMPLES) ]
        error = sorted(errors)[0]
        all_errors.append({ "x": clusters, "y": error })
      chart = {"chart": {"defaultSeriesType": "line"},
               "xAxis": {"title": {"text": "Clusters"}, "min": 1 },
               "yAxis": {"title": {"text": "Mean Squared Error"}},
               "title": {"text": "K-means Results"},
               "series": [ {"data": all_errors } ]}
      return chart

class HAC(tftask.MultipleChartTask):
  def __init__(self):
    tftask.MultipleChartTask.__init__(self, 4)

  def get_priority(self):
    return 2

  def create_chart(self, dataset, clusters, attr1_name, attr1_index, attr2_name, attr2_index):
    # Create the points to plot
    plot_pts = []
    series = []
    index = 0
    for cluster in clusters:
      index += 1
      series_data = []
      for pt in cluster:
        data = dataset[pt]
        series_data.append({"x": data[attr1_index], "y": data[attr2_index]})
      series.append({ "data": series_data, "name": "Cluster %d" % index})
    chart = {"chart": {"defaultSeriesType": "scatter"},
             "xAxis": {"title": {"text": attr1_name}},
             "yAxis": {"title": {"text": attr2_name}},
             "title": {"text": self.get_name() + " Results"},
             "series": series }
    return chart

  def run_hac(self, dataset):
    return clust.min_hac(dataset, 4)

  def get_name(self):
    return 'Min HAC'

  def get_description(self):
    return ("Implement the %s algorithm and run it on adults-small.txt." % self.get_name())

  def get_dataset(self):
    return clust.parse_input(open('adults-small.txt', 'r'), 100)

  def task(self):
    # Run the min hac algorithm
    dataset = self.get_dataset()
    clusters = self.run_hac(dataset)
    charts = []
    # Create the pie chart
    data = []
    for i in xrange(len(clusters)):
      data.append([ 'Cluster %d (%d points)' % (i + 1, len(clusters[i])), len(clusters[i]) ])
    charts.append({"chart": {"defaultSeriesType": "pie"},
                   "plotOptions": { "pie": { "dataLabels": { "enabled": True }}},
                   "title": {"text": self.get_name() + " Results"},
                   "series": [{ "data": data }]})
    charts.append(self.create_chart(dataset, clusters, "Age", 0, "Education", 1))
    charts.append(self.create_chart(dataset, clusters, "Education", 1, "Income", 2))
    charts.append(self.create_chart(dataset, clusters, "Age", 0, "Income", 2))
    return charts

class MaxHAC(HAC):
  def get_name(self):
    return "Max HAC"
  def get_priority(self):
    return 3
  def run_hac(self, dataset):
    return clust.max_hac(dataset, 4)

class MeanHAC(HAC):
  def get_name(self):
    return "Mean HAC"
  def get_priority(self):
    return 4
  def get_dataset(self):
    return clust.parse_input(open('adults-small.txt', 'r'), 200)
  def run_hac(self, dataset):
    return clust.mean_hac(dataset, 4)

class CentroidHAC(MeanHAC):
  def get_name(self):
    return "Centroid HAC"
  def get_priority(self):
    return 5
  def run_hac(self, dataset):
    return clust.centroid_hac(dataset, 4)

def main(argv):
    return tftask.main()

if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv))
