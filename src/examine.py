"""Examine a graph
"""
from collections import namedtuple
import csv
from util import _load_graph

from graph_tool import Graph
from graph_tool import load_graph_from_csv
from graph_tool.clustering import global_clustering
from graph_tool.stats import vertex_average
from graph_tool.topology import extract_largest_component
import numpy as np
from sklearn.linear_model import LinearRegression
import powerlaw

Params = namedtuple("Params", ["communities", "density", "maxdeg", "exp", "overlap", "sizevar", "vertices"])
Fieldnames = [
    "experiment", "name", "real", "vertices", "edges", "min_degree", "max_degree", "avg_degree",
    "largest_connected_component", "islands", "clustering_coefficient",
    "directed", "exponent", "percentile_95"
]
Properties = namedtuple("Properties", Fieldnames)

REAL = [
    "web-polblogs", "web-google", "web-edu", "web-EPA", "web-BerkStan", "web-webbase-2001", "web-indochina-2004",
    "web-spam", "web-sk-2005", "web-frwikinews-user-edits", "web-NotreDame", "web-arabic-2005", "web-italycnr-2000",
    "web-Stanford", "web-baidu-baike-related", "web-it-2004", "web-google-dir", "web-BerkStan-dir",
    "web-wikipedia2009", "web-uk-2005", "web-wiki-ch-internal", "web-hudong", "web-baidu-baike",
    "web-wikipedia-growth", "web-wikipedia-link-de",
    # too big to run in a reasonable amount of time
    # , "web-wikipedia-link-it", "web-wikipedia-link-fr",
    # "web-indochina-2004-all", "web-uk-2002-all", "web-sk-2005-all", "web-webbase-2001-all", "web-it-2004-all"
]

REAL_UNDIRECTED = [
    "web-arabic-2005", "web-BerkStan", "web-edu", "web-google", "web-indochina-2004", "web-it-2004", "web-polblogs",
    "web-sk-2005", "web-spam", "web-uk-2005", "web-webbase-2001", "web-wikipedia2009"
]

GENERATED = {
    "distribution": [
        Params(32, 1.0, 0.05, -3.5, 2.0, 5.0, 20000),
        Params(32, 1.0, 0.05, -3.0, 2.0, 5.0, 20000),
        Params(32, 1.0, 0.05, -2.5, 2.0, 5.0, 20000),
        Params(32, 1.0, 0.05, -2.0, 2.0, 5.0, 20000),
        Params(32, 1.0, 0.05, -1.5, 2.0, 5.0, 20000),
        Params(32, 1.0, 0.05, -1.0, 2.0, 5.0, 20000),
        Params(32, 1.0, 0.05, -0.5, 2.0, 5.0, 20000),
        Params(32, 1.0, 0.05, 0.0, 2.0, 5.0, 20000),
    ],
    "communitysize": [
        Params(2, 1.0, 0.05, -2.1, 2.0, 5.0, 32768),
        Params(4, 1.0, 0.05, -2.1, 2.0, 5.0, 32768),
        Params(8, 1.0, 0.05, -2.1, 2.0, 5.0, 32768),
        Params(16, 1.0, 0.05, -2.1, 2.0, 5.0, 32768),
        Params(32, 1.0, 0.05, -2.1, 2.0, 5.0, 32768),
        Params(64, 1.0, 0.05, -2.1, 2.0, 5.0, 32768),
        Params(128, 1.0, 0.05, -2.1, 2.0, 5.0, 32768),
        Params(256, 1.0, 0.05, -2.1, 2.0, 5.0, 32768),
#         Params(512, 1.0, 0.05, -2.1, 2.0, 5.0, 32768),
#         Params(1024, 1.0, 0.05, -2.1, 2.0, 5.0, 32768),
#         Params(2048, 1.0, 0.05, -2.1, 2.0, 5.0, 32768),
    ],
    "overlap": [
        Params(32, 1.0, 0.05, -2.1, 1.0, 5.0, 20000),
        Params(32, 1.0, 0.05, -2.1, 1.5, 5.0, 20000),
        Params(32, 1.0, 0.05, -2.1, 2.0, 5.0, 20000),
        Params(32, 1.0, 0.05, -2.1, 2.5, 5.0, 20000),
        Params(32, 1.0, 0.05, -2.1, 3.0, 5.0, 20000),
        Params(32, 1.0, 0.05, -2.1, 3.5, 5.0, 20000),
        Params(32, 1.0, 0.05, -2.1, 4.0, 5.0, 20000),
        Params(32, 1.0, 0.05, -2.1, 4.5, 5.0, 20000),
        Params(32, 1.0, 0.05, -2.1, 5.0, 5.0, 20000),
        Params(32, 1.0, 0.05, -2.1, 5.5, 5.0, 20000),
        Params(32, 1.0, 0.05, -2.1, 6.0, 5.0, 20000),
    ],
    "sizevariation": [
        Params(32, 1.0, 0.05, -2.1, 2.0, 1.0, 20000),
        Params(32, 1.0, 0.05, -2.1, 2.0, 1.5, 20000),
        Params(32, 1.0, 0.05, -2.1, 2.0, 2.0, 20000),
        Params(32, 1.0, 0.05, -2.1, 2.0, 2.5, 20000),
        Params(32, 1.0, 0.05, -2.1, 2.0, 3.0, 20000),
        Params(32, 1.0, 0.05, -2.1, 2.0, 3.5, 20000),
        Params(32, 1.0, 0.05, -2.1, 2.0, 4.0, 20000),
        Params(32, 1.0, 0.05, -2.1, 2.0, 4.5, 20000),
        Params(32, 1.0, 0.05, -2.1, 2.0, 5.0, 20000),
        Params(32, 1.0, 0.05, -2.1, 2.0, 5.5, 20000),
        Params(32, 1.0, 0.05, -2.1, 2.0, 6.0, 20000),
    ],
    "supernode": [
        Params(32, 1.0, 0.0005, -2.1, 2.0, 5.0, 20000),
        Params(32, 1.0, 0.001, -2.1, 2.0, 5.0, 20000),
        Params(32, 1.0, 0.005, -2.1, 2.0, 5.0, 20000),
        Params(32, 1.0, 0.01, -2.1, 2.0, 5.0, 20000),
        Params(32, 1.0, 0.05, -2.1, 2.0, 5.0, 20000),
        Params(32, 1.0, 0.1, -2.1, 2.0, 5.0, 20000),
        Params(32, 1.0, 0.5, -2.1, 2.0, 5.0, 20000),
    ],
    "scaling": [
        Params(11, 1.0, 0.05, -2.1, 2.0, 5.0, 1000),
        Params(14, 1.0, 0.05, -2.1, 2.0, 5.0, 2000),
        Params(18, 1.0, 0.05, -2.1, 2.0, 5.0, 4000),
        Params(23, 1.0, 0.05, -2.1, 2.0, 5.0, 8000),
        Params(29, 1.0, 0.05, -2.1, 2.0, 5.0, 16000),
        Params(37, 1.0, 0.05, -2.1, 2.0, 5.0, 32000),
        Params(48, 1.0, 0.05, -2.1, 2.0, 5.0, 64000),
        Params(61, 1.0, 0.05, -2.1, 2.0, 5.0, 128000),
        Params(78, 1.0, 0.05, -2.1, 2.0, 5.0, 256000),
        Params(99, 1.0, 0.05, -2.1, 2.0, 5.0, 512000),
        Params(126, 1.0, 0.05, -2.1, 2.0, 5.0, 1024000),
    ],
    "sparsity": [
        Params(32, 1.0, 0.05, -2.1, 2.0, 5.0, 20000),
        Params(32, 0.9, 0.05, -2.1, 2.0, 5.0, 20000),
        Params(32, 0.8, 0.05, -2.1, 2.0, 5.0, 20000),
        Params(32, 0.7, 0.05, -2.1, 2.0, 5.0, 20000),
        Params(32, 0.6, 0.05, -2.1, 2.0, 5.0, 20000),
        Params(32, 0.5, 0.05, -2.1, 2.0, 5.0, 20000),
        Params(32, 0.4, 0.05, -2.1, 2.0, 5.0, 20000),
        Params(32, 0.3, 0.05, -2.1, 2.0, 5.0, 20000),
        Params(32, 0.2, 0.05, -2.1, 2.0, 5.0, 20000),
        Params(32, 0.1, 0.05, -2.1, 2.0, 5.0, 20000),
    ]
}


def examine_graph(graph: Graph, experiment: str, graphname: str, real: bool, directed: bool = True) -> Properties:
    vertices = graph.num_vertices()
    edges = graph.num_edges()
    total_degrees = graph.get_total_degrees(np.arange(vertices))
    min_degree = np.min(total_degrees)
    max_degree = np.max(total_degrees)
    avg_degree = vertex_average(graph, "total")[0]
    largest_component = extract_largest_component(graph, directed=False).num_vertices()
    num_islands = np.sum(total_degrees == 0)
    cc = global_clustering(graph)[0]
    # _degrees, _counts = np.unique(total_degrees, return_counts=True)
    # log_degrees = np.log(_degrees)
    # log_counts = np.log(_counts)
    # regressor = LinearRegression()
    # regressor.fit(log_degrees.reshape(-1, 1), log_counts)
    # exponent = regressor.coef_[0]
    result = powerlaw.Fit(total_degrees, xmin=1, discrete=True,
                          xmax=max_degree)
    exponent = -result.alpha
    percentile = np.percentile(total_degrees, 95)
    # print("Exponent for this graph is: ", exponent)
    # print("Using powerlaw package: e = {} xmin = {} xmax = {}".format(
    #     exponent2, result.xmin, result.xmax))
    # print("degrees: {}\ncounts: {}".format(_degrees[:20], _counts[:20]))
    return Properties(experiment, graphname, real, vertices, edges, min_degree,
                      max_degree, avg_degree, largest_component, num_islands,
                      cc, directed, exponent, percentile)
# End of examine_graph()


if __name__ == "__main__":
    # Examine synthetic graphs
    print("=====Examining Synthetic Graphs=====")
    props = list()
    for experiment, graph_params in GENERATED.items():
        for param in graph_params:
            graphtype = "test_{0}_{1}_{2}_{3}".format(param.communities,
                                                      param.maxdeg, param.exp,
                                                      param.density)
            difficulty = "{0}Overlap_{1}BlockSizeVar".format(param.overlap, param.sizevar)
            print("experiment = {0}, graphname = {1}_{2}".format(experiment, graphtype, difficulty))
            filename = "../../data/{0}/{1}/{0}_{1}_{2}_nodes".format(graphtype, difficulty, param.vertices)
            graph = _load_graph(filename)
            props.append(examine_graph(graph, experiment,
                                       "{}_{}".format(graphtype, difficulty),
                                       False, True))
    with open('synthetic_graphs_examined.csv', mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=Fieldnames)
        writer.writeheader()
        for prop in props:
            writer.writerow(prop._asdict())
    # Examine real world graphs
    print("=====Examining Real World Graphs=====")
    with open('real_graphs_examined.csv', mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=Fieldnames)
        writer.writeheader()
    for graphname in REAL:
        print("graphname = ", graphname)
        filename = "../../data/{0}/unkOverlap_unkBlockSizeVar/{0}_unkOverlap_unkBlockSizeVar_-1_nodes".format(graphname)
        if graphname in REAL_UNDIRECTED:
            graph = load_graph_from_csv(filename + ".tsv", False, csv_options={'delimiter': ' '})
            directed=False
        else:
            graph = load_graph_from_csv(filename + ".tsv", True, csv_options={'delimiter': ' '})
            directed=True
        print("done loading graph")
        prop = examine_graph(graph, "real", graphname, True, directed)
        print("done examining graph")
        with open('real_graphs_examined.csv', mode='a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=Fieldnames)
            writer.writerow(prop._asdict())


