import pyglet
import pickle as pkl
import sys
import networkx as nx
import matplotlib.pyplot as plt

# game_window = pyglet.window.Window(800, 600)

# if __name__ == '__main__':
#     with open("multiagentnfv/scenarios/topo/netGraph_abilene.pickle", 'rb') as f:
#         if sys.version_info > (3, 0):
#             graph = dict(pkl.load(f, encoding='latin1'))
#         else:
#             graph = dict(pkl.load(f))
#         nodes = graph['nodes']
#         edges = graph['edges']  # (u,v)
#         graph_topo = dict()
#         for item in edges:
#             if graph_topo.get(item[0]) == None:
#                 graph_topo[item[0]] = [item[1]]
#             else:
#                 graph_topo[item[0]].append(item[1])
#         for item in nodes:
#             if graph_topo.get(item) == None:
#                 graph_topo[item] = []
#         G = nx.from_dict_of_lists(graph_topo)
#         pos = nx.spring_layout(G)
#         nx.draw(G)
#         plt.axis("off")         
#         plt.show()
#     pyglet.app.run()
"""
==========
Javascript
==========

Example of writing JSON format graph data and using the D3 Javascript library
to produce an HTML/Javascript drawing.

You will need to download the following directory:

- https://github.com/networkx/networkx/tree/master/examples/javascript/force
"""
import json

import flask
import networkx as nx
from networkx.readwrite import json_graph

G = nx.barbell_graph(6, 3)
# this d3 example uses the name attribute for the mouse-hover value,
# so add a name to each node
for n in G:
    G.nodes[n]["name"] = n
# write json formatted data
d = json_graph.node_link_data(G)  # node-link format to serialize
# write json
with open("force.json", 'w') as f:
    json.dump(d, f)
print("Wrote node-link JSON data to force/force.json")

# Serve the file over http to allow for cross origin requests
app = flask.Flask(__name__, static_folder="force")


@app.route("/")
def static_proxy():
    return app.send_static_file("force.html")


print("\nGo to http://localhost:8000 to see the example\n")
app.run(port=8000)
