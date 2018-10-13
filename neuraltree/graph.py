import igraph


class NoNameException(Exception):
    def __init__(self):
        super().__init__("Layer does not have a unique name.")


class NonUniqueNameException(Exception):
    def __init__(self, layer_name):
        super().__init__("Layer with name, {}, already exists in graph.".format(layer_name))


class NonExistentLayerException(Exception):
    def __init__(self, layer_name):
        super().__init__("No such layer with name, {}, exists in graph.".format(layer_name))


class LayerGraph:
    def __init__(self):
        self.graph = igraph.Graph()
        self.layer_name_to_layer = {}

    def add_layer(self, existing_layer_name, new_layer):
        graph_size = len(self.graph.vs)

        if graph_size == 0:
            if not hasattr(new_layer, "name"):
                raise NoNameException()
            else:
                self.graph.add_vertex(new_layer.name)

        if existing_layer_name not in self.layer_name_to_layer:
            raise NonExistentLayerException(existing_layer_name)
        elif not hasattr(new_layer, "name"):
            raise NoNameException()
        elif new_layer.name in self.layer_name_to_layer:
            raise NonUniqueNameException(new_layer.name)

        self.layer_name_to_layer[new_layer.name] = new_layer
        self.graph.add_edge(existing_layer_name, new_layer.name)

    def add_graph(self, existing_layer_name, new_layer_graph):
        new_vertices = new_layer_graph.graph.vs
        self.graph.add_vertices(new_vertices["name"])

        self.graph.add_edge(existing_layer_name, new_layer_graph.graph.vs[0]["name"])

        new_edge_tuples_by_name = [
            (new_vertices[edge.source], new_vertices[edge.target])
            for edge in new_layer_graph.graph.es
        ]
        self.graph.add_edges(new_edge_tuples_by_name)

        self.layer_name_to_layer.update(new_layer_graph.layer_name_to_layer)
