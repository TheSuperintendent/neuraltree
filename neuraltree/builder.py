import abc


class NeuralBuilder(abc.ABC):
    def __init__(self):
        self.layers = {}

    @abc.abstractmethod
    def build(self):
        return


class RootSystemBuilder(NeuralBuilder):
    def __init__(self):
        super().__init__()

    def build(self):
        # TODO: implement this method
        # TODO: name each layer and add to layer
        pass

    def import_secondary_root(self, secondary_root):
        # TODO: implement this method
        pass
