import abc

from neuraltree.builder import RootSystemBuilder, BranchSystemBuilder


class NeuralSystem(abc.ABC):
    def __init__(self, name: str):
        self.name = name

        self.builder = None
        self.model = None

        self.sub_models = {}

    def __import_sub_models(self, neural_system):
        self.sub_models[neural_system.name] = neural_system.model
        self.sub_models.update(neural_system.sub_models)


class RootSystem(NeuralSystem):
    def __init__(self, name: str):
        super().__init__(name)

        self.builder = RootSystemBuilder()
        self.model = self.builder.build()

    def import_root(self, hardpoint_layer_name, root_system):
        self.builder.import_root_system(hardpoint_layer_name, root_system.builder)
        self.model = self.builder.build()
        self.__import_sub_models(root_system)


class BranchSystem(NeuralSystem):
    def __init__(self, name: str):
        super().__init__(name)

        self.builder = BranchSystemBuilder()
        self.model = self.builder.build()

    def import_branch(self, hardpoint_layer_name, branch_system):
        self.builder.import_branch_system(hardpoint_layer_name, branch_system.builder)
        self.model = self.builder.build()
        self.__import_sub_models(branch_system)
