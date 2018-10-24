import abc

from neuraltree.builder import RootSystemBuilder, BranchSystemBuilder, TrunkBuilder, NeuralTreeBuilder


class NeuralSystem(abc.ABC):
    def __init__(self, name: str, builder):
        self.name = name

        self.builder = builder
        self.model = builder.build()

        self.sub_models = {}

    def import_sub_models(self, neural_subsystem):
        self.sub_models[neural_subsystem.name] = neural_subsystem.model
        self.sub_models.update(neural_subsystem.sub_models)

    @staticmethod
    def create_from_model(model):
        pass


class RootSystem(NeuralSystem):
    def __init__(self, name: str, builder):
        super().__init__(name, builder)

    def import_root(self, hardpoint_layer_name, root_system):
        self.builder.import_root_system(hardpoint_layer_name, root_system.builder)
        self.model = self.builder.build()
        self.import_sub_models(root_system)


class BranchSystem(NeuralSystem):
    def __init__(self, name: str, builder):
        super().__init__(name, builder)

    def import_branch(self, hardpoint_layer_name, branch_system):
        self.builder.import_branch_system(hardpoint_layer_name, branch_system.builder)
        self.model = self.builder.build()
        self.import_sub_models(branch_system)


class TrunkSystem(NeuralSystem):
    def __init__(self, name: str, builder):
        super().__init__(name, builder)

        self.builder = TrunkBuilder()
        self.model = self.builder.build()


class NeuralTree:
    def __init__(self,
                 name: str,
                 root_system: RootSystem,
                 trunk_system: TrunkSystem,
                 branch_system: BranchSystem,
                 roots_to_trunk_map: dict,
                 trunk_to_branches_map: dict):
        self.name = name

        self.branch_system = branch_system
        self.trunk_system = trunk_system
        self.root_system = root_system

        self.builder = NeuralTreeBuilder(
            self.root_system.builder,
            self.trunk_system.builder,
            self.branch_system.builder,
            roots_to_trunk_map,
            trunk_to_branches_map
        )
        self.model = self.builder.build()

    def fit(self, xtrn, ytrn, xdev, ydev):
        pass

    def predict(self, X):
        pass
