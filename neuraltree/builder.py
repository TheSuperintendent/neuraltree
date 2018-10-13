import abc
import keras.backend as K
import numpy as np

from keras.layers import Dense, Reshape, Concatenate
from keras.models import Model

from neuraltree.graph import NonUniqueNameException, NonExistentLayerException


def get_transition_layers(outgoing_layer_input_shape, incoming_layer_name, outgoing_layer_name):
    transition_layer_output_shape = K.int_shape(outgoing_layer_input_shape)
    hidden_units = np.prod(transition_layer_output_shape)

    transition_layer = Dense(
        units=hidden_units,
        activation="relu",
        name=incoming_layer_name + "_to_" + outgoing_layer_name + "_transition_layer"
    )

    reshaped_transition_layer = Reshape(
        target_shape=transition_layer_output_shape,
        name=transition_layer.name + "_reshaped"
    )

    return transition_layer, reshaped_transition_layer


def get_layers_from_names(name_to_layer: dict, layer_names: list) -> list:
    neural_layers = []
    for name in layer_names:
        if name in name_to_layer:
            neural_layers.append(name_to_layer[name])
        else:
            raise NonExistentLayerException(name)

    return neural_layers


class NeuralBuilder(abc.ABC):
    def __init__(self,
                 name_to_layer: dict,
                 incoming_layers_by_name: dict,
                 outgoing_layers_by_name: dict,
                 input_layers: list,
                 output_layers: list):
        self.name_to_layer = name_to_layer

        # keep track of layers' incoming layer and outgoing layers
        self.incoming_layers_by_name = incoming_layers_by_name
        self.outgoing_layers_by_name = outgoing_layers_by_name

        self.input_layers = input_layers
        self.output_layers = output_layers

    def build(self):
        name_to_connected_layer = {}

        for name in self.incoming_layers_by_name.keys():
            curr_layer = self.name_to_layer[name] \
                if name not in name_to_connected_layer \
                else name_to_connected_layer[name]

            incoming_layer_names = self.incoming_layers_by_name[name]
            incoming_layers = get_layers_from_names(name_to_connected_layer, incoming_layer_names)

            if len(incoming_layers) > 1:
                concat_layer = Concatenate(incoming_layers)
                curr_layer_connected = curr_layer(concat_layer)
            else:
                curr_layer_connected = curr_layer(incoming_layers[0])

            name_to_connected_layer[name] = curr_layer_connected

        return Model(inputs=self.input_layers, output=self.output_layers)

    def __import_dicts(self, imported_neural_arch):
        self.__import_dict_by_attr_name(imported_neural_arch, "name_to_layer")
        self.__import_dict_by_attr_name(imported_neural_arch, "incoming_layers_by_name")
        self.__import_dict_by_attr_name(imported_neural_arch, "outgoing_layers_by_name")

    def __import_dict_by_attr_name(self, imported_neural_arch, dict_attr_name):
        if not hasattr(self, dict_attr_name) or not hasattr(imported_neural_arch, dict_attr_name):
            raise AttributeError()

        self_dict_attr = getattr(self, dict_attr_name)
        imported_dict_attr = getattr(imported_neural_arch, dict_attr_name)

        names_diff_set = self_dict_attr.keys() - imported_dict_attr.keys()
        if len(names_diff_set) > 0:
            raise NonUniqueNameException(next(iter(names_diff_set)))

        self_dict_attr.update(imported_dict_attr)


class RootSystemBuilder(NeuralBuilder):
    def __init__(self,
                 name_to_layer: dict = {},
                 incoming_layers_by_name: dict = {},
                 outgoing_layers_by_name: dict = {},
                 input_layers: list = [],
                 output_layers: list = []):
        super().__init__(name_to_layer, incoming_layers_by_name, outgoing_layers_by_name, input_layers, output_layers)

    def import_root_system(self, hardpoint_layer_name, root_system):
        attach_layer = root_system.get_attach_layer()
        hardpoint_layer = self.name_to_layer[hardpoint_layer_name]

        transition_layer, reshaped_transition_layer = get_transition_layers(
            hardpoint_layer.input_shape,
            attach_layer.name,
            hardpoint_layer_name
        )

        self.__import_dicts(root_system)

        self.incoming_layers_by_name[hardpoint_layer_name].append(reshaped_transition_layer.name)
        self.incoming_layers_by_name[transition_layer.name] = [attach_layer.name]
        self.incoming_layers_by_name[reshaped_transition_layer.name] = [transition_layer.name]

        self.outgoing_layers_by_name[attach_layer.name] = [transition_layer.name]
        self.outgoing_layers_by_name[transition_layer.name] = [reshaped_transition_layer.name]
        self.outgoing_layers_by_name[reshaped_transition_layer.name] = [hardpoint_layer.name]

        self.input_layers.append(root_system.input_layers)
        self.output_layers.append(root_system.output_layers)


class BranchSystemBuilder(NeuralBuilder):
    def __init__(self,
                 name_to_layer: dict = {},
                 incoming_layers_by_name: dict = {},
                 outgoing_layers_by_name: dict = {},
                 input_layers: list = [],
                 output_layers: list = []):
        super().__init__(name_to_layer, incoming_layers_by_name, outgoing_layers_by_name, input_layers, output_layers)

    def import_branch_system(self, hardpoint_layer_name, branch_system):
        attach_layer = branch_system.get_attach_layer()
        hardpoint_layer = self.name_to_layer[hardpoint_layer_name]

        transition_layer, reshaped_transition_layer = get_transition_layers(
            attach_layer.input_shape,
            hardpoint_layer_name,
            attach_layer.name
        )

        self.__import_dicts(branch_system)

        self.incoming_layers_by_name[attach_layer.name] = [reshaped_transition_layer.name]
        self.incoming_layers_by_name[transition_layer.name] = [hardpoint_layer.name]
        self.incoming_layers_by_name[reshaped_transition_layer.name] = [transition_layer.name]

        self.outgoing_layers_by_name[hardpoint_layer_name].append(transition_layer.name)
        self.outgoing_layers_by_name[transition_layer.name] = [reshaped_transition_layer.name]
        self.outgoing_layers_by_name[reshaped_transition_layer.name] = [attach_layer.name]

        self.input_layers.append(branch_system.input_layers)
        self.output_layers.append(branch_system.output_layers)


class TrunkBuilder(NeuralBuilder):
    def __init__(self,
                 name_to_layer: dict = {},
                 incoming_layers_by_name: dict = {},
                 outgoing_layers_by_name: dict = {},
                 input_layers: list = [],
                 output_layers: list = []):
        super().__init__(name_to_layer, incoming_layers_by_name, outgoing_layers_by_name, input_layers, output_layers)

    def import_bark(self, hardpoint_input_name, hardpoint_output_name, bark):
        self.__import_dicts(bark)
        self.__attach_bark_input(hardpoint_input_name, bark)
        self.__attach_bark_output(hardpoint_output_name, bark)

    def __attach_bark_input(self, hardpoint_layer_name, bark):
        attach_layer = bark.get_input_attach_layer()
        hardpoint_layer = self.name_to_layer[hardpoint_layer_name]

        transition_layer, reshaped_transition_layer = get_transition_layers(
            attach_layer.input_shape,
            hardpoint_layer_name,
            attach_layer.name
        )

        self.incoming_layers_by_name[attach_layer.name] = [reshaped_transition_layer.name]
        self.incoming_layers_by_name[transition_layer.name] = [hardpoint_layer.name]
        self.incoming_layers_by_name[reshaped_transition_layer.name] = [transition_layer.name]

        self.outgoing_layers_by_name[hardpoint_layer_name].append(transition_layer.name)
        self.outgoing_layers_by_name[transition_layer.name] = [reshaped_transition_layer.name]
        self.outgoing_layers_by_name[reshaped_transition_layer.name] = [attach_layer.name]

        self.input_layers.append(bark.input_layers)

    def __attach_bark_output(self, hardpoint_layer_name, bark):
        attach_layer = bark.get_output_attach_layer()
        hardpoint_layer = self.name_to_layer[hardpoint_layer_name]

        transition_layer, reshaped_transition_layer = get_transition_layers(
            hardpoint_layer.input_shape,
            attach_layer.name,
            hardpoint_layer_name
        )

        self.incoming_layers_by_name[hardpoint_layer_name].append(reshaped_transition_layer.name)
        self.incoming_layers_by_name[transition_layer.name] = [attach_layer.name]
        self.incoming_layers_by_name[reshaped_transition_layer.name] = [transition_layer.name]

        self.outgoing_layers_by_name[attach_layer.name] = [transition_layer.name]
        self.outgoing_layers_by_name[transition_layer.name] = [reshaped_transition_layer.name]
        self.outgoing_layers_by_name[reshaped_transition_layer.name] = [hardpoint_layer.name]

        self.output_layers.append(bark.output_layers)


class NeuralTreeBuilder:
    def __init__(self):
        self.root_builder = RootSystemBuilder()
        self.trunk_builder = TrunkBuilder()
        self.branch_builder = BranchSystemBuilder()

        self.incoming_layers_by_name = {}

    def build(self):
        # TODO: fill roots_to_trunk_map and trunk_to_branches_map
        roots_to_trunk_map = {}
        trunk_to_branches_map = {}
        self.build_roots_to_trunk(roots_to_trunk_map)
        self.build_trunk_to_branches(trunk_to_branches_map)

    def build_roots_to_trunk(self, roots_to_trunk_map):
        root_output_layers = self.root_builder.output_layers

        for root_layer in root_output_layers:
            trunk_outgoing_layers = get_layers_from_names(
                self.trunk_builder.name_to_layer,
                roots_to_trunk_map[root_layer.name]
            )
            for trunk_layer in trunk_outgoing_layers:
                transition_layer, reshaped_transition_layer = get_transition_layers(
                    trunk_layer.input_shape,
                    root_layer.name,
                    trunk_layer.name
                )

                self.incoming_layers_by_name[trunk_layer].append(reshaped_transition_layer.name)
                self.incoming_layers_by_name[transition_layer.name] = [root_layer.name]
                self.incoming_layers_by_name[reshaped_transition_layer.name] = [transition_layer.name]

    def build_trunk_to_branches(self, trunk_to_branches_map):
        trunk_output_layers = self.trunk_builder.output_layers

        for trunk_layer in trunk_output_layers:
            branch_outgoing_layers = get_layers_from_names(
                self.branch_builder.name_to_layer,
                trunk_to_branches_map[trunk_layer.name]
            )
            for branch_layer in branch_outgoing_layers:
                transition_layer, reshaped_transition_layer = get_transition_layers(
                    branch_layer.input_shape,
                    trunk_layer.name,
                    branch_layer.name
                )

                self.incoming_layers_by_name[branch_layer].append(reshaped_transition_layer.name)
                self.incoming_layers_by_name[transition_layer.name] = [trunk_layer.name]
                self.incoming_layers_by_name[reshaped_transition_layer.name] = [transition_layer.name]
