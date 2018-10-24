import abc
import numpy as np
import keras.backend as K

from keras.layers import Dense, Reshape, Concatenate
from keras.models import Model

from neuraltree.graph import NonUniqueNameException, NonExistentLayerException


def get_transition_layers(outgoing_layer_input_shape, incoming_layer_name, outgoing_layer_name):
    transition_layer_output_shape = outgoing_layer_input_shape[1:]
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


def get_tensor_layers_from_names(name_to_unlinked_layer: dict, name_to_linked_layer: dict, layer_names: list) -> list:
    neural_layers = []
    for name in layer_names:
        tensor_layer = None
        try:
            if name in name_to_unlinked_layer:
                if K.is_keras_tensor(name_to_unlinked_layer[name]):
                    tensor_layer = name_to_unlinked_layer[name]
            else:
                raise NonExistentLayerException(name)
        except ValueError:
            if name in name_to_linked_layer:
                if K.is_keras_tensor(name_to_linked_layer[name]):
                    tensor_layer = name_to_linked_layer[name]
            else:
                raise NonExistentLayerException(name)

        neural_layers.append(tensor_layer)

    return neural_layers


def get_layers_from_names(name_to_layer: dict, layer_names: list) -> list:
    neural_layers = []
    for name in layer_names:
        if name in name_to_layer:
            neural_layers.append(name_to_layer[name])
        else:
            raise NonExistentLayerException(name)

    return neural_layers


def get_name_to_unlinked_layer_dict(model):
    name_to_unlinked_layer = {layer.name: layer for layer in model.layers}
    for input_layer in model.inputs:
        name_to_unlinked_layer[parse_out_unlinked_name(input_layer.name)] = input_layer

    return name_to_unlinked_layer


def get_incoming_and_outgoing_layers(model):
    incoming_layers_by_name = {}
    outgoing_layers_by_name = {}
    layer_build_order_by_name = []

    layers_by_depth = model._layers_by_depth

    for depth in layers_by_depth:
        layers = layers_by_depth[depth]
        for layer in layers:
            incoming_layers = []
            outgoing_layers = []
            try:
                incoming_layers.append(parse_out_unlinked_name(layer.input.name))
                outgoing_layers.append(parse_out_unlinked_name(layer.output.name))
            except AttributeError:
                i = 0
                while True:
                    try:
                        incoming_layers.append(parse_out_unlinked_name(layer.get_input_at(i).name))
                        outgoing_layers.append(parse_out_unlinked_name(layer.get_output_at(i).name))
                    except ValueError:
                        break
            incoming_layers_by_name[layer.name] = incoming_layers
            outgoing_layers_by_name[layer.name] = outgoing_layers

            layer_build_order_by_name.append(layer.name)

    layer_build_order_by_name.reverse()

    return incoming_layers_by_name, outgoing_layers_by_name, layer_build_order_by_name[1:]


def parse_out_unlinked_name(layer_name: str):
    parsed_by_slash = layer_name.split("/")[0]
    return parsed_by_slash.split(":")[0]


def get_input_and_output_layers(model):
    return model.inputs, model.outputs


class NeuralBuilder(abc.ABC):
    def __init__(self,
                 name_to_unlinked_layer: dict,
                 incoming_layers_by_name: dict,
                 outgoing_layers_by_name: dict,
                 layer_build_order_by_name: list,
                 input_layers: list,
                 output_layers: list):
        self.name_to_unlinked_layer = name_to_unlinked_layer

        # keep track of layers' incoming layers and outgoing layers
        self.incoming_layers_by_name = incoming_layers_by_name
        self.outgoing_layers_by_name = outgoing_layers_by_name

        self.layer_build_order_by_name = layer_build_order_by_name

        self.input_layers = input_layers
        self.output_layers = output_layers

    def build(self):
        name_to_linked_layer = {}

        for name in self.layer_build_order_by_name:
            curr_layer = self.name_to_unlinked_layer[name] \
                if name not in name_to_linked_layer \
                else name_to_linked_layer[name]

            # print("incoming layers by name: %s" % str(self.incoming_layers_by_name))
            # print("name to unlinked layers: %s" % str(self.name_to_unlinked_layer))
            # print("name to linked layers: %s" % str(name_to_linked_layer))

            incoming_layer_names = self.incoming_layers_by_name[name]
            incoming_layers = get_tensor_layers_from_names(self.name_to_unlinked_layer,
                                                           name_to_linked_layer,
                                                           incoming_layer_names)

            # print(incoming_layers)
            # print(K.is_keras_tensor(incoming_layers[0]))

            if len(incoming_layers) > 1:
                concat_layer = Concatenate(incoming_layers)
                curr_layer_connected = curr_layer(concat_layer)
            else:
                curr_layer_connected = curr_layer(incoming_layers[0])

            name_to_linked_layer[name] = curr_layer_connected

        # TODO: call model.compile to add optimizer, loss and metrics
        return Model(inputs=self.input_layers, output=self.output_layers)

    def import_dicts(self, imported_neural_arch):
        self.__import_dict_by_attr_name(imported_neural_arch, "name_to_unlinked_layer")
        self.__import_dict_by_attr_name(imported_neural_arch, "incoming_layers_by_name")
        self.__import_dict_by_attr_name(imported_neural_arch, "outgoing_layers_by_name")

    def __import_dict_by_attr_name(self, imported_neural_arch, dict_attr_name):
        if not hasattr(self, dict_attr_name) or not hasattr(imported_neural_arch, dict_attr_name):
            raise AttributeError()

        self_dict_attr = getattr(self, dict_attr_name)
        imported_dict_attr = getattr(imported_neural_arch, dict_attr_name)

        names_diff_set = self_dict_attr.keys() - imported_dict_attr.keys()
        if len(names_diff_set) == 0:
            raise NonUniqueNameException(next(iter(names_diff_set)))

        self_dict_attr.update(imported_dict_attr)


class RootSystemBuilder(NeuralBuilder):
    def __init__(self,
                 name_to_unlinked_layer: dict = {},
                 incoming_layers_by_name: dict = {},
                 outgoing_layers_by_name: dict = {},
                 layer_build_order_by_name: list = [],
                 input_layers: list = [],
                 output_layers: list = []):
        super().__init__(
            name_to_unlinked_layer,
            incoming_layers_by_name,
            outgoing_layers_by_name,
            layer_build_order_by_name,
            input_layers,
            output_layers
        )

    def import_root_system(self, hardpoint_layer_name, root_system):
        attach_layer = root_system.get_attach_layer()
        hardpoint_layer = self.name_to_unlinked_layer[hardpoint_layer_name]

        transition_layer, reshaped_transition_layer = get_transition_layers(
            hardpoint_layer.input_shape,
            attach_layer.name,
            hardpoint_layer_name
        )

        self.import_dicts(root_system)

        self.incoming_layers_by_name[hardpoint_layer_name].append(reshaped_transition_layer.name)
        self.incoming_layers_by_name[transition_layer.name] = [attach_layer.name]
        self.incoming_layers_by_name[reshaped_transition_layer.name] = [transition_layer.name]

        self.outgoing_layers_by_name[attach_layer.name] = [transition_layer.name]
        self.outgoing_layers_by_name[transition_layer.name] = [reshaped_transition_layer.name]
        self.outgoing_layers_by_name[reshaped_transition_layer.name] = [hardpoint_layer.name]

        self.input_layers.extend(root_system.input_layers)
        self.output_layers.extend(root_system.output_layers)

    def get_attach_layer(self):
        return self.input_layers[0]


class BranchSystemBuilder(NeuralBuilder):
    def __init__(self,
                 name_to_unlinked_layer: dict = {},
                 incoming_layers_by_name: dict = {},
                 outgoing_layers_by_name: dict = {},
                 layer_build_order_by_name: list = [],
                 input_layers: list = [],
                 output_layers: list = []):
        super().__init__(
            name_to_unlinked_layer,
            incoming_layers_by_name,
            outgoing_layers_by_name,
            layer_build_order_by_name,
            input_layers,
            output_layers
        )

    def import_branch_system(self, hardpoint_layer_name, branch_system):
        attach_layer = branch_system.get_attach_layer()
        hardpoint_layer = self.name_to_unlinked_layer[hardpoint_layer_name]

        transition_layer, reshaped_transition_layer = get_transition_layers(
            attach_layer.input_shape,
            hardpoint_layer_name,
            attach_layer.name
        )

        self.import_dicts(branch_system)

        self.incoming_layers_by_name[attach_layer.name] = [reshaped_transition_layer.name]
        self.incoming_layers_by_name[transition_layer.name] = [hardpoint_layer.name]
        self.incoming_layers_by_name[reshaped_transition_layer.name] = [transition_layer.name]

        self.outgoing_layers_by_name[hardpoint_layer_name].append(transition_layer.name)
        self.outgoing_layers_by_name[transition_layer.name] = [reshaped_transition_layer.name]
        self.outgoing_layers_by_name[reshaped_transition_layer.name] = [attach_layer.name]

        self.input_layers.extend(branch_system.input_layers)
        self.output_layers.extend(branch_system.output_layers)


class TrunkBuilder(NeuralBuilder):
    def __init__(self,
                 name_to_unlinked_layer: dict = {},
                 incoming_layers_by_name: dict = {},
                 outgoing_layers_by_name: dict = {},
                 layer_build_order_by_name: list = [],
                 input_layers: list = [],
                 output_layers: list = []):
        super().__init__(
            name_to_unlinked_layer,
            incoming_layers_by_name,
            outgoing_layers_by_name,
            layer_build_order_by_name,
            input_layers,
            output_layers
        )

    def import_bark(self, hardpoint_input_name, hardpoint_output_name, bark):
        self.import_dicts(bark)
        self.__attach_bark_input(hardpoint_input_name, bark)
        self.__attach_bark_output(hardpoint_output_name, bark)

    def __attach_bark_input(self, hardpoint_layer_name, bark):
        attach_layer = bark.get_input_attach_layer()
        hardpoint_layer = self.name_to_unlinked_layer[hardpoint_layer_name]

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

        self.input_layers.extend(bark.input_layers)

    def __attach_bark_output(self, hardpoint_layer_name, bark):
        attach_layer = bark.get_output_attach_layer()
        hardpoint_layer = self.name_to_unlinked_layer[hardpoint_layer_name]

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

        self.output_layers.extend(bark.output_layers)


class NeuralTreeBuilder:
    def __init__(self, root_builder, trunk_builder, branch_builder, roots_to_trunk_map, trunk_to_branches_map):
        self.root_builder = root_builder
        self.trunk_builder = trunk_builder
        self.branch_builder = branch_builder

        self.roots_to_trunk_map = roots_to_trunk_map
        self.trunk_to_branches_map = trunk_to_branches_map

        self.incoming_layers_by_name = {}

    def build(self):
        self.build_roots_to_trunk(self.roots_to_trunk_map)
        self.build_trunk_to_branches(self.trunk_to_branches_map)

    def build_roots_to_trunk(self, roots_to_trunk_map):
        for root_layer_name in roots_to_trunk_map.keys():
            trunk_outgoing_layers = get_layers_from_names(
                self.trunk_builder.name_to_unlinked_layer,
                roots_to_trunk_map[parse_out_unlinked_name(root_layer_name)]
            )
            for trunk_layer in trunk_outgoing_layers:
                transition_layer, reshaped_transition_layer = get_transition_layers(
                    trunk_layer.input_shape,
                    parse_out_unlinked_name(root_layer_name),
                    trunk_layer.name
                )

                self.incoming_layers_by_name[trunk_layer].append(reshaped_transition_layer.name)
                self.incoming_layers_by_name[transition_layer.name] = [parse_out_unlinked_name(root_layer_name.name)]
                self.incoming_layers_by_name[reshaped_transition_layer.name] = [transition_layer.name]

    def build_trunk_to_branches(self, trunk_to_branches_map):
        trunk_output_layers = self.trunk_builder.output_layers

        for trunk_layer in trunk_output_layers:
            branch_outgoing_layers = get_layers_from_names(
                self.branch_builder.name_to_unlinked_layer,
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
