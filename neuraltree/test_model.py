import keras.backend as K

from keras.layers import Input, Dense
from keras.models import Model

from neuraltree.builder import \
    get_name_to_unlinked_layer_dict, \
    get_incoming_and_outgoing_layers, \
    get_input_and_output_layers, \
    RootSystemBuilder, BranchSystemBuilder, TrunkBuilder
from neuraltree.model import RootSystem, BranchSystem, TrunkSystem
from neuraltree.model import NeuralTree


def create_sample_model():
    # init
    input_layer = Input(shape=(50,))
    hidden_layer_1 = Dense(units=20)
    hidden_layer_2 = Dense(units=20)
    output_layer = Dense(units=4)

    # build
    hidden_layer_1_linked = hidden_layer_1(input_layer)
    hidden_layer_2_linked = hidden_layer_2(hidden_layer_1_linked)
    output_layer_linked = output_layer(hidden_layer_2_linked)

    model = Model(inputs=[input_layer], outputs=[output_layer_linked])
    model.compile(optimizer="rmsprop", loss="mse")

    return model


def create_root_system():
    sample_root_model = create_sample_model()

    incoming_layers_by_name, \
        outgoing_layers_by_name, \
        layer_build_order_by_name = get_incoming_and_outgoing_layers(sample_root_model)

    input_layers, output_layers = get_input_and_output_layers(sample_root_model)
    root_builder = RootSystemBuilder(
        get_name_to_unlinked_layer_dict(sample_root_model),
        incoming_layers_by_name,
        outgoing_layers_by_name,
        layer_build_order_by_name,
        input_layers,
        output_layers
    )

    return RootSystem("", root_builder)


def create_branch_system():
    sample_branch_model = create_sample_model()

    incoming_layers_by_name, \
        outgoing_layers_by_name, \
        layer_build_order_by_name = get_incoming_and_outgoing_layers(sample_branch_model)

    input_layers, output_layers = get_input_and_output_layers(sample_branch_model)
    branch_builder = BranchSystemBuilder(
        get_name_to_unlinked_layer_dict(sample_branch_model),
        incoming_layers_by_name,
        outgoing_layers_by_name,
        layer_build_order_by_name,
        input_layers,
        output_layers
    )

    return BranchSystem("", branch_builder)


def create_trunk_system():
    sample_trunk_model = create_sample_model()

    incoming_layers_by_name, \
        outgoing_layers_by_name, \
        layer_build_order_by_name = get_incoming_and_outgoing_layers(sample_trunk_model)

    input_layers, output_layers = get_input_and_output_layers(sample_trunk_model)
    trunk_builder = TrunkBuilder(
        get_name_to_unlinked_layer_dict(sample_trunk_model),
        incoming_layers_by_name,
        outgoing_layers_by_name,
        layer_build_order_by_name,
        input_layers,
        output_layers
    )

    return TrunkSystem("", trunk_builder)


def test_create_root_system():
    root_system = create_root_system()
    assert RootSystem == type(root_system)
    assert not {} == root_system.builder.name_to_unlinked_layer
    print(root_system.builder.name_to_unlinked_layer)


def test_create_branch_system():
    branch_system = create_branch_system()
    assert BranchSystem == type(branch_system)
    assert not {} == branch_system.builder.name_to_unlinked_layer
    print(branch_system.builder.name_to_unlinked_layer)


def test_create_trunk_system():
    trunk_system = create_trunk_system()
    assert TrunkSystem == type(trunk_system)
    assert not {} == trunk_system.builder.name_to_unlinked_layer
    print(trunk_system.builder.name_to_unlinked_layer)


def test_create_tree():
    root_system = create_root_system()
    branch_system = create_branch_system()
    trunk_system = create_trunk_system()

    # --- Root System --- #
    # input
    # hidden
    # hidden-------------------
    # output                    |
    # --- Trunk System --- #    |
    # input                     |
    # hidden<------------------
    # hidden-------------------
    # output                    |
    # --- Branch System --- #   |
    # input                     |
    # hidden<------------------
    # hidden
    # output

    roots_to_trunk_map = {
        "dense_2": "dense_1"
    }
    trunk_to_branches_map = {
        "dense_2": "dense_1"
    }

    tree = NeuralTree("", root_system, trunk_system, branch_system, roots_to_trunk_map, trunk_to_branches_map)
