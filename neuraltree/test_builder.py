from keras.layers import Input, Dense
from keras.models import Model

from neuraltree.builder import RootSystemBuilder


root_1_input_layer = Input(shape=(1,))
root_1_hidden_layer = Dense(units=5)
root_1_output_layer = Dense(units=2)

# returns a name that does not correspond with name param
root_1_dict = {
    root_1_input_layer.name: root_1_input_layer,
    root_1_hidden_layer.name: root_1_hidden_layer,
    root_1_output_layer.name: root_1_output_layer
}

root_1_incoming_layers_by_name = {
    root_1_hidden_layer.name: [root_1_input_layer.name],
    root_1_output_layer.name: [root_1_hidden_layer.name]
}
root_1_outgoing_layers_by_name = {
    root_1_input_layer.name: [root_1_hidden_layer.name],
    root_1_hidden_layer.name: [root_1_output_layer.name]
}
root_1_input_layers = [root_1_input_layer]
root_1_output_layers = [root_1_output_layer]

root_1_hidden_layer_linked = root_1_hidden_layer(root_1_input_layer)
root_1_output_layer_linked = root_1_output_layer(root_1_hidden_layer_linked)
root_1_model = Model(inputs=root_1_input_layers, outputs=[root_1_output_layer_linked])

root_builder_1 = RootSystemBuilder(
    name_to_unlinked_layer=root_1_dict,
    incoming_layers_by_name=root_1_incoming_layers_by_name,
    outgoing_layers_by_name=root_1_outgoing_layers_by_name,
    input_layers=root_1_input_layers,
    output_layers=root_1_output_layers
)


def test_root_system_builder():
    assert root_builder_1.name_to_unlinked_layer == root_1_dict
    assert root_builder_1.incoming_layers_by_name == {
        root_1_hidden_layer.name: root_1_input_layer.name,
        root_1_output_layer.name: root_1_hidden_layer.name
    }
    assert root_builder_1.outgoing_layers_by_name == {
        root_1_input_layer.name: root_1_hidden_layer.name,
        root_1_hidden_layer.name: root_1_output_layer.name
    }
    assert root_builder_1.input_layers == [root_1_input_layer]
    assert root_builder_1.output_layers == [root_1_output_layer]


def test_root_system_builder_import_root():
    root_2_input_layer = Input(shape=(1,))

    root_2_dict = {root_2_input_layer.name: root_2_input_layer}
    root_2_incoming_layers_by_name = {}
    root_2_outgoing_layers_by_name = {}
    root_2_input_layers = [root_2_input_layer]
    root_2_output_layers = []

    root_builder_2 = RootSystemBuilder(
        name_to_unlinked_layer=root_2_dict,
        incoming_layers_by_name=root_2_incoming_layers_by_name,
        outgoing_layers_by_name=root_2_outgoing_layers_by_name,
        input_layers=root_2_input_layers,
        output_layers=root_2_output_layers
    )
    root_builder_1.import_root_system(root_1_hidden_layer.name, root_builder_2)

    assert root_builder_1.incoming_layers_by_name == {
        root_1_hidden_layer.name: [
            root_1_input_layer.name,
            "{}_to_{}_transition_layer_reshaped".format(root_2_input_layer.name, root_1_hidden_layer.name)
        ],
        "{}_to_{}_transition_layer_reshaped".format(root_2_input_layer.name, root_1_hidden_layer.name): [
            "{}_to_{}_transition_layer".format(root_2_input_layer.name, root_1_hidden_layer.name)
        ],
        "{}_to_{}_transition_layer".format(root_2_input_layer.name, root_1_hidden_layer.name): [
            root_2_input_layer.name
        ],
        root_1_output_layer.name: [
            root_1_hidden_layer.name
        ]
    }
    assert root_builder_1.outgoing_layers_by_name == {
        root_1_input_layer.name: [root_1_hidden_layer.name],
        root_2_input_layer.name: [
            "{}_to_{}_transition_layer".format(root_2_input_layer.name, root_1_hidden_layer.name)
        ],
        "{}_to_{}_transition_layer".format(root_2_input_layer.name, root_1_hidden_layer.name): [
            "{}_to_{}_transition_layer_reshaped".format(root_2_input_layer.name, root_1_hidden_layer.name)
        ],
        "{}_to_{}_transition_layer_reshaped".format(root_2_input_layer.name, root_1_hidden_layer.name): [
            root_1_hidden_layer.name
        ],
        root_1_hidden_layer.name: [root_1_output_layer.name]
    }
    assert root_builder_1.input_layers == [root_1_input_layer, root_2_input_layer]
    assert root_builder_1.output_layers == [root_1_output_layer]
