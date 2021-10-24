from .mlp_model import MlpModel


def make(classifier_type, node_list, activation_list):
    if classifier_type == 'mlp':
        return MlpModel(node_list,
                            activation_list)