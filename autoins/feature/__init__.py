from .mlp_model import MlpModel


def make(feature_type,
            node_list,
            activation_list):

    if feature_type == 'mlp':
        return MlpModel(node_list, activation_list)