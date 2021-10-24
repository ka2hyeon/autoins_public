from .asymmetric import AsymmetricModel

def make(similarity_type, units):
    if similarity_type == 'asymmetric':
        return AsymmetricModel(units = units)