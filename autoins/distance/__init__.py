from .distance import DistanceType0, DistanceType1, DistanceType2, DistanceType3

mapper = {
    'dist_type0' : DistanceType0,
    'dist_type1' : DistanceType1,
    'dist_type2' : DistanceType2,
    'dist_type3' : DistanceType3,  
}


def make(name, **kwargs):
    return mapper[name](**kwargs)