import numpy as np

from abc import  abstractmethod

class Distance():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, *args):

        if len(args) == 1:
            return self._compute_dist(args[0])
        elif len(args) == 2:
            delta_t = args[1]-args[0]
            return self._compute_dist(delta_t)
        else:
            raise ValueError

    def plot(self):
        pass

    @abstractmethod
    def _compute_dist(self, x, y):
        raise NotImplementedError
    
class DistanceType0(Distance):
    def _compute_dist(self, delta_t):
        scale = self.kwargs.get('scale')
        
        d = np.zeros_like(delta_t)
        p_idx =  np.where(delta_t >= 0)
        n_idx = np.where(delta_t<0) 
        d[p_idx] = delta_t[p_idx]*scale
        d[n_idx] = -delta_t[n_idx]*scale
        return d

class DistanceType1(Distance):
    def _compute_dist(self, delta_t):
        scale = self.kwargs.get('scale')
        p_alpha = self.kwargs.get('p_alpha')
        m_alpha = self.kwargs.get('m_alpha')

        d = np.zeros_like(delta_t)
        p_idx =  np.where(delta_t >= 0)
        n_idx = np.where(delta_t<0)

        d[p_idx] = p_alpha*delta_t[p_idx]*scale
        d[n_idx] = -m_alpha*delta_t[n_idx]*scale
        return d
    
class DistanceType2(Distance):
    def _compute_dist(self, delta_t):
        scale = self.kwargs.get('scale')

        d = np.zeros_like(delta_t)
        p_idx =  np.where(delta_t >= 0)
        n_idx = np.where(delta_t<0)

        d[p_idx] = 1.-np.exp(-delta_t[p_idx]*scale)
        d[n_idx] = 1.
        return d

class DistanceType3(Distance):
    def _compute_dist(self, delta_t):
        
        d = np.zeros_like(delta_t)
        pp_index = np.where(delta_t > 1)
        p_idx =  np.where( delta_t >=0 )
        n_idx = np.where(delta_t<0)
        assert len(pp_index[0]) == 0

        d[p_idx] = np.abs(delta_t[p_idx])
        d[n_idx] = 1.
        return d