import numpy as np
from scipy.constants import speed_of_light

class ElectricWave():
    """
    A class used to represent electric field wave
    
    ...

    Attributes
    ----------
    E: np.ndarray
        electric field vector (in V / m)
    frequency:
        wave frequency (in Herz)
    wavevector:
        wavevector (in 1 / m)
    dielectric constant:
        dielectric constant of medium in which wave propagates
    
    """
    def __init__(self,
                 E: np.ndarray,
                 freq,
                 k: np.ndarray,
                 eps: float):
        """Create an instance of ElectricWave class.

        Keyword arguments:
        E_x, E_y, E_z -- form electric field vector (in V / m)
        freq -- frequency of wave (in Herz)
        k-- wavevector, non-dimensional, is transformed to unit vector internally
        eps -- medium dielectric constant (non-dimensional)
        """
        if np.ndim(E) != 1:
            raise ValueError("Electric field vector isn't one-dimensional")

        if len(E) != 3:
            raise ValueError("Electric field does not have all three")

        if np.ndim(k) != 1:
            raise ValueError("Wavevector isn't one-dimensional")

        if len(k) != 3:
            raise ValueError("Wavevector does not have all three")

        if np.dot(E, k) != 0:
            raise ValueError("k must be orthogonal to E")

        self.E = E
        self.frequency = freq
        self.dielectric_constant = eps
        self.refractive_index = np.power(eps, 2)
        self.k_modulus = freq * self.refractive_index / speed_of_light
        
        self.wavevector = self.k_modulus * self._get_unit_vector(k)
        self.amplitude = np.linalg.norm(self.E)

    def _get_unit_vector(self, vector: np.ndarray) -> np.ndarray:
        """ Create unit vector from arbitrary vector
        
        """
        if np.linalg.norm(vector) != 1:
            return vector / np.linalg.norm(vector)

        return vector

    def _get_normal_component(self, surface_vector: np.ndarray, point=np.array([0, 0, 0])):
        """ Finds normal component of electric field vector at certain point
        """
        return np.dot(self.local_value(point), surface_vector) * surface_vector

    def _get_tangential_component(self, surface_vector: np.ndarray, point=np.array([0, 0, 0])):
        """
        finds tangential component of electric field vector at certain point

        Parameters:
        -----------
        surface_vector: numpy array
            unit vector of surface
        point: numpy array
            radius vector to point of interest

        Returns:
        -----------
        float
            tangential component of electric field vector at certain point
        """
        return -np.cross(surface_vector, np.cross(surface_vector, self.local_vector(point)))

    # def _is_orthogonal(vector_1: np.ndarray, vector_2: np.ndarray):
    #     if np.dot(vector_1, vector_2) == 0:
    #         return True
    #     else:
    #         return False

    def intensity(self) -> float:
        """Returns intensity of wave"""
        return np.power(self.amplitude, 2)

    def refraction(self, incident_angle: float, n_2: float):
        """Returns refracted electric wave"""
        ky_reflected = self.k[1]
        kz_reflected = np.sqrt(np.power(n_2, 2) / np.power(self.refractive_index, 2) * np.power(self.k_modulus, 2) - np.power(self.k[1], 2))
        
        pass
    
    def local_value(self, radius: np.ndarray, t=0) -> float:
        """value of electric field at certain point in space"""
        return self.amplitude * np.exp(self.frequency * t * 1j) * np.exp( - 1j * np.dot(self.wavevector, radius))

    def local_vector(self, point, t=0):
        constant = self.amplitude * np.exp(self.frequency *t * 1j)
        result = []
        for coord, wave in zip(point, self.wavevector):
            result.append(constant * np.exp(-1j * coord * wave))

        return result

if __name__ == '__main__':
    k = np.array([1, 1, 0])
    E_0 = np.array([0, 0, 2])

    inc_wave = ElectricWave(E_0, 300, k, 1.21)

    print(inc_wave._get_normal_component(surface_vector=np.array([0, 0, 1])))
    #print(inc_wave._get_tangential_component(surface_vector=np.array([0, 0, 1])))
    r_0 = np.array([10, 0, 1])
    print(inc_wave.local_value(r_0))
    print(inc_wave.local_vector(r_0))