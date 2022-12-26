import numpy as np
import pandas as pd

from em_waves import ElectricWave
from polarizability import polarizability, ellipsoid_polarizability

from scipy.constants import speed_of_light

import matplotlib.pyplot as plt

def set_diagonal_elements(radius, part_eps: float, med_eps: float, *, a1=None, a2=None, a3=None) -> np.ndarray:
    """ sets values for diagonal elements of interaction matrix
    """
    if a1 is None:
        return polarizability(radius, part_eps, med_eps, k=0) * np.identity(3)
    else:
        A = np.identity(3)
        A[0][0] = ellipsoid_polarizability(a1, a2, a3, along='a1')
        A[1][1] = ellipsoid_polarizability(a1, a2, a3, along='a2')
        A[3][3] = ellipsoid_polarizability(a1, a2, a3, along='a3')
        return A

def set_nondiagonal_elements(r_i, r_j, k):
    """ sets values for non-diagonal elements of interaction matrix
    """
    r = abs(np.array(r_i) - (r_j))
    r_hat = (np.array(r_i) - np.array(r_j)) / r
    exponent = np.exp(1j * abs(k) * abs(r)) / abs(r)
    first = np.power(k, 2) * (np.outer(r_hat, r_hat) - np.identity(3))
    second = (1j * k * r - 1) / np.power(r, 2) * (3 * np.outer(r_hat, r_hat) - np.identity(3))
    return exponent * (first + second)


def extinction_cross_section(incident_wave, polarization, k):
    coeff = k
    return coeff*np.imag(np.sum([np.dot(np.conj(e), p) for e, p in zip(incident_wave, polarization)]))

def scattering_cross_section():
    pass

def absorbing_cross_section():
    pass

def interaction_matrix(rs, coordinates, part_eps, med_eps, k):
    interaction_matrix = np.ndarray([len(rs), len(rs), 3, 3])
    result_matrix = np.ndarray([3*len(rs), 3*len(rs)])
    
    for i in range(len(rs)):
        for j in range(len(rs)):
            if i == j:
                interaction_matrix[i][j] = set_diagonal_elements(rs[i], part_eps, med_eps)
            else:
                interaction_matrix[i][j] = set_nondiagonal_elements(coordinates[i], coordinates[j], k=k)


    k = 0
    for i in range(len(rs)):
        for line in range(3):
            result_matrix[k] = np.concatenate(([interaction_matrix[i][j][line] for j in range(len(rs))]))
            k += 1

    return result_matrix

def incident_light(coordinates):
    incident_vector = []
    for point in coordinates:
        incident_vector.append(incident_field.local_vector(point))

    incident_vector = np.asarray(incident_vector).reshape(len(rs)*3, 1)

    return incident_vector

def solve_matrix(interaction_matrix, incident_light):
    return np.linalg.solve(interaction_matrix, incident_light)

if __name__=="__main__":
    # rs = np.array([5, 5])
    # coordinates = np.array([[10, 10, 10], [20, 20, 20]])
    home = '/home/bohdan/git_projects/ddasimulation-python'

    coordinates = np.genfromtxt(f'{home}/systems/dipoles.csv', skip_header=1, usecols=(0,1,2), delimiter=',')
    rs = np.genfromtxt(f'{home}/systems/dipoles.csv', skip_header=1, usecols=(3,4,5), delimiter=',')
    print(rs)
    med_eps = 1.5

    incident_field = ElectricWave(np.array([1, 1, 0]), freq=300, k=np.array([0, 0, 1]), eps=med_eps)

    part_eps = pd.read_csv(f'{home}/eps/ag.csv')
    part_eps['eps'] = part_eps['real_eps'] + 1j * part_eps['imag_eps']
    print(part_eps)

    extinction=[]
    waves = []
    for freq, eps in zip(part_eps['freq'], part_eps['eps']):
        solution = []
        k = freq / speed_of_light
        waves.append(k)

        # solution.append(solve_matrix(rs, coordinates, k=k, part_eps=eps, med_eps=med_eps))
        solution.append(solve_matrix(interaction_matrix(rs, coordinates, k=k, part_eps=eps, med_eps=med_eps), incident_light=incident_light(coordinates)))
        
        extinction.append(-extinction_cross_section(np.reshape(incident_light(coordinates), (len(rs), 3)), np.reshape(solution, (len(rs), 3)), k=k))

    print(np.reshape(incident_light(coordinates), (len(rs), 3)))
    
    plt.scatter(waves, extinction)
    plt.show()