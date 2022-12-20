import numpy as np
from scipy.integrate import quad

def geometrical_factor(a_1, a_2, a_3, along='a1'):
    coefficient = a_1 * a_2 * a_3 / 2
    root = lambda q: np.sqrt((a_1**2 +q) * (a_2**2 + q) * (a_3**2 + q))
    
    if along == 'a1':
        denom = lambda q: (a_1**2 + q) * root(q)
    elif along == 'a2':
        denom = lambda q: (a_2**2 + q) * root(q)
    elif along == 'a3':
        denom = lambda q: (a_3**2 + q) * root(q)
    else:
        raise ValueError("along argument is not correct")


    f = lambda q: 1 / denom(q)

    integral, rest = quad(f, 0, np.inf)

    print(integral)
    
    return coefficient*integral

def ellipsoid_polarizability(a_1, a_2, a_3, *, along='a1', med_eps, part_eps):
    L_i = 1 / 3

    if along == 'a1':
        L_i = geometrical_factor(a_1, a_2, a_3)
    elif along == 'a2':
        L_i = geometrical_factor(a_1, a_2, a_3, along='a_2')
    elif along == 'a3':
        L_i = geometrical_factor(a_1, a_2, a_3, along='a3')
    return 4 * np.pi * a_1 * a_2 * a_3 * (part_eps - med_eps) / (3 * med_eps * 3 * L_i * (part_eps - med_eps))

def Clausius_Mosotti_relation(radius: float, part_eps: float, med_eps: float, figure="sphere") -> float:
    """ calculates Clasius-Mosotti polarizability
    
    Parameters
    -----------
    radius: float
        radius of particle
    
    part_eps: float
        dielectric constant of particle

    med_eps: float
        dielectric constant of medium
    """
    if figure == "sphere":
        return 4 * np.pi * np.power(radius, 3) * (part_eps - med_eps) / (part_eps + 2 * med_eps)
    elif figure == "ellipse":
        return 
    else:
        raise ValueError("incorrect value for figure parameter")

def polarizability(radius: float, part_eps:float, med_eps: float, k: float) -> float:
    """ calculates polarizability of sphere with radiation term
    """
    return Clausius_Mosotti_relation(radius, part_eps, med_eps) / (1 - (2 / 3) * np.power(k, 3) * Clausius_Mosotti_relation(radius, part_eps, med_eps))


# print(geometrical_factor(3, 4, 5) + geometrical_factor(3, 4, 5, along='a2') + geometrical_factor(3, 4, 5, along='a3'))

# print(ellipsoid_polarizability(3, 4, 5, med_eps=1.5, part_eps=1.3))
