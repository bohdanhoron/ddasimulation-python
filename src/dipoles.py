import numpy as np
from material import Material

class Dipole():
    def __init__(self, coordinates: list, radius: float):
        if np.ndim(coordinates) != 1:
            raise ValueError("array of coordinates is not vector")

        if len(coordinates) != 3:
            raise ValueError("array of coordinates is not three-dimensional")
        
        self.coordinates = np.array(coordinates)
        self.x, self.y, self.z = coordinates
        self.radius = radius

    def __str__(self) -> str:
        return f"({self.x}, {self.y}, {self.z}), r = {self.radius}"

class Dipoles():

    def __init__(self, dipoles: list):
        self.dipoles = np.array(dipoles)

    def __getitem__(self, index):
        return self.dipoles[index]

    def __len__(self) -> int:
        return len(self.dipoles)

    def __str__(self) -> str:
        return "it is Dipoles object"


if __name__=="__main__":

    dip1 = Dipole([0, 0, 0], 10)
    dip2 = Dipole([15, 15, 15], 10)

    dips = Dipoles([dip1, dip2])
    
    for dip in dips:
        print(dip)