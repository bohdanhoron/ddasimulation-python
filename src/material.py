import numpy as np

class Material():
    """ A class used to represent material optical behavior"""

    def __init__(self, frequencies: list, epsilons: list, *, name=None):
        self.frequencies = np.array(frequencies)
        self.epsilons = np.array(epsilons)
        self.name = name
    
    def values(self):
        return zip(self.frequencies, self.epsilons)

    def init_from_file(self, filepath: str):
        pass

    def get_pair(self):
        for pair in zip(self.frequencies, self.epsilons):
            yield pair

    def __str__(self) -> str:
        return f"it is {self.name}"

if __name__=="__main__":
    freq =  [10, 15, 20]
    eps = [1, 2, 3]

    iron = Material(freq, eps, name="iron")

    print(iron)

    for pair in iron.values():
        print(pair)