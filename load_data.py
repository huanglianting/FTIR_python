import numpy as np

def load_data(data):
    if 'AB' in data:
        AB = data['AB'][:, 1:]
        x = data['AB'][:, 0]
    else:
        TR = data['TR'][:, 1:]
        x = data['TR'][:, 0]
        AB = -np.log10(TR)

    return x, AB
