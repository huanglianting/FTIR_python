import numpy as np

def load_data(data):
    Sc = data['Sc']
    S0 = data['S0']
    y_bg = S0[:, 1][4:-2]
    x = Sc[:, 0]
    return x, y_bg, Sc
