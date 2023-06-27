import numpy as np

def show_equiv_label(predictions): 
    shading= {
                    0: 'A',
                    1: 'B',
                    2: 'C', 
                    3: 'D', 
                    4: 'E',
                    5: 'Exception'
                }
    pred = np.apply_along_axis(lambda x: np.round(x), 1, predictions)
    for i in range(len(pred)):
        for j in range(len(pred[i])):
            if pred[i][j] == 1:
                return shading[j]