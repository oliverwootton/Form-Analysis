import numpy as np

def str_to_int(y_data):
    cnn_y_data = []
    for word in y_data:
        if word == 'el-ex':
            cnn_y_data.append(1)
        elif word == 'misc1':
            cnn_y_data.append(1)
        elif word == 'misc2':
            cnn_y_data.append(2)
        elif word == 'misc3':
            cnn_y_data.append(3)
        elif word == 'misc4':
            cnn_y_data.append(4)
        elif word == 'misc5':
            cnn_y_data.append(5)
    return np.array(cnn_y_data)

def int_to_str(y_data):
    cnn_y_data = []
    for num in y_data:
        if num == 1 or num == 0:
            cnn_y_data.append('misc1')
        elif num == 2:
            cnn_y_data.append('misc2')
        elif num == 3:
            cnn_y_data.append('misc3')
        elif num == 4:
            cnn_y_data.append('misc4')
        elif num == 5:
            cnn_y_data.append('misc5')
    return cnn_y_data