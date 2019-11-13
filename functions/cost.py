def mse(y_pred, y_true):
    return (y_true - y_pred)**2

def loss_ann(Y, Y_pred):
    return sum([mse(y_pred, y_true) for y_pred, y_true in (Y, Y_pred)]) / len(Y)