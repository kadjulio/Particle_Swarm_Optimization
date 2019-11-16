def mse(y_pred, y_true):
    return (y_true - y_pred)**2

def loss_ann(Y, Y_pred):
    loss = 0.0
    for y_pred, y_true in zip(Y_pred, Y):
        loss += mse(y_pred,y_true)
    return  loss / len(Y)