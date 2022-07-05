import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loss_array import loss_train, loss_eval
from scipy import interpolate

def plot_loss(epoch, loss_arr, e_loss_arr):
    axis = np.linspace(1, epoch, epoch)
    x = np.arange(0, len(e_loss_arr)) * ((epoch-1) / (len(e_loss_arr)-1))
    f = interpolate.interp1d(x, e_loss_arr)
    x = np.arange(0, epoch)
    e_loss_arr = f(x)
    label = 'Loss'
    fig = plt.figure()
    plt.title(label)
    plt.plot(axis, loss_arr, label='train loss')
    plt.plot(axis, e_loss_arr, label='valid loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    plot_loss(len(loss_train), loss_train, loss_eval)

