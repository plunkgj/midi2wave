"""
Gary Plunkett, March 2019
Simple debugging functions for midi wavenet
"""

import torch

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def VerifyTrainData(x, y, y_pred, iteration):

    x = x[0]
    y = y[0]
    y_pred = y_pred[0]
    
    print(x.size())
    print(y.size())
    print(y_pred.size())

    print("Nonzero of midi input:")
    print(x.nonzero().size())
    
    audio = mu_law_decode(y)
    print(torch.max(audio))
    audio = audio.cpu().numpy()
    print(audio.shape)
    write("verifyData/y_" + str(iteration) + ".wav", 16000, audio)

    plt.cla()    
    #fig, ax = plt.subplots()
    plt.imshow(x.cpu().numpy(), aspect="auto", origin="lower")
    plt.savefig("verifyData/midi_" + str(iteration) + ".png")

    plt.cla()
    plt.imshow(y_pred.detach().cpu().numpy(), cmap="inferno", interpolation="nearest", aspect="auto", origin="lower")
    plt.savefig("verifyData/y_pred_" + str(iteration) + ".png")
    
def plot_probs(probs):
    plt.cla()
    plt.imshow(logits, cmap="inferno", interpolation="nearest", aspect="auto", origin="lower")
    plt.savefig("verifyData/logit_probs.png")

    
def tprobe(X, name):
    print ()
    print("probing " + name)
    print("X max: " + str(torch.max(X)))
    print("X min: " + str(torch.min(X)))
