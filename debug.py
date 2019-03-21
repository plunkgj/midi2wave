"""
Gary Plunkett, March 2019
Simple debugging functions for midi wavenet
"""

import torch
import torch.nn.functional as F
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
    print(name + " max: " + str(torch.max(X)))
    print(name + " min: " + str(torch.min(X)))
    print(name + " mean: " + str(torch.mean(x)))
    

def AnalyzeMidiSignal(act_data, signal_writer):
    in_acts = act_data[0].detach().cpu()
    cond_acts = act_data[1].detach().cpu()

    cosim = torch.mean(F.cosine_similarity(in_acts, cond_acts, dim=2))
    pairwise_distance = torch.mean(F.pairwise_distance(in_acts, cond_acts))
    in_act_mag = torch.mean(torch.pow(torch.sum(torch.pow(in_acts, 2), dim=2), 0.5))
    cond_act_mag = torch.mean(torch.pow(torch.sum(torch.pow(cond_acts, 2), dim=2), 0.5))

    av_cond_act = torch.mean(cond_acts, dim=3, keepdim=True)
    cond_act_dev = torch.mean(F.cosine_similarity(cond_acts, av_cond_act))

    return (cosim, pairwise_distance, in_act_mag, cond_act_mag)
    
