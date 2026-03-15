import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable


def init_loss_target(loss_threshold):
    return (loss_threshold, loss_threshold, loss_threshold, loss_threshold)

def compute_loss_target(loss_orth, loss_cons, loss_smap, loss_lbpr, 
                        kapp_orth, kapp_cons, kapp_smap, kapp_lbpr,
                        kappa, dt, eps=1.0e-6):
    kapp_orth_updated = max(eps, kapp_orth + 0.0 * dt * ((0.0 + loss_orth).cpu().detach().numpy() - kapp_orth))
    kapp_cons_updated = max(eps, kapp_cons + kappa * dt * ((0.0 + loss_cons).cpu().detach().numpy() - kapp_cons))
    kapp_lbpr_updated = max(eps, kapp_lbpr + kappa * dt * ((0.0 + loss_lbpr).cpu().detach().numpy() - kapp_lbpr))
    kapp_smap_updated = max(eps, kapp_smap + kappa * dt * ((0.0 + loss_smap).cpu().detach().numpy() - kapp_smap))
    return (kapp_orth_updated,
            kapp_cons_updated,
            kapp_smap_updated,
            kapp_lbpr_updated)


def compute_loss_weights_simple(loss_orth, loss_cons, loss_smap, loss_lbpr, 
                                kapp_orth, kapp_cons, kapp_smap, kapp_lbpr,
                                eps=1.0e-6):
    w_orth = 1.0
    w_smap = 1.0
    w_cons = 1.0
    loss_orth_numpy = (0.0 + loss_orth).cpu().detach().numpy() / max(eps, kapp_orth)
    loss_cons_numpy = (0.0 + loss_cons).cpu().detach().numpy() / max(eps, kapp_cons)
    loss_smap_numpy = (0.0 + loss_smap).cpu().detach().numpy() / max(eps, kapp_smap)
    loss_lbpr_numpy = (0.0 + loss_lbpr).cpu().detach().numpy() / max(eps, kapp_lbpr)


    w_cons = np.exp(-loss_orth_numpy)
    w_lbpr = np.exp(-max(loss_cons_numpy, loss_orth_numpy))
    w_smap = np.exp(-max(loss_cons_numpy, loss_orth_numpy, loss_lbpr_numpy))

    w_summ = w_orth + w_smap + w_cons + w_lbpr
    w_orth = w_orth / w_summ
    w_cons = w_cons / w_summ
    w_smap = w_smap / w_summ
    w_lbpr = w_lbpr / w_summ
    return (w_orth, w_cons, w_smap, w_lbpr)



def compute_neural_network_parameters(nnet, nepoch, nbatch, lr,
                                      xembed_numpy, ylabel_numpy, ylprob_numpy, wgraph_numpy,
                                      idx_train, idx_ttest, idx_valid,
                                      loss_threshold=1.0e-2, kappa=1.0e1, eps=1.0e-6):
    print('compute_neural_network_parameters')
    nnet.fc_smat.requires_grad_(True)
    nnet.cl_smat.requires_grad_(True)
    optimizer = optim.Adam(nnet.parameters(), lr=lr, weight_decay=0.0)
    loss_data_train = np.zeros((nepoch + 1, 6))
    loss_data_ttest = np.zeros((nepoch + 1, 6))
    loss_data_valid = np.zeros((nepoch + 1, 6))
    kapp_orth, kapp_cons, kapp_smap, kapp_lbpr = init_loss_target(loss_threshold)

    wgraph = torch.from_numpy(wgraph_numpy).double()
    xembed = torch.from_numpy(xembed_numpy).double()
    ylprob = torch.from_numpy(ylprob_numpy).double()
    ylabel = ylabel_numpy.copy()


    for kepoch in range(nepoch):
        optimizer.zero_grad()
        loss_orth, loss_cons, loss_smap, loss_lbpr, loss_accs = nnet(Variable(xembed), ylabel, ylprob, wgraph, idx_valid)
        kapp_orth, kapp_cons, kapp_smap, kapp_lbpr = compute_loss_target(loss_orth, loss_cons, loss_smap, loss_lbpr,
                                                                         kapp_orth, kapp_cons, kapp_smap, kapp_lbpr,
                                                                         kappa, lr)

        w_orth, w_cons, w_smap, w_lbpr = compute_loss_weights_simple(loss_orth, loss_cons, loss_smap, loss_lbpr,
                                                                     kapp_orth, kapp_cons, kapp_smap, kapp_lbpr)

        loss = w_orth * loss_orth + w_cons * loss_cons + w_smap * loss_smap + w_lbpr * loss_lbpr
        loss_data_valid[kepoch, 0] = loss_orth.item() 
        loss_data_valid[kepoch, 1] = loss_cons.item()
        loss_data_valid[kepoch, 2] = loss_smap.item()
        loss_data_valid[kepoch, 3] = loss_lbpr.item()
        loss_data_valid[kepoch, 4] = loss.item()
        loss_data_valid[kepoch, 5] = loss_accs

        print('kepoch = ' + str(kepoch) + ' of ' + str(nepoch))
        print('kepoch = ' + str(kepoch) + ' w_orth_valid = ' + str(w_orth))
        print('kepoch = ' + str(kepoch) + ' w_cons_valid = ' + str(w_cons))
        print('kepoch = ' + str(kepoch) + ' w_smap_valid = ' + str(w_smap))
        print('kepoch = ' + str(kepoch) + ' w_lbpr_valid = ' + str(w_lbpr))

        print('kepoch = ' + str(kepoch) + ' loss_orth_valid.item() = ' + str(loss_data_valid[kepoch, 0]))
        print('kepoch = ' + str(kepoch) + ' loss_cons_valid.item() = ' + str(loss_data_valid[kepoch, 1]))
        print('kepoch = ' + str(kepoch) + ' loss_smap_valid.item() = ' + str(loss_data_valid[kepoch, 2]))
        print('kepoch = ' + str(kepoch) + ' loss_lbpr_valid.item() = ' + str(loss_data_valid[kepoch, 3]))
        print('kepoch = ' + str(kepoch) + ' loss_comb_valid.item() = ' + str(loss_data_valid[kepoch, 4]))
        print('kepoch = ' + str(kepoch) + ' loss_accs_valid = ' + str(loss_data_valid[kepoch, 5]))

        loss_orth, loss_cons, loss_smap, loss_lbpr, loss_accs = nnet(Variable(xembed), ylabel, ylprob, wgraph, idx_ttest)
        kapp_orth, kapp_cons, kapp_smap, kapp_lbpr = compute_loss_target(loss_orth, loss_cons, loss_smap, loss_lbpr,
                                                                         kapp_orth, kapp_cons, kapp_smap, kapp_lbpr,
                                                                         kappa, lr)

        w_orth, w_cons, w_smap, w_lbpr = compute_loss_weights_simple(loss_orth, loss_cons, loss_smap, loss_lbpr,
                                                                     kapp_orth, kapp_cons, kapp_smap, kapp_lbpr)

        loss = w_orth * loss_orth + w_cons * loss_cons + w_smap * loss_smap + w_lbpr * loss_lbpr
        loss_data_ttest[kepoch, 0] = loss_orth.item() 
        loss_data_ttest[kepoch, 1] = loss_cons.item()
        loss_data_ttest[kepoch, 2] = loss_smap.item()
        loss_data_ttest[kepoch, 3] = loss_lbpr.item()
        loss_data_ttest[kepoch, 4] = loss.item()
        loss_data_ttest[kepoch, 5] = loss_accs

        print('kepoch = ' + str(kepoch) + ' of ' + str(nepoch))
        print('kepoch = ' + str(kepoch) + ' w_orth_ttest = ' + str(w_orth))
        print('kepoch = ' + str(kepoch) + ' w_cons_ttest = ' + str(w_cons))
        print('kepoch = ' + str(kepoch) + ' w_smap_ttest = ' + str(w_smap))
        print('kepoch = ' + str(kepoch) + ' w_lbpr_ttest = ' + str(w_lbpr))

        print('kepoch = ' + str(kepoch) + ' loss_orth_ttest.item() = ' + str(loss_data_ttest[kepoch, 0]))
        print('kepoch = ' + str(kepoch) + ' loss_cons_ttest.item() = ' + str(loss_data_ttest[kepoch, 1]))
        print('kepoch = ' + str(kepoch) + ' loss_smap_ttest.item() = ' + str(loss_data_ttest[kepoch, 2]))
        print('kepoch = ' + str(kepoch) + ' loss_lbpr_ttest.item() = ' + str(loss_data_ttest[kepoch, 3]))
        print('kepoch = ' + str(kepoch) + ' loss_comb_ttest.item() = ' + str(loss_data_ttest[kepoch, 4]))
        print('kepoch = ' + str(kepoch) + ' loss_accs_ttest = ' + str(loss_data_ttest[kepoch, 5]))

        loss_orth, loss_cons, loss_smap, loss_lbpr, loss_accs = nnet(Variable(xembed), ylabel, ylprob, wgraph, idx_train)
        kapp_orth, kapp_cons, kapp_smap, kapp_lbpr = compute_loss_target(loss_orth, loss_cons, loss_smap, loss_lbpr,
                                                                         kapp_orth, kapp_cons, kapp_smap, kapp_lbpr,
                                                                         kappa, lr)

        w_orth, w_cons, w_smap, w_lbpr = compute_loss_weights_simple(loss_orth, loss_cons, loss_smap, loss_lbpr,
                                                                     kapp_orth, kapp_cons, kapp_smap, kapp_lbpr)

        loss = w_orth * loss_orth + w_cons * loss_cons + w_smap * loss_smap + w_lbpr * loss_lbpr
        loss_data_train[kepoch, 0] = loss_orth.item() 
        loss_data_train[kepoch, 1] = loss_cons.item()
        loss_data_train[kepoch, 2] = loss_smap.item()
        loss_data_train[kepoch, 3] = loss_lbpr.item()
        loss_data_train[kepoch, 4] = loss.item()
        loss_data_train[kepoch, 5] = loss_accs

        print('kepoch = ' + str(kepoch) + ' of ' + str(nepoch))
        print('kepoch = ' + str(kepoch) + ' w_orth_train = ' + str(w_orth))
        print('kepoch = ' + str(kepoch) + ' w_cons_train = ' + str(w_cons))
        print('kepoch = ' + str(kepoch) + ' w_smap_train = ' + str(w_smap))
        print('kepoch = ' + str(kepoch) + ' w_lbpr_train = ' + str(w_lbpr))

        print('kepoch = ' + str(kepoch) + ' loss_orth_train.item() = ' + str(loss_data_train[kepoch, 0]))
        print('kepoch = ' + str(kepoch) + ' loss_cons_train.item() = ' + str(loss_data_train[kepoch, 1]))
        print('kepoch = ' + str(kepoch) + ' loss_smap_train.item() = ' + str(loss_data_train[kepoch, 2]))
        print('kepoch = ' + str(kepoch) + ' loss_lbpr_train.item() = ' + str(loss_data_train[kepoch, 3]))
        print('kepoch = ' + str(kepoch) + ' loss_comb_train.item() = ' + str(loss_data_train[kepoch, 4]))
        print('kepoch = ' + str(kepoch) + ' loss_accs_train = ' + str(loss_data_train[kepoch, 5]))

        loss.backward()
        optimizer.step()


    loss_orth, loss_cons, loss_smap, loss_lbpr, loss_accs = nnet(Variable(xembed), ylabel, ylprob, wgraph, idx_valid)
    kapp_orth, kapp_cons, kapp_smap, kapp_lbpr = compute_loss_target(loss_orth, loss_cons, loss_smap, loss_lbpr, 
                                                                     kapp_orth, kapp_cons, kapp_smap, kapp_lbpr,
                                                                     kappa, lr)

    w_orth, w_cons, w_smap, w_lbpr = compute_loss_weights_simple(loss_orth, loss_cons, loss_smap, loss_lbpr,
                                                                 kapp_orth, kapp_cons, kapp_smap, kapp_lbpr)
    loss = w_orth * loss_orth + w_cons * loss_cons + w_smap * loss_smap + w_lbpr * loss_lbpr
    loss_data_valid[kepoch, 0] = loss_orth.item() 
    loss_data_valid[kepoch, 1] = loss_cons.item()
    loss_data_valid[kepoch, 2] = loss_smap.item()
    loss_data_valid[kepoch, 3] = loss_lbpr.item()
    loss_data_valid[kepoch, 4] = loss.item()
    loss_data_valid[kepoch, 5] = loss_accs

    loss_orth, loss_cons, loss_smap, loss_lbpr, loss_accs = nnet(Variable(xembed), ylabel, ylprob, wgraph, idx_ttest)
    kapp_orth, kapp_cons, kapp_smap, kapp_lbpr = compute_loss_target(loss_orth, loss_cons, loss_smap, loss_lbpr, 
                                                                     kapp_orth, kapp_cons, kapp_smap, kapp_lbpr,
                                                                     kappa, lr)

    w_orth, w_cons, w_smap, w_lbpr = compute_loss_weights_simple(loss_orth, loss_cons, loss_smap, loss_lbpr,
                                                                 kapp_orth, kapp_cons, kapp_smap, kapp_lbpr)
    loss = w_orth * loss_orth + w_cons * loss_cons + w_smap * loss_smap + w_lbpr * loss_lbpr
    loss_data_ttest[kepoch, 0] = loss_orth.item() 
    loss_data_ttest[kepoch, 1] = loss_cons.item()
    loss_data_ttest[kepoch, 2] = loss_smap.item()
    loss_data_ttest[kepoch, 3] = loss_lbpr.item()
    loss_data_ttest[kepoch, 4] = loss.item()
    loss_data_ttest[kepoch, 5] = loss_accs

    loss_orth, loss_cons, loss_smap, loss_lbpr, loss_accs = nnet(Variable(xembed), ylabel, ylprob, wgraph, idx_train)
    kapp_orth, kapp_cons, kapp_smap, kapp_lbpr = compute_loss_target(loss_orth, loss_cons, loss_smap, loss_lbpr, 
                                                                     kapp_orth, kapp_cons, kapp_smap, kapp_lbpr,
                                                                     kappa, lr)

    w_orth, w_cons, w_smap, w_lbpr = compute_loss_weights_simple(loss_orth, loss_cons, loss_smap, loss_lbpr,
                                                                 kapp_orth, kapp_cons, kapp_smap, kapp_lbpr)
    loss = w_orth * loss_orth + w_cons * loss_cons + w_smap * loss_smap + w_lbpr * loss_lbpr
    loss_data_train[kepoch, 0] = loss_orth.item() 
    loss_data_train[kepoch, 1] = loss_cons.item()
    loss_data_train[kepoch, 2] = loss_smap.item()
    loss_data_train[kepoch, 3] = loss_lbpr.item()
    loss_data_train[kepoch, 4] = loss.item()
    loss_data_train[kepoch, 5] = loss_accs

    return (loss_data_train, loss_data_ttest, loss_data_valid)




















