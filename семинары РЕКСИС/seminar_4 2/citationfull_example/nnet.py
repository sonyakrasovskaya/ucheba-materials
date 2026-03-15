import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score


class Sheaf_NNet(nn.Module):
    def __init__(self, nvert, dimx, dimy, nlab, nconv=3, nsmat=64):
        super(Sheaf_NNet, self).__init__()
        self.nvert = nvert
        self.nconv = nconv
        self.dimx = dimx
        self.dimy = dimy
        self.nlab = nlab
        self.cl_smat = nn.Sequential(nn.Linear(self.dimx, self.nlab),
                                     nn.LogSoftmax(dim=1))
        
        self.fc_smat = nn.Sequential(nn.Linear(self.dimx, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, nsmat),
                                     nn.ReLU(),
                                     nn.Linear(nsmat, self.dimy * self.dimx))


    def forward(self, xembed, ylabel, ylprob, wgraph, idvert):
        loss = 0.0
        smat = torch.reshape(self.fc_smat(xembed), (-1, self.dimy, self.dimx))
        tmat = torch.transpose(smat, 1, 2)
        xmaped = 0.0 + xembed
        for idx_conv in range(self.nconv):
            ymaped = torch.bmm(smat, torch.reshape(xmaped, (-1, self.dimx, 1)))
            qmaped = torch.tensordot(wgraph, ymaped, dims=([1], [0]))
            xmaped = torch.reshape(torch.bmm(tmat, qmaped), (-1, self.dimx))
            smat = torch.reshape(self.fc_smat(xmaped), (-1, self.dimy, self.dimx))
            tmat = torch.transpose(smat, 1, 2)

        loss_smap = torch.mean((xmaped - xembed) * (xmaped - xembed)) * self.dimx
        glprob = self.cl_smat(xmaped[idvert[:], :])
        kl_div = torch.sum(torch.exp(ylprob[idvert[:], :]) * (ylprob[idvert[:], :] - glprob[:, :]), 1)
        loss_lbpr = torch.mean(kl_div)
        # loss_lbpr = -torch.mean(self.cl_smat(xmaped[idvert[:], :])[:, ylabel[idvert[:]]])


        if self.dimy <= self.dimx:
            rmat = torch.torch.bmm(smat, torch.permute(smat, (0, 2, 1)))
            target_matrix = torch.zeros((rmat.shape[0], self.dimy, self.dimy))
            for idx in range(target_matrix.shape[0]):
                target_matrix[idx, :, :] = torch.eye(self.dimy)

        else:
            rmat = torch.torch.bmm(torch.permute(smat, (0, 2, 1)), smat)
            target_matrix = torch.zeros((rmat.shape[0], self.dimx, self.dimx))
            for idx in range(target_matrix.shape[0]):
                target_matrix[idx, :, :] = torch.eye(self.dimx)
        
        rmat = rmat - target_matrix
        loss_orth = torch.sqrt(torch.mean(rmat * rmat) * self.dimy * self.dimy)
        
        smat = torch.reshape(self.fc_smat(xembed), (-1, self.dimy, self.dimx))
        smat_proj = torch.reshape(self.fc_smat(xmaped), (-1, self.dimy, self.dimx))
        loss_cons = torch.mean((smat_proj - smat) * (smat_proj - smat)) * self.dimy * self.dimx

        yscore = self.cl_smat(xmaped[idvert[:], :]).cpu().detach().numpy()
        ynumer = np.zeros((yscore.shape[0]))
        for idx in range(ynumer.size):
            max_val = 0.0
            for idx_max in range(yscore.shape[1]):
                if max_val < np.exp(yscore[idx, idx_max]):
                    max_val = np.exp(yscore[idx, idx_max])
                    ynumer[idx] = idx_max
        loss_accs = accuracy_score(ylabel[idvert[:]], ynumer)
        return (loss_orth, loss_cons, loss_smap, loss_lbpr, loss_accs)






