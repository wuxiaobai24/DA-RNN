import torch
from torch import nn
from torch.autograd import Variable
from torch import optim

class InputAttnEncoder(nn.Module):

    def __init__(self, n_feat, n_hidden, T):
        super(InputAttnEncoder, self).__init__()
        self.n_feat = n_feat
        self.n_hidden = n_hidden
        self.T = T

        self.lstm = nn.LSTMCell(n_feat, n_hidden)
        self.attn1 = nn.Linear(2 * n_hidden + n_feat, T + 1)
        self.attn2 = nn.Linear(T + 1, n_feat)

    def forward(self, X):
        # X: [n_batch, T, n_feat]
        h = torch.zeros([X.size(0), self.n_hidden]).to(X.device)
        c = torch.zeros([X.size(0), self.n_hidden]).to(X.device)
        hs, cs = [], []
        for i in range(X.size(1)):
            xi = torch.cat([X[:, i, :], h, c], dim=1)
            xi = torch.tanh(self.attn1(xi))
            xi = self.attn2(xi)
            xi = xi * X[:, i, :]
            h, c = self.lstm(xi, (h, c))
            hs.append(h)
            cs.append(c)
        # [n_batch, T, n_hidden]
        return torch.stack(hs).permute(1, 0, 2), torch.stack(cs).permute(1, 0, 2)

class TemporalAttenDecoder(nn.Module):

    def __init__(self, n_feat, n_target, n_hidden, T):
        super(TemporalAttenDecoder, self).__init__()

        self.n_feat = n_feat
        self.n_hidden = n_hidden
        self.n_target = n_target
        self.T = T

        self.lstm = nn.LSTMCell(n_target, n_hidden)
        self.attn1 = nn.Linear(n_hidden * 2 + n_feat, n_feat)
        self.attn2 = nn.Linear(n_feat, T + 1)
        self.comb_fc = nn.Linear(n_feat + n_target, n_target)
        self.fc = nn.Linear(n_hidden + n_feat, n_target)

    def forward(self, feat, target):
        # feat [B, T + 1, n_feat]
        # target [B, T, n_target]
        h = torch.zeros([feat.size(0), self.n_hidden]).to(feat.device)
        c = torch.zeros([feat.size(0), self.n_hidden]).to(feat.device)
        for i in range(self.T):
            xi = torch.cat([h, c, feat[:, i, :]], dim=1)
            xi = torch.tanh(self.attn1(xi))
            xi = torch.softmax(self.attn2(xi), 1)
            feat_i = (xi.view(xi.size(0), xi.size(1), 1) * feat).sum(1)
            xi = torch.cat([target[:, i].view(-1, 1), feat_i], 1)
            xi = self.comb_fc(xi)
            h, c = self.lstm(xi, (h, c))
            # hs.append(h)
            # cs.append(c)
        xt = torch.cat([h, feat_i], dim=1)
        out = self.fc(xt)
        return out



class DARNN(nn.Module):

    def __init__(self, n_feat, n_target, n_encoder_hidden, n_decoder_hidden, T):
        super(DARNN, self).__init__()
        self.n_feat = n_feat
        self.n_target = n_target
        self.n_encoder_hidden = n_encoder_hidden
        self.n_decoder_hidden = n_decoder_hidden
        self.T = T
        self.encoder = InputAttnEncoder(n_feat, n_encoder_hidden, T)
        self.decoder = TemporalAttenDecoder(n_encoder_hidden, n_target, n_decoder_hidden, T)

    def forward(self, feat, target):
        out, cell = self.encoder(feat)
        return self.decoder(out, target)
