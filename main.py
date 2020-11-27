import time
from torch import mode
from data import NasdaqDataset
from model import DARNN
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse



def MAE(pred, target):
    return (pred - target).abs().mean()


def MSE(pred, target):
    return torch.pow(pred - target, 2).mean()


def RMSE(pred, target):
    return MSE(pred, target).sqrt()


def MAPE(pred, target):
    return ((pred - target).abs() / (target.abs() + 1e-8)).mean()

parser = argparse.ArgumentParser(description='DA-RNN')
parser.add_argument('--path', type=str, default='./dataset/nasdaq100/small/nasdaq100_padding.csv')
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--encoder_hidden', type=int, default=128)
parser.add_argument('--decoder_hidden', type=int, default=128)
parser.add_argument('--timestep', type=int, default=10)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.01)

args = parser.parse_args()
print(args)
# constans parameters
N_FEATURE = 81
N_ENCODER_HIDDEN = args.encoder_hidden
N_DECODER_HIDDEN = args.decoder_hidden
N_TAGET = 1
T = args.timestep
BATCH_SIZE = args.batchsize
N_EPOCHS = args.epochs
SEED = 24
LEARNING_RATE = args.lr

# set seed

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


PATH = './nasdaq100_padding.csv'
train_data = NasdaqDataset(PATH, T, 'train')
val_data = NasdaqDataset(PATH, T, 'val', scaler=train_data.scaler, 
                        target_scaler=train_data.target_scaler)
test_data = NasdaqDataset(PATH, T, 'test', scaler=train_data.scaler,
                        target_scaler=train_data.target_scaler)

print("Train data's len is", len(train_data))
print("Val data's len is", len(val_data))
print("Test data's len is", len(test_data))

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)


# encoder = InputAttnEncoder(N_FEATURE, N_ENCODER_HIDDEN, T)
# decoder = TemporalAttenDecoder(N_ENCODER_HIDDEN, N_TAGET, N_DECODER_HIDDEN, T)
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
print('device is', device)

model = DARNN(N_FEATURE, N_TAGET, N_ENCODER_HIDDEN,
                N_DECODER_HIDDEN, T).to(device)

loss_func = torch.nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(1, N_EPOCHS + 1):
    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    start_time = time.time()
    train_losses = []
    for i, data in pbar:
        (feat, target), y = data
        if use_cuda:
            feat = feat.cuda()
            target = target.cuda()
            y = y.cuda()
        prepare_time = start_time - time.time()

        optimizer.zero_grad()
        pred = model(feat, target)
        loss = loss_func(pred.view(-1), y)
        train_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        process_time = start_time-time.time()-prepare_time
        pbar.set_description("Compute efficiency: {:.2f}, epoch: {}/{}".format(
            process_time/(process_time + prepare_time), epoch, N_EPOCHS + 1))
        start_time = time.time()

    if epoch % 1 == 0:
        model.eval()
        pbar = tqdm(enumerate(val_loader), total=len(val_loader))
        losses = {
            'MAE': [],
            'RMSE': [],
            'MAPE': []
        }
        for i, data in pbar:
            (feat, target), y = data
            if use_cuda:
                feat = feat.cuda()
                target = target.cuda()
                y = y.cuda()
            pred = model(feat, target)

            losses['MAE'].append(MAE(pred, y.view(-1)).item())
            losses['RMSE'].append(RMSE(pred, y.view(-1)).item())
            losses['MAPE'].append(MAPE(pred, y.view(-1)).item())
        
        print('Epoch {:d}: MAE = {:.2f}, RMSE = {:.2f}, MAPE = {:.2f}'.format(
            epoch, np.mean(losses['MAE']),
            np.mean(losses['RMSE']),
            np.mean(losses['MAPE'])))

pbar = tqdm(enumerate(test_loader), total=len(val_loader))
losses = {
    'MAE': [],
    'RMSE': [],
    'MAPE': []
}
for i, data in pbar:
    (feat, target), y = data
    if use_cuda:
        feat = feat.cuda()
        target = target.cuda()
        y = y.cuda()
    pred = model(feat, target)
    losses['MAE'].append(MAE(pred, y.view(-1)).item())
    losses['RMSE'].append(RMSE(pred, y.view(-1)).item())
    losses['MAPE'].append(MAPE(pred, y.view(-1)).item())
print('Test: MAE = {:.2f}, RMSE = {:.2f}, MAPE = {:.2f}'.format(
    np.mean(losses['MAE']),
    np.mean(losses['RMSE']),
    np.mean(losses['MAPE'])))
