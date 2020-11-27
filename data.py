import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class NasdaqDataset(Dataset):
    
    def __init__(self, path, T, type = 'train', target_name = 'NDX', scaler = None, target_scaler = None):
        super(NasdaqDataset, self).__init__()
        self.T = T
        self.scaler = scaler
        self.target_scaler = target_scaler
        self.type = type
        self.target_name = target_name

        if type == 'train':
            start, end =0, 35100
        elif type == 'val':
            start, end = 35100, 35100 + 2730
        else:
            start, end = 35100 + 2730, 35100 + 2730 * 2
        
        df = pd.read_csv(path)[start:end]
        driving = df.loc[:, [x for x in df.columns.tolist() if x != target_name]].to_numpy()
        target = np.array(df[target_name]).reshape(-1, 1)
        data = np.hstack([driving, target.reshape(-1, 1)])
        if self.scaler is None:
            # self.scaler = StandardScaler().fit(driving)
            self.scaler = StandardScaler().fit(data)
        if self.target_scaler is None:
            self.target_scaler = StandardScaler().fit(target)
        
        data = torch.Tensor(self.scaler.transform(data))
        # driving = torch.Tensor(self.scaler.transform(driving))
        # target = torch.Tensor(self.target_scaler.transform(target))
        
        self.X, self.y = [], []
        # for i in range(driving.shape[0] - T):
        #     self.X.append((driving[i:i+T+1, :], target[i:i+T]))
        #     self.y.append(target[i+T])

        for i in range(data.shape[0] - T):
            self.X.append((data[i:i+T+1, :-1], data[i:i+T, -1]))
            self.y.append(data[i+T, -1])


    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx], self.y[idx]