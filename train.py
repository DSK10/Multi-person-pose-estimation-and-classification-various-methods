import torch
from torch import nn
# from tqdm.auto import tqdm
import pandas  as pd
import numpy as np
from Custommodel import ModelTrain


df = pd.read_csv('data.csv',header=None)
target = df[99].values
target1 = torch.tensor(target, dtype=torch.float64)
df = df.drop(99,axis=1)
dataX = torch.tensor(np.array(df))



model = ModelTrain(inp=99,output=1)
# model.load_state_dict(torch.load('weight.pth'))

criterion = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(),lr=0.001)

for i in range(100):
    for data,label in zip(dataX,target1):
        data = data.to(torch.float32)
        label = label.to(torch.float32)
        opt.zero_grad()
        out = model(data)
        loss = criterion(out,label)
        print("loss :",loss.item())
        loss.backward()
        opt.step()

torch.save(model.state_dict(),"weight2.pth")
