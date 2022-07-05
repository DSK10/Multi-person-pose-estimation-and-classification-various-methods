from torch import nn


class ModelTrain(nn.Module):
    def __init__(self, inp=99, output=1, hidden=100):
        super(ModelTrain, self).__init__()
        self.network = nn.Sequential(
            self.block(inp,hidden*2),
            self.block(hidden*2,hidden),
            self.block(hidden,output)

        )

    def block(self, input,output):
        return nn.Sequential(
            nn.Linear(input,output),
            nn.Tanh(),
        )

    def forward(self,x):
        return self.network(x)