from torch import nn


class Dense(nn.Module):

    def __init__(self, in_features, out_features, bias=True, dropout=0.1, nonlinearity=None):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.nonlinearity = getattr(nn, nonlinearity)() if nonlinearity else None
        self.reset_parameters()

    def forward(self, x):
        x = self.dropout(self.fc(x))
        if self.nonlinearity is not None:
            x = self.nonlinearity(x)
        return x

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc.weight)



