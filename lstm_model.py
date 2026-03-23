from torch import nn

class VolatilityLSTM(nn.Module):
    def __init__(self, hidden_size=64, dropout_rate=0.1, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=4, hidden_size=hidden_size, dropout=dropout_rate, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        output, (hn, cn) = self.lstm(x)
        lastTimestep = output[:, -1, :]
        return self.linear(self.dropout(lastTimestep))