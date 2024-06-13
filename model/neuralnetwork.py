import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset, DataLoader,TensorDataset


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer_1 = nn.Linear(input_size, hidden_size1)
        nn.init.kaiming_uniform_(self.layer_1.weight, nonlinearity="relu")
        self.layer_2 = nn.Linear(hidden_size1, hidden_size2)
        nn.init.kaiming_uniform_(self.layer_2.weight, nonlinearity="relu")
        self.layer_3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x
    
# Convert data to torch tensors
class Data(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len

#performs model evaluation
def evaluate(model, dataloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100 * (correct / total)


# Train the model
def train_model(model,optimizer,train_dataloader,loss_fn):
    for epoch in range(100):
        for X, y in train_dataloader:
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y.long())
            loss.backward()
            optimizer.step()

 # Convert data to DataLoader format
def data_format(X_train_split,y_train_split,X_val_split,y_val_split):
    train_data = Data(X_train_split, y_train_split)
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
    val_data = Data(X_val_split, y_val_split)
    val_loader = DataLoader(dataset=val_data, batch_size=64, shuffle=True)
    return train_loader,val_loader

