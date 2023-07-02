import pandas as pd

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import csv
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from data_prep import get_paths_and_labels

torch.manual_seed(1)


class CobreDataset(Dataset):
    def __init__(self, data, labels):
        self.x = torch.from_numpy(data)
        self.y = torch.from_numpy(labels)
        self.n_samples = labels.shape[0]
        self.targets = labels

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples


class NeuralNetwork(nn.Module):

    def __init__(self, input_shape):
        super().__init__()
        self.lin1 = nn.Linear(input_shape, 64)
        self.dropout1 = nn.Dropout(0.4)
        self.lin2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.4)
        self.lin4 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.tanh(self.lin1(x))
        x = self.dropout1(x)
        x = torch.relu(self.lin2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.lin4(x))
        return x


def save_model(model):
    now = datetime.now()
    time_str = now.strftime("%d_%m_%Y_%H%M%S")
    path = "./model_" + time_str + ".pth"
    torch.save(model.state_dict(), path)


def training():
    losses = []
    val_losses = []
    accuracies = []
    val_acc = []
    for epoch in range(num_epochs):
        print('EPOCH:', epoch + 1)
        model.train(True)
        running_loss = 0.0
        len_data = 0
        correct = 0
        for i, (inputs, labels) in enumerate(train_batches):

            outputs = model(inputs.cuda())
            loss = loss_f(outputs, labels.cuda().unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for o, l in zip(outputs, labels):
                if o >= 0.5:
                    b_o = 1.0
                else:
                    b_o = 0.0
                if b_o == l:
                    correct += 1
            running_loss += loss.item()
            len_data += batch_size
        accuracy = correct / len_data
        avg_loss = running_loss / len_data
        print(avg_loss)
        print("Accuracy:", accuracy)
        accuracies.append(accuracy)
        losses.append(avg_loss)
        model.train(False)


        valid_loss = 0.0
        len_val_data = 0
        correct = 0
        for i, (inputs, labels) in enumerate(val_batches):
            outputs = model(inputs.cuda())
            loss = loss_f(outputs, labels.cuda().unsqueeze(1))
            optimizer.zero_grad()
            for o, l in zip(outputs, labels):
                if o >= 0.5:
                    b_o = 1.0
                else:
                    b_o = 0.0
                if b_o == l:
                    correct += 1
            valid_loss += loss.item()
            len_val_data += batch_size
        accuracy = correct / len_data
        print(valid_loss / len_val_data)
        print("Val accuracy:", accuracy)
        val_acc.append(accuracy)
        val_losses.append(valid_loss / len_val_data)

    save_model(model)
    plt.plot(losses, label='train loss')
    plt.plot(val_losses, label='validation loss')
    plt.legend()
    plt.show()

    plt.plot(accuracies, label='train accuracy')
    plt.plot(val_acc, label='validation accuracy')
    plt.legend()
    plt.show()
    return model


if __name__ == '__main__':

    source_path = '../COBRE_fmri/'
    csv_path = pd.read_csv(source_path + 'cobre_model_group.csv')
    filepaths, binary_class = get_paths_and_labels(csv_path, source_path)
    csvFC_path = 'nn_FCs.csv'
    corr_mats_in_line = []
    with open(csvFC_path, 'r') as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            corr_mat = np.array(row[0].strip('][').split(', ')).reshape(20, -1)
            corr_mats_in_line.append(np.array(row[0].strip('][').split(', ')).astype(np.float32))
    dataset = CobreDataset(np.array(corr_mats_in_line, dtype=np.float32), np.array(binary_class, dtype=np.float32))
    train_indices, test_indices, _, _ = train_test_split(range(len(dataset)),
                                                         dataset.targets,
                                                         stratify=dataset.targets,
                                                         test_size=0.2,
                                                         random_state=16)

    batch_size = 5
    num_epochs = 30

    train_split, val_split, test_split = random_split(dataset, (0.65, 0.15, 0.2))
    train_batches = DataLoader(dataset=train_split, batch_size=batch_size, shuffle=True)
    val_batches = DataLoader(dataset=val_split, batch_size=batch_size)
    test_batches = DataLoader(dataset=test_split, batch_size=batch_size)

    device = torch.device('cpu')
    if torch.cuda.is_available():
        print('Cuda is available')
        device = torch.device('cuda')

    model = NeuralNetwork(input_shape=400)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_f = nn.BCELoss()

    fetch_saved = True  # put True for testing a model saved in the path variable, False for training and testing a new model

    if fetch_saved:
        path = "NN_model_from_paper.pth"
        print("Testing with model", path)
        model.load_state_dict(torch.load(path))
        model.to(device)
    else:
        training()

    with torch.no_grad():
        correct_hc, correct_sz, false_hc, false_sz = 0, 0, 0, 0
        running_accuracy = 0
        total = 0
        for i, (inputs, labels) in enumerate(test_batches):
            predicted = model(inputs.cuda())
            for p, lab in zip(predicted, labels):
                if p.item() >= 0.5:
                    binary_pred = 1.0
                else:
                    binary_pred = 0.0
                if lab == binary_pred:
                    running_accuracy += 1
                    if binary_pred == 0.0:
                        correct_hc += 1
                    else:
                        correct_sz += 1
                else:
                    if binary_pred == 0.0:
                        false_hc += 1
                    else:
                        false_sz += 1
                total += 1
    print("Accuracy:", running_accuracy/total)
    print("Correct HC:", correct_hc)
    print("Correct SZ:", correct_sz)
    print("False HC:", false_hc)
    print("False SZ:", false_sz)
