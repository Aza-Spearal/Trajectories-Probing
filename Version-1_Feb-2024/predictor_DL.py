from taker import Model
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import torchmetrics as metrics

import random

import torch.nn.functional as F

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LabeledTokenizer:
    def __init__(self, model):
        self.model: Model = model
        self.map = dict()
        self.cpt = dict()
        self.reverse_map = dict()
        self.num_labels = 0

    def get_label_id(self, label, n_tokens):
        # add label to map if not already present
        if label not in self.map:
            self.map[label] = self.num_labels
            self.reverse_map[self.num_labels] = label
            self.num_labels += 1
            self.cpt[label] = n_tokens
        else:
            self.cpt[label] += n_tokens
        return self.map[label]

    def tokenize(self, data, label_y):

        input_list = []
        label_list = []

        for prompt in range(len(data)):

            if data[prompt]['split_text'] is not None:

                for index in range(len(data[prompt]['split_text'])):

                    text = data[prompt]['split_text'][index]['text']
                    label = data[prompt]['split_text'][index][label_y]

                    if model_name=="roberta-large":
                        if len(text) < 810:
                            residual = m.get_residual_stream(text)
                            n_tokens = residual.shape[1]

                            if n_tokens<token_max:

                                label_id = self.get_label_id(label, n_tokens)
                                label_idx = torch.full((n_tokens,), label_id)
                                input_list.append(residual)

                                label_list.append(label_idx)
                    else:

                        residual = m.get_residual_stream(text)
                        n_tokens = residual.shape[1]

                        if n_tokens<token_max:

                            label_id = self.get_label_id(label, n_tokens)
                            label_idx = torch.full((n_tokens,), label_id)
                            input_list.append(residual)

                            label_list.append(label_idx)

        input_torch = torch.cat(input_list, dim=1)
        label_torch = torch.cat(label_list, dim=0)
        print(self.cpt)
        return input_torch, label_torch

class Net(L.LightningModule):
    def __init__(self, inp_shape, n_classes):
        super().__init__()
        interm_layer = 512
        pdrop = 0.2
        self.linear1 = torch.nn.Linear(inp_shape, interm_layer)
        self.norm1 = nn.BatchNorm1d(interm_layer)
        self.dropout1 = nn.Dropout(pdrop)

        self.linear2 = torch.nn.Linear(interm_layer, interm_layer)
        self.norm2 = nn.BatchNorm1d(interm_layer)
        self.dropout2 = nn.Dropout(pdrop)

        self.linear3 = torch.nn.Linear(interm_layer, n_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.dropout1(self.relu(self.norm1(self.linear1(x))))
        x = self.dropout2(self.relu(self.norm2(self.linear2(x))))
        x = self.linear3(x) 
        return x
    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = criterion(output, target)
        return loss
    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs)
        loss = criterion(output, target)
        a = torch.argmax(output, dim=1)
        b = torch.argmax(target, dim=1)
        acc = accuracy(a, b).item()
        macc = macro_accuracy(a, b).item()
        fscore = f1(a, b).item()
        acclass = metrics.functional.accuracy(a,b, task='multiclass', num_classes=n_classes, average='none').to(device)
        self.log_dict({'F1-Score': fscore, 'Macro-Accuracy': macc, 'Accuracy': acc, 'CrossEntropy_loss': loss, 'class0': acclass[0], 'class0': acclass[0], 'class1': acclass[1], 'class2': acclass[2], 'class3': acclass[3], 'class4': acclass[4], 'class5': acclass[5]})
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())


model_name = 'hello' #for roberta tokenizer, don't touch it

#Comment 2 of these lines:
m = Model("nickypro/tinyllama-15M")
#model_name = "roberta-large"; m = Model(model_name)
#m = Model('mistralai/Mistral-7B-Instruct-v0.2', dtype='int8')


labeled_tokenizer = LabeledTokenizer(m)

data_files=['mistral1.json', 'mistral2.json', 'mistral4.json']
data = list()
for fil in data_files:
    with open(fil, 'r') as infile:
        data.extend(json.load(infile))


random.shuffle(data)
data = data[int(len(data)*0.5):]
idx = int(len(data)*0.5)
train_data = data[:idx]
test_data = data[idx:]

label_y = 'genre'

token_max = 512


x_train, y_train = labeled_tokenizer.tokenize(train_data, label_y)
x_test, y_test = labeled_tokenizer.tokenize(test_data, label_y)

n_classes = torch.unique(y_train).numel()

y_test = F.one_hot(y_test, num_classes = n_classes).to(device).to(torch.float)
y_train = F.one_hot(y_train, num_classes = n_classes).to(device).to(torch.float)


criterion = nn.CrossEntropyLoss().to(device)
f1 = metrics.F1Score(task="multiclass", num_classes=n_classes, average = 'macro').to(device)
accuracy = metrics.Accuracy(task="multiclass", num_classes=n_classes, average='micro').to(device)
macro_accuracy = metrics.Accuracy(task="multiclass", num_classes=n_classes, average='macro').to(device)


save_train = x_train
save_test = x_test


for layer in range(m.cfg.n_layers*2 + 1):

    x_train = save_train[layer].to(device).to(torch.float)
    x_test = save_test[layer].to(device).to(torch.float)

    train_dataset = TensorDataset(x_train, y_train)
    test_dataset = TensorDataset(x_test, y_test)

    print('train dataset size:', len(train_dataset))
    print('test dataset size:', len(test_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))

    net = Net(m.cfg.d_model, n_classes).to(device)

    early_stop_callback = EarlyStopping(monitor="CrossEntropy_loss", mode="min", patience=3, verbose=False)
    trainer = L.Trainer(callbacks=[early_stop_callback], accelerator="auto", devices = 1, max_epochs=-1)

    trainer.fit(net, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    o = trainer.validate(net, dataloaders=val_dataloader, verbose=False)

    print('Inferring', label_y, ', Layer', layer)
    print('F1-Score ' + str(round(o[0]['F1-Score'], 2)) + ' Macro-Accuracy ' + str(round(o[0]['Macro-Accuracy'], 2)) + ' Accuracy ' + str(round(o[0]['Accuracy'], 2)) + ' CE ' + str(round(o[0]['CrossEntropy_loss'], 2)))
    print('class0 ' + str(round(o[0]['class0'], 2)) + ' class1 ' + str(round(o[0]['class1'], 2)) + ' class2 ' + str(round(o[0]['class2'], 2)) + ' class3 ' + str(round(o[0]['class3'], 2)) + ' class4 ' + str(round(o[0]['class4'], 2)) + ' class5 ' + str(round(o[0]['class5'], 2)))
