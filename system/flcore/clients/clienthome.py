import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *
from torch.utils.data import DataLoader
from imblearn.over_sampling import SMOTE


class clientHome(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.opt_pred = torch.optim.SGD(self.model.predictor.parameters(), lr=self.learning_rate)
        self.trainloader_rep = None

        # differential privacy
        if self.privacy:
            check_dp(self.model)
            initialize_dp(self.model, self.optimizer, self.sample_rate, self.dp_sigma)

    def train(self):
        trainloader = self.load_train_data()
        
        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        max_local_steps = self.local_steps
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.optimizer.zero_grad()
                output = self.model(x)
                loss = self.loss(output, y)
                loss.backward()
                if self.privacy:
                    dp_step(self.optimizer, i, len(trainloader))
                else:
                    self.optimizer.step()

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        if self.privacy:
            res, DELTA = get_dp_params(self.optimizer)
            print(f"Client {self.id}", f"(ε = {res[0]:.2f}, δ = {DELTA}) for α = {res[1]}")

    def train_pred(self):
        self.model.train()
        for i, (x, y) in enumerate(self.trainloader_rep):
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            self.opt_pred.zero_grad()
            output = self.model.predictor(x)
            loss = self.loss(output, y)
            loss.backward()
            self.opt_pred.step()

    def generate_data(self):
        train_data_rep = []
        train_data_y = []
        trainloader = self.load_train_data()
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                train_data_rep.append(self.model.base(x).detach().cpu().numpy())
                train_data_y.append(y.detach().cpu().numpy())
        train_data_rep = np.concatenate(train_data_rep, axis=0)
        train_data_y = np.concatenate(train_data_y, axis=0)
        if len(np.unique(train_data_y)) > 1:
            smote = SMOTE()
            X, Y = smote.fit_resample(train_data_rep, train_data_y)
        else:
            X, Y = train_data_rep, train_data_y
        print(f'Client {self.id} data ratio: ', '{:.2f}%'.format(100*(len(Y))/len(train_data_y)))
        X_train = torch.Tensor(X).type(torch.float32)
        y_train = torch.Tensor(Y).type(torch.int64)
        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        self.trainloader_rep = DataLoader(train_data, self.batch_size, drop_last=True, shuffle=False)
