import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *


class clientPrune(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        self.mask = [torch.ones_like(p) for p in self.model.parameters()]
        self.topk = args.topk

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
                self.pruneGrad()
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
        
    def set_parameters(self, model):
        for new_param, old_param, m in zip(model.parameters(), self.model.parameters(), self.mask):
            old_param.data = new_param.data.clone() * m

    def pruneGrad(self):
        for p, m in zip(self.model.parameters(), self.mask):
            p.grad.data = p.grad * m

    def get_mask(self):
        for p, m in zip(self.model.parameters(), self.mask):
            m.data = top_mask(abs(p.grad).view(-1), self.topk, abs(p.grad))
            p.data = p * m


def top_mask(x1, topk, x2):
    res, ind = torch.sort(x1)
    threshold = res[-int(topk / 100 * len(x1))]
    return (x2 >= threshold).detach()