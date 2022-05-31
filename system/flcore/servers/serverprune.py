import time
import torch
from flcore.clients.clientprune import clientPrune
from flcore.servers.serverbase import Server
from threading import Thread


class FedPrune(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientPrune)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.mask = [torch.zeros_like(p) for p in self.global_model.parameters()]


    def train(self):
        for i in range(self.global_rounds+1):
            for client in self.clients:
                client.train()

        for client in self.clients:
            client.get_mask()

        self.done = False
        i = 0
        while not self.done:
        # for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.intersection_mask()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()
                self.compression_rate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            self.done = self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt)
            i += 1

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

    def add_parameters(self, w, client_model):
        for server_param, client_param, m in zip(self.global_model.parameters(), client_model.parameters(), self.mask):
            server_param.data += client_param.data.clone() * w * (m!=0) / m * self.join_clients

    def compression_rate(self):
        a, b = 0, 0
        for c in self.clients:
            for p in c.model.parameters():
                a += torch.sum(p!=0).item()
                b += p.numel()
        print("Compression rate: {:.4}".format(a/b))

    def intersection_mask(self):
        self.mask = [torch.zeros_like(p) + 1e-5 for p in self.global_model.parameters()]
        for c in self.selected_clients:
            for m, m1 in zip(self.mask, c.mask):
                m.data = m + m1
        