import time
from flcore.clients.clientprox import clientProx
from flcore.servers.serverbase import Server
from threading import Thread


class FedProx(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(args, clientProx)


        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []


    def train(self):
        self.done = False
        i = 0
        while not self.done:
        # for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

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

        
        self.selected_clients = self.clients
        finetune_acc = []
        self.done = False
        i = 0
        while not self.done:
        # for i in range(self.global_rounds+1):
            s_t = time.time()

            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized model")
                self.evaluate(acc=finetune_acc)

            for client in self.clients:
                client.train()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            self.done = self.check_done(acc_lss=[finetune_acc], top_cnt=self.top_cnt)
            i += 1

        print("\nBest global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(max(finetune_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()
