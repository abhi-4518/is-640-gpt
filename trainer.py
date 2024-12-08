"""
import torch

class Trainer:
    def __init__(self, model, data_obj, max_iters=5000, eval_interval=500, eval_iters=200, batch_size=64, learning_rate=3e-4):
        self.model = model
        self.data = data_obj
        self.max_iters = max_iters
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.batch_size = batch_size
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        self.train_data, self.val_data = self.data.split_data()

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.data.get_batch(split, self.batch_size, self.train_data, self.val_data)
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.model.train()
        return out

    def train(self):
        for iter in range(self.max_iters):
            if iter % self.eval_interval == 0 or iter == self.max_iters - 1:
                losses = self.estimate_loss()
                print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

            # sample a batch of data
            xb, yb = self.data.get_batch('train', self.batch_size, self.train_data, self.val_data)

            # evaluate the loss
            _, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        # final loss value after training
        final_losses = self.estimate_loss()
        print(f"Final train loss {final_losses['train']:.4f}, val loss {final_losses['val']:.4f}")
"""
import torch
import torch.nn as nn

class Trainer:
    def __init__(self, data, model):
        self.data = data
        self.model = model

    def train(self, iterations):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for _ in range(iterations):
            # get a small sequence of data
            # Suppose we take the first 11 characters as a single batch (batch_size=1, seq_len=11)
            batch = self.data.data[:11].unsqueeze(0)  # shape [1, 11]

            # input to model is everything except last token
            input_tokens = batch[:, :-1]  # shape [1, 10]
            # target is the next token for each input token
            target_tokens = batch[:, 1:]  # shape [1, 10]

            output = self.model(input_tokens)  # shape [1, 10, vocab_size]

            # reshape for CrossEntropyLoss
            B, T, V = output.shape
            output = output.reshape(B*T, V)
            target_tokens = target_tokens.reshape(B*T)

            loss = criterion(output, target_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
