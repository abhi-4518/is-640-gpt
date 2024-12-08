import torch
from data import Data
from model import GPTLanguageModel
from trainer import Trainer

if __name__ == "__main__":
    # hyperparameters
    batch_size = 16
    block_size = 8
    max_iters = 100
    eval_interval = 20
    eval_iters = 200
    learning_rate = 1e-3
    n_embd = 32
    n_head = 6
    n_layer = 4
    dropout = 0.2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
     # load data
    data_obj = Data(file_path='input.txt', device=device, block_size=block_size)

    # create model
    model = GPTLanguageModel(
        vocab_size=data_obj.vocab_size,
        n_embd=n_embd, 
        n_head=n_head, 
        n_layer=n_layer, 
        block_size=block_size, 
        dropout=dropout,
        device=device
    ).to(device)

    # train model
    trainer = Trainer(model, data_obj, max_iters=max_iters, eval_interval=eval_interval, eval_iters=eval_iters, batch_size=batch_size, learning_rate=learning_rate)
    trainer.train()

    # generate text
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    #generated_ids = model.generate(context, max_new_tokens=100)
    generated = model.generate(context, WORD_COUNT)[0].tolist()
    print(data_obj.decode(generated_ids[0].tolist()))
