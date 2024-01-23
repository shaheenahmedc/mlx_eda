import torch
import tqdm
from model import CBOW

def train_cbow(n_epochs, model, loss_function, optimizer, dataloader):
    for epoch in range(n_epochs):
        total_loss = 0
        i = 0
        for context, target in tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}", unit="batch"):
            optimizer.zero_grad()
            log_probs = model(context)
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss}")
        torch.save(model.state_dict(), f"./cbow_epoch_{epoch+1}.pt")

def train_two_tower(n_epochs, model, loss_function, optimizer, dataloader):
    batch_lim = 100
    for epoch in range(n_epochs):
        # Wrap your data loader with tqdm for a nice progress bar
        progress_bar = tqdm.tqdm(dataloader, desc=f'Epoch {epoch+1}/{n_epochs}', unit='batch')

        for batch_query, label, batch_sentence in enumerate(progress_bar, start=1):
            optimizer.zero_grad()

            query_vector, sentence_vector = model(batch_query, batch_sentence)
            loss = loss_function(query_vector, sentence_vector, label)

            loss.backward()
            optimizer.step()
