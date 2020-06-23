#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
import random
import re
import string
import time

from save import torch_save

import numpy as np

import nltk
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# In[ ]:
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#print(string.punctuation)

SOS_token = 0
EOS_token = 1
# In[ ]:


message_path = "./messages.tsv"

DATASET_PATH = "./message_dataset.pth"
DATALOADER_PATH = "./message_dataloader.pth"

# In[120]:


class MessageDataset(Dataset):
    def __init__(self, message_file):
        self.filepath = message_file
        self.punctuation = {"?", "!", ":", "/", ";"}

        self.SOS_token = 0
        self.EOS_token = 1

        self.VOCAB_INDEX = {}
        self.INDEX_VOCAB = {0: "<sos>", 1: "<eos>"}
        self.word_num = 2

        self.prompt = []
        self.response = []

        self.max_message = 0

        with open(self.filepath) as message_file:
            reader = list(csv.reader(message_file, delimiter="\t"))
            message_len = len(reader)
            for i,row in enumerate(reader):
                you = row[0]
                me = row[1]

                you_tokens = self.tokenize(you)
                me_tokens = self.tokenize(me)

                for tokens in (you_tokens, me_tokens):
                    if len(tokens) > self.max_message:
                        self.max_message = len(tokens)
                    for token in tokens:
                        if token not in self.VOCAB_INDEX:
                            self.VOCAB_INDEX[token] = self.word_num
                            self.INDEX_VOCAB[self.word_num] = token
                            self.word_num += 1
                
                self.prompt.append(you)
                self.response.append(me)
                
                if i % 1000 == 0:
                    print("Processed {} out of {} rows in message file".format(i, message_len))
        
        self.VOCAB_LEN = len(self.INDEX_VOCAB)

    def __getitem__(self, index):
        you = self.tokenize(self.prompt[index])
        me = self.tokenize(self.response[index])

        you_indices = np.zeros((len(you)+1, 1))
        me_indices = np.zeros((len(me)+1, 1))

        for i,tok in enumerate(you):
            ind = self.VOCAB_INDEX[tok]
            you_indices[i] = ind
        you_indices[len(you)] = self.EOS_token
        
        for i,tok in enumerate(me):
            ind = self.VOCAB_INDEX[tok]
            me_indices[i] = ind
        me_indices[len(me)] = self.EOS_token

        you_np = np.asarray(you_indices)
        me_np = np.asarray(me_indices)

        return torch.LongTensor(you_np), torch.LongTensor(me_np)

    def __len__(self):
        return len(self.prompt)
    
    def tokenize(self, text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = word_tokenize(text)
        return words


# torch_save(dataset, DATASET_PATH)

# In[121]:

dataset = MessageDataset(message_path)

# try:
#     dataset = torch.load(DATASET_PATH)
#     dataloader = torch.load(DATALOADER_PATH)
#     print("Loaded dataset and dataloader from paths")
# except:
#     print("Did not load dataset and dataloader from storage")
#     dataset = MessageDataset(message_path)
#     dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=2)
#     torch.save(dataset, DATASET_PATH)
#     torch.save(dataloader, DATALOADER_PATH)
#     print("Saved dataset and dataloader at {} and {}".format(DATASET_PATH, DATALOADER_PATH))


# In[61]:


#nprint(dataset.VOCAB_INDEX)


# In[ ]:


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# In[ ]:


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# In[152]:


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_tensor = torch.transpose(input_tensor, 0, 1).squeeze(1).squeeze(1)
    target_tensor = torch.transpose(target_tensor, 0, 1).squeeze(1).squeeze(1)

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
            if decoder_input.item() == EOS_token:
                break

    if loss != 0:
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.item() / target_length
    else:
        return loss


# In[71]:


# learning_rate = 0.01
# # NEED TO SET
# vocab_words = dataset.word_num
# max_length = dataset.max_message
# print("Max length of message is {}".format(max_length))


# In[131]:


def trainIters(encoder, decoder, dataloader, epochs, temp_path, learning_rate=0.01):
    try:
        checkpoint = torch.load(temp_path)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder.load_state_dict(checkpoint["decoder"])
        print("Loaded previous temp state dict")
    except:
        print("Old state dict not loaded")
    
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    prev_loss_avg = None

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for i in range(epochs):
        for prompt,response in dataloader:
            input_tensor = prompt.cuda()
            target_tensor = response.cuda()

            loss = train(input_tensor, target_tensor, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion, max_length)
            print_loss_total += loss
            plot_loss_total += loss
        else:
            print_loss_avg = print_loss_total / len(dataloader)
            print_loss_total = 0
            print("Epoch {} - Loss: {}".format(i, print_loss_avg))
            if prev_loss_avg is None or print_loss_avg < prev_loss_avg:
                torch.save({"encoder": encoder.state_dict(), 
                            "decoder": decoder.state_dict(),}, temp_path)
                prev_loss_avg = print_loss_avg
                print("Saved model at temp path!")


# In[151]:


# hidden_size = 256
# encoder = EncoderRNN(vocab_words, hidden_size).to(device)
# decoder = AttnDecoderRNN(hidden_size, vocab_words, max_length, dropout_p=0.1).to(device)

# epochs = 20
# temp_path = "./bot_temp_v2.pth"
# final_path = "./bot_v2.pth"

# In[145]:


def evaluate(encoder, decoder, dataset, text, max_length):
    with torch.no_grad():
        input_tensor = build_input(text, dataset)
        if device == "cuda:0":
            input_tensor = input_tensor.cuda()
        print(input_tensor.size())
        input_length = input_tensor.size(0)
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(dataset.INDEX_VOCAB[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


# In[146]:


def build_input(text, dataset):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    indices = np.zeros((len(words), 1))
    i = 0
    for tok in words:
        try:
            ind = dataset.VOCAB_INDEX[tok]
            indices[i] = ind
            i += 1
        except:
            indices[i] = dataset.VOCAB_INDEX["what"]
            i += 1
    ind_np = np.asarray(indices)

    return torch.LongTensor(ind_np)


# In[ ]:


def play(text, encoder, decoder, dataset):
    max_length = dataset.max_message
    output_words, attentions = evaluate(encoder, decoder, dataset, text, max_length)
    output_sentence = ' '.join(output_words)
    print(output_sentence)
    return output_sentence

# checkpoint = torch.load(temp_path)
# encoder.load_state_dict(checkpoint["encoder"])
# decoder.load_state_dict(checkpoint["decoder"])

if __name__ == "__main__":

    print("hi")
    # torch_save(dataset, DATASET_PATH)

    print(dataset.VOCAB_INDEX)
    trainIters(encoder, decoder, dataloader, epochs, temp_path, learning_rate=0.0001)
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, final_path)
    print("Saved model at final path!")

    playing = True
    while playing:
        print("Don't enter anything if you want to stop")
        input_text = input("Text me: ")
        if input_text == "":
            playing = False
        else:
            play(input_text, encoder, decoder, dataset)
