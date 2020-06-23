import encoder_decoder
from encoder_decoder import EncoderRNN, AttnDecoderRNN, evaluate, play

import torch

from flask import Flask, request, jsonify
app = Flask(__name__)

DATASET_PATH = "./message_dataset.pth"
DATALOADER_PATH = "./message_dataloader.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device is {}".format(device))

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/text', methods=["GET", "POST"])
def respond():
    answer = {"response": None}
    text = request.json["text"]
    response = play(text, encoder, decoder, dataset)
    response = response.replace(" <EOS>", "")
    answer["response"] = response
    return jsonify(answer)

import numpy as np

from torch.utils.data import Dataset

import nltk
from nltk.tokenize import word_tokenize

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

dataset = torch.load(DATASET_PATH)
vocab_words = dataset.word_num
max_length = dataset.max_message

# vocab_words = 20202
# max_length = 801
# dataloader = torch.load(DATALOADER_PATH)
# print("Loaded dataset and dataloader from paths")

hidden_size = 256
encoder = EncoderRNN(vocab_words, hidden_size).to(device)
decoder = AttnDecoderRNN(hidden_size, vocab_words, max_length, dropout_p=0.1).to(device)

temp_path = "./bot_temp_v2.pth"
final_path = "./bot_v2.pth"

checkpoint = torch.load(temp_path, map_location=torch.device('cpu'))
encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

if __name__ == "__main__":

    app.run()

