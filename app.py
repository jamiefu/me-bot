from encoder_decoder import device, dataset, EncoderRNN, AttnDecoderRNN, evaluate, play

import torch

import csv
import re
import string

from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/text', methods=["GET", "POST"])
def respond():
    answer = {"response": None}
    text = request.json["text"]
    response = play(text, encoder, decoder, dataset)
    if response == "<EOS>":
        response = "my bot said there was nothing to say, so here it is: nothing"
    else:
        response = response.replace(" <EOS>", "")
    answer["response"] = response
    return jsonify(answer)

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

