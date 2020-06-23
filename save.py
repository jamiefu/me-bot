import torch

def torch_save(thing, path):
    torch.save(thing, path)
    print("I saved it")