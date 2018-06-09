import matplotlib
matplotlib.use('Agg')
import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from neural_nets_library import training
from tree_to_sequence.program_datasets import *
from tree_to_sequence.translating_trees import *

from functools import partial
import argparse
import os

# Counts the number of matches between the prediction and target.
def count_matches(prediction, target):
    matches = 0
    if int(prediction.value) == int(target.value):
        matches += 1
    for i in range(min(len(target.children), len(prediction.children))):
        matches += count_matches(prediction.children[i], target.children[i])
    return matches

# Program accuracy (1 if completely correct, 0 otherwise)
def program_accuracy(prediction, target):
    if decoder_type == "sequence":
        return 1 if target.tolist() == prediction else 0
    if prediction.size() == count_matches(prediction, target) and prediction.size() == target.size():
        return 1
    else:
        return 0

# Calculate validation accuracy (this could either be program or token accuracy)
def validation_criterion(prediction, target):
    return program_accuracy(prediction, target)

parser = argparse.ArgumentParser()

parser.add_argument('--model', required=True, help='Name of model file.')
# parser.add_argument('--problem_number', type=int, required=True, help='Number of the program translation problem. 0 corresponds to for-lambda while 1 is javascript-coffeescript')
parser.add_argument('--decoder_type', required=True, help='Name of decoder. Should be either grammar, sequence, or tree.')
parser.add_argument('--device_number', default=0, help='Number of device to test on. Default is 0')
opt = parser.parse_args()

torch.cuda.set_device(opt.device_number)
decoder_type = opt.decoder_type
num_vars = 10
num_ints = 11
one_hot = False
binarize_input = True
binarize_output = (decoder_type == "tree")
eos_token = (decoder_type != "grammar")
long_base_case = True
input_as_seq = False
output_as_seq = (decoder_type == "sequence")

# Test
model = torch.load("test_various_models/" + opt.model)
model = model.cuda(opt.device_number)


# if opt.problem_number == 0:
dset_test = ForLambdaDataset("ANC/MainProgramDatasets/ForLambda/test_For.json",
                                       binarize_input=binarize_input, binarize_output=binarize_output, 
                                       eos_token=eos_token, one_hot=one_hot,
                                       num_ints=num_ints, num_vars=num_vars,
                                       long_base_case=long_base_case, 
                                       input_as_seq=input_as_seq, 
                                       output_as_seq=output_as_seq)

# elif opt.problem_number == 1:

#     dset_test = JsCoffeeDataset("ANC/MainProgramDatasets/CoffeeJavascript/test_CS.json", "ANC/MainProgramDatasets/CoffeeJavascript/test_JS.json",
#                                  binarize_input=binarize_input, binarize_output=binarize_output, 
#                                  eos_token=eos_token, one_hot=one_hot, num_ints=num_ints, num_vars=num_vars,
#                                  long_base_case=long_base_case, input_as_seq=input_as_seq, output_as_seq=output_as_seq)


mean_acc = training.test_model_tree_to_tree(model, dset_test, validation_criterion, use_cuda=True) 

print("mean accuracy", mean_acc)
