import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt

from neural_nets_library import training
from tree_to_sequence.tree_encoder import TreeEncoder
from tree_to_sequence.tree_decoder import TreeDecoder
from tree_to_sequence.tree_to_sequence_attention import TreeToSequenceAttention
from tree_to_sequence.grammar_tree_decoder import GrammarTreeDecoder
from tree_to_sequence.multilayer_lstm_cell import MultilayerLSTMCell
from tree_to_sequence.program_datasets import *
from tree_to_sequence.translating_trees import *
from tree_to_sequence.tree_to_tree_attention import TreeToTreeAttention

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

def reset_all_parameters_uniform(model, stdev):
    for param in model.parameters():
        nn.init.uniform_(param, -stdev, stdev)

parser = argparse.ArgumentParser()

parser.add_argument('--decoder_type', required=True, help='Name of decoder. Should be either grammar, sequence, or tree.')
parser.add_argument('--save_file', required=True, help='Name of save file')
parser.add_argument('--problem_number', type=int, required=True, help='Number of the program translation problem. 0 corresponds to for-lambda while 1 is javascript-coffeescript')
parser.add_argument('--save_folder', default="test_various_models", help='Name of folder to save files in. Defaults to test_various_models/')
parser.add_argument('--cuda_device', type=int, default=0, help='Number of cuda device. Not relevant if cuda is disabled. Default is 0.')
parser.add_argument('--num_vars', type=int, default=10, help='Number of variable names. Default is 10.')
parser.add_argument('--num_ints', type=int, default=11, help='Number of possible integer literals. Default is 11')
parser.add_argument('--one_hot', action='store_true', help='Use one hot vectors instead of embeddings.')
parser.add_argument('--binarize_input', action='store_true', help="Binarize the input. Default is not to.")
parser.add_argument('--binarize_output', action='store_true', help="Binarize the output. Default is not to.")
parser.add_argument('--binary_tree_lstm_cell', action='store_true', help="Use a binary tree lstm cell. Default is not to.")
parser.add_argument('--no_long_base_case', action='store_true', help="Use a more minimal tree (mainly dropping out tokens that don't add any information)")
parser.add_argument('--lr', type=float, default=0.005, help='learning rate for model using adam, default=0.005')
parser.add_argument('--dropout', type=float, default=False, help='Dropout probability. The default is not to use dropout.')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train for. The default is 5.')
parser.add_argument('--no_cuda', action='store_true', help='Disables cuda')
opt = parser.parse_args()

decoder_type = opt.decoder_type
save_file = opt.save_file
save_folder = opt.save_folder
use_cuda = not opt.no_cuda
torch.cuda.set_device(opt.cuda_device)

num_vars = opt.num_vars
num_ints = opt.num_ints
one_hot = opt.one_hot
binarize_input = opt.binarize_input
binarize_output = opt.binarize_output
eos_token = (decoder_type != "grammar")
long_base_case = not opt.no_long_base_case
input_as_seq = False
output_as_seq = (decoder_type == "sequence")

if opt.problem_number == 0:
    # Make dataset
    dset_train = ForLambdaDataset("ANC/MainProgramDatasets/ForLambda/training_For.json",
                                       binarize_input=binarize_input, binarize_output=binarize_output, 
                                       eos_token=eos_token, one_hot=one_hot,
                                       num_ints=num_ints, num_vars=num_vars,
                                       long_base_case=long_base_case, 
                                       input_as_seq=input_as_seq, 
                                       output_as_seq=output_as_seq)

    dset_val = ForLambdaDataset("ANC/MainProgramDatasets/ForLambda/validation_For.json",
                                       binarize_input=binarize_input, binarize_output=binarize_output, 
                                       eos_token=eos_token, one_hot=one_hot,
                                       num_ints=num_ints, num_vars=num_vars,
                                       long_base_case=long_base_case, 
                                       input_as_seq=input_as_seq, 
                                       output_as_seq=output_as_seq)

    dset_test = ForLambdaDataset("ANC/MainProgramDatasets/ForLambda/test_For.json",
                                       binarize_input=binarize_input, binarize_output=binarize_output, 
                                       eos_token=eos_token, one_hot=one_hot,
                                       num_ints=num_ints, num_vars=num_vars,
                                       long_base_case=long_base_case, 
                                       input_as_seq=input_as_seq, 
                                       output_as_seq=output_as_seq)
elif opt.problem_number == 1:
    dset_train = JsCoffeeDataset("ANC/MainProgramDatasets/CoffeeJavascript/training_CS.json", 
                                 "ANC/MainProgramDatasets/CoffeeJavascript/training_JS.json",
                                  binarize_input=binarize_input, binarize_output=binarize_output, 
                                  eos_token=eos_token, one_hot=one_hot, num_ints=num_ints, num_vars=num_vars,
                                  long_base_case=long_base_case, input_as_seq=input_as_seq, output_as_seq=output_as_seq)

    dset_val = JsCoffeeDataset("ANC/MainProgramDatasets/CoffeeJavascript/validation_CS.json", "ANC/MainProgramDatasets/CoffeeJavascript/validation_JS.json",
                                binarize_input=binarize_input, binarize_output=binarize_output, 
                                eos_token=eos_token, one_hot=one_hot, num_ints=num_ints, num_vars=num_vars,
                                long_base_case=long_base_case, input_as_seq=input_as_seq, output_as_seq=output_as_seq)

    dset_test = JsCoffeeDataset("ANC/MainProgramDatasets/CoffeeJavascript/test_CS.json", "ANC/MainProgramDatasets/CoffeeJavascript/test_JS.json",
                                 binarize_input=binarize_input, binarize_output=binarize_output, 
                                 eos_token=eos_token, one_hot=one_hot, num_ints=num_ints, num_vars=num_vars,
                                 long_base_case=long_base_case, input_as_seq=input_as_seq, output_as_seq=output_as_seq)
else:
    raise ValueError("Problem number must be either 0 or 1.")

if decoder_type != "sequence":
    max_size = int(max([x[1].size() for x in dset_train] + [x[1].size() for x in dset_val] + [x[1].size() for x in dset_test]))

embedding_size = 256
hidden_size = 256
num_layers = 1
alignment_size = 50
align_type = 1

if opt.problem_number == 0:
    encoder_input_size = num_vars + num_ints + len(for_ops)
    nclass = num_vars + num_ints + len(lambda_ops)
    num_categories = len(LambdaGrammar)
    num_possible_parents = len(Lambda)
    max_num_children = 2 if binarize_output else 4
    parent_to_category = partial(parent_to_category_LAMBDA, num_vars, num_ints)
    category_to_child = partial(category_to_child_LAMBDA, num_vars, num_ints)
else:
    encoder_input_size = num_vars + num_ints + len(javascript_ops)
    nclass = num_vars + num_ints + len(coffee_ops)
    num_categories = len(CoffeeGrammar)
    num_possible_parents = len(Coffee)
    max_num_children = 2 if binarize_output else 3
    parent_to_category = partial(parent_to_category_coffee, num_vars, num_ints)
    category_to_child = partial(category_to_child_coffee, num_vars, num_ints)

plot_every = 100
save_every=5000

def save_plots():
    # Save plots
    plt.plot([x * plot_every for x in range(len(train_plot_losses))], train_plot_losses)
    plt.plot([x * plot_every for x in range(len(val_plot_losses))], val_plot_losses)
    plt.title("Training Loss")
    plt.xlabel('Training Examples')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend(("Train", "Validation"))
    plt.savefig(save_folder + "/" + save_file + "_train_plot.png")
    plt.close()

    plt.plot([x * plot_every for x in range(len(train_plot_accuracies))], train_plot_accuracies)
    plt.plot([x * plot_every for x in range(len(val_plot_accuracies))], val_plot_accuracies)
    plt.title("Training Program Accuracy")
    plt.xlabel('Training Examples')
    plt.ylabel('Percent Accurate Programs')
    plt.legend(("Train", "Validation"))
    plt.savefig(save_folder + "/" + save_file + "_accuracy_plot.png")

def save_test_accuracy():
    with open(save_folder + "/" + save_file + "_test.txt", "a") as file:
        file.write(str(test_accuracy))
    print('really done')

def make_model():
    encoder = TreeEncoder(encoder_input_size, hidden_size, num_layers, [1, 2, 3, 4, 5], attention=True, one_hot=one_hot, 
                          binary_tree_lstm_cell=opt.binary_tree_lstm_cell)

    if decoder_type == "grammar":
        decoder = GrammarTreeDecoder(embedding_size, hidden_size, num_categories, 
                                     num_possible_parents, parent_to_category, 
                                     category_to_child, share_linear=True, share_lstm_cell=True, 
                                     num_ints_vars=num_ints + num_vars)
        
        program_model = TreeToTreeAttention(encoder, decoder, 
                                            hidden_size, embedding_size, 
                                            nclass=nclass, root_value=nclass,
                                            alignment_size=alignment_size, 
                                            align_type=align_type, max_size=max_size)
    elif decoder_type == "tree":
        decoder = TreeDecoder(embedding_size, hidden_size, max_num_children, nclass=nclass)
        program_model = TreeToTreeAttention(encoder, decoder, 
                                            hidden_size, embedding_size, 
                                            nclass=nclass, root_value=nclass,
                                            alignment_size=alignment_size, 
                                            align_type=align_type, max_size=max_size)
        
    elif decoder_type == "sequence":
        decoder = MultilayerLSTMCell(embedding_size + hidden_size, hidden_size, num_layers)
        program_model = TreeToSequenceAttention(encoder, decoder, hidden_size, nclass, 
                                                embedding_size, alignment_size=alignment_size, 
                                                align_type=align_type)
        
    if use_cuda:
        program_model = program_model.cuda()
        
    reset_all_parameters_uniform(program_model, 0.1)
    encoder.initialize_forget_bias(3)
    decoder.initialize_forget_bias(3)
    
    return program_model

test_accuracies = []
program_model = make_model()

# Optimizer
optimizer = optim.Adam(program_model.parameters(), lr=opt.lr)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=500, factor=0.8)

# Train
os.system("mkdir "+ save_folder)
model, train_plot_losses, train_plot_accuracies, val_plot_losses, val_plot_accuracies = training.train_model_tree_to_tree(
                                program_model, 
                                dset_train, 
                                optimizer, 
                                lr_scheduler=lr_scheduler, 
                                num_epochs=opt.num_epochs, plot_every=plot_every,
                                batch_size=100, 
                                print_every=200, 
                                validation_dset = dset_val,                                                          
                                validation_criterion=validation_criterion,
                                use_cuda=use_cuda, 
                                plateau_lr=True,
                                save_file=save_file,
                                save_folder=save_folder,
                                save_every=save_every)

# Test
test_accuracy = training.test_model_tree_to_tree(model, dset_test, validation_criterion, use_cuda=True)
test_accuracies.append(test_accuracy)
# save_plots()
print("test", test_accuracy)
    
save_test_accuracy()
