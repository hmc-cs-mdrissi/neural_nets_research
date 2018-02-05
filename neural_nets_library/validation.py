import torch
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def category_from_output(output):
    """
    Get the index of the maximum value in output.data tensor
    """
    _, category_i = output.data.max(1) # Tensor out of Variable with .data
    return category_i

def category_and_prob(output):
    """
    Get the maximum and index of maximum in the output.data tensor
    """
    top_value, category_i = output.data.max(1) # Tensor out of Variable with .data
    return top_value, category_i

def confusion_matrix(model, data_loader, all_categories):
    """
    Create a confusion matrix using a model, data loader, and a categories list
    """
    n_categories = len(all_categories)
    # Keep track of correct guesses in a confusion matrix
    confusion = torch.zeros(n_categories, n_categories)

    for inputs, labels in data_loader:
        inputs = Variable(inputs.cuda())
        output = model(inputs)

        guesses = category_from_output(output)

        for category_i, guess_i in zip(labels, guesses):
            confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure(figsize=(16, 16), dpi=160)
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()

def missed_confidences(model, data_loader, logsoftmax_used = False):
    """
    Plots a histogram of a model's missed guesses.
    Probability assigned to model's guess when wrong.
    """
    missed_confidence_list = []

    for inputs, labels in data_loader:
        inputs = Variable(inputs.cuda())
        output = model(inputs)

        if logsoftmax_used:
            output.exp_()
        else:
            output = F.softmax(output)

        values, guesses = category_and_prob(output)

        for value, guess, category in zip(values, guesses, labels):
            if guess != category:
                missed_confidence_list.append(value)

    plt.hist(missed_confidence_list, bins=100)
    plt.show()

def confidences_for_true_when_wrong(model, data_loader, logsoftmax_used = False):
    """
    Plots a histogram of probabilities.
    Probability assigned to correct answer when model's guess is wrong.
    """
    correct_confidence_list = []

    for inputs, labels in data_loader:
        inputs = Variable(inputs.cuda())
        output = model(inputs)

        if logsoftmax_used:
            output.exp_()
        else:
            output = F.softmax(output)

        guesses = category_from_output(output)

        for batch_id, (guess, category) in enumerate(zip(guesses, labels)):
            if guess != category:
                correct_confidence_list.append(output.data[batch_id, category])

    plt.hist(correct_confidence_list, bins=100)
    plt.show()

def compare_models(model1, model2, dset_loader):
    """
    Compares two models on data set provided by loader
    """
    running_matches = 0

    for inputs, _ in dset_loader:
        # wrap them in Variable
        inputs = Variable(inputs.cuda())

        # forward
        outputs1 = model1(inputs)
        _, preds1 = outputs1.data.max(1)

        outputs2 = model2(inputs)
        _, preds2 = outputs2.data.max(1)

        running_matches += torch.sum(preds1 == preds2)

    return running_matches/(len(dset_loader) * dset_loader.batch_size)
