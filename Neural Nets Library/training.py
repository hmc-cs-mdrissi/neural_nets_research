#   The purpose of this file is to contain a collection of useful
#   helper functions for training and testing supervised models.

import copy
import time

import torch
from torch.autograd import Variable
import torch.utils.data as data

def split_dataset(dset):
    """
    Given a dataset, samples elements randomly from a given list of indices, without replacement,
    to create a training, test, and validation sets.
    These sets are then used to return training, testing, and validation DataLoaders.
    """
    sampler_dset_train = data.sampler.SubsetRandomSampler(list(range(int(0.7*len(dset)))))
    sampler_dset_test = data.sampler.SubsetRandomSampler(list(range(int(0.7*len(dset)),
                                                                    int(0.85*len(dset)))))
    sampler_dset_validation = data.sampler.SubsetRandomSampler(list(range(int(0.85*len(dset)),
                                                                          len(dset))))

    loader_dset_train = data.DataLoader(
        dset, batch_size=128, num_workers=4,
        pin_memory=True, sampler=sampler_dset_train)
    loader_dset_test = data.DataLoader(
        dset, batch_size=128, num_workers=4,
        pin_memory=True, sampler=sampler_dset_test)
    loader_dset_validation = data.DataLoader(
        dset, batch_size=128, num_workers=4,
        pin_memory=True, sampler=sampler_dset_validation)

    return loader_dset_train, loader_dset_test, loader_dset_validation

def train_model_with_validation(model, train_loader, validation_loader, criterion,
                                optimizer, lr_scheduler, num_epochs=20):
    """
    Trains a model while printing updates on loss and accuracy. Once training is complete,
    it is tested on the validation data set.
    """
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train(True)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        optimizer = lr_scheduler(optimizer, epoch)

        running_loss = 0.0
        running_corrects = 0

        current_batch = 0
        # Iterate over data.
        for inputs, labels in train_loader:
            current_batch += 1

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), \
                             Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == labels.data)

            if current_batch % 250 == 0:
                curr_acc = running_corrects / (current_batch * train_loader.batch_size)
                curr_loss = running_loss / (current_batch * train_loader.batch_size)
                time_elapsed = time.time() - since

                print('Epoch Number: {}, Batch Number: {}, Loss: {:.4f}, Acc: {:.4f}'.format(
                    epoch, current_batch, curr_loss, curr_acc))
                print('Time so far is {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))



        validation_acc = test_model(model, validation_loader)
        print('Epoch Number: {}, Validation Accuracy: {:.4f}'.format(epoch, validation_acc))

        # deep copy the model
        if validation_acc > best_acc:
            best_acc = validation_acc
            best_model = copy.deepcopy(model)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.train(False)

    return best_model

def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=7):
    """
    Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.
    """
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer

def train_model(model, dset_loader, criterion, optimizer, lr_scheduler = exp_lr_scheduler, num_epochs=20,
                print_every=200, plot_every=100):
    """
    Trains a model while printing updates on loss and accuracy.
    """
    since = time.time()

    best_model = model
    best_loss = float('inf')
    model.train(True)

    running_loss = 0.0
    total_batch_number = 0
    all_losses = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        optimizer = lr_scheduler(optimizer, epoch)

        epoch_running_loss = 0.0
        current_batch = 0
        # Iterate over data.
        for inputs, labels in dset_loader:
            current_batch += 1
            total_batch_number += 1

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), \
                             Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()

            # statistics
            epoch_running_loss += loss.data[0]
            running_loss += loss.data[0]

            if total_batch_number % print_every == 0:
                curr_loss = epoch_running_loss / (current_batch * dset_loader.batch_size)
                time_elapsed = time.time() - since

                print('Epoch Number: {}, Batch Number: {}, Loss: {:.4f}'.format(
                    epoch, current_batch, curr_loss))
                print('Time so far is {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))

            if total_batch_number % plot_every == 0:
                all_losses.append(running_loss/ (total_batch_number * dset_loader.batch_size))


        # deep copy the model
        if epoch_running_loss < best_loss:
            best_loss = epoch_running_loss
            best_model = copy.deepcopy(model)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best loss: {:4f}'.format(best_loss))

    model.train(False)

    return best_model, all_losses

def test_model(model, dset_loader):
    """
    Tests a model on a given data set and returns the accuracy of the model
    on the set.
    """
    model.train(False)

    running_corrects = 0

    for inputs, labels in dset_loader:
        # wrap them in Variable
        inputs, labels = Variable(inputs.cuda()), \
                         Variable(labels.cuda())

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        running_corrects += torch.sum(preds == labels.data)

    return running_corrects/(len(dset_loader) * dset_loader.batch_size)
