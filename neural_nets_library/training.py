#   The purpose of this file is to contain a collection of useful
#   helper functions for training and testing supervised models.

import copy
import time

import torch
import torch.utils.data as data
import torch.optim as optim
import numpy as np

def split_dataset(dset, batch_size=128, thread_count=4):
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
        dset, batch_size=batch_size, num_workers=thread_count,
        pin_memory=True, sampler=sampler_dset_train)
    loader_dset_test = data.DataLoader(
        dset, batch_size=batch_size, num_workers=thread_count,
        pin_memory=True, sampler=sampler_dset_test)
    loader_dset_validation = data.DataLoader(
        dset, batch_size=batch_size, num_workers=thread_count,
        pin_memory=True, sampler=sampler_dset_validation)

    return loader_dset_train, loader_dset_test, loader_dset_validation

def train_model_with_validation(model, train_loader, validation_loader, criterion,
                                optimizer, lr_scheduler=None, num_epochs=20):
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

        if lr_scheduler is not None:
            lr_scheduler.step(epoch)

        running_loss = 0.0
        running_corrects = 0

        current_batch = 0
        # Iterate over data.
        for inputs, labels in train_loader:
            start_time = time.time()
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
            running_loss += float(loss)
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

def clip_grads(net):
    """Clip the gradients to -5 to 5."""
    for p in net.parameters():
        if p.grad is not None:
            p.grad.data.clamp_(-5, 5)

def train_model(model, dset_loader, training_criterion, optimizer, 
                lr_scheduler=None, num_epochs=20,
                print_every=200, plot_every=100, deep_copy_desired=False, validation_criterion=None,
                plateau_lr=False, use_cuda=True):

    since = time.time()

    best_model = model
    best_loss = float('inf')
    model.train(True)
    train_plot_losses = []
    validation_plot_losses = []
    running_train_plot_loss = 0.0
    running_validation_plot_loss = 0.0
    running_train_print_loss = 0.0
    running_validation_print_loss = 0.0
    total_batch_number = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        epoch_running_loss = 0.0
        current_batch = 0

        if lr_scheduler is not None and not plateau_lr:
            lr_scheduler.step(epoch)

        # Iterate over data.
        for inputs, labels in dset_loader:
            total_batch_number += 1
            current_batch += 1

            # wrap them in Variable
            if use_cuda:
                inputs, labels = Variable(inputs.cuda()), \
                               Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            loss = training_criterion(outputs, labels)

            if validation_criterion is not None:
                validation_loss = float(validation_criterion(outputs, labels))
                running_validation_plot_loss += validation_loss
                running_validation_print_loss += validation_loss

            if plateau_lr:
                lr_scheduler.step(float(loss))
            # backward
            loss.backward()
            clip_grads(model)
            optimizer.step()

            # statistics
            epoch_running_loss += float(loss)
            running_train_plot_loss += float(loss)
            running_train_print_loss += float(loss)


            if total_batch_number % print_every == 0:
                curr_loss = running_train_print_loss / print_every
                time_elapsed = time.time() - since
                
                if validation_criterion is not None:
                    curr_validation_loss = running_validation_print_loss / print_every
                    print('Epoch Number: {}, Batch Number: {}, Training Loss: {:.4f}, Validation Metric: {:.4f}'.format(
                    epoch, current_batch, curr_loss, curr_validation_loss))
                else:
                    print('Epoch Number: {}, Batch Number: {}, Loss: {:.4f}'.format(
                    epoch, current_batch, curr_loss))
                    print('Time so far is {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))

                running_train_print_loss = 0.0
                running_validation_print_loss = 0.0

            if total_batch_number % plot_every == 0:
                train_plot_losses.append(running_train_plot_loss/plot_every)
                running_train_plot_loss = 0.0
                if validation_criterion is not None:
                    validation_plot_losses.append(running_validation_plot_loss/plot_every)
                    running_validation_plot_loss = 0.0

        # deep copy the model
        if epoch_running_loss < best_loss:
            best_loss = epoch_running_loss/len(dset_loader)
            if deep_copy_desired:
                best_model = copy.deepcopy(model)

    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best loss: {:4f}'.format(best_loss))

    model.train(False)

    return best_model, train_plot_losses, validation_plot_losses

def train_model_anc(model,
                    dset_loader,
                    optimizer,
                    lr_scheduler=None,
                    num_epochs=20,
                    print_every=200,
                    plot_every=100,
                    validation_criterion=None,
                    batch_size=50,
                    deep_copy_desired=False,
                    use_cuda=False,
                    plateau_lr=False):
    since = time.time()

    best_model = model
    best_loss = float('inf')
    model.train(True)
    train_plot_losses = []
    validation_plot_losses = []
    running_train_plot_loss = 0.0
    running_validation_plot_loss = 0.0
    running_train_print_loss = 0.0
    running_validation_print_loss = 0.0
    total_batch_number = 0

    # Loss used for batches
    loss = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        epoch_running_loss = 0.0
        current_batch = 0

        if lr_scheduler is not None and not plateau_lr:
            lr_scheduler.step(epoch)

        # Iterate over data.
        for input, target in dset_loader:
            if use_cuda:
                input, target = input.cuda(), target.cuda()

            total_batch_number += 1
            current_batch += 1

            # forward
            iteration_loss = model.forward_train(input, target)

            if validation_criterion is not None:
                output = model.forward_prediction(input)
                validation_loss = validation_criterion(output, target)
                running_validation_plot_loss += validation_loss
                running_validation_print_loss += validation_loss

            loss += iteration_loss

            if total_batch_number % batch_size == 0:
                loss /= batch_size
                loss.backward()
                clip_grads(model)
                optimizer.step()
                
                if plateau_lr:
                    lr_scheduler.step(float(loss))
                
                loss = 0
                
                # zero the parameter gradients
                optimizer.zero_grad()

            # statistics
            epoch_running_loss += float(iteration_loss)
            running_train_plot_loss += float(iteration_loss)
            running_train_print_loss += float(iteration_loss)


            if total_batch_number % print_every == 0:
                curr_loss = running_train_print_loss / print_every
                time_elapsed = time.time() - since

                if validation_criterion is not None:
                    curr_validation_loss = running_validation_print_loss / print_every
                    print('Epoch Number: {}, Batch Number: {}, Validation Metric: {:.4f}'.format(
                    epoch, current_batch, curr_validation_loss))
                    running_validation_print_loss = 0.0

                print('Epoch Number: {}, Batch Number: {}, Training Loss: {:.4f}'.format(
                epoch, current_batch, curr_loss))
                print('Time so far is {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

                running_train_print_loss = 0.0

            if total_batch_number % plot_every == 0:
                train_plot_losses.append(running_train_plot_loss/plot_every)
                running_train_plot_loss = 0.0
                if validation_criterion is not None:
                    validation_plot_losses.append(running_validation_plot_loss/plot_every)
                    running_validation_plot_loss = 0.0

        # deep copy the model
        if epoch_running_loss < best_loss:
            best_loss = epoch_running_loss/len(dset_loader)
            if deep_copy_desired:
                best_model = copy.deepcopy(model)

    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
          time_elapsed // 60, time_elapsed % 60))
    print('Best loss: {:4f}'.format(best_loss))

    model.train(False)
    return best_model, train_plot_losses, validation_plot_losses

def train_model_tree_to_anc(model,
                    dset_loader,
                    optimizer,
                    lr_scheduler=None,
                    num_epochs=20,
                    print_every=200,
                    plot_every=100,
                    batch_size=50,
                    deep_copy_desired=False,
                    plateau_lr=False):
    since = time.time()

    best_model = model
    best_loss = float('inf')
    model.train(True)
    train_plot_losses = []
    running_train_plot_loss = 0.0
    running_train_print_loss = 0.0
    total_batch_number = 0

    # Loss used for batches
    loss = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        epoch_running_loss = 0.0
        current_batch = 0

        if lr_scheduler is not None and not plateau_lr:
            lr_scheduler.step(epoch)

        # Iterate over data.
        for input, target in dset_loader:
            total_batch_number += 1
            current_batch += 1

            iteration_loss = model(input, target)
            loss += iteration_loss
            
            if total_batch_number % batch_size == 0:
                loss /= batch_size
                loss.backward()
                clip_grads(model)
                optimizer.step()

                if plateau_lr:
                    lr_scheduler.step(float(loss))

                loss = 0

                # zero the parameter gradients
                optimizer.zero_grad()

            # statistics
            epoch_running_loss += float(iteration_loss)
            running_train_plot_loss += float(iteration_loss)
            running_train_print_loss += float(iteration_loss)

            if total_batch_number % print_every == 0:
                curr_loss = running_train_print_loss / print_every
                time_elapsed = time.time() - since

                print('Epoch Number: {}, Batch Number: {}, Training Loss: {:.4f}'.format(
                    epoch, current_batch, curr_loss))
                print('Time so far is {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))

                running_train_print_loss = 0.0

            if total_batch_number % plot_every == 0:
                train_plot_losses.append(running_train_plot_loss / plot_every)
                running_train_plot_loss = 0.0

        # deep copy the model
        if epoch_running_loss < best_loss:
            best_loss = epoch_running_loss / len(dset_loader)
            if deep_copy_desired:
                best_model = copy.deepcopy(model)

    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best loss: {:4f}'.format(best_loss))

    model.train(False)
    return best_model, train_plot_losses

def train_model_tree_to_ntm(model,
                    dset_loader,
                    optimizer,
                    lr_scheduler=None,
                    num_epochs=20,
                    print_every=200,
                    plot_every=100,
                    batch_size=50,
                    deep_copy_desired=False,
                    use_cuda=True,
                    plateau_lr=False):
    since = time.time()

    best_model = model
    best_loss = float('inf')
    model.train(True)
    train_plot_losses = []
    running_train_plot_loss = 0.0
    running_train_print_loss = 0.0
    total_batch_number = 0

    # Loss used for batches
    loss = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        epoch_running_loss = 0.0
        current_batch = 0

        if lr_scheduler is not None and not plateau_lr:
            lr_scheduler.step(epoch)

        # Iterate over data.
        for tree, input, target in dset_loader:
            target = Variable(target)
            input = Variable(input)
            total_batch_number += 1
            current_batch += 1

            iteration_loss = model.forward_train((tree,input), target)
            loss += iteration_loss
            
            if total_batch_number % batch_size == 0:
                loss /= batch_size
                loss.backward()
                clip_grads(model)
                optimizer.step()

                if plateau_lr:
                    lr_scheduler.step(float(loss))

                loss = 0

                # zero the parameter gradients
                optimizer.zero_grad()

            # statistics
            epoch_running_loss += float(iteration_loss)
            running_train_plot_loss += float(iteration_loss)
            running_train_print_loss += float(iteration_loss)

            if total_batch_number % print_every == 0:
                curr_loss = running_train_print_loss / print_every
                time_elapsed = time.time() - since

                print('Epoch Number: {}, Batch Number: {}, Training Loss: {:.4f}'.format(
                    epoch, current_batch, curr_loss))
                print('Time so far is {:.0f}m {:.0f}s'.format(
                    time_elapsed // 60, time_elapsed % 60))
                print('Example diff:')
                example_outs = []
                for i in range(len(target)):
                    example_outs.append(model.forward_prediction((tree,input[i])).data[0][0])
                print('Example Outs: ', example_outs)
                print('Expected Outs: ', target)
                running_train_print_loss = 0.0

            if total_batch_number % plot_every == 0:
                train_plot_losses.append(running_train_plot_loss / plot_every)
                running_train_plot_loss = 0.0

        # deep copy the model
        if epoch_running_loss < best_loss:
            best_loss = epoch_running_loss / len(dset_loader)
            if deep_copy_desired:
                best_model = copy.deepcopy(model)

    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best loss: {:4f}'.format(best_loss))

    model.train(False)
    return best_model, train_plot_losses


import time
def train_model_tree_to_tree(model,
                    dset_loader,
                    optimizer,
                    lr_scheduler=None,
                    num_epochs=20,
                    print_every=200,
                    plot_every=100,
                    batch_size=50,
                    validation_criterion=None,
                    validation_dset = None,
                    plateau_lr=False,
                    use_cuda=False,
                    skip_output_cuda=False,
                    skip_input_cuda=False,
                    save_file=False,
                    save_folder=False,
                    save_every=1000,
                    tokens=None,
                    save_current_only=False,
                    input_tree_form=True,
                    save_model_every=10):
    since = time.time()

    model.train(True)
    
    train_plot_losses = []
    train_plot_accuracies = []
    
    curr_train_plot_losses = []
    curr_train_plot_accuracies = []
    
    val_plot_losses = []
    val_plot_accuracies = []
    
    curr_val_plot_losses = []
    curr_val_plot_accuracies = []
    
    running_train_plot_loss = 0.0
    running_train_print_loss = 0.0
    
    train_running_plot_accuracy = 0.0
    train_running_print_accuracy = 0.0
    
    running_val_plot_loss = 0.0
    val_running_plot_accuracy = 0.0
    
    total_batch_number = 0

    # Loss used for batches
    loss = 0  # MUST UNCOMMENT IF YOU WANT TO BE ABLE TO BACKPROP AFTER MORE THAN 1

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        epoch_running_loss = 0.0
        current_batch = 0

        if lr_scheduler is not None and not plateau_lr:
            lr_scheduler.step(epoch)

        # Iterate over data.
        for input_tree, target_tree in dset_loader:
            start_time = time.time()
            loss = 0
            if use_cuda:
                if not skip_input_cuda:
                    input_tree = input_tree.cuda()
                if not skip_output_cuda:
                    if type(target_tree) == list:
                        target_tree = [actual_tree.cuda() for actual_tree in target_tree]
                    else:
                        target_tree = target_tree.cuda()
            
            total_batch_number += 1
            current_batch += 1

            model.train()
            iteration_loss = model.forward_train(input_tree, target_tree, input_tree_form=input_tree_form)
            loss += iteration_loss
            
            if total_batch_number % batch_size == 0:
                loss /= batch_size
                loss.backward()
                
                clip_grads(model)
                optimizer.step()

                if plateau_lr:
                    lr_scheduler.step(float(loss))

#                 del loss
#                 loss = 0

                # zero the parameter gradients
                optimizer.zero_grad()

            if validation_criterion is not None:
                model.eval()
                output = model.forward_prediction(input_tree)
                validation_loss = validation_criterion(output, target_tree)
                train_running_plot_accuracy += float(validation_loss)
                train_running_print_accuracy += float(validation_loss)
                if validation_dset:
                    input_val, target_val = validation_dset[total_batch_number % len(validation_dset)]
                    if use_cuda:
                        if not skip_input_cuda:
                            input_val = input_val.cuda()
                        if not skip_output_cuda:                            
                            if type(target_val) == list:
                                target_val = [actual_tree.cuda() for actual_tree in target_val]
                            else:
                                target_val = target_val.cuda()
                    model.eval()
                    output_val = model.forward_prediction(input_val)
                    val_running_plot_accuracy += float(validation_criterion(output_val, target_val))
                
            # statistics
            epoch_running_loss += float(iteration_loss)
            running_train_plot_loss += float(iteration_loss)
            if validation_dset:
                input_val, target_val = validation_dset[total_batch_number % len(validation_dset)]
                if use_cuda:
                    if not skip_input_cuda:
                        input_val = input_val.cuda()
                    if not skip_output_cuda:
                        if type(target_val) == list:
                            target_val = [actual_tree.cuda() for actual_tree in target_val]
                        else:
                            target_val = target_val.cuda()
                        
                model.train()
                running_val_plot_loss += float(model.forward_train(input_val, target_val, input_tree_form=input_tree_form))
            running_train_print_loss += float(iteration_loss)

            if total_batch_number % print_every == 0:
                curr_loss = running_train_print_loss / print_every
                time_elapsed = time.time() - since

                print('Epoch Number: {}, Batch Number: {}, Training Loss: {:.4f}'.format(
                    epoch, current_batch, curr_loss))
                print('Time so far is {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                running_train_print_loss = 0.0
                
                if validation_criterion is not None:
                    curr_validation_loss = train_running_print_accuracy / print_every
                    print('Epoch Number: {}, Batch Number: {}, Validation Metric: {:.4f}'.format(
                    epoch, current_batch, curr_validation_loss))
                    print('Example output:')
                    model.eval()
                    model.print_img_tree_example(input_tree, target_tree, tokens) #THIS BREAKS A LOT OF STUFF
                    train_running_print_accuracy = 0.0
                
            if total_batch_number % plot_every == 0:
                train_plot_losses.append(running_train_plot_loss / plot_every)
                curr_train_plot_losses.append(running_train_plot_loss / plot_every)
                running_train_plot_loss = 0.0
                
                if validation_dset:
                    val_plot_losses.append(running_val_plot_loss / plot_every)
                    curr_val_plot_losses.append(running_val_plot_loss / plot_every)
                    running_val_plot_loss = 0.0
                    
                    val_plot_accuracies.append(val_running_plot_accuracy/plot_every)
                    curr_val_plot_accuracies.append(val_running_plot_accuracy/plot_every)
                    val_running_plot_accuracy = 0.0
                
                
                if validation_criterion is not None:
                    train_plot_accuracies.append(train_running_plot_accuracy/plot_every)
                    curr_train_plot_accuracies.append(train_running_plot_accuracy/plot_every)
                    train_running_plot_accuracy = 0.0
                    
            if save_file and save_folder and total_batch_number % save_every == 0:
                # Save losses
                with open(save_folder + "/" + save_file + "_train_loss.txt", "a") as file:
                    for val in curr_train_plot_losses:
                        file.write(str(val) + ",")
                curr_train_plot_losses = []
                # Save accuracies
                with open(save_folder + "/" + save_file + "_train_accuracy.txt", "a") as file:
                    for val in curr_train_plot_accuracies:
                        file.write(str(val) + ",")
                curr_train_plot_accuracies = []
                if validation_dset:
                    # Save losses
                    with open(save_folder + "/" + save_file + "_val_loss.txt", "a") as file:
                        for val in curr_val_plot_losses:
                            file.write(str(val) + ",")
                    curr_val_plot_losses = []
                    # Save accuracies
                    with open(save_folder + "/" + save_file + "_val_accuracy.txt", "a") as file:
                        for val in curr_val_plot_accuracies:
                            file.write(str(val) + ",")
                    curr_val_plot_accuracies = []
#             print("DONE WITH BATCH")
#             check_allocation()
            end_time = time.time()
#             print("One batch took: ", end_time - start_time)
#             get_gpu_memory_map()
    
        # Save model
        if save_file and save_folder:
            if save_current_only:
                torch.save(model, save_folder + "/" + save_file + "_model")
            elif epoch % save_model_every == 0:
                torch.save(model, save_folder + "/" + save_file + "_epoch_" + str(epoch) + "_model")
            

    print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    model.train(False)
    return model, train_plot_losses, train_plot_accuracies, val_plot_losses, val_plot_accuracies

import subprocess

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    print("Current usage: %i of 11178" % gpu_memory_map[1])


import gc
def check_allocation():
    print("STARTING ALLOCATION CHECK")
    total_objects = 0
    bad_stuff = 0
    num_tensors = 0
    num_one_tensors = 0
    num_bigger_tensors = 0
    non_tensors = 0
    for obj in gc.get_objects():
        total_objects += 1
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#                 print(type(obj), obj.size())
                if torch.is_tensor(obj):
                    num_tensors += 1
                    if len(obj.size()) == 1 and obj.size()[0] == 1:
                        num_one_tesnors += 1
                    else:
                        num_bigger_tensors += 1
                else:
                    non_tensors += 1
#                 
        except:
            bad_stuff += 1
#             print("allcoation check error")
    print("total", total_objects, "bad", bad_stuff, "num tesnors", num_tensors, "bigger", num_bigger_tensors, "non_tensors", non_tensors)
    print("ENDING ALLOCATION CHECK")
#     assert False


# This one is for testing batching!!!
def yet_another_train_func(model,
                    dset_loader,
                    optimizer,
                    lr_scheduler=None,
                    num_epochs=20,
                    print_every=200,
                    plot_every=100,
                    batch_size=50,
                    validation_criterion=None,
                    validation_dset = None,
                    plateau_lr=False,
                    use_cuda=True, # unused argument, for compatibility with other training funcs
                    save_file=False,
                    save_folder=False,
                    print_example=False,
                    save_every=1000):
    
    since = time.time()

    model.train(True)
    
    train_plot_losses = []
    train_plot_accuracies = []
    
    curr_train_plot_losses = []
    curr_train_plot_accuracies = []
    
    val_plot_losses = []
    val_plot_accuracies = []
    
    curr_val_plot_losses = []
    curr_val_plot_accuracies = []
    
    running_train_plot_loss = 0.0
    running_train_print_loss = 0.0
    
    train_running_plot_accuracy = 0.0
    train_running_print_accuracy = 0.0
    
    running_val_plot_loss = 0.0
    val_running_plot_accuracy = 0.0
    
    total_batch_number = 0

    # Loss used for batches
    loss = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        epoch_running_loss = 0.0
        current_batch = 0

        if lr_scheduler is not None and not plateau_lr:
            lr_scheduler.step(epoch)
            
            
        for batch in dset_loader:
            tree_list = batch
            input_tree_list = [tree_tuple[0] for tree_tuple in tree_list]
            target_tree_list = [tree_tuple[1] for tree_tuple in tree_list]
                               
            total_batch_number += 1
            current_batch += 1

            iteration_loss = model.forward_train(input_tree_list, target_tree_list)
            loss += iteration_loss
            
            loss /= batch_size
            loss.backward()
            clip_grads(model)
            optimizer.step()

            if plateau_lr:
                lr_scheduler.step(float(loss))

            loss *= 0 #TODO:CHANGE BACK!!

            # zero the parameter gradients
            optimizer.zero_grad()

            if validation_criterion is not None:
                output = model.forward_prediction(input_tree)
                validation_loss = validation_criterion(output, target_tree)
                train_running_plot_accuracy += validation_loss
                train_running_print_accuracy += validation_loss
                if validation_dset:
                    input_val, target_val = validation_dset[total_batch_number % len(validation_dset)]
                    if use_cuda:
                        input_val, target_val = input_val.cuda(), target_val.cuda()
                    output_val = model.forward_prediction(input_val)
                    val_running_plot_accuracy += validation_criterion(output_val, target_val)
                
            # statistics
            epoch_running_loss += float(iteration_loss)
            running_train_plot_loss += float(iteration_loss)
            if validation_dset:
                input_val, target_val = validation_dset[total_batch_number % len(validation_dset)]
                if use_cuda:
                    input_val, target_val = input_val.cuda(), target_val.cuda()
                running_val_plot_loss += float(model.forward_train(input_val, target_val))
            running_train_print_loss += float(iteration_loss)

            if total_batch_number % print_every == 0:
                curr_loss = running_train_print_loss / print_every
                time_elapsed = time.time() - since

                print('Epoch Number: {}, Batch Number: {}, Training Loss: {:.4f}'.format(
                    epoch, current_batch, curr_loss))
                print('Time so far is {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
                running_train_print_loss = 0.0
                
                if validation_criterion is not None:
                    curr_validation_loss = train_running_print_accuracy / print_every
                    print('Epoch Number: {}, Batch Number: {}, Validation Metric: {:.4f}'.format(
                    epoch, current_batch, curr_validation_loss))
                    print('Example output:')
                    if print_examples:
                        model.print_example(input_tree_list[0], target_tree_list[0])
                    train_running_print_accuracy = 0.0
                
            if total_batch_number % plot_every == 0:
                train_plot_losses.append(running_train_plot_loss / plot_every)
                curr_train_plot_losses.append(running_train_plot_loss / plot_every)
                running_train_plot_loss = 0.0
                
                if validation_dset:
                    val_plot_losses.append(running_val_plot_loss / plot_every)
                    curr_val_plot_losses.append(running_val_plot_loss / plot_every)
                    running_val_plot_loss = 0.0
                    
                    val_plot_accuracies.append(val_running_plot_accuracy/plot_every)
                    curr_val_plot_accuracies.append(val_running_plot_accuracy/plot_every)
                    val_running_plot_accuracy = 0.0
                
                
                if validation_criterion is not None:
                    train_plot_accuracies.append(train_running_plot_accuracy/plot_every)
                    curr_train_plot_accuracies.append(train_running_plot_accuracy/plot_every)
                    train_running_plot_accuracy = 0.0
                    
            if save_file and save_folder and total_batch_number % save_every == 0:
                # Save losses
                with open(save_folder + "/" + save_file + "_train_loss.txt", "a") as file:
                    for val in curr_train_plot_losses:
                        file.write(str(val) + ",")
                curr_train_plot_losses = []
                # Save accuracies
                with open(save_folder + "/" + save_file + "_train_accuracy.txt", "a") as file:
                    for val in curr_train_plot_accuracies:
                        file.write(str(val) + ",")
                curr_train_plot_accuracies = []
                if validation_dset:
                    # Save losses
                    with open(save_folder + "/" + save_file + "_val_loss.txt", "a") as file:
                        for val in curr_val_plot_losses:
                            file.write(str(val) + ",")
                    curr_val_plot_losses = []
                    # Save accuracies
                    with open(save_folder + "/" + save_file + "_val_accuracy.txt", "a") as file:
                        for val in curr_val_plot_accuracies:
                            file.write(str(val) + ",")
                    curr_val_plot_accuracies = []
        print("just finished epoch " + str(epoch))            
    
        # Save model
#         if save_file and save_folder:
#             torch.save(model, save_folder + "/" + save_file + "_epoch_" + str(epoch) + "_model")

    print("all done!!!")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    model.train(False)
    return model, train_plot_losses, train_plot_accuracies



def test_model(model, dset_loader):
    """
    Tests a model on a given data set and returns the accuracy of the model
    on the set.
    """
    model.train(False)

    running_corrects = 0

    for inputs, labels in dset_loader:
        # wrap them in Variable
#         inputs, labels = Variable(inputs.cuda()), \
#                          Variable(labels.cuda())

        # forward
        outputs = model.forward_prediction(inputs)
        _, preds = torch.max(outputs.data, 1)

        running_corrects += torch.sum(preds == labels.data)

    return running_corrects/(len(dset_loader) * dset_loader.batch_size)

def test_model_tree_to_tree(model, dset_loader, metric, use_cuda=False):
    """
    Tests a model on a given data set and returns the accuracy of the model
    on the set.
    """
    model.train(False)
    model.eval()

    running_corrects = 0
    accuracies = []

    for inputs, labels in dset_loader:
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        
        # forward
        output = model.forward_prediction(inputs)
        accuracies.append(metric(output, labels))
        
    mean = np.mean(accuracies)

    return mean
