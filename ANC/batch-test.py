

# Test a bunch of times
num_trials = 20

num_original_convergences = 0
num_0_losses = 0
num_better_convergences = 0
otherPrograms = []

num_examples = 100

for i in range(num_trials):
    print("Trial ", i)
    
    M = 5
    dataset = AccessTaskDataset(M, num_examples)
    data_loader = data.DataLoader(dataset, batch_size = 1) # Don't change this batch size.  You have been warned.

    controller = Controller(first_arg = first_arg, 
                        second_arg = second_arg, 
                        output = target, 
                        instruction = instruction, 
                        initial_registers = init_registers, 
                        stop_threshold = .9, 
                        multiplier = 2,
                        correctness_weight = 1, 
                        halting_weight = 5, 
                        efficiency_weight = 0.1, 
                        confidence_weight = 0.5, 
                        t_max = 50) 
    
    best_model, train_plot_losses, validation_plot_losses = training.train_model_anc(
        controller, 
        data_loader,  
        optimizer, 
        num_epochs = 15, 
        print_every = 5, 
        plot_every = plot_every, 
        deep_copy_desired = False, 
        validation_criterion = anc_validation_criterion, 
        batch_size = 1) # In the paper, they used batch sizes of 1 or 5
    
    percent_orig = compareOutput()
    if percent_orig > .99:
        num_original_convergences += 1
    end_losses = validation_plot_losses[-2:]
    if sum(end_losses) < .01:
        num_0_losses += 1
    if percent_orig < .99 and sum(end_losses) < .01:
        num_better_convergences += 1
        otherPrograms.append((controller.output, controller.instruction, controller.first_arg, controller.second_arg, controller.registers))
print("LOSS CONVERGENCES", num_0_losses * 1.0 / num_trials)
print("ORIG CONVERGENCES", num_original_convergences * 1.0 / num_trials)
print("BETTER CONVERGENCES", num_better_convergences * 1.0 / num_trials)

# penguin