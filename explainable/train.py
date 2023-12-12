import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import os
from ExplainableNode import ExplainableNode  # Import ExplainableNode
import wandb

scaler = torch.cuda.amp.GradScaler()

def train(model, dataloader, optimizer, criterion, scaler=scaler, explainable_node=None, device='cuda'):
    model.train()
    running_loss = 0
    correct, total = 0, 0

    # if explainable_node is not None:
    #
    #     explainable_node.register_hooks(model, [model.layer1, model.layer2, model.layer3, model.layer4],
    #                                            ['layer1', 'layer2', 'layer3', 'layer4', 'layer5'])

    for inputs, targets in tqdm(dataloader, desc='Training', leave=False):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        if explainable_node is not None:
            explainable_node.calculate_metrics()
            explainable_node.clear_values()

    accuracy = correct / total

    if explainable_node is not None:
        aggregated_epoch_metrics = explainable_node.get_aggregated_epoch_metrics()
        print(f"Aggregated Epoch Metrics: {aggregated_epoch_metrics}")
        explainable_node.clear_values()  # Clear for next epoch

    return running_loss / len(dataloader), accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Validation'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return running_loss / len(dataloader), correct / total

def test(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='Testing'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    return correct / total



def experiment(model, train_loader, val_loader, optimizer, scheduler, criterion,
               config, device='cuda', explainable_node=None, test_loader=None):


    # wandb.login(key="your_key")
    # run = wandb.init(
    #     name="Resnet-Update",
    #     reinit=True,
    #     # id = '',
    #     # resume = "must",
    #     project="ExplainableNode",
    #     config=config
    # )

    # path = './'

    best_val_acc = 0

    for epoch in range(config['epochs']):
        if explainable_node:
            explainable_node.clear_values()

        train_loss, train_acc = train(model, train_loader, optimizer, criterion, explainable_node=explainable_node)
        print(
            f'Epoch [{epoch + 1}/{config["epochs"]}] - Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.4f}')

        if val_loader is not None:
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            print(
                f'Epoch [{epoch + 1}/{config["epochs"]}] - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

            # Update the best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc

            # Step the scheduler based on the validation loss
            if scheduler is not None:
                scheduler.step(val_loss)

            # curr_lr = float(optimizer.param_groups[0]['lr'])
            # wandb.log({"train_loss": train_loss, 'train_Acc': train_acc, 'validation_Acc': val_acc,
            #             'validation_loss': val_loss, "learning_Rate": curr_lr})

            # torch.save(
            #     {'model_state_dict': model.state_dict(),
            #      'optimizer_state_dict': optimizer.state_dict(),
            #      'scheduler_state_dict': scheduler.state_dict(),
            #      metric[0]: metric[1],
            #      'epoch': epoch},
            #     path
            # )
    if test_loader is not None:
        test_acc =  test(model, test_loader, device)
        print(
            f'Test Acc: {test_acc:.4f}')

    # run.finish()
    explainable_node.epoch_metrics.clear()
    return best_val_acc
