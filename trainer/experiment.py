import torch
import numpy as np
from trainer import inputs
from trainer import model


def train(args, model, dataloader, optimizer, device):
    """Create the training loop for one epoch.

    Args:
      model: The transformer model that you are training, based on
      nn.Module
      dataloader: The training dataset
      optimizer: The selected optmizer to update parameters and gradients
      device: device
    """
    model.train()
    for i, batch in enumerate(dataloader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            if i == 0 or i % args.log_every == 0 or i+1 == len(dataloader):
                print("Progress: {:3.0f}% - Batch: {:>4.0f}/{:<4.0f} - Loss: {:<.4f}".format(
                    100. * (1+i) / len(dataloader), # Progress
                    i+1, len(dataloader), # Batch
                    loss.item())) # Loss


def evaluate(model, dataloader, device):
    """Create the evaluation loop.
    Args:
      model: The transformer model that you are training, based on
      nn.Module
      dataloader: The development or testing dataset
      device: device
    """
    print("\nStarting evaluation...")
    model.eval()
    with torch.no_grad():
        eval_preds = []
        eval_labels = []

        for _, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            preds = model(input_ids, attention_mask=attention_mask, labels=labels)
            preds = preds[1].argmax(dim=-1)
            eval_preds.append(preds.cpu().numpy())
            eval_labels.append(batch['labels'].cpu().numpy())

    print("Done evaluation")
    return np.concatenate(eval_labels), np.concatenate(eval_preds)


def run(args):
    """Load the data, train, evaluate, and export the model for serving and
     evaluating.

    Args:
      args: experiment parameters.
    """
    cuda_availability = torch.cuda.is_available()
    if cuda_availability:
      device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
      device = 'cpu'
    print('\n*************************')
    print('`cuda` available: {}'.format(cuda_availability))
    print('Current Device: {}'.format(device))
    print('*************************\n')

    torch.manual_seed(args.seed)

    # Open our dataset
    train_loader, eval_loader, test_loader, counts = inputs.load_data(args)

    # Print counts
    print("Number of tokens removed:")
    print(f"- Training prem: {counts['train_prem_count']}")
    print(f"- Training hypo: {counts['train_hypo_count']}")
    print(f"- Dev prem: {counts['dev_prem_count']}")
    print(f"- Dev hypo: {counts['dev_hypo_count']}")
    print(f"- Test prem: {counts['test_prem_count']}")
    print(f"- Test hypo: {counts['test_hypo_count']}")

    # Create the model, loss function, and optimizer
    bert_model, optimizer = model.create(args, device)

    # Train / Test the model
    for epoch in range(1, args.epochs + 1):
        train(args, bert_model, train_loader, optimizer, device)
        dev_labels, dev_preds = evaluate(bert_model, eval_loader, device)
        # Print validation accuracy
        dev_accuracy = (dev_labels == dev_preds).mean()
        print("\nDev accuracy after epoch {}: {}".format(epoch, dev_accuracy))

    # Evaluate the model
    print("Evaluate the model using the testing dataset")
    test_labels, test_preds = evaluate(bert_model, test_loader, device)
    # Print validation accuracy
    test_accuracy = (test_labels == test_preds).mean()
    print("\nTest accuracy after epoch {}: {}".format(args.epochs, test_accuracy))

    # Export the trained model
    torch.save(bert_model.state_dict(), args.model_name)

    # Save the model to GCS
    if args.job_dir:
        inputs.save_model(args)
