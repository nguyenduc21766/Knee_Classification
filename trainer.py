from models import MyModel
from args import get_args
import torch.nn as nn
import torch.optim as optim


def train_model(model, train_loader, val_loader):
    args = get_args()

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epoch):
        train_loss = 0
        # starting the training -> setting the model to training mode
        model.train()
        for batch in train_loader:
            inputs = batch['image']
            targets = batch['label']

            # resetting the gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        print("Epoch-{}: {}".format(epoch+1, train_loss/len(train_loader)))

        val_loss = validate_model(model, val_loader, criterion)
        print("Validation loss: {}".format(val_loss))

def validate_model(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    for batch in val_loader:
        inputs = batch['image']
        targets = batch['label']

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss.item()

    return val_loss/len(val_loader)