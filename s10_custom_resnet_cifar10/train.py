import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
from torchsummary import summary

def train_custom_resnet(model, optimizer, criterion, train_loader, test_loader, num_epochs, max_at_epoch, lr_max, device):
    writer = SummaryWriter()

    summary(model, input_size=(3, 32, 32))

    scheduler = OneCycleLR(optimizer, max_lr=lr_max, steps_per_epoch=len(train_loader), epochs=num_epochs, pct_start=max_at_epoch / num_epochs, anneal_strategy='linear', div_factor=10)

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        train_accuracy = 100 * train_correct / train_total
        writer.add_scalar("train_loss", train_loss / len(train_loader), epoch)
        writer.add_scalar("train_accuracy", train_accuracy, epoch)

        # Evaluation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += targets.size(0)
                test_correct += (predicted == targets).sum().item()

        test_accuracy = 100 * test_correct / test_total
        writer.add_scalar("test_loss", test_loss / len(test_loader), epoch)
        writer.add_scalar("test_accuracy", test_accuracy, epoch)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_accuracy:.2f}%')

        if test_accuracy >= 90:
            print('Target accuracy reached!')
            break

    writer.close()