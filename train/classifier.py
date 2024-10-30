import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .datasets.loader import caltech_256_train, caltech_256_val
from models.encoder import DEFAULT_2D_ENCODER, Encoder2D
from torch.optim.lr_scheduler import ReduceLROnPlateau
BATCH_SIZE = 32
ACCUMULATION_STEPS = 1
MAX_GRADIENT = 1
TRAIN_TOTAL = len(caltech_256_train)

class Encoder2DClassifier(nn.Module):
    def __init__(self, encoder: Encoder2D, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        self.encoder = encoder
        context_token_dim = encoder.d_model
        self.classifier = nn.Linear(context_token_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        o,x = self.encoder(x)
        output = self.classifier(x).squeeze(1)
        return output

# Training function
def train_encoder_classifier(model:Encoder2DClassifier, train_loader: DataLoader,
                             val_loader: DataLoader, num_epochs: int, learning_rate: float,
                             device: torch.device, weight_decay=1e-5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    writer = SummaryWriter()
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        optimizer.zero_grad()
        model.zero_grad()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            batch_loss = criterion(outputs, labels) / ACCUMULATION_STEPS
            _, predicted = outputs.max(1)
            batch_correct = predicted.eq(labels).sum().item()
            train_correct += batch_correct
            train_loss += batch_loss.item()

            batch_loss.backward()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRADIENT)
                optimizer.step()
                model.zero_grad()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100. * train_correct / TRAIN_TOTAL

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100. * val_correct / val_total
        scheduler.step(avg_val_loss)

        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.2f}%")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    writer.close()
# Example usage
if __name__ == "__main__":
    # Set up data loaders (you'll need to adjust this based on your dataset)

    train_loader = DataLoader(caltech_256_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(caltech_256_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    # Create the model
    encoder = DEFAULT_2D_ENCODER
    num_classes = len(caltech_256_train.classes)
    model = Encoder2DClassifier(encoder, num_classes)

    # Train the model
    device = torch.device("cuda")
    weight_decay = 1e-5  # Adjust this value as needed
    train_encoder_classifier(model, train_loader, val_loader, num_epochs=30,
                             learning_rate=1e-4, device=device, weight_decay=weight_decay)