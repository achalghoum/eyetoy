from typing import Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .datasets.loader import caltech_256_train, caltech_256_val
from models.encoder import DEFAULT_2D_ENCODER, DEFAULT_SIMPLIFIED_2D_ENCODER, Encoder2D, SimplifiedEncoder2D
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.utils as utils

class Encoder2DClassifier(nn.Module):
    def __init__(self, encoder: Union[Encoder2D,SimplifiedEncoder2D], num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        self.encoder = encoder
        context_token_dim = encoder.global_attention.d_model
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(context_token_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        _,x = self.encoder(x)
        output = self.classifier(x).squeeze(1)
        return output

# Training function
def train_encoder_classifier(model: Union[Encoder2DClassifier,SimplifiedEncoder2D], train_loader: DataLoader, 
                             val_loader: DataLoader, num_epochs: int, learning_rate: float, 
                             device: torch.device, weight_decay=1e-5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

    best_val_loss = float('inf')
    patience = 5
    counter = 0
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        loss_list = []
        correct_list = []
        total_list = []
        window_size = 10  # Adjust this value as needed
        optimizer.zero_grad()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient monitoring
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if grad_norm > 1.0:
                        print(f"Epoch {epoch}, Batch {batch_idx}: Large gradient in {name}: {grad_norm}")
                    elif grad_norm < 1e-4:
                        print(f"Epoch {epoch}, Batch {batch_idx}: Small gradient in {name}: {grad_norm}")

            optimizer.step()
            loss_list.append(loss.item())
            _, predicted = outputs.max(1)
            batch_correct = predicted.eq(labels).sum().item()
            batch_total = labels.size(0)
            correct_list.append(batch_correct)
            total_list.append(batch_total)
            
            # Keep only the most recent values
            loss_list = loss_list[-window_size:]
            correct_list = correct_list[-window_size:]
            total_list = total_list[-window_size:]
            
            # Calculate moving averages
            moving_loss = sum(loss_list) / len(loss_list)
            moving_correct = sum(correct_list)
            moving_total = sum(total_list)
            moving_accuracy = 100. * moving_correct / moving_total if moving_total > 0 else 0.0

            print(f"Epoch {epoch+1}, Batch {batch_idx}, Moving Loss: {moving_loss:.4f}, "
                  f"Moving Accuracy: {moving_accuracy:.2f}%")

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
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1} - Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {100.*train_correct/train_total:.2f}%, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val Accuracy: {100.*val_correct/val_total:.2f}%")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

# Example usage
if __name__ == "__main__":
    # Set up data loaders (you'll need to adjust this based on your dataset)
        
    train_loader = DataLoader(caltech_256_train, batch_size=16, shuffle=True, num_workers=1)
    val_loader = DataLoader(caltech_256_val, batch_size=32, shuffle=False, num_workers=1)
    
    # Create the model
    encoder = DEFAULT_2D_ENCODER
    num_classes = len(caltech_256_train.classes)
    model = Encoder2DClassifier(encoder, num_classes)
    
    # Train the model
    device = torch.device("cuda")
    weight_decay = 1e-4  # Adjust this value as needed
    train_encoder_classifier(model, train_loader, val_loader, num_epochs=10, 
                             learning_rate=1e-3, device=device, weight_decay=weight_decay)
