from typing import Union
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from .datasets.loader import caltech_256_train, caltech_256_val
from models.encoder import DEFAULT_2D_ENCODER, Encoder2D
from torch.optim.lr_scheduler import ReduceLROnPlateau
from  torch.cuda.amp import autocast

torch.set_default_tensor_type(torch.HalfTensor)
torch.cuda.set_device(0)  # Set the default GPU device

class Encoder2DClassifier(nn.Module):
    def __init__(self, encoder: Encoder2D, num_classes: int, dropout_rate: float = 0.5):
        super().__init__()
        self.encoder = encoder
        context_token_dim = encoder.d_model
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(context_token_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        _,x = self.encoder(x)
        output = self.classifier(x).squeeze(1)
        return output

# Training function
def train_encoder_classifier(model:Encoder2DClassifier, train_loader: DataLoader,
                             val_loader: DataLoader, num_epochs: int, learning_rate: float, 
                             device: torch.device, weight_decay=1e-5):
    with autocast():
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
        best_val_loss = float('inf')
        patience = 5
        counter = 0
        model.to(device)
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            optimizer.zero_grad()
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)

                batch_loss.backward()

                # Gradient monitoring
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if grad_norm > 1.0:
                            print(f"Epoch {epoch}, Batch {batch_idx}: Large gradient in {name}: {grad_norm}")
                        elif grad_norm < 1e-4:
                            print(f"Epoch {epoch}, Batch {batch_idx}: Small gradient in {name}: {grad_norm}")
                optimizer.step()
                model.zero_grad()


                # Keep only the most recent values

                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {batch_loss.item():.4f}, ")

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
    val_loader = DataLoader(caltech_256_val, batch_size=16, shuffle=False, num_workers=1)
    
    # Create the model
    encoder = DEFAULT_2D_ENCODER
    num_classes = len(caltech_256_train.classes)
    model = Encoder2DClassifier(encoder, num_classes)
    
    # Train the model
    device = torch.device("cuda")
    weight_decay = 1e-4  # Adjust this value as needed
    train_encoder_classifier(model, train_loader, val_loader, num_epochs=10, 
                             learning_rate=1e-3, device=device, weight_decay=weight_decay)
