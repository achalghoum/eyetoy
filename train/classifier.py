import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .datasets.loader import DATASETS
from models.encoder import DEFAULT_2D_ENCODER, Encoder2D
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import argparse
from torchvision.transforms.v2 import CutMix, MixUp, RandomChoice
from torch.utils.data.dataloader import default_collate

def compute_accuracy_from_distributions(outputs, targets):
    """
    Compute accuracy when targets are distributions over classes.

    Args:
        outputs (torch.Tensor): Model outputs (logits or probabilities), shape (batch_size, num_classes).
        targets (torch.Tensor): Target distributions, shape (batch_size, num_classes).

    Returns:
        accuracy (float): Accuracy as the proportion of correct predictions.
    """
    # Predicted classes from model outputs
    _, preds = torch.max(outputs, 1)

    # True classes from target distributions
    _, true_classes = torch.max(targets, 1)

    # Compare predictions with true classes
    correct = preds.eq(true_classes)

    return sum(correct)

BATCH_SIZE = 64
ACCUMULATION_STEPS = 1
MAX_GRADIENT = 1
TRAIN_TOTAL = 0

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
                             device: torch.device, weight_decay=1e-5, resume_from=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = OneCycleLR(optimizer, epochs=num_epochs, steps_per_epoch=len(train_loader), max_lr=learning_rate)

    # Initialize training state
    start_epoch = 0
    best_val_loss = float('inf')
    patience = 100
    counter = 0
    
    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        counter = checkpoint['early_stop_counter']

    # Calculate total steps for OneCycleLR
    total_steps = len(train_loader) * (num_epochs - start_epoch)

    writer = SummaryWriter()
    model.to(device)
    
    try:
        for epoch in range(start_epoch, num_epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            optimizer.zero_grad()
            model.zero_grad()
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                # Check for NaNs and Inf in the output

                batch_loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(labels).sum().item()
                train_correct += batch_correct
                train_loss += batch_loss.item()
                writer.add_scalar('Batch Loss/train', batch_loss, batch_idx+(len(train_loader)*(epoch)))
                writer.add_scalar('Batch Accuracy/train', (100*batch_correct)/len(labels), batch_idx+(len(train_loader)*(epoch)))
                writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], batch_idx+(len(train_loader)*epoch))

                batch_loss.backward()

                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    model.zero_grad()
                    scheduler.step()

            avg_train_loss = train_loss / (len(train_loader))
            train_accuracy = 100. * train_correct / TRAIN_TOTAL

            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            top5_correct = 0  # Initialize top-5 correct counter
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    # Calculate top-5 accuracy
                    top5_correct += (outputs.topk(5, dim=1)[1] == labels.view(-1, 1)).sum().item()

            avg_val_loss = val_loss / (len(val_loader))
            val_accuracy = 100. * val_correct / val_total
            top5_accuracy = 100. * top5_correct / val_total  # Calculate top-5 accuracy

            writer.add_scalar('Loss/train', avg_train_loss, epoch+1)
            writer.add_scalar('Loss/val', avg_val_loss, epoch+1)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch+1)
            writer.add_scalar('Accuracy/val', val_accuracy, epoch+1)
            writer.add_scalar('Top5 Accuracy/val', top5_accuracy, epoch+1)  # Log top-5 accuracy

            print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, "
                  f"Train Accuracy: {train_accuracy:.2f}%, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Val Accuracy: {val_accuracy:.2f}%, "
                  f"Top-5 Val Accuracy: {top5_accuracy:.2f}%")  # Print top-5 accuracy

            # Save checkpoint every epoch
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'early_stop_counter': counter,
            }
            torch.save(checkpoint, 'checkpoint.pth')


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

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'early_stop_counter': counter,
        }
        torch.save(checkpoint, 'interrupt_checkpoint.pth')
        print("Checkpoint saved. You can resume training using this checkpoint.")
        
    finally:
        writer.close()

# Example usage
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train Encoder2D Classifier')
    parser.add_argument('--dataset', type=str, required=True, 
                        choices=DATASETS.keys(), help='Name of the dataset to use')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, help='Epochs', default=30)
    parser.add_argument("--lr", type=float, help="Learning Rate", default=1e-3)
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=64)
    parser.add_argument("--weight_decay", type=float, help="Weighz Decay", default=1e-5)
    args = parser.parse_args()
    BATCH_SIZE= args.batch_size
    # Load the specified dataset
    train_dataset, val_dataset = DATASETS[args.dataset]
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    TRAIN_TOTAL = len(train_dataset)
    # Create the model
    encoder = DEFAULT_2D_ENCODER
    num_classes = train_dataset.num_classes
    model = Encoder2DClassifier(encoder, num_classes)

    # Train the model
    device = torch.device("cuda")
    weight_decay = args.weight_decay 
    epochs = args.epochs
    learning_rate = args.lr


    train_encoder_classifier(model, train_loader, val_loader, num_epochs=epochs,
                             learning_rate=learning_rate, device=device, weight_decay=weight_decay,
                             resume_from=args.resume)