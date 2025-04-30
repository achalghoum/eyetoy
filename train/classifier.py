import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from .datasets.loader import DATASETS
from models.encoder import DEFAULT_2D_ENCODER, Encoder2D
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import argparse
from torchvision.transforms.v2 import CutMix, MixUp, RandomChoice
from torch.utils.data.dataloader import default_collate
import sys


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
        o, x = self.encoder(x)
        output = self.classifier(x).squeeze(1)
        return output


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_encoder_classifier(
    model: Encoder2DClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    rank: int,
    world_size: int,
    weight_decay=1e-5,
    resume_from=None,
):
    # Move model to device first
    model = model.to(device)
    
    # Wrap model with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    
    # Initialize training state
    start_epoch = 0
    best_val_loss = float("inf")
    patience = 100
    counter = 0
    
    # Setup function for scheduler to make it easy to recreate after resume
    def create_scheduler(optimizer, num_epochs, start_epoch, train_loader):
        return OneCycleLR(
            optimizer,
            epochs=num_epochs - start_epoch,
            steps_per_epoch=len(train_loader),
            max_lr=learning_rate,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=10000.0,
        )
    
    # Initial scheduler creation
    scheduler = create_scheduler(optimizer, num_epochs, start_epoch, train_loader)

    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        if world_size > 1:
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Fix optimizer state device mismatch
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        
        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["best_val_loss"]
        counter = checkpoint["early_stop_counter"]
        
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            scheduler = create_scheduler(optimizer, num_epochs, start_epoch, train_loader)
        
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if "numpy_rng_state" in checkpoint and "numpy" in sys.modules:
            import numpy as np
            np.random.set_state(checkpoint["numpy_rng_state"])

    # Calculate total steps for OneCycleLR
    total_steps = len(train_loader) * (num_epochs - start_epoch)

    # Only create SummaryWriter on rank 0
    writer = SummaryWriter() if rank == 0 else None

    try:
        for epoch in range(start_epoch, num_epochs):
            # Set epoch for distributed sampler
            if world_size > 1:
                train_loader.sampler.set_epoch(epoch)
            
            model.train()
            train_loss = 0
            train_correct = 0
            optimizer.zero_grad()
            model.zero_grad()

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(labels).sum().item()
                train_correct += batch_correct
                train_loss += batch_loss.item()
                
                if rank == 0:
                    writer.add_scalar(
                        "Batch Loss/train",
                        batch_loss,
                        batch_idx + (len(train_loader) * (epoch)),
                    )
                    writer.add_scalar(
                        "Batch Accuracy/train",
                        (100 * batch_correct) / len(labels),
                        batch_idx + (len(train_loader) * (epoch)),
                    )
                    writer.add_scalar(
                        "Learning Rate",
                        optimizer.param_groups[0]["lr"],
                        batch_idx + (len(train_loader) * epoch),
                    )

                batch_loss = batch_loss / ACCUMULATION_STEPS
                batch_loss.backward()

                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    optimizer.step()
                    model.zero_grad()
                    scheduler.step()

            # Synchronize metrics across all processes
            if world_size > 1:
                dist.all_reduce(torch.tensor(train_loss).to(device))
                dist.all_reduce(torch.tensor(train_correct).to(device))
                train_loss = train_loss / world_size
                train_correct = train_correct / world_size

            avg_train_loss = train_loss / (len(train_loader))
            train_accuracy = 100.0 * train_correct / TRAIN_TOTAL

            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            top5_correct = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    top5_correct += (outputs.topk(5, dim=1)[1] == labels.view(-1, 1)).sum().item()

            # Synchronize validation metrics
            if world_size > 1:
                dist.all_reduce(torch.tensor(val_loss).to(device))
                dist.all_reduce(torch.tensor(val_correct).to(device))
                dist.all_reduce(torch.tensor(val_total).to(device))
                dist.all_reduce(torch.tensor(top5_correct).to(device))
                val_loss = val_loss / world_size
                val_correct = val_correct / world_size
                val_total = val_total / world_size
                top5_correct = top5_correct / world_size

            avg_val_loss = val_loss / (len(val_loader))
            val_accuracy = 100.0 * val_correct / val_total
            top5_accuracy = 100.0 * top5_correct / val_total

            if rank == 0:
                writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
                writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)
                writer.add_scalar("Accuracy/train", train_accuracy, epoch + 1)
                writer.add_scalar("Accuracy/val", val_accuracy, epoch + 1)
                writer.add_scalar("Top5 Accuracy/val", top5_accuracy, epoch + 1)

                print(
                    f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, "
                    f"Train Accuracy: {train_accuracy:.2f}%, "
                    f"Val Loss: {avg_val_loss:.4f}, "
                    f"Val Accuracy: {val_accuracy:.2f}%, "
                    f"Top-5 Val Accuracy: {top5_accuracy:.2f}%"
                )

                # Save checkpoint every epoch
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.module.state_dict() if world_size > 1 else model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "early_stop_counter": counter,
                    "torch_rng_state": torch.get_rng_state(),
                }
                
                if "numpy" in sys.modules:
                    import numpy as np
                    checkpoint["numpy_rng_state"] = np.random.get_state()
                    
                torch.save(checkpoint, "checkpoint.pth")

                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    counter = 0
                    torch.save(model.module.state_dict() if world_size > 1 else model.state_dict(), "best_model.pth")
                else:
                    counter += 1
                    if counter >= patience:
                        print(f"Early stopping triggered after {epoch + 1} epochs")
                        break

    except KeyboardInterrupt:
        if rank == 0:
            print("\nTraining interrupted. Saving checkpoint...")
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.module.state_dict() if world_size > 1 else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "early_stop_counter": counter,
                "torch_rng_state": torch.get_rng_state(),
            }
            
            if "numpy" in sys.modules:
                import numpy as np
                checkpoint["numpy_rng_state"] = np.random.get_state()
            
            torch.save(checkpoint, "interrupt_checkpoint.pth")
            print("Checkpoint saved. You can resume training using this checkpoint.")

    finally:
        if rank == 0 and writer is not None:
            writer.close()

def train(rank, world_size, args):
    setup(rank, world_size)
    
    # Load the specified dataset
    train_dataset_func, val_dataset_func = DATASETS[args.dataset]
    train_dataset = train_dataset_func()
    val_dataset = val_dataset_func()
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=1,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=1,
        pin_memory=True
    )
    
    global TRAIN_TOTAL
    TRAIN_TOTAL = len(train_dataset)
    
    # Create the model
    encoder = DEFAULT_2D_ENCODER
    num_classes = train_dataset.num_classes
    model = Encoder2DClassifier(encoder, num_classes)
    
    # Train the model
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    weight_decay = args.weight_decay
    epochs = args.epochs
    learning_rate = args.lr

    train_encoder_classifier(
        model,
        train_loader,
        val_loader,
        num_epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        rank=rank,
        world_size=world_size,
        weight_decay=weight_decay,
        resume_from=args.resume,
    )
    
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Encoder2D Classifier")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=DATASETS.keys(),
        help="Name of the dataset to use",
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, help="Epochs", default=30)
    parser.add_argument("--lr", type=float, help="Learning Rate", default=1e-3)
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=64)
    parser.add_argument("--weight_decay", type=float, help="Weight Decay", default=1e-5)
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(
            train,
            args=(world_size, args),
            nprocs=world_size,
            join=True
        )
    else:
        train(0, 1, args)
