import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from torch.utils.tensorboard.writer import SummaryWriter
from .datasets.loader import DATASETS
from models.encoder import DEFAULT_2D_ENCODER, Encoder2D
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import argparse
import time
import traceback
import sys
from datetime import timedelta

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
    
    # Set timeout to a large value to prevent timeouts during dataset loading
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    
    # Initialize the process group with a timeout
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=30))


def cleanup():
    dist.destroy_process_group()


def train_encoder_classifier(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    device,
    rank,
    world_size,
    weight_decay=1e-5,
    resume_from=None,
):
    # Move model to device first
    model = model.to(device)
    
    # Wrap model with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)
        if rank == 0:
            print(f"Model wrapped with DDP, using {world_size} GPUs")
    
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

    # Only create SummaryWriter on rank 0
    writer = SummaryWriter() if rank == 0 else None

    try:
        for epoch in range(start_epoch, num_epochs):
            # Set epoch for distributed sampler
            if world_size > 1:
                if hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)
            
            model.train()
            train_loss = 0
            train_correct = 0
            optimizer.zero_grad()
            model.zero_grad()

            # Use a separate variable to track batch progress
            total_batches = len(train_loader)
            
            if rank == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Starting training with {total_batches} batches")
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if rank == 0:
                    print(f"Rank {rank}: Starting Batch {batch_idx}")

                if rank == 0 and batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}/{total_batches}")
                
                # Move data to device
                inputs, labels = inputs.to(device), labels.to(device)
                if rank == 0:
                    print(f"Rank {rank}: Data moved to device for Batch {batch_idx}")
                
                # Forward pass
                if rank == 0:
                    print(f"Rank {rank}: Starting forward pass for Batch {batch_idx}")
                outputs = model(inputs)
                if rank == 0:
                    print(f"Rank {rank}: Finished forward pass for Batch {batch_idx}")

                batch_loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(labels).sum().item()
                
                # Track metrics locally
                train_correct += batch_correct
                train_loss += batch_loss.item()
                
                # Log batch metrics if rank 0
                if rank == 0 and writer is not None:
                    writer.add_scalar(
                        "Batch Loss/train",
                        batch_loss.item(),
                        batch_idx + (len(train_loader) * epoch),
                    )
                    writer.add_scalar(
                        "Batch Accuracy/train",
                        (100 * batch_correct) / len(labels),
                        batch_idx + (len(train_loader) * epoch),
                    )
                    writer.add_scalar(
                        "Learning Rate",
                        optimizer.param_groups[0]["lr"],
                        batch_idx + (len(train_loader) * epoch),
                    )

                # Accumulate gradients and optimize
                batch_loss = batch_loss / ACCUMULATION_STEPS
                batch_loss.backward()

                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRADIENT)
                    optimizer.step()
                    model.zero_grad()
                    scheduler.step()
                
                # Synchronize processes after each batch to prevent hanging
                if world_size > 1:
                    dist.barrier()

            # Make sure all processes finish training before validation
            if world_size > 1:
                # Convert metrics to tensors for all_reduce
                train_loss_tensor = torch.tensor(train_loss).to(device)
                train_correct_tensor = torch.tensor(train_correct).to(device)
                
                # Synchronize metrics across processes
                dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)
                
                # Get the reduced values
                train_loss = train_loss_tensor.item() / world_size
                train_correct = train_correct_tensor.item() / world_size

            # Compute average metrics
            avg_train_loss = train_loss / len(train_loader)
            train_total = TRAIN_TOTAL if TRAIN_TOTAL > 0 else len(train_loader) * BATCH_SIZE
            train_accuracy = 100.0 * train_correct / train_total

            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            top5_correct = 0
            
            if rank == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Starting validation")
            
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
                # Create tensors for reduction
                val_loss_tensor = torch.tensor(val_loss).to(device)
                val_correct_tensor = torch.tensor(val_correct).to(device)
                val_total_tensor = torch.tensor(val_total).to(device)
                top5_correct_tensor = torch.tensor(top5_correct).to(device)
                
                # Reduce metrics across processes
                dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_correct_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(val_total_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(top5_correct_tensor, op=dist.ReduceOp.SUM)
                
                # Get reduced values
                val_loss = val_loss_tensor.item() / world_size
                val_correct = val_correct_tensor.item() / world_size
                val_total = val_total_tensor.item() / world_size
                top5_correct = top5_correct_tensor.item() / world_size

            # Compute validation metrics
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100.0 * val_correct / val_total
            top5_accuracy = 100.0 * top5_correct / val_total

            # Log metrics on rank 0
            if rank == 0:
                if writer is not None:
                    writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
                    writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)
                    writer.add_scalar("Accuracy/train", train_accuracy, epoch + 1)
                    writer.add_scalar("Accuracy/val", val_accuracy, epoch + 1)
                    writer.add_scalar("Top5 Accuracy/val", top5_accuracy, epoch + 1)

                print(
                    f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, "
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
            
            # Make sure all processes are synchronized before next epoch
            if world_size > 1:
                dist.barrier()

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
    
    except Exception as e:
        if rank == 0:
            print(f"Error during training: {str(e)}")
            traceback.print_exc()

    finally:
        if rank == 0 and writer is not None:
            writer.close()


def train(rank, world_size, args):
    try:
        setup(rank, world_size)
        
        # Load the specified dataset with error handling
        if rank == 0:
            print(f"Loading dataset: {args.dataset}")
            
        # Retry mechanism for dataset loading
        max_retries = 3
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                train_dataset_func, val_dataset_func = DATASETS[args.dataset]
                
                if rank == 0:
                    print(f"Initializing training dataset (attempt {attempt+1}/{max_retries})...")
                train_dataset = train_dataset_func()
                
                if rank == 0:
                    print(f"Initializing validation dataset (attempt {attempt+1}/{max_retries})...")
                val_dataset = val_dataset_func()
                
                # If we got here, dataset loading succeeded
                break
            except Exception as e:
                if rank == 0:
                    print(f"Error loading dataset (attempt {attempt+1}/{max_retries}): {str(e)}")
                    traceback.print_exc()
                
                if attempt < max_retries - 1:
                    if rank == 0:
                        print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    if rank == 0:
                        print("Failed to load dataset after multiple attempts. Exiting.")
                    cleanup()
                    return
        
        # Make sure all processes get past the dataset loading
        if world_size > 1:
            dist.barrier()
        
        # Safely get dataset properties
        if not hasattr(train_dataset, 'num_classes'):
            if rank == 0:
                print("Warning: Dataset doesn't have num_classes attribute. Attempting to determine from data...")
            # Try to determine num_classes from the dataset
            try:
                # Get a sample and check its label dimension
                sample_data = train_dataset[0]
                if isinstance(sample_data, tuple) and len(sample_data) >= 2:
                    if isinstance(sample_data[1], int):
                        num_classes = max(sample_data[1], 1) + 1  # Assuming 0-indexed classes
                        if rank == 0:
                            print(f"Detected {num_classes} classes")
                    elif isinstance(sample_data[1], torch.Tensor) and sample_data[1].dim() == 1:
                        num_classes = sample_data[1].size(0)
                        if rank == 0:
                            print(f"Detected {num_classes} classes from tensor dimension")
                    else:
                        num_classes = 1000  # Fallback to a common value
                        if rank == 0:
                            print(f"Using default value of {num_classes} classes")
                else:
                    num_classes = 1000  # Fallback to a common value
                    if rank == 0:
                        print(f"Using default value of {num_classes} classes")
            except:
                num_classes = 1000  # Fallback to a common value
                if rank == 0:
                    print(f"Failed to determine num_classes. Using default value of {num_classes}")
        else:
            num_classes = train_dataset.num_classes
        
        # Synchronize num_classes across all processes
        if world_size > 1:
            num_classes_tensor = torch.tensor(num_classes, device=f"cuda:{rank}")
            dist.broadcast(num_classes_tensor, src=0)
            num_classes = num_classes_tensor.item()
        
        # Create distributed samplers with proper error handling
        try:
            if rank == 0:
                print("Creating data samplers...")
                
            train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
            val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
        except Exception as e:
            if rank == 0:
                print(f"Error creating distributed samplers: {str(e)}")
                print("Falling back to non-distributed sampling")
            train_sampler = None
            val_sampler = None
        
        # Wait for samplers to be created
        if world_size > 1:
            dist.barrier()
        
        # Create safe data loaders with error handling
        try:
            if rank == 0:
                print("Creating data loaders...")
            
            # Reduce batch size if memory is a concern
            effective_batch_size = args.batch_size
            
            # DataLoader kwargs that are conditional
            num_workers = 4 # <-- CHANGE HERE: Start with 4 workers per DataLoader
            dataloader_kwargs = {
                'batch_size': effective_batch_size,
                'pin_memory': True, # Keep pin_memory=True for faster CPU->GPU transfer
                'num_workers': num_workers,
            }
            # Only add timeout when using workers
            if num_workers > 0:
                 dataloader_kwargs['timeout'] = 120 # Increase timeout slightly
            
            train_loader = DataLoader(
                train_dataset,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                drop_last=True,  # Prevent issues with uneven batch sizes
                **dataloader_kwargs
            )
            
            val_loader = DataLoader(
                val_dataset,
                shuffle=False,
                sampler=val_sampler,
                drop_last=False,  # Keep all validation samples
                **dataloader_kwargs
            )
            
            # Update global variable safely
            if rank == 0:
                global TRAIN_TOTAL
                try:
                    TRAIN_TOTAL = len(train_dataset)
                except:
                    TRAIN_TOTAL = effective_batch_size * len(train_loader)
                    if rank == 0:
                        print(f"Couldn't determine dataset size, using batch_size * num_batches: {TRAIN_TOTAL}")
                
            # Wait for data loaders to be created
            if world_size > 1:
                dist.barrier()
                
        except Exception as e:
            if rank == 0:
                print(f"Error creating data loaders: {str(e)}")
                traceback.print_exc()
            cleanup()
            return
        
        try:
            # Create the model
            if rank == 0:
                print("Creating model...")
            encoder = DEFAULT_2D_ENCODER
            model = Encoder2DClassifier(encoder, num_classes)
            
            # Train the model
            device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
            weight_decay = args.weight_decay
            epochs = args.epochs
            learning_rate = args.lr

            # Adjust learning rate based on world_size for stability
            if world_size > 1:
                learning_rate = learning_rate * world_size
                if rank == 0:
                    print(f"Adjusting learning rate for distributed training: {learning_rate}")

            if rank == 0:
                print(f"Starting training on device: {device}")
                
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
        except Exception as e:
            if rank == 0:
                print(f"Error during model creation or training: {str(e)}")
                traceback.print_exc()
    except Exception as e:
        if rank == 0:
            print(f"Unhandled exception in train function: {str(e)}")
            traceback.print_exc()
    finally:
        try:
            cleanup()
        except:
            pass


if __name__ == "__main__":
    # Import datetime here to avoid circular imports
    from datetime import timedelta
    
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
    
    # Set NCCL environment variables for better reliability
    os.environ['NCCL_DEBUG'] = 'INFO'  # Set to INFO for debugging, WARN for production
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['NCCL_IB_TIMEOUT'] = '30'  # 30 second timeout instead of default
    
    # Suppress P2P disabled warnings
    os.environ['NCCL_IGNORE_DISABLED_P2P'] = '1'
    
    world_size = torch.cuda.device_count()
    if world_size > 1:
        try:
            # Try with spawn first
            print(f"Starting distributed training with {world_size} GPUs")
            mp.spawn(
                train,
                args=(world_size, args),
                nprocs=world_size,
                join=True
            )
        except Exception as e:
            print(f"Error in distributed training: {str(e)}")
            traceback.print_exc()
            print("Falling back to single GPU training...")
            train(0, 1, args)
    else:
        print("Using single GPU or CPU for training")
        train(0, 1, args)
