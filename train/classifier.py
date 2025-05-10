import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard.writer import SummaryWriter
from .datasets.loader import DATASETS
from models.encoder import DEFAULT_2D_ENCODER, Encoder2D
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import argparse
import time
import traceback
import sys
from datetime import timedelta
from typing import cast
from functools import partial
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

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
        self.classifier_head = nn.Linear(context_token_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier_head.weight)
        nn.init.zeros_(self.classifier_head.bias)

    def forward(self, x):
        o, x = self.encoder(x)
        output = self.classifier_head(x).squeeze(1)
        return output


def setup_distributed():
    """Initializes the distributed environment."""
    if not dist.is_available() or not torch.cuda.is_available() or torch.cuda.device_count() <= 1:
        print("Distributed training not available or not required.")
        return 0, 0, 1 # local_rank 0, global rank 0, World Size 1

    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        print("RANK or WORLD_SIZE not set, assuming single process.")
        return 0, 0, 1 # local_rank 0, global rank 0, World Size 1

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank)) # Get local rank if set

    print(f"Initializing process group: Rank {rank}/{world_size}, Local Rank {local_rank}")

    # MASTER_ADDR and MASTER_PORT should be set by torchrun/launch
    if 'MASTER_ADDR' not in os.environ:
         os.environ['MASTER_ADDR'] = 'localhost' # Default for single node
    if 'MASTER_PORT' not in os.environ:
         os.environ['MASTER_PORT'] = '12355' # Default port

    # Set NCCL options for better stability
    os.environ['NCCL_DEBUG'] = os.environ.get('NCCL_DEBUG', 'WARN') # Default to WARN
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['NCCL_IB_TIMEOUT'] = '30'
    os.environ['NCCL_SOCKET_TIMEOUT'] = '300' # Added longer socket timeout
    os.environ['NCCL_IGNORE_DISABLED_P2P'] = '1'
    
    # Enable timeout detection
    timeout = timedelta(minutes=60) # Increased timeout for large models

    # Initialize the process group with error checking
    try:
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=timeout
        )
    except Exception as e:
        print(f"Error initializing process group: {e}")
        # Try alternative backends if NCCL fails
        try:
            print("Retrying with Gloo backend")
            dist.init_process_group(
                backend="gloo",
                rank=rank,
                world_size=world_size,
                timeout=timeout
            )
        except Exception as e2:
            print(f"Error initializing with alternative backend: {e2}")
            print("Running in non-distributed mode")
            return 0, 0, 1

    # Set the device for this process
    torch.cuda.set_device(local_rank)
    print(f"Rank {rank} initialization complete. Using device cuda:{local_rank}")
    
    # Make sure the process group is working before proceeding
    try:
        dist.barrier()
        print(f"Rank {rank}: Process group barrier test successful")
    except Exception as e:
        print(f"Rank {rank}: Process group barrier test failed: {e}")
        cleanup_distributed()
        return 0, 0, 1

    return local_rank, rank, world_size

def cleanup_distributed():
    """Cleans up the distributed environment."""
    if dist.is_initialized():
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
    # Move model to device before wrapping
    model = model.to(device)
    local_rank = device.index if device.type == 'cuda' else -1

    # --- Use regular DDP instead of FSDP ---
    if world_size > 1:
        if rank == 0: 
            print("Wrapping model with DDP instead of FSDP for better stability")
        # Make sure all processes are synchronized before DDP wrap
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dist.barrier()
        
        # Wrap with DDP instead of FSDP - much more reliable
        model = DDP(model, device_ids=[local_rank] if local_rank != -1 else None)
        
        if rank == 0:
            print(f"DDP Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    # --- End DDP Configuration ---

    # Define mixed precision dtype
    mp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    # --- Optimizer: Must be initialized AFTER DDP wrapping ---
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=(torch.cuda.is_available()))
    if rank == 0: print("Optimizer initialized")
    # --- End Optimizer ---

    # ADD BACK: Criterion definition
    criterion = nn.CrossEntropyLoss()

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

    # --- GradScaler for Mixed Precision ---
    scaler = GradScaler()
    if rank == 0: print(f"GradScaler Initialized: Enabled={scaler.is_enabled()}")
    # --- End GradScaler ---

    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        if rank == 0:
            print(f"Resuming training from {resume_from}")
        
        # Make sure all processes are synchronized before loading
        if world_size > 1:
            dist.barrier()
        
        try:
            # Load checkpoint on CPU first to avoid device mismatches
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank} if torch.cuda.is_available() else 'cpu'
            checkpoint = torch.load(resume_from, map_location=map_location)
            
            # Load model state dict if available - no need for FSDP context
            if "model_state_dict" in checkpoint and checkpoint["model_state_dict"] is not None:
                try:
                    # For DDP, we need to handle the module prefix if loading from non-DDP checkpoint
                    # Extract the state dict
                    state_dict = checkpoint["model_state_dict"]
                    
                    # Let's create a new state dict to avoid in-place modifications
                    new_state_dict = {}
                    
                    # Handle module prefix depending on current model type and checkpoint format
                    if world_size > 1 and isinstance(model, DDP):
                        # We're using DDP now
                        is_ddp_checkpoint = any(k.startswith('module.') for k in state_dict.keys())
                        if not is_ddp_checkpoint:
                            # Add 'module.' prefix to keys because current model is DDP but checkpoint isn't
                            for k, v in state_dict.items():
                                new_state_dict[f'module.{k}'] = v
                        else:
                            # Checkpoint is already DDP format, keep as is
                            new_state_dict = state_dict
                    else:
                        # We're not using DDP now
                        is_ddp_checkpoint = any(k.startswith('module.') for k in state_dict.keys())
                        if is_ddp_checkpoint:
                            # Remove 'module.' prefix because checkpoint is DDP but current model isn't
                            for k, v in state_dict.items():
                                if k.startswith('module.'):
                                    new_state_dict[k[7:]] = v  # 7 is the length of 'module.'
                                else:
                                    new_state_dict[k] = v
                        else:
                            # Neither is DDP, keep as is
                            new_state_dict = state_dict
                    
                    # Load the processed state dict
                    model.load_state_dict(new_state_dict)
                    if rank == 0:
                        print("Successfully loaded model state")
                except Exception as e:
                    if rank == 0:
                        print(f"Warning: Error loading model state: {e}")
            else:
                if rank == 0:
                    print("No model state found in checkpoint")
            
            # Ensure all processes have loaded model before continuing
            if world_size > 1:
                torch.cuda.synchronize()
                dist.barrier()

            # Load optimizer state 
            if "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    if rank == 0:
                        print("Loaded optimizer state")
                except Exception as e:
                    if rank == 0:
                        print(f"Warning: Could not load optimizer state: {e}")

            # Load scaler state
            if "scaler_state_dict" in checkpoint and scaler is not None:
                try:
                    scaler.load_state_dict(checkpoint["scaler_state_dict"])
                    if rank == 0:
                        print("Loaded GradScaler state")
                except Exception as e:
                    if rank == 0:
                        print(f"Warning: Could not load scaler state: {e}")

            # Load scheduler state
            if "scheduler_state_dict" in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    if rank == 0:
                        print("Loaded scheduler state")
                except Exception as e:
                    if rank == 0:
                        print(f"Warning: Couldn't load scheduler state: {e}. Recreating scheduler.")
                    scheduler = create_scheduler(optimizer, num_epochs, checkpoint.get("epoch", 0), train_loader)
            else:
                 # Recreate if not found
                 scheduler = create_scheduler(optimizer, num_epochs, checkpoint.get("epoch", 0), train_loader)

            # Load other states
            start_epoch = checkpoint.get("epoch", 0)
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            counter = checkpoint.get("early_stop_counter", 0)
            
            # Load RNG states
            if "torch_rng_state" in checkpoint:
                torch.set_rng_state(checkpoint["torch_rng_state"])
                if rank == 0:
                    print("Loaded PyTorch RNG state")
                
            if "cuda_rng_state" in checkpoint and torch.cuda.is_available():
                # Only load CUDA RNG state if available
                try:
                    torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
                    if rank == 0:
                        print("Loaded CUDA RNG state")
                except Exception as e:
                    if rank == 0:
                        print(f"Warning: Couldn't load CUDA RNG state: {e}")
                    
            # Final synchronization after loading checkpoint
            if world_size > 1:
                dist.barrier()
                
            if rank == 0:
                print(f"Successfully resumed from checkpoint at epoch {start_epoch}")
                
        except Exception as e:
            if rank == 0:
                print(f"Error resuming from checkpoint: {e}")
                import traceback
                traceback.print_exc()
            # Continue with fresh start
            start_epoch = 0
            best_val_loss = float("inf")
            counter = 0

    # Only create SummaryWriter on rank 0
    writer = SummaryWriter() if rank == 0 else None

    try:
        for epoch in range(start_epoch, num_epochs):
            if world_size > 1:
                if hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)
            
            model.train()
            train_loss = 0
            train_correct = 0
            optimizer.zero_grad(set_to_none=True)

            total_batches = len(train_loader)
            
            if rank == 0:
                print(f"Epoch {epoch+1}/{num_epochs} - Starting training with {total_batches} batches")
            
            # Make sure all processes start the epoch together
            if world_size > 1:
                torch.cuda.synchronize()
                dist.barrier()
                
            # Track processing time per batch for debugging
            batch_start_time = time.time()
            epoch_start_time = time.time()
            
            # Remove log buffer for console output
            # But keep track of epoch stats
            train_start = time.time()
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # Move tensors to device first
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # --- Mixed Precision: Autocast Forward Pass ---
                with autocast('cuda', dtype=mp_dtype):
                    outputs = model(inputs)
                    batch_loss = criterion(outputs, labels)
                # --- End Autocast ---

                loss_val = batch_loss.item() # Get loss value before scaling
                
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(labels).sum().item()
                
                train_correct += batch_correct
                train_loss += loss_val
                
                # Keep full TensorBoard logging for all batches
                if rank == 0 and writer is not None:
                    writer.add_scalar("Batch Loss/train", loss_val, batch_idx + (len(train_loader) * epoch))
                    writer.add_scalar("Batch Accuracy/train", (100 * batch_correct) / len(labels), batch_idx + (len(train_loader) * epoch))
                    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], batch_idx + (len(train_loader) * epoch))
                    
                    # Add more detailed metrics to TensorBoard
                    if batch_idx % 100 == 0:  # Log detailed stats periodically
                        writer.add_histogram("Activations/outputs", outputs, batch_idx + (len(train_loader) * epoch))
                        writer.add_histogram("Gradients/batch_loss", batch_loss, batch_idx + (len(train_loader) * epoch))

                # --- Mixed Precision: Scale Loss and Backward ---
                # Scale the loss
                scaled_loss = scaler.scale(batch_loss / ACCUMULATION_STEPS)
                # Backward pass with scaled loss
                scaled_loss.backward()
                # --- End Scaling ---

                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    # --- Mixed Precision: Unscale, Clip, Step, Update ---
                    # Unscale gradients before clipping
                    scaler.unscale_(optimizer)

                    # Gradient clipping - update to use torch.nn.utils instead of FSDP-specific method
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRADIENT)
                    
                    # Check for valid gradients
                    valid_gradients = True
                    for param in model.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                valid_gradients = False
                                break
                    
                    if not valid_gradients:
                        if rank == 0:
                            print(f"Warning: Invalid gradients at step {batch_idx}, skipping optimizer step.")
                        optimizer.zero_grad(set_to_none=True) # Still zero grad if skipping
                    else:
                        # Optimizer step (uses unscaled gradients)
                        scaler.step(optimizer)
                        # Update the scaler for the next iteration
                        scaler.update()
                        # Zero grad after step
                        optimizer.zero_grad(set_to_none=True)
                    # --- End Mixed Precision Steps ---

                    scheduler.step()
                
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
            
            # Make sure all processes start validation together
            if world_size > 1:
                torch.cuda.synchronize()
                dist.barrier()
                
            val_start_time = time.time()
            
            with torch.no_grad():
                val_batches = len(val_loader)
                for val_idx, (inputs, labels) in enumerate(val_loader):
                    # Process validation batch (no console logging)
                    inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    with autocast('cuda', dtype=mp_dtype):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    top5_correct += (outputs.topk(5, dim=1)[1] == labels.view(-1, 1)).sum().item()
                    
                    # Add detailed TensorBoard logging for validation
                    if rank == 0 and writer is not None and val_idx % 10 == 0:
                        writer.add_scalar("Batch Loss/val", loss.item(), val_idx + (len(val_loader) * epoch))
                        batch_accuracy = 100.0 * predicted.eq(labels).sum().item() / labels.size(0)
                        writer.add_scalar("Batch Accuracy/val", batch_accuracy, val_idx + (len(val_loader) * epoch))
                        if val_idx % 50 == 0:  # Less frequent for validation
                            writer.add_histogram("Val Activations/outputs", outputs, val_idx + (len(val_loader) * epoch))
            
            # Log validation time - only at the end
            if rank == 0:
                val_time = time.time() - val_start_time
                print(f"Validation completed in {val_time:.2f}s")
                # Print epoch total time
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")

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
                
                # Add synchronization point after reduction operations
                dist.barrier()

            # Compute validation metrics
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100.0 * val_correct / val_total
            top5_accuracy = 100.0 * top5_correct / val_total

            # Log metrics on rank 0
            if rank == 0:
                # Add comprehensive TensorBoard metrics for the entire epoch
                if writer is not None:
                    writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
                    writer.add_scalar("Loss/val", avg_val_loss, epoch + 1)
                    writer.add_scalar("Accuracy/train", train_accuracy, epoch + 1)
                    writer.add_scalar("Accuracy/val", val_accuracy, epoch + 1)
                    writer.add_scalar("Top5 Accuracy/val", top5_accuracy, epoch + 1)
                    
                    # Add epoch time metrics
                    writer.add_scalar("Time/epoch_total", time.time() - epoch_start_time, epoch + 1)
                    writer.add_scalar("Time/train", val_start_time - epoch_start_time, epoch + 1)
                    writer.add_scalar("Time/validation", time.time() - val_start_time, epoch + 1)
                    
                    # Log gradients and parameter norms
                    # Be selective about which parameters to log to avoid excessive data
                    grad_samples = 0
                    weight_samples = 0
                    max_samples = 10  # Limit the number of parameters we log
                    
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            # Only log a limited subset of parameters to avoid overwhelming TensorBoard
                            if 'weight' in name and weight_samples < max_samples:
                                try:
                                    writer.add_histogram(f"Weights/{name}", param.detach().cpu(), epoch + 1)
                                    writer.add_scalar(f"Norms/weight_{name}", param.detach().norm().item(), epoch + 1)
                                    weight_samples += 1
                                except:
                                    pass  # Skip if tensor conversion fails
                                    
                            if param.grad is not None and grad_samples < max_samples:
                                try:
                                    writer.add_histogram(f"Gradients/{name}", param.grad.detach().cpu(), epoch + 1)
                                    writer.add_scalar(f"Norms/grad_{name}", param.grad.detach().norm().item(), epoch + 1) 
                                    grad_samples += 1
                                except:
                                    pass  # Skip if tensor conversion fails

                # Print detailed epoch summary
                print(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Train Accuracy: {train_accuracy:.2f}%, "
                    f"Val Loss: {avg_val_loss:.4f}, "
                    f"Val Accuracy: {val_accuracy:.2f}%, "
                    f"Top-5 Val Accuracy: {top5_accuracy:.2f}%, "
                    f"Time: {time.time() - epoch_start_time:.2f}s"
                )

                # --- FSDP: Save Checkpoint (No scaler state) ---
                # The current FSDP state_dict_type context manager is being deprecated
                # and is likely causing issues. Let's use a simpler approach.
                if rank == 0:
                    try:
                        # Save state_dict instead of full model to avoid pickling errors
                        model_state_dict = None
                        try:
                            # For DDP models, we need to access the .module attribute
                            # to get the underlying model's state_dict
                            if isinstance(model, DDP):
                                model_state_dict = model.module.state_dict()
                            else:
                                model_state_dict = model.state_dict()
                        except Exception as e:
                            print(f"Warning: Could not get model state_dict: {e}")
                        
                        # Prepare the checkpoint
                        checkpoint = {
                            "epoch": epoch + 1,
                            "model_state_dict": model_state_dict,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
                            "best_val_loss": best_val_loss,
                            "early_stop_counter": counter,
                            "torch_rng_state": torch.get_rng_state(),
                            "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                        }
                        
                        # Save checkpoint with epoch number
                        checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
                        torch.save(checkpoint, checkpoint_path)
                        
                        # Also save as latest checkpoint (overwrite)
                        torch.save(checkpoint, "checkpoint_latest.pth")
                        
                        print(f"Saved checkpoint for epoch {epoch+1}")

                        # Early stopping check (still uses avg_val_loss)
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            counter = 0
                            # Save best checkpoint
                            torch.save(checkpoint, "checkpoint_best.pth")
                            print(f"New best model saved (val_loss: {avg_val_loss:.4f})")
                        else:
                            counter += 1
                            if counter >= patience:
                                print(f"Early stopping triggered after {epoch + 1} epochs")
                                break
                    except Exception as e:
                        print(f"Warning: Error saving checkpoint: {e}")
                        import traceback
                        traceback.print_exc()
                # --- End FSDP Save ---

            # Make sure we have a barrier at the end of each epoch
            if world_size > 1:
                # Make sure all processes reach this point before proceeding to next epoch
                torch.cuda.synchronize()  # Ensure CUDA operations are complete
                dist.barrier()

    except KeyboardInterrupt:
        if rank == 0:
            print("\nTraining interrupted. Saving checkpoint...")
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.module.state_dict() if world_size > 1 else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict() if scaler is not None else None, # Save scaler state
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
        cleanup_distributed()


def main(args):
    """Main function to setup and run training."""
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and world_size > 0 else "cpu")

    # --- Batch Size Automation --- 
    if args.batch_size is None:
        if torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(device)
                total_mem_gb = props.total_memory / (1024**3)
                # HEURISTIC: Estimate ~2GB VRAM needed per sample? Cap at 256.
                # This is highly approximate and model-dependent!
                heuristic_batch_size = min(256, max(8, int(total_mem_gb / 2.0)))
                args.batch_size = heuristic_batch_size # Set the calculated value
                if rank == 0:
                    print(f"[Warning] --batch_size not set. Using heuristic value based on VRAM ({total_mem_gb:.1f}GB): {args.batch_size}")
                    print(f"[Warning] This is EXPERIMENTAL. Monitor GPU memory ('nvidia-smi') and tune --batch_size manually for optimal performance.")
            except Exception as e:
                if rank == 0:
                    print(f"[Warning] Failed to determine heuristic batch size ({e}). Falling back to default 64.")
                args.batch_size = 64 # Fallback default
        else: # CPU
            args.batch_size = 64 # Default for CPU
            if rank == 0:
                 print(f"[Info] No GPU detected. Using default batch size: {args.batch_size}")

        # Synchronize the calculated/default batch size across all processes
        if world_size > 1:
            bs_tensor = torch.tensor(args.batch_size, device=device, dtype=torch.long)
            dist.broadcast(bs_tensor, src=0) # Broadcast from rank 0
            args.batch_size = bs_tensor.item() # Update args on all ranks
            # Add barrier to ensure all ranks have the BS before proceeding
            dist.barrier()
    # --- End Batch Size Automation ---

    # --- Dataset Loading --- (Moved from old 'train' function)
    if rank == 0:
        print(f"Loading dataset: {args.dataset}")
    # ... (Retry mechanism for dataset loading, use 'rank' for rank == 0 checks) ...
    # Load datasets (ensure this happens on all ranks or is synchronized)
    try:
        train_dataset_func, val_dataset_func = DATASETS[args.dataset]
        # Consider downloading/preparing only on rank 0 and using barrier
        # if dist.is_initialized() and rank != 0:
        #     dist.barrier() # Wait for rank 0
        train_dataset = train_dataset_func()
        val_dataset = val_dataset_func()
        # if dist.is_initialized() and rank == 0:
        #     dist.barrier() # Signal completion
    except Exception as e:
        if rank == 0:
            print(f"Error loading dataset: {str(e)}")
            traceback.print_exc()
        cleanup_distributed()
        return
    # --- End Dataset Loading ---

    if world_size > 1: dist.barrier() # Ensure datasets loaded everywhere

    # --- Determine Num Classes --- (Moved from old 'train' function)
    # ... (Safely get num_classes, use 'rank' for rank == 0 checks)
    # ... (Synchronize num_classes using dist.broadcast if world_size > 1)
    try:
        num_classes = getattr(train_dataset, 'num_classes', 1000) # Default
        if rank == 0 and not hasattr(train_dataset, 'num_classes'):
             print(f"Warning: Using default {num_classes} classes.")
    except Exception:
         num_classes = 1000
         if rank == 0:
              print(f"Error getting num_classes. Using default {num_classes}.")

    if world_size > 1:
        num_classes_tensor = torch.tensor(num_classes, device=device)
        dist.broadcast(num_classes_tensor, src=0)
        num_classes = num_classes_tensor.item()
    # --- End Num Classes ---

    # --- Calculate Optimal DataLoader Workers ---
    available_cpus = os.cpu_count()
    # Calculate workers per process, ensuring minimum of 2 if max_workers allows
    num_workers = min(args.max_workers, available_cpus if available_cpus else args.max_workers) 
    num_workers = max(2, num_workers) # Ensure at least 2 workers if possible
    if rank == 0:
        print(f"System CPU Count: {available_cpus}. Using {num_workers} DataLoader workers per process.")
        if num_workers > 8:
             print("Warning: High number of workers detected. Monitor CPU/memory usage.")
    # --- End Worker Calculation ---

    # --- Samplers and Loaders --- (Moved from old 'train' function)
    # ... (Create samplers and loaders, use 'rank' for rank == 0 checks)
    # ... (Set num_workers=4 or from args)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    # Use calculated num_workers
    dataloader_kwargs = {
        'batch_size': args.batch_size, # Keep batch size from args
        'pin_memory': True,
        'num_workers': num_workers, # Use calculated value
        'persistent_workers': num_workers > 0, 
    }
    if num_workers > 0:
        dataloader_kwargs['timeout'] = 120

    train_loader = DataLoader(
        train_dataset,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
        **dataloader_kwargs
    )
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **dataloader_kwargs
    )

    # Determine TRAIN_TOTAL safely
    global TRAIN_TOTAL
    try:
        # Check if dataset object has __len__
        if hasattr(train_dataset, '__len__'):
            TRAIN_TOTAL = len(train_dataset)
        else:
            raise TypeError # Fallback if no len
    except TypeError:
        TRAIN_TOTAL = args.batch_size * len(train_loader) * world_size # Estimate
        if rank == 0:
            print(f"Couldn't determine dataset size via len(), estimated total samples: {TRAIN_TOTAL}")
    # --- End Samplers/Loaders ---

    # --- Model Creation --- (Moved from old 'train' function)
    if rank == 0:
        print("Creating model...")
    # --- Type Hint Fix Attempt ---
    encoder = cast(Encoder2D, DEFAULT_2D_ENCODER)
    # Ensure num_classes is int
    num_classes = int(num_classes)
    model = Encoder2DClassifier(encoder, num_classes)
    # --- End Type Hint Fix ---

    # Adjust learning rate
    learning_rate = args.lr
    if world_size > 1:
        # Simple linear scaling rule
        learning_rate = learning_rate * world_size
        if rank == 0:
            print(f"Adjusting learning rate for distributed training ({world_size} GPUs): {learning_rate:.6f}")

    if rank == 0:
        print(f"Starting training on device: {device} (Global Rank {rank}/{world_size})")

    # --- Call Training Loop ---
    try:
        train_encoder_classifier(
            model,
            train_loader,
            val_loader,
            num_epochs=args.epochs,
            learning_rate=learning_rate,
            device=device,
            rank=rank,
            world_size=world_size,
            weight_decay=args.weight_decay,
            resume_from=args.resume,
        )
    except Exception as e:
        if rank == 0:
            print(f"Error during training loop: {str(e)}")
            traceback.print_exc()
    finally:
        cleanup_distributed()
    # --- End Training Call ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Encoder2D Classifier", formatter_class=argparse.ArgumentDefaultsHelpFormatter) # Add formatter
    parser.add_argument(
        "--dataset", type=str, required=True, choices=DATASETS.keys(),
        help="Name of the dataset to use"
    )
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Base Learning Rate (scaled by world_size)")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-GPU Batch Size. If unset, uses heuristic based on VRAM (EXPERIMENTAL)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight Decay")
    parser.add_argument("--max_workers", type=int, default=8, help="Max DataLoader workers per process (limited by CPU cores)")
    args = parser.parse_args()

    main(args)
