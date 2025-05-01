import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
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

    # Set NCCL options (moved here for consistency)
    os.environ['NCCL_DEBUG'] = os.environ.get('NCCL_DEBUG', 'WARN') # Default to WARN
    os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
    os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['NCCL_IB_TIMEOUT'] = '30'
    os.environ['NCCL_IGNORE_DISABLED_P2P'] = '1'

    # Initialize the process group
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=30)
    )

    # Set the device for this process
    torch.cuda.set_device(local_rank)
    print(f"Rank {rank} initialization complete. Using device cuda:{local_rank}")

    return local_rank, rank, world_size

def cleanup_distributed():
    """Cleans up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_fsdp_wrap_policy():
    # Import layers needed for the policy inside the function
    # Assuming Encoder2D is the main transformer block container
    # and Encoder2DClassifier has self.encoder and self.classifier_head
    fsdp_auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        # Specify the names of the transformer block classes within your model
        # For now, let's target the top-level Encoder2D and the classifier Linear layer
        transformer_layer_cls={
             Encoder2D, # Wrap the whole encoder block
             nn.Linear, # Wrap Linear layers (like the classifier head)
        }
    )
    return fsdp_auto_wrap_policy


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
    # FSDP requires model on the correct device *before* wrapping
    model = model.to(device)
    local_rank = device.index

    # --- FSDP Configuration ---
    # Choose Sharding Strategy: FULL_SHARD saves most memory
    fsdp_sharding_strategy = ShardingStrategy.FULL_SHARD
    # Define Mixed Precision policy (BF16 recommended for Ampere/Hopper GPUs like 4090)
    # Use FP16 if BF16 is not supported or causes issues
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16, # USING bfloat16
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    # Get the auto-wrap policy
    fsdp_auto_wrap_policy = get_fsdp_wrap_policy()
    # Set device ID for FSDP
    fsdp_device_id = torch.cuda.current_device() # Should match local_rank device

    # Wrap the model with FSDP
    if rank == 0: print("Wrapping model with FSDP...")
    model = FSDP(
        model,
        auto_wrap_policy=fsdp_auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=fsdp_sharding_strategy,
        device_id=fsdp_device_id, # Use the explicitly set device
        # use_orig_params=True, # Often needed for torch.compile compatibility later
        # cpu_offload=CPUOffload(offload_params=True), # Optional: If memory is extremely tight
        # backward_prefetch=BackwardPrefetch.BACKWARD_PRE, # Optional: Sometimes helps overlap
    )
    if rank == 0: print(f"FSDP Model Info:\n{model}")
    # --- End FSDP Configuration ---

    # --- Optimizer: Must be initialized AFTER FSDP wrapping ---
    # FSDP flattens parameters, so optimizer needs to see the FSDP model params
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, fused=(torch.cuda.is_available())) # Try fused=True
    if rank == 0: print("Optimizer initialized after FSDP wrapping.")
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

    # --- FSDP Checkpointing Configuration ---
    # Use the new FSDP checkpointing API
    full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    # --- End FSDP Checkpointing ---

    # Resume from checkpoint if specified
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming training from {resume_from}")
        # Load checkpoint on CPU first to avoid device mismatches
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank} if torch.cuda.is_available() else 'cpu'
        checkpoint = torch.load(resume_from, map_location=map_location)

        # --- FSDP: Load Model State ---
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
             model_state_dict = checkpoint.get("model_state_dict", None)
             if model_state_dict:
                  model.load_state_dict(model_state_dict)
                  if rank == 0: print("Loaded FSDP Model state.")
        # --- End FSDP Load ---

        # Load optimizer state (FSDP handles sharding, but load full state dict for simplicity first)
        # Note: Loading sharded optimizer state is more complex but memory efficient
        opt_state_dict = checkpoint.get("optimizer_state_dict", None)
        if opt_state_dict:
             # Need to shard the loaded state dict before loading into FSDP optimizer
             # Simple load might work if optimizer was created *before* wrapping (not recommended)
             # Proper way: Load full dict -> FSDP.scatter_full_optim_state_dict -> optimizer.load_state_dict
             # Start with simple load and see if it errors / works correctly
             try:
                  optimizer.load_state_dict(opt_state_dict)
                  if rank == 0: print("Loaded Optimizer state (attempted simple load).")
             except Exception as e:
                  if rank == 0: print(f"Warning: Could not load optimizer state directly: {e}. Optimizer state reset.")

        # Load scheduler state
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if rank == 0: print("Loaded Scheduler state.")
        else:
             # Recreate if not found
             scheduler = create_scheduler(optimizer, num_epochs, start_epoch, train_loader)

        # Load other states
        start_epoch = checkpoint.get("epoch", 0)
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        counter = checkpoint.get("early_stop_counter", 0)

    # Get the desired dtype from the FSDP policy for input casting
    input_dtype = mp_policy.param_dtype if mp_policy else torch.float32
    if rank == 0:
        print(f"Casting input tensor to: {input_dtype}")

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
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if rank == 0 and batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}/{total_batches}")
                
                # Move tensors to device first
                inputs = inputs.to(device)
                labels = labels.to(device)

                # --- Explicitly cast input to target dtype --- 
                inputs = inputs.to(input_dtype)
                # --- End input cast ---
                
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels)
                
                loss_val = batch_loss.item()
                
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(labels).sum().item()
                
                train_correct += batch_correct
                train_loss += loss_val
                
                if rank == 0 and writer is not None:
                    writer.add_scalar("Batch Loss/train", loss_val, batch_idx + (len(train_loader) * epoch))
                    writer.add_scalar("Batch Accuracy/train", (100 * batch_correct) / len(labels), batch_idx + (len(train_loader) * epoch))
                    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], batch_idx + (len(train_loader) * epoch))

                # --- FSDP handles scaling/backward ---
                # Loss already scaled internally if using MixedPrecision
                (batch_loss / ACCUMULATION_STEPS).backward() # Simple backward call
                # --- End FSDP backward ---

                if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                    # Gradient clipping (Needs to happen *after* FSDP unshards gradients)
                    # model.clip_grad_norm_(MAX_GRADIENT) # FSDP provides this method
                    # Clip based on global norm across all ranks
                    total_norm = model.clip_grad_norm_(max_norm=MAX_GRADIENT)
                    if rank == 0 and total_norm.isinf() or total_norm.isnan():
                        print(f"Warning: Gradient norm is {total_norm} at step {batch_idx}, skipping optimizer step.")
                        optimizer.zero_grad(set_to_none=True) # Still zero grad if skipping
                    else:
                         optimizer.step()
                         optimizer.zero_grad(set_to_none=True) # Zero grad after step

                    scheduler.step()
                
                if world_size > 1:
                    dist.barrier()

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

                # --- FSDP: Save Checkpoint ---
                # Ensure optimizer state is consolidated before saving if needed
                # Use the new FSDP StateDictType context manager for model state
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                     model_state = model.state_dict()
                     # Optimizer state might need similar handling for sharded state,
                     # but saving full optimizer state on rank 0 is simpler to start.
                     opt_state = optimizer.state_dict() # Might be large if not sharded

                if rank == 0: # Save only on rank 0
                     checkpoint = {
                          "epoch": epoch + 1,
                          "model_state_dict": model_state, # Use CPU state dict from rank 0
                          "optimizer_state_dict": opt_state, # Full optimizer state from rank 0
                          "scheduler_state_dict": scheduler.state_dict(),
                          "best_val_loss": best_val_loss,
                          "early_stop_counter": counter,
                          "torch_rng_state": torch.get_rng_state(),
                     }
                     # ... (add numpy state if needed) ...
                     torch.save(checkpoint, "checkpoint.pth")

                     # Early stopping check (still uses avg_val_loss)
                     if avg_val_loss < best_val_loss:
                          best_val_loss = avg_val_loss
                          counter = 0
                          # Save best model state dict using the same FSDP context
                          with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                               best_model_state = model.state_dict()
                          torch.save(best_model_state, "best_model.pth") # Save only the state dict
                     else:
                          counter += 1
                          if counter >= patience:
                              print(f"Early stopping triggered after {epoch + 1} epochs")
                              break
                # --- End FSDP Save ---

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
