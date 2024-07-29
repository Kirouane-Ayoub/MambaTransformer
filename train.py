import argparse
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import settings
from dataset import train_dataset, val_dataset
from model import MambaTransformer
from tokenizer import new_tokenizer


def train(
    model: MambaTransformer,
    train_data: DataLoader,
    val_data: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int = 10,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    clip_grad_norm: float = 1.0,
    lr_scheduler=None,
):
    """Trains the Mistral model.

    Args:
        model: The Mistral model to train.
        train_data: A DataLoader for the training dataset.
        optimizer: The optimizer to use for training.
        epochs: The number of training epochs.
        device: The device to use for training (e.g., 'cuda' or 'cpu').
        clip_grad_norm: The maximum norm of the gradients to clip.
        lr_scheduler: An optional learning rate scheduler.
    """

    model = model.to(device)
    model.train()

    print("Training...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        total_loss = 0.0
        start_time = time.time()

        for batch in tqdm(train_data, leave=False):
            input_ids, labels = batch

            input_ids, labels = input_ids.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids)

            # Calculate loss (use cross-entropy loss for language modeling)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs.view(-1, new_tokenizer.vocab_size), labels.view(-1))

            # Backward pass
            loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            # Update weights
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step(loss.detach().item())

            total_loss += loss.item()

        avg_loss = total_loss / len(train_data)
        elapsed_time = time.time() - start_time
        print(f"Average loss: {avg_loss:.4f} | Elapsed time: {elapsed_time:.2f}s")

        # Evaluation Phase
        model.eval()
        eval_loss = 0
        with torch.no_grad():
            for step, batch in enumerate(val_data):
                # Get input_ids and labels from the batch
                input_ids, labels = batch
                input_ids = input_ids.to(device)  # Send input_ids to the device
                labels = labels.to(device)  # Send labels to the device

                # Forward pass
                outputs = model(input_ids)

                # Calculate loss
                loss = F.cross_entropy(
                    outputs.view(-1, new_tokenizer.vocab_size),
                    labels.view(-1),
                    ignore_index=new_tokenizer.pad_token_id,
                )
                eval_loss += loss.item()
        avg_eval_loss = eval_loss / len(val_data)
        print(f"Epoch: {epoch+1}, Evaluation Loss: {avg_eval_loss:.4f}")
    model_save_path = "hybrid.pt"
    torch.save(model.state_dict(), model_save_path)
    print("Training complete!")


def main(args):
    # Create an instance of the MambaTransformer model
    model = MambaTransformer(
        num_tokens=settings.VOCAB_SIZE,  # Number of tokens in the input sequence
        dim=settings.MEDEL_DIMENSION,  # Dimension of the model
        heads=8,  # Number of attention heads
        depth=4,  # Number of transformer layers
        dim_head=64,  # Dimension of each attention head
        d_state=512,  # Dimension of the state
        dropout=0.1,  # Dropout rate
        ff_mult=4,  # Multiplier for the feed-forward layer dimension
        return_embeddings=False,  # Whether to return the embeddings,
        transformer_depth=settings.NUMBER_OF_TRANSFORMER_BLOCKS,  # Number of transformer blocks
        mamba_depth=settings.NUMBER_OF_MAMBA_BLOCKS,  # Number of Mamba blocks,
        use_linear_attn=True,  # Whether to use linear attention
    )

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Optionally, set up a learning rate scheduler if needed
    lr_scheduler = None
    if args.lr_scheduler:
        # Set up the scheduler here (if applicable)
        pass

    # Call the train function with the arguments
    train(
        model=model,
        train_data=train_dataset,
        val_data=val_dataset,
        optimizer=optimizer,
        epochs=args.epochs,
        device=args.device,
        clip_grad_norm=args.clip_grad_norm,
        lr_scheduler=lr_scheduler,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Mamba Transformer model")

    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (e.g., 'cuda' or 'cpu')",
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=1.0,
        help="Maximum norm of the gradients to clip",
    )
    parser.add_argument(
        "--lr_scheduler",
        action="store_true",
        help="Whether to use a learning rate scheduler",
    )

    args = parser.parse_args()

    main(args)
