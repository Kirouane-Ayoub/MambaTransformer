import argparse

import torch
import torch.nn.functional as F

import settings
from model import MambaTransformer
from tokenizer import new_tokenizer


def generate_text(model, device, max_length, num_return_sequences, prompt):
    # Encode the prompt
    tokens = new_tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)

    x = tokens.to(device)

    while x.size(1) < max_length:
        with torch.no_grad():
            outputs = model(x)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, -1, ix)
            x = torch.cat((x, xcol), dim=1)

    # Print the generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = new_tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"> {decoded}")


def main(args):
    # Initialize the model
    model = MambaTransformer(
        num_tokens=settings.VOCAB_SIZE,
        dim=settings.MODEL_DIMENSION,
        heads=8,
        depth=4,
        dim_head=64,
        d_state=512,
        dropout=0.1,
        ff_mult=4,
        return_embeddings=False,
        transformer_depth=settings.NUMBER_OF_TRANSFORMER_BLOCKS,
        mamba_depth=settings.NUMBER_OF_MAMBA_BLOCKS,
        use_linear_attn=True,
    )

    # Load the model weights (modify the path as needed)
    model.load_state_dict(torch.load("hybrid.pt"))

    # Set the device
    device = torch.device(args.device)
    model.to(device)

    # Generate text
    generate_text(
        model=model,
        device=device,
        max_length=args.max_length,
        num_return_sequences=args.num_return_sequences,
        prompt=args.prompt,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text with the Mamba Transformer model"
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=30,
        help="Maximum length of the generated sequence",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=10,
        help="Number of sequences to generate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for generation (e.g., 'cuda' or 'cpu')",
    )
    parser.add_argument(
        "--prompt", type=str, default="نتا ", help="Prompt text to start generation"
    )

    args = parser.parse_args()

    main(args)
