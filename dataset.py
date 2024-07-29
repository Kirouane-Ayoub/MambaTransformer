from torch.utils.data import DataLoader, Dataset

import settings
from tokenizer import new_tokenizer


class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, encoding="utf-8") as f:
            self.texts = f.readlines()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )  # Ensure PyTorch Tensor output

        input_ids = encoding["input_ids"].squeeze()

        # Assuming you want to use the input_ids as labels for language modeling

        # Shift labels
        labels = input_ids.clone()

        labels[:-1] = input_ids[1:]  # Shift labels
        return input_ids, labels  # Return both input_ids and labels

    # Create dataset and dataloaders


train_dataset = TextDataset(
    settings.TRAINING_FILE, new_tokenizer, max_length=settings.MAX_LENGTH
)
train_loader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=False)
# Create dataset and dataloaders
val_dataset = TextDataset(
    settings.EVAL_FILE, new_tokenizer, max_length=settings.MAX_LENGTH
)
val_loader = DataLoader(val_dataset, batch_size=settings.BATCH_SIZE, shuffle=False)
