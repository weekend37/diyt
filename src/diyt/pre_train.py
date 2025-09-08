import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import yaml

from argparse import ArgumentParser
from pathlib import Path
from pydantic import BaseModel
from typing import Generator
from tqdm import tqdm

from tokenizer import Encoded, SimpleTokenizer
from model import ModelConfig, Transformer
from paths import DATA_DIR, ASSETS_DIR


class TrainingConfig(BaseModel):
    batch_size: int
    learning_rate: float
    num_epochs: int
    device: str
    model_checkpoint_folder: Path
    start_from_epoch: int
    save_frequency: int
    start_from_checkpoint: str | None = None


def batchify(encoded: Encoded, batch_size: int) -> Generator[Encoded, None, None]:
    for i in range(0, len(encoded), batch_size):
        batch = Encoded(
            token_ids=encoded.token_ids[i : i + batch_size, :],
            attention_mask=encoded.attention_mask[i : i + batch_size, :]
        )
        yield batch


def train(
    model: Transformer,
    tokenizer: SimpleTokenizer,
    texts: list[str],
    training_config: TrainingConfig,
    model_config: ModelConfig
):
    seperator = f" {tokenizer.end_of_string_token} "
    pad_token_id = tokenizer.token_to_id[tokenizer.pad_token]
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=training_config.learning_rate)

    average_losses = []
    epochs = range(training_config.start_from_epoch, training_config.start_from_epoch + training_config.num_epochs)
    progress_bar = tqdm(epochs, desc="🚀 Here we go!")
    for epoch in progress_bar:

        random.shuffle(texts)
        data = seperator.join(texts) + seperator
        encoded_data = tokenizer.encode(data)
        # Trim off last (incomplete) context window of corpus
        C = model.context_length
        N = len(encoded_data) // C
        token_ids = encoded_data.token_ids[: N * C].reshape(N, C).to(training_config.device)  # [N, C]
        attention_mask = encoded_data.attention_mask[: N * C].reshape(N, C).to(training_config.device)  # [N, C]
        encoded_data_reshaped = Encoded(token_ids=token_ids, attention_mask=attention_mask)

        model.train()
        epoch_losses = []
        for batch in batchify(encoded_data_reshaped, batch_size=training_config.batch_size):
            optimizer.zero_grad()
            targets = batch.token_ids[:, 1:]  # [B, C-1]
            model_output = model(batch.token_ids, batch.attention_mask)  # [B, C] -> [B, C, V]
            next_token_logits = model_output[:, :-1, :]  # [B, C-1, V]
            loss = loss_function(next_token_logits.transpose(-1, -2), targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.detach().cpu())
        average_loss = np.mean(epoch_losses)
        average_losses.append(average_loss)
        progress_bar.set_description(f"Epoch {epoch+1} | average loss: {average_loss}")

        if (epoch + 1) % (training_config.save_frequency) == 0 or epoch == training_config.num_epochs - 1:
            training_config.model_checkpoint_folder.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': average_loss,
                'model_config': model_config.model_dump(),
                'training_config': training_config.model_dump()
            }
            checkpoint_path = training_config.model_checkpoint_folder / f"checkpoint_{epoch + 1}.pth"
            torch.save(checkpoint, checkpoint_path)

    plt.plot(epochs, average_losses)
    plt.savefig("assets/average_losses.png")


if __name__ == "__main__":

    parser = ArgumentParser(description="Training script with config file")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default="src/diyt/pre_train_config.yaml",
        help="Path to config yaml file"
    )
    args = parser.parse_args()
    print(f"⚙️  using config: {args.config}")
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    model_config = ModelConfig.model_validate(config_dict["model_config"])
    training_config = TrainingConfig.model_validate(config_dict["training_config"])

    with open(DATA_DIR / "harry_potter" / "train.json", "r", encoding="utf-8") as f:
        texts = json.load(f)

    tokenizer = SimpleTokenizer.load(ASSETS_DIR / "tokenizer.json")
    model = Transformer(
        context_length=model_config.context_length,
        token_embedding_size=model_config.token_embedding_size,
        decoder_dim=model_config.decoder_dim,
        n_heads=model_config.n_heads,
        n_decoder_blocks=model_config.n_decoder_blocks,
        dropout_ratio=model_config.dropout_ratio,
        vocab_size=tokenizer.vocab_size
    )
    if training_config.start_from_checkpoint is not None:
        checkpoint_path = training_config.model_checkpoint_folder / training_config.start_from_checkpoint
        print(f"🏷️  Starting from checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
    train(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        training_config=training_config,
        model_config=model_config
    )
