# GPT_Model_Trainer

## Overview

The `GPT_Model_Trainer` project is designed to train GPT-2 models with support for multi-format data ingestion, real-time loss monitoring, and integration with the Hugging Face architecture. This project leverages PyTorch and the Hugging Face `transformers` library to provide a flexible and efficient training pipeline.

## Features

### Multi-Format Ingestion

- **Solr**: Fetch data from a Solr server using a specified query.
- **File**: Load data from a local file.
- **URL**: Retrieve data from a specified URL.
- **Directory**: Load multiple text files from a specified directory.

### Real-Time Monitoring

- **RealTimeLossMonitor**: Monitor and plot training and validation loss in real-time.
- **Early Stopping**: Stop training early if the validation loss does not improve for a specified number of epochs (patience).

### Hugging Face Architecture Support

- **GPT-2 Model**: Utilize the GPT-2 model architecture from the Hugging Face `transformers` library.
- **Tokenization**: Use the Hugging Face `GPT2Tokenizer` for tokenizing input text.
- **Model Checkpointing**: Save and load model checkpoints to resume training.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/kmcowan/GPT_Model_Trainer.git
    cd GPT_Model_Trainer
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Configuration**: Create a `config.json` file with the necessary configuration parameters. Example:
    ```json
    {
        "data_source_type": "file",
        "file_path": "data/training_data.txt",
        "num_epochs": 5,
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 6,
        "learning_rate": 3e-4,
        "model_name": "gpt2_model.pth",
        "use_dir": false,
        "directory_path": "data/"
    }
    ```

2. **Run the Training Script**:
    ```sh
    python main.py
    ```

## Project Structure

- `main.py`: Main script to load configuration, fetch data, initialize the model, and start training.
- `GPT2Model.py`: Custom implementation of the GPT-2 model.
- `trainer.py`: Contains the training loop and model saving functionality.
- `RealTimeLossMonitor.py`: Implements real-time loss monitoring and early stopping.
- `evaluator.py`: Functions to evaluate the model's performance.
- `data_pull.py`: Functions to fetch data from various sources.
- `tokenizer.py`: Tokenization utilities using Hugging Face `GPT2Tokenizer`.

## Example

To train a model using data from a local file, ensure your `config.json` is set up correctly and run the training script:

```sh
python main.py
```

The training process will print the device being used (CPU or GPU), and after each epoch, it will display the training and validation loss. The model checkpoints will be saved in the `checkpoint/` directory.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
