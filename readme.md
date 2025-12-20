# GuangKE

## About The Project

This project focuses on Named Entity Recognition (NER) tasks, with the core implementation of a BicLSTM-CRF model. The model integrates 1D CNN, BiLSTM and CRF to address traditional BiLSTM's weakness in capturing local text features, while retaining its advantage in modeling long-range sequential dependencies and CRF's ability to optimize global label sequence consistency.

### Core Components

The model adopts a modular design, including Embedding layer (word vector mapping), 1D-CNN layer (local n-gram feature extraction), BiLSTM layer (bidirectional temporal dependency modeling), Linear layer (label emission score generation) and CRF layer (global label sequence optimization).

### Key Features

- Feature fusion: Combines CNN's local feature extraction and BiLSTM's sequential modeling for complementary advantages.

- End-to-end training: Supports unified end-to-end training and inference via CRF-based negative log-likelihood loss.

- Configurable: Adapts to different datasets/tasks via external parameters (e.g., embedding dimension, hidden size, dropout rate).

### Application Scenarios

Suitable for NER tasks, especially Chinese NER (lacking explicit word boundaries) and professional domain NER (e.g., product/medical entity recognition) with simple parameter/data adaptation.

## Install

```bash
conda create -n guangke python=3.10
conda activate guangke
pip install -r requirements.txt
```

## Usage

```bash
conda activate guangke
python run.py task=train model=BiLSTM_Small data=PD hyper.num_epochs=5
```

## Author

- Github : [Guangyaochenyu](https://github.com/Guangyaochenyu)

## Contributing

Contributions, issues and feature requires are welcome!

## Show your support
Give a ⭐️ if you like this project!

## License

Copytright (c) 2025 [Guangyaochenyu](https://github.com/Guangyaochenyu)

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details
