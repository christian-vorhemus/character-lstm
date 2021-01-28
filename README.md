# Character-based LSTM
A LSTM that can be used as a starting point to train a character-based prediction model. This example tries to classify if a first name is female or male based on the dataset in the /data folder.

## Prerequisites

- Make sure that you have a newer [version of Python and PIP installed](https://www.python.org/downloads/)
- PyTorch is required. Some platforms may require that you have additional packages installed before you can run PyTorch (see [the install pages](https://pytorch.org/))
- Open up a terminal and type

  ```
  git clone https://github.com/christian-vorhemus/character-lstm.git
  cd character-lstm
  pip install -r requirements.txt
  ```

## Model training

Start the training with

  ```
  python main.py train
  ```

## Inference

Check if a name is predicted as male or female with

  ```
  python main.py anna
  anna is probably a female name
  ```
