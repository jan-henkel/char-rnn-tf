# char-rnn-tf
Train a character-level RNN on an input textfile and generate samples. Inspired by Andrej Karpathy's char-rnn.

# train.py usage

Call train.py to train the model. Example:
```lang-none
python train.py '/path/to/input.txt'
```

General usage as described by the help message:
```lang-none
train.py [-h] [--seq-len SEQ_LENGTH] [--stride STRIDE]
                [--val-frac VAL_FRAC] [--reprocess] [--type {lstm,rnn,gru}]
                [--layers NUM_LAYERS] [--layer-norm]
                [--embed-dim EMBEDDING_DIM] [--hidden-dim HIDDEN_DIM]
                [--nobias] [--save-dir SAVE_DIR] [--restore-last]
                [--clear-model] [--iter ITERATIONS] [--lr LEARNING_RATE]
                [--batch-size BATCH_SIZE] [--dropout DROPOUT_KEEP_PROB]
                [--print-every PRINT_EVERY]
                PATH

positional arguments:
  PATH                  path of the input file

optional arguments:
  -h, --help            show this help message and exit

input preprocessing:
  --seq-len SEQ_LENGTH  sequence length
  --stride STRIDE       stride, defaults to sequence length
  --val-frac VAL_FRAC   fraction of data used for validation set
  --reprocess           do preprocessing again (otherwise preprocessing
                        arguments will be ignored if preprocessed data is
                        already present)

model parameters:
  --type {lstm,rnn,gru}
                        rnn type
  --layers NUM_LAYERS   number of layers
  --layer-norm          use layer normalization. has no effect on anything
                        other than lstm
  --embed-dim EMBEDDING_DIM
                        embedding dimension. defaults to one-hot encoding if
                        not specified
  --hidden-dim HIDDEN_DIM
                        rnn hidden layer dimension
  --nobias              don't learn bias for character scores

model save and restore settings:
  --save-dir SAVE_DIR   optional, defaults to path derived from input file
  --restore-last        restore last model rather than best performing model.
                        does nothing if no previous model is present
  --clear-model         clear previous model parameters

training parameters:
  --iter ITERATIONS     number of iterations
  --lr LEARNING_RATE    learning rate
  --batch-size BATCH_SIZE
                        batch size
  --dropout DROPOUT_KEEP_PROB
                        dropout keep probability
  --print-every PRINT_EVERY
                        number of iterations between progress reports and
                        checkpoints
```

# sample.py usage

Call sample.py to generate samples from a previously trained model. Example:
```lang-none
python sample.py '/path/to/input.txt' --prime-text 'The meaning of life is '
```

General usage as described by the help message:
```lang-none
sample.py [-h] [--save-dir SAVE_DIR] [--restore-last]
                 [--len SAMPLE_LENGTH] [--prime-text PRIME_TEXT]
                 [--temp TEMPERATURE]
                 path

positional arguments:
  path                  path of the input file

optional arguments:
  -h, --help            show this help message and exit

model restore settings:
  --save-dir SAVE_DIR   model save directory, defaults to path derived from
                        input file
  --restore-last        restore last model rather than best performing model

sample settings:
  --len SAMPLE_LENGTH   sample length
  --prime-text PRIME_TEXT
                        prime the rnn with an input
  --temp TEMPERATURE    sampling temperature, smaller values lead to more
                        conservative guesses
```
