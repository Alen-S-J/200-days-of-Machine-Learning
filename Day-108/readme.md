# Natural Language Processing (NLP) with LSTM, RNN, and CNN

## Introduction

Natural Language Processing (NLP) involves the use of machine learning techniques to understand and process human language. In this README, we'll explore three popular architectures for NLP tasks: Long Short-Term Memory (LSTM), Recurrent Neural Network (RNN), and Convolutional Neural Network (CNN). We'll provide theoretical overviews of each architecture and include code explanations and implementations in both PyTorch and TensorFlow.

## Long Short-Term Memory (LSTM)

### Theory

LSTM is a type of recurrent neural network (RNN) designed to handle the vanishing gradient problem, which is common in traditional RNNs. LSTMs introduce memory cells and gating mechanisms to capture long-term dependencies in sequential data. The key components include:

- **Memory Cell**: Maintains a cell state to remember information over long sequences.
- **Input Gate**: Controls the flow of information to update the cell state.
- **Forget Gate**: Manages what information to discard from the cell state.
- **Output Gate**: Determines the output based on the updated cell state.

### Code Explanation

The provided PyTorch code implements a simple TextLSTM model for predicting the next character in a sequence. The model is trained on sequences of characters, and the LSTM architecture helps capture dependencies in the input data.

```python
# Code excerpt for TextLSTM in PyTorch
class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        input = X.transpose(0, 1)  # X : [n_step, batch_size, n_class]

        hidden_state = torch.zeros(1, len(X), n_hidden)  # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        cell_state = torch.zeros(1, len(X), n_hidden)     # [num_layers(=1) * num_directions(=1), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model
```

## Recurrent Neural Network (RNN)

### Theory

RNNs are a class of neural networks designed for sequential data by maintaining hidden states that capture information about previous inputs. However, traditional RNNs suffer from the vanishing gradient problem, limiting their ability to capture long-range dependencies.

### Code Explanation

The provided PyTorch code includes a basic example of an RNN-based language model. The RNN layer processes input sequences, and the model is trained for binary classification (e.g., sentiment analysis).

```python
# Code excerpt for TextRNN in PyTorch
class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        self.rnn = nn.RNN(input_size=n_class, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, hidden, X):
        X = X.transpose(0, 1) # X : [n_step, batch_size, n_class]
        outputs, hidden = self.rnn(X, hidden)
        outputs = outputs[-1] # [batch_size, n_hidden]
        model = self.W(outputs) + self.b # model : [batch_size, n_class]
        return model
```

## Convolutional Neural Network (CNN)

### Theory

CNNs, widely used in image processing, can also be applied to sequential data like text. In NLP, CNNs use filters to capture local patterns, providing a different approach to feature extraction compared to RNNs and LSTMs.

### Code Explanation

The provided PyTorch code demonstrates a TextCNN model for text classification. Convolutional layers with different filter sizes capture various n-gram features, and global max-pooling is applied to obtain the most relevant features.

```python
# Code excerpt for TextCNN in PyTorch
class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.num_filters_total = num_filters * len(filter_sizes)
        self.W = nn.Embedding(vocab_size, embedding_size)
        self.Weight = nn.Linear(self.num_filters_total, num_classes, bias=False)
        self.Bias = nn.Parameter(torch.ones([num_classes]))
        self.filter_list = nn.ModuleList([nn.Conv2d(1, num_filters, (size, embedding_size)) for size in filter_sizes])

    def forward(self, X):
        embedded_chars = self.W(X)
        embedded_chars = embedded_chars.unsqueeze(1)

        pooled_outputs = []
        for i, conv in enumerate(self.filter_list):
            h = F.relu(conv(embedded_chars))
            mp = nn.MaxPool2d((sequence_length - filter_sizes[i] + 1, 1))
            pooled = mp(h).permute(0, 3, 2, 1)
            pooled_outputs.append(pooled)

        h_pool = torch.cat(pooled_outputs, len(filter_sizes))
        h_pool_flat = torch.reshape(h_pool, [-1, self.num_filters_total])
        model = self.Weight(h_pool_flat) + self.Bias
        return model
```

## Conclusion

Understanding the theoretical foundations of LSTM, RNN, and CNN in the context of NLP is crucial for designing effective models. The provided code snippets offer practical implementations, showcasing how these architectures can be applied to various NLP tasks. Experimenting with these models and adapting them to specific use cases is key to achieving optimal performance.