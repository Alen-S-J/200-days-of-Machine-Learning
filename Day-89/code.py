from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import nlpaug.augmenter.word as naw

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Load and preprocess your new sentiment analysis dataset
# Replace with your data loading and preprocessing steps

# Augment the data
aug = naw.SynonymAug()

augmented_train_texts = []
for text in train_texts:
    augmented_text = aug.augment(text)
    augmented_train_texts.append(augmented_text)

# Combine the original and augmented texts
combined_train_texts = train_texts + augmented_train_texts
combined_train_labels = train_labels + train_labels  # Assuming labels remain the same for augmented data

# Tokenize the combined texts and convert labels to tensors
combined_train_encodings = tokenizer(combined_train_texts, truncation=True, padding=True)

combined_train_dataset = TensorDataset(torch.tensor(combined_train_encodings['input_ids']),
                                       torch.tensor(combined_train_labels))

# Fine-tuning the BERT model on the combined dataset
combined_train_loader = DataLoader(combined_train_dataset, batch_size=16, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

model.train()
for epoch in range(3):  # Replace with the desired number of epochs
    for batch in combined_train_loader:
        optimizer.zero_grad()
        input_ids, labels = batch
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluate the fine-tuned model on the original test set
test_encodings = tokenizer(test_texts, truncation=True, padding=True)
test_dataset = TensorDataset(torch.tensor(test_encodings['input_ids']),
                             torch.tensor(test_labels))

test_loader = DataLoader(test_dataset, batch_size=16)
predictions = []
model.eval()
with torch.no_grad():
    for batch in test_loader:
        input_ids, labels = batch
        outputs = model(input_ids)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        predictions.extend(preds.tolist())

accuracy = accuracy_score(test_labels, predictions)
print(f"Accuracy after fine-tuning with augmented data: {accuracy}")
