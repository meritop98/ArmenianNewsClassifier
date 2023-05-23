import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
import pandas as pd
from config import CFG

class NewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        title = self.data.iloc[index]['Title']
        text = self.data.iloc[index]['Text']
        label = self.data.iloc[index]['Category']

        encoding = self.tokenizer.encode_plus(
            title,
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (size, embedding_dim)) for size in filter_sizes
        ])
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, input_ids, attention_mask):
        embedded = self.embedding(input_ids)
        embedded = embedded.unsqueeze(1)  # Add channel dimension
        conved = [nn.functional.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.size(2)).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, dim=1)
        output = self.fc(cat)
        return output

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    train_acc = 0

    for batch in tqdm(train_loader, desc='Training'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, dim=1)
        train_acc += (predicted == labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)

    return train_loss, train_acc


def evaluate(model, device, data_loader, criterion):
    model.eval()
    eval_loss = 0
    eval_f1 = 0

    with torch.no_grad():
        all_labels = []
        all_predictions = []

        for batch in tqdm(data_loader, desc='Evaluation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            eval_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        eval_loss /= len(data_loader.dataset)
        eval_f1 = f1_score(all_labels, all_predictions, average='weighted')

    return eval_loss, eval_f1

def train_cnn_classifier(df):
    label_encoder = LabelEncoder()
    label_encoder.fit(df['Category'])
    df['Category'] = label_encoder.transform(df['Category'])

    train_data, val_data = train_test_split(df, test_size=0.15, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    max_length = 100
    vocab_size = len(tokenizer)
    embedding_dim = 100
    num_filters = 100
    filter_sizes = [3, 4, 5]
    num_classes = len(label_encoder.classes_)

    train_dataset = NewsDataset(train_data, tokenizer, max_length)
    val_dataset = NewsDataset(val_data, tokenizer, max_length)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = CNNClassifier(vocab_size, embedding_dim, num_filters, filter_sizes, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_models = [(0, None), (0, None)]

    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss, train_f1 = train(model, device, train_loader, optimizer, criterion)
        val_loss, val_f1 = evaluate(model, device, val_loader, criterion)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Training Loss: {train_loss:.4f} | Training F1 Score: {train_f1:.4f}')
        print(f'Validation Loss: {val_loss:.4f} | Validation F1 Score: {val_f1:.4f}')
        print('-' * 50)

        if val_f1 > best_models[0][0]:
            best_models[0] = (val_f1, model.state_dict())
        elif val_f1 > best_models[1][0]:
            best_models[1] = (val_f1, model.state_dict())
    for i, (val_f1, state_dict) in enumerate(best_models):
        if state_dict is not None:
            checkpoint_name = f'checkpoint_val_f1_{val_f1:.4f}.pt'
            torch.save(state_dict, checkpoint_name)
    return best_models
def predict(model, tokenizer, max_length, title_text, label_encoder):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    title = title_text['Title']
    text = title_text['Text']

    # Preprocess the inputs
    encoding = tokenizer.encode_plus(
        title,
        text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].squeeze().to(device)
    attention_mask = encoding['attention_mask'].squeeze().to(device)

    # Pass the inputs through the model
    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
        _, predicted = torch.max(outputs, dim=1)

    # Convert the predicted label to the initial category
    predicted_label = label_encoder.inverse_transform([predicted.item()])[0]

    return predicted_label
if __name__ == "__main__":
    import os
    # df = pd.read_csv(os.path.join(CFG.data_dir, 'data_cleaned.csv'))
    # train_cnn_classifier(df)


