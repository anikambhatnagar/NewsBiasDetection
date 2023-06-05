import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, AdamW
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, AdamW
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
print('...finished imports')


filepath = r'C:\Users\user1\Desktop\AI\SURP\nlp\data\content_df.csv'
df = pd.read_csv(filepath)




c_subset_df = df[df['politics'] == 0]
l_subset_df = df[df['politics'] == 1]




# Renaming the columns
c_df_longformer = c_subset_df.drop(columns=['age', 'gender', 'newsOutlet', 'politics', 'url', 'id','country_US', 'language_en', 'ind_politics', 'bias_sum', 'id_count','ratio', 'title' ])
c_df_longformer.rename(columns={'content': 'text', 'bias-question': 'label'}, inplace=True)


print("\nDataFrame after renaming columns")
print(c_df_longformer.columns, c_df_longformer.head())



# split the data
train_texts, val_texts, train_labels, val_labels = train_test_split(c_df_longformer['text'], c_df_longformer['label'], test_size=0.2, random_state=42)

# initialize the tokenizer
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')

print(max(len(text) for text in train_texts))

# tokenize the texts
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=512)

# create a dataset object
class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# create train and validation datasets
train_dataset = TextDataset(train_encodings, train_labels.tolist())
val_dataset = TextDataset(val_encodings, val_labels.tolist())

print('initialized the model')
# Initialize the model
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=2)

# Load the saved model
model.load_state_dict(torch.load('model_path.pt'))

# initialize the data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True)

# initialize the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5, no_deprecation_warning=True)

# start training
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

train_loss = []
val_loss = []
print('\n\n', train_dataset)



for epoch in range(3):  # number of epochs
    print(f"epoch: {epoch}")
    model.train()
    for batch in tqdm(train_loader):
        print(f"batch: {batch}")
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        train_loss.append(loss.item())

    
        print(f'Validation loss: {loss.item()}')
        val_loss.append(loss.item())
        optimizer.step()

    model.eval()
    print('eval mode:')
    for batch in val_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            print(f'Validation loss: {loss.item()}')

# Save the model after training
torch.save(model.state_dict(), 'model_path.pt')


# Plot training and validation loss over epochs
plt.plot(train_loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

