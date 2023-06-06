# Scott Nelson & Anika Bhatnagar
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, AdamW
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import torch
import os
print('...finished imports')


training = True


class  BagOfWords:
    """Bag of words model for text classification."""
    
    def __init__(self, classifier=None, vectorizer=None):
        self.classifier = MultinomialNB() if classifier is None else classifier
        self.vectorizer = CountVectorizer(stop_words='english') if vectorizer is None else vectorizer

    def fit(self, text_train, labels_train, training=True):
        if training:
            features = self.vectorizer.fit_transform(text_train)
            self.classifier.fit(features, labels_train)
        else:
            self.load_model()

    def predict(self, text_test, labels_test=None, training=True):
        features = self.vectorizer.transform(text_test)
        return self.classifier.predict(features)

    def evaluate(self, text_test, labels_test):
        predictions = self.predict(text_test)
        print(classification_report(labels_test, predictions))
        print('Accuracy: ', accuracy_score(labels_test, predictions))
        
    def cross_validate(self, text, labels, cv=5):
        scores = cross_val_score(self.classifier, self.vectorizer.transform(text), labels, cv=cv)
        print('Cross validation scores:', scores)
        print('Mean cross validation score:', scores.mean())
        
    def save_model(self, dir_path='models/', file_name='model.joblib'):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        joblib.dump((self.classifier, self.vectorizer), dir_path + file_name)

    def load_model(self, file_path):
        self.classifier, self.vectorizer = joblib.load(file_path)

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

class LongformerModel:
    def __init__(self, df, num_labels=4, batch_size=16, epochs=3):
        self.df = df
        self.tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
        self.model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=num_labels)
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def preprocess_data(self):
        processed_df = self.df.drop(columns=['age', 'gender', 'newsOutlet', 'politics', 'url', 'id','country_US', 'language_en', 'ind_politics', 'bias_sum', 'id_count','ratio', 'title' ])
        processed_df.rename(columns={'content': 'text', 'bias-question': 'label'}, inplace=True)
        return processed_df

    def create_datasets(self, c_df_longformer):
        train_texts, val_texts, train_labels, val_labels = train_test_split(c_df_longformer['text'], c_df_longformer['label'], test_size=0.2, random_state=42)
        train_encodings = self.tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512)
        val_encodings = self.tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=512)
        train_dataset = TextDataset(train_encodings, train_labels.tolist())
        val_dataset = TextDataset(val_encodings, val_labels.tolist())
        return train_dataset, val_dataset

    def train(self, train_dataset, model_name):
        if training:
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            optimizer = AdamW(self.model.parameters(), lr=1e-5, no_deprecation_warning=True)
            self.model.to(self.device)
            train_loss = []

            for epoch in range(self.epochs):
                print(f"Epoch: {epoch}")
                self.model.train()
                for batch in tqdm(train_loader):
                    optimizer.zero_grad()
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    train_loss.append(loss.item())
                    optimizer.step()
            print(model_name)
            print(type(model_name))
            torch.save(self.model.state_dict(),model_name)
            return train_loss
        else:
            self.model.load_state_dict(torch.load('model_path.pt'))

    def evaluate(self, val_dataset):
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.eval()
        val_loss=[]

        with torch.no_grad():
            for batch in tqdm(val_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss.append(loss.item())
        return val_loss
                

    def plot_loss(self, train_loss, val_loss, model_name='model_loss.png'):
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label='Training loss')
        plt.plot(val_loss, label='Validation loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('model_loss.png')


def load_and_sanitize(data_file):
    df = pd.read_csv(data_file, encoding='latin-1')
    # df = df.dropna(axis='columns')
    # df = df.rename(columns={'v1': 'label', 'v2': 'text'})
    return df

def process_df(df, politics_value):
    subset_df = df[df['politics'] == politics_value]
    processed_df = subset_df.drop(columns=['age', 'gender', 'newsOutlet', 'politics', 'url', 'id','country_US', 'language_en', 'ind_politics', 'bias_sum', 'id_count','ratio', 'title' ])
    processed_df.rename(columns={'content': 'text', 'bias-question': 'label'}, inplace=True)
    return processed_df

if __name__ == "__main__":
    csv_file_path = r'/home/csstudent/Desktop/NewsBiasDetection/combo_df.csv'
    df = load_and_sanitize(csv_file_path)

    dfs = [process_df(df, i) for i in range(3)]
    model_names = ['c_bow_model.joblib', 'l_bow_model.joblib', 'i_bow_model.joblib']

    for df, model_name in zip(dfs, model_names):
        y_train, y_test, x_train, x_test = train_test_split(df['label'], df['text'], test_size=0.2)
    
        print("Training BagOfWords model...")
        bow_model = BagOfWords()
        bow_model.fit(x_train, y_train)

        print("Evaluating model...")
        bow_model.evaluate(x_test, y_test)

        print("Performing cross-validation...")
        bow_model.cross_validate(df['text'], df['label'])

        print(f"Saving Model...{model_name}")
        bow_model.save_model(file_name=model_name)

        print("Model training and evaluation complete. Model saved.")

    longformer_model_names = ['c_lf_model.joblib', 'l_lf_model.joblib', 'i_lf_model.joblib']

    for df, model_name in zip(dfs, longformer_model_names):

        longformer_model = LongformerModel(csv_file_path)
        train_dataset, val_dataset = longformer_model.create_datasets(df)
        print(f"Training Longformer model for {model_name}...")

        train_loss = longformer_model.train(train_dataset, model_name)
        print("Evaluating Longformer model...")

        val_loss = longformer_model.evaluate(val_dataset)
        longformer_model.plot_loss(train_loss, val_loss, model_name=model_name)
        print("Saving Longformer model...")

        print(f"model name: {model_name}")
        torch.save(longformer_model.model.state_dict(), model_name)
        print("Longformer model training and evaluation complete. Model saved.")
