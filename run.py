print("%"*80, '\n')
import sys
import this
import torch
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, AdamW
from sklearn.metrics import classification_report

import logging
import os
import joblib

from tqdm import tqdm


# Set up logging
logging.basicConfig(filename='training.log', level=logging.INFO)

l_split_model_path = 'l_model.pt'
r_split_model_path = 'r_model.pt'
combo_model_path = 'combo_model.pt'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
quit()

"""

The architecture planning and implementation described below.
I will have a large dataframe of user demographics, a document (content), and the label: (bias-question). I will use
 multiple longformers and a decision tree for a machine learning pipeline. Here is how the pipeline will be trained
  on each example, X_i. First, the longformers are initialized, one combo model trained on content: label and then two
  longformers 1.) politics==0 content:label 2.) politics==1 content:label. These longformers will be trained simultaneously
  as the decision tree is trained. Furthermore, the decision-tree will be trained on the user demographics in
  addition to the longformer outputs for each X_i training case. So the combo model will pass a 0 or 1 label prediction
  to the decision-tree at each X_i, and the two split longformers will pass a length 4 list of confidence probabilities
  (the values in the list must add to one since it is a distribution)
  for each example X_i  [politics==0;label==0, politics==0;label==1, politics==1;label==0, politics==1;label==1].
  This list of probabilities will be passed into the decision-tree along with the user demographics and the
   combo model output for each training example X_i"""


"""
1. Initialize the longformers:
    a.) combo_model
    b.) (l_model, r_model)

2. Feature Generation: transform the content of each example into a set of features.
    a.) prediction from COMBO MODEL
    b.) probabilities from L_MODEL & R_MODEL denoting the confidence of the model that the example belongs to each of the four classes
        (politics = 0 or 1, label = 0 or 1)

3. Concatenate the Longformer's outputs with the user demographics:
    a.) Once the Longformer outputs are obtained, concatenate them with the user demographics data for each example X_i to get the final feature set
        that will be input into the decision tree

4. Train a Decision Tree:
    a.) Using the final feature set generated from step 3, now train a decision tree classifier with the target variable 'bias-question'
"""


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class CustomDataset(Dataset):
    def __init__(self, content, labels, tokenizer):
        self.content = content
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        text = self.content[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten()}, torch.tensor(label)


class Longformers:
    def __init__(self):
        self.tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')

        self.l_model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=2)
        self.r_model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=2)
        self.combo_model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=2)

    def load_models(self):
        # add conditions to load the trained model if it's available
        try:
            self.l_model.load_state_dict(torch.load("l_split_model_path"))
            self.r_model.load_state_dict(torch.load("r_split_model_path"))
            self.combo_model.load_state_dict(torch.load("combo_model_path"))
        except FileNotFoundError:
            print("Trained models not found. Training will start from scratch.")


    def encode(self, text):
        return self.tokenizer.encode_plus(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')

    def classify(self, text):
        with torch.no_grad():
            encoded_text = self.encode(text)

            l_output = self.l_model(**encoded_text)
            r_output = self.r_model(**encoded_text)
            combo_output = self.combo_model(**encoded_text)

            # return the probabilities rather than applying the sigmoid function to the output logits.
            #   ...we want thesis prob's to act as confidence scores for each class, for each political leaning.
            # softmax gives us probabilities that sum to 1 which will represent the confidence scores better.
            l_classification = torch.softmax(l_output.logits, dim=1).numpy()
            r_classification = torch.softmax(r_output.logits, dim=1).numpy()
            combo_classification = torch.softmax(combo_output.logits, dim=1).numpy()

        return l_classification, r_classification, combo_classification


    def train_combo_model(self, content, labels):
        # Create a dataset and a dataloader
        dataset = CustomDataset(content, labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Define a loss function and an optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = AdamW(self.combo_model.parameters(), lr=1e-5)

        # Iterate over the data and update the model weights
        for epoch in tqdm(range(10), desc="Batches"):
            for batch in dataloader:
                inputs, targets = batch
                outputs = self.combo_model(**inputs)
                loss = loss_fn(outputs.logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Save the model weights
        torch.save(self.combo_model.state_dict(), "combo_model_path")


    def train_l_model(self, content, labels):
        # Create a dataset and a dataloader
        dataset = CustomDataset(content, labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Define a loss function and an optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = AdamW(self.l_model.parameters(), lr=1e-5)

        # Iterate over the data and update the model weights
        for epoch in tqdm(range(10), desc="Batches"):
            for batch in dataloader:
                inputs, targets = batch
                outputs = self.l_model(**inputs)
                loss = loss_fn(outputs.logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # save the weights
        torch.save(self.l_model.state_dict(), l_split_model_path)

    def train_r_model(self, content, labels):
        # Create a dataset and a dataloader
        dataset = CustomDataset(content, labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        # Define a loss function and an optimizer
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = AdamW(self.r_model.parameters(), lr=1e-5)

        # Iterate over the data and update the model weights
        for epoch in tqdm(range(10), desc="Batches"):
            for batch in dataloader:
                inputs, targets = batch
                outputs = self.r_model(**inputs)
                loss = loss_fn(outputs.logits, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # save the weights
        torch.save(self.r_model.state_dict(), r_split_model_path)


    def combo_predict(self, content):
        return self._predict(content, self.combo_model)

    def split_predict(self, content):
        l_pred = self._predict(content, self.l_model)
        r_pred = self._predict(content, self.r_model)
        return l_pred, r_pred

    def _predict(self, content, model):
        tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
        encodings = tokenizer(content, truncation=True, padding=True, max_length=512, return_tensors='pt')

        #Move to GPU if available
        for key in encodings:
            encodings[key] = encodings[key].to(device)

        # set model to eval mode and move to GPU
        model.eval()
        model.to(device)

        with torch.no_grad():
            output = model(**encodings)

        # Extract logits and compute softmax to get probabilities
        logits = output.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()

        prediction = torch.argmax(logits, dim=1).cpu().numpy()
        prediction_tuple = (prediction, probabilities)

        return prediction_tuple

    def extract_features(self, text):
        # get classifications from the longformers
        l_classification, r_classification, combo_classification = self.classify(text)

        # construct the probabilities list according to [politics==0;label==0, politics==0;label==1, politics==1;label==0, politics==1;label==1]
        probabilities = [
            l_classification[0][0],  # politics==0;label==0
            l_classification[0][1],  # politics==0;label==1
            r_classification[0][0],  # politics==1;label==0
            r_classification[0][1],  # politics==1;label==1
        ]

        # return combo prediction and the probabilities list
        combo_prediction = np.argmax(combo_classification, axis=1)[0]
        return [combo_prediction] + probabilities

# main function
def main():
    logging.info('Started training')

    # Load and preprocess your data
    # df = pd.read_csv('/home/csstudent/Desktop/NewsBiasDetection/content.csv')
    df = pd.read_csv('/content/drive/MyDrive/SURP_notebook/cleaned_batches/content_df.csv')
    logging.info(f'Loaded data with shape: {df.shape}')

    #
    target = 'bias-question'

    # load models and create an instance of the Longformers class
    longformers = Longformers()

    # preprocess the data
    df_l = df[df['politics'] == 0].reset_index(drop=True)
    df_r = df[df['politics'] == 1].reset_index(drop=True)


    # load the models if they are already trained
    longformers.load_models()

    # Train models only if they aren't already trained
    if not os.path.isfile("combo_model_path"):
        logging.info('Training combo model')
        longformers.train_combo_model(df['content'], df['bias-question'])
        logging.info('Combo model trained')

    if not os.path.isfile("l_split_model_path"):
        logging.info('Training l_model')
        longformers.train_l_model(df_l['content'], df_l['bias-question'])
        logging.info('l_model trained')

    if not os.path.isfile("r_split_model_path"):
        logging.info('Training r_model')
        longformers.train_r_model(df_r['content'], df_r['bias-question'])
        logging.info('r_model trained')

    # Create df_batches, for instance using np.array_split
    df_batches = np.array_split(df, 10)  # adjust to the size of your batches

    # generate features in a batch-wise manner
    for i, df_batch in enumerate(tqdm(df_batches, desc="Batches")):
        logging.info(f'Processing batch {i + 1}/{len(df_batches)}')
        df_batch['features'] = df_batch['content'].apply(longformers.extract_features)
        joblib.dump(df_batch, f'df_batch_{i + 1}.pkl')  # Save intermediate results to disk
        logging.info(f'Finished processing batch {i + 1}/{len(df_batches)}')

    # After all batches are processed, combine the results and continue with your script
    df_batches = [joblib.load(f'df_batch_{i + 1}.pkl') for i in range(len(df_batches))]
    df = pd.concat(df_batches)

    # Convert politics and gender to int
    df['politics'] = df['politics'].astype(int)
    df['gender'] = df['gender'].astype(int)





    # Break down the concatenation step and check the structures
    df_longformer_features = df['features'].apply(pd.Series)
    df_longformer_features.columns = [f'feature_{i}' for i in range(df_longformer_features.shape[1])]
    print("Generated Features: \n", df_longformer_features)

    # Prepare your final feature set
    df_demographics = df.drop(columns=['content', 'bias-question', 'features'])
    print("Demographics: \n", df_demographics)

    # Concatenate the longformer_features with df_demographics
    X = pd.concat([df_longformer_features, df_demographics], axis=1)
    print("Final features shape: ", X.shape)
    print("Final features head: ", X.head())

    y = df[target]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the decision tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    logging.info('Trained decision tree classifier')

    # Save the decision tree model
    joblib.dump(dt, 'decision_tree_model.pkl')
    logging.info('Saved decision tree model')

    y_pred = dt.predict(X_test)

    report = classification_report(y_test, y_pred)
    print(report)
    logging.info(f'Classification report: \n {report}')

    logging.info('Finished training')

if __name__ == '__main__':
    main()
