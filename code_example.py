import numpy as np
import torch
from taker import Model

import json

from sklearn import linear_model
from sklearn.model_selection import train_test_split

import lazypredict
from lazypredict.Supervised import LazyClassifier

class LabeledTokenizer:
    def __init__(self, model):
        self.model: Model = model
        self.map = dict()
        self.reverse_map = dict()
        self.num_labels = 0

        # Define a list of ANSI color codes
        self.colors = [
            "\033[31m",  # Red
            "\033[32m",  # Green
            "\033[33m",  # Yellow
            "\033[34m",  # Blue
            "\033[35m",  # Magenta
            "\033[36m",  # Cyan
            "\033[91m",  # Light Red
            "\033[92m",  # Light Green
            "\033[93m",  # Light Yellow
            "\033[94m"   # Light Blue
            # Add more colors if needed
        ]
        self.reset_color = "\033[0m"

    def print_colorful_text(self, tokens, labels):
        # Reset color to default

        # Print tokens with colors based on labels
        token_strings = self.model.tokenizer.convert_ids_to_tokens(tokens)
        token_strings = [s.replace("<0x0A>", "\n") for s in token_strings]
        labels_seen = set()

        labels = np.array(labels)
        for token_str, label in zip(token_strings, labels):
            color = self.colors[label % len(self.colors)]  # Get color for label
            print(f"{color}{token_str}{self.reset_color}", end='')  # Print with color and reset afterwards
            labels_seen.add(label)

        self.print_color_legend(labels_seen)

    def print_color_legend(self, labels_seen):
        print("\nLegend of Colours:")
        for label in list(labels_seen):
            color = self.colors[label % len(self.colors)]
            label_text = self.reverse_map[label]
            print(f"{color}█ - {label_text}{self.reset_color}")


    def get_label_id(self, label):
        # add label to map if not already present
        if label not in self.map:
            self.map[label] = self.num_labels
            self.reverse_map[self.num_labels] = label
            self.num_labels += 1

        return self.map[label]


    def tokenize(self, data, uses_start_token=True):

        input_list = []
        label_list = []

        for topic in range(len(data)):

            for index in range(len(data[topic]['split_text'])):

                text = data[topic]['split_text'][index]['text']
                label = data[topic]['split_text'][index]['label']

                input_idx = labeled_tokenizer .model.get_ids(text + "\n")
                # Remove <s> token from the start since it's not the start of the text
                if uses_start_token and index > 0:
                    input_idx = input_idx[:, 1:]

                label_id = labeled_tokenizer.get_label_id(label)
                label_idx = torch.zeros_like(input_idx, dtype=torch.int32)
                label_idx[...] = label_id

                input_list.append(input_idx)
                label_list.append(label_idx)

        input_torch = torch.cat(input_list, dim=1)
        label_torch = torch.cat(label_list, dim=1)
        return input_torch, label_torch[0]



def train_predictor_layer(m: Model, input_stream, y):

    scores = []
    Y_predicted, best_score, best_layer = None, 0.0, 0

    # Train a linear probe on each layer
    for layer in range(m.cfg.n_layers*2 + 1):
        x = np.array(input_stream[layer].cpu())
        X_train, X_test, y_train, y_test  = train_test_split(x, y, test_size=0.2)

        # Initialise Simple Regression Model
        predictor = linear_model.LogisticRegression(max_iter=1000)

        # Train the Regression
        predictor = predictor.fit(X_train, y_train)

        scores.append( predictor.score(X_test, y_test) )
        print('Layer', layer, scores[-1])

        if scores[-1] > best_score:
            best_score = scores[-1]
            best_layer = len(scores) - 1
            Y_predicted = predictor.predict(X_test)

    print(f"best result in layer {best_layer}: {best_score}\n")
    #labeled_tokenizer.print_colorful_text(code_ids_test[0], Y_predicted)
    

#in case you don't have enough memory, we use a loop to get the input activation
def get_stream(model, input_tok, batch):
    n_tokens = input_tok[0].shape[0]
    #we initialise the tensor with the expected final size:
    input_stream = torch.empty(m.cfg.n_layers*2 + 1, n_tokens, m.cfg.d_model)

    for i in range((n_tokens//batch)+1):
        n = i*batch
        if i == n_tokens//batch: #for the last round of the loop
            residual = m.get_residual_stream(input_ids=input_tok[:, n:])
            input_stream[:, n:] = residual
        else: #otherwise
            residual = m.get_residual_stream(input_ids=input_tok[:, n:n+batch])
            input_stream[:, n:n+batch] = residual

    return input_stream


m = Model("nickypro/tinyllama-15M")

labeled_tokenizer = LabeledTokenizer(m)

f = open('examples.json')
data = json.load(f)

input_tok, label_tok = labeled_tokenizer.tokenize(data)
y = label_tok.cpu()

#to avoid cuda out of memory:
batch = 1000
input_stream = get_stream(m, input_tok, batch)


#if you want try a model on all layer:
train_predictor_layer(m, input_stream, y)


#if you want use lazyclassifier (very long, for me 9mn):
#if you want use that, perhaps you will have errors and warning, tell me (Eloise) if its the case
'''
layer = 11

x = np.array(input_stream[layer].cpu())
X_train, X_test, y_train, y_test  = train_test_split(x, y, test_size=0.2)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
print('Layer', layer)
print(predictions)
'''
