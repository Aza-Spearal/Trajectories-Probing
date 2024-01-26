import numpy as np
import torch
from sklearn import linear_model

from taker import Model
from code_data import code_short, code_long

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

    def get_label_id(self, label):
        # add label to map if not already present
        if label not in self.map:
            self.map[label] = self.num_labels
            self.reverse_map[self.num_labels] = label
            self.num_labels += 1

        return self.map[label]

    def tokenize(self, data, uses_start_token=True):
        input_ids_list = []
        label_ids_list = []

        for index, (text, label) in enumerate(data):
            input_ids = self.model.get_ids(text + "\n")

            # Remove <s> token from the start since it's not the start of the text
            if uses_start_token and index > 0:
                input_ids = input_ids[:, 1:]

            label_id = self.get_label_id(label)
            label_ids = torch.zeros_like(input_ids, dtype=torch.int32)
            label_ids[...] = label_id

            input_ids_list.append(input_ids)
            label_ids_list.append(label_ids)

            print(input_ids.shape, label_ids.shape)

        input_ids = torch.cat(input_ids_list, dim=1)
        label_ids = torch.cat(label_ids_list, dim=1)

        return input_ids, label_ids[0]

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


def train_predictor_layer(m: Model):
    # Get the Training Data
    # Run the recipe into the model, and collect the residual stream activations at each layer
    # layers are: 0: input, 1: attn_0, 2: mlp_0, ....... (2n-1): attn_(n-1),
    # (2n): mlp_(n-1) == output
    labeled_tokenizer = LabeledTokenizer(m)

    # Get the Training text and labels
    code_ids_train, labels_train = labeled_tokenizer.tokenize(code_long)
    stream_train = m.get_residual_stream(input_ids=code_ids_train).to('cpu')

    # Get the Test Data
    code_ids_test, labels_test = labeled_tokenizer.tokenize(code_short)
    stream_test = m.get_residual_stream(input_ids=code_ids_test).to('cpu')

    labeled_tokenizer.print_colorful_text(code_ids_test[0], labels_test)

    scores = []
    Y_predicted, best_score, best_layer = None, 0.0, 0

    # Train a linear probe on each layer
    for layer in range(m.cfg.n_layers*2 + 1):
        # Initialise Simple Regression Model
        #reg = linear_model.LinearRegression()
        reg = linear_model.LogisticRegression(max_iter=1000)

        # Generate the Training data
        X_train = np.array(stream_train[layer])
        Y_train = labels_train

        # Generate the Test data
        X_test = np.array(stream_test[layer])
        Y_test = labels_test

        # Train the Regression
        reg = reg.fit(X_train, Y_train)

        scores.append( reg.score(X_test, Y_test) )
        print('Layer', layer, scores[-1])

        if scores[-1] > best_score:
            best_score = scores[-1]
            best_layer = len(scores) - 1
            Y_predicted = reg.predict(X_test)

    print(f"best result in layer {best_layer}: {best_score}")
    labeled_tokenizer.print_colorful_text(code_ids_test[0], Y_predicted)

m = Model("nickypro/tinyllama-15M")
train_predictor_layer(m)