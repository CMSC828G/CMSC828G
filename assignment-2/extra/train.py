from argparse import ArgumentParser
import os
import time
from tqdm import tqdm

import pandas as pd
import torch
from gcn import NativeGCNGraphClassifier, CustomGCNGraphClassifier, normalize_adjacency


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--test_dataset", type=str, required=True)
    parser.add_argument("--hidden_units", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kernel-type", choices=["triton", "torch"], default="torch")
    return parser.parse_args()


def load_dataset(dataset_path, device):
    ds = torch.load(dataset_path, weights_only=True)

    print(f"Dataset: {ds['dataset_name']} ({ds['split']})")
    print(f"Number of samples: {len(ds['X'])}")
    print(f"Number of features: {ds['num_features']}")
    print(f"Number of classes: {ds['num_classes']}")

    # apply normalization to the adjacency matrix
    for i in range(len(ds['X'])):
        ds['A'][i] = normalize_adjacency(ds['A'][i])

        ds['X'][i] = ds['X'][i].to(device)
        ds['A'][i] = ds['A'][i].to(device)
        ds['y'][i] = ds['y'][i].to(device)

    return ds


def train(model, train_ds, test_ds, epochs, lr, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()

        num_samples = len(train_ds['X'])
        X_train, A_train, y_train = train_ds['X'], train_ds['A'], train_ds['y']

        train_loss = 0
        train_accuracy = 0
        train_start = time.time()

        for X, A, y in tqdm(zip(X_train, A_train, y_train), total=num_samples):
            optimizer.zero_grad()
            y_pred = model(X, A).unsqueeze(0)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_accuracy += (y_pred.argmax(dim=1) == y).sum().item()

        train_duration = time.time() - train_start
        train_iteration_time = train_duration / num_samples
        train_loss /= num_samples
        train_accuracy /= num_samples
        print(f"Epoch {epoch + 1}/{epochs} (train): loss={train_loss:.4f}, accuracy={train_accuracy:.4f}, duration={train_duration:.3f}s, iteration_time={train_iteration_time:.5f}s")

        model.eval()
        X_test, A_test, y_test = test_ds['X'], test_ds['A'], test_ds['y']

        test_samples = len(X_test)
        test_loss = 0
        test_accuracy = 0
        test_start = time.time()

        for X, A, y in zip(X_test, A_test, y_test):
            y_pred = model(X, A).unsqueeze(0)
            loss = criterion(y_pred, y)

            test_loss += loss.item()
            test_accuracy += (y_pred.argmax(dim=1) == y).sum().item()

        test_duration = time.time() - test_start
        test_iteration_time = test_duration / test_samples
        test_loss /= test_samples
        test_accuracy /= test_samples
        print(f"Epoch {epoch + 1}/{epochs} (test): loss={test_loss:.4f}, accuracy={test_accuracy:.4f}, duration={test_duration:.3f}s, iteration_time={test_iteration_time:.5f}s")

    return test_loss, test_accuracy, train_iteration_time, test_iteration_time

def main():
    args = get_args()

    # set seed for reproducibility
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load the dataset
    train_ds = load_dataset(args.train_dataset, device)
    test_ds = load_dataset(args.test_dataset, device)
    num_features = train_ds['num_features']
    num_classes = train_ds['num_classes']

    # create the model
    model_cls = CustomGCNGraphClassifier if args.kernel_type == "triton" else NativeGCNGraphClassifier
    model = model_cls(input_dim=num_features, hidden_dim=args.hidden_units, output_dim=args.hidden_units//2, num_classes=num_classes)
    model.to(device)

    test_loss, test_acc, train_iter_time, test_iter_time = train(model, train_ds, test_ds, args.epochs, args.lr, device)

    results_dict = {
        "model": str(model.__class__),
        "dataset_name": train_ds['dataset_name'],
        "num_features": num_features,
        "num_classes": num_classes,
        "hidden_units": args.hidden_units,
        "lr": args.lr,
        "epochs": args.epochs,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "train_iteration_time": train_iter_time,
        "test_iteration_time": test_iter_time
    }
    results_df = pd.DataFrame([results_dict])
    results_df.to_csv("results.csv", mode="a", header=not os.path.exists('results.csv'), index=False)



if __name__ == "__main__":
    main()
