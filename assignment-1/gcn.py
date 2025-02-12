import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from relu_kernel import custom_relu
from max_pool_kernel import custom_max_pool
from matmul_kernel import custom_matmul


class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, X, A):
        return torch.matmul(A, torch.matmul(X, self.weight.t()))


class CustomGCNLayer(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(CustomGCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.empty((output_dim, input_dim)))

    def forward(self, X, A):
        """ compute (A @ X @ W^T) """
        return torch.matmul(A, torch.matmul(X, self.weight.t()))


class NativeGCNGraphClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_classes):
        super(NativeGCNGraphClassifier, self).__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, num_classes)

    def forward(self, X, A):
        # forward layers 1 and 2
        X = self.gcn1(X, A)
        X = F.relu(X)
        X = self.gcn2(X, A)
        X = F.relu(X)

        # max pooling
        X = torch.max(X, dim=0).values

        # fully connected layer
        X = self.fc(X)
        return X



class CustomGCNGraphClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_classes):
        super(CustomGCNGraphClassifier, self).__init__()
        self.gcn1 = CustomGCNLayer(input_dim, hidden_dim)
        self.gcn2 = CustomGCNLayer(hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim, num_classes)

    def forward(self, X, A):
        # forward layers 1 and 2
        X = self.gcn1(X, A)
        X = custom_relu(X)
        X = self.gcn2(X, A)

        # max pooling
        X = custom_max_pool(X)

        # fully connected layer
        X = self.fc(X)
        return X


def normalize_adjacency(A):
    A_hat = A + torch.eye(A.size(0))
    D = torch.diag(torch.pow(A_hat.sum(1), -0.5))
    return D @ A_hat @ D