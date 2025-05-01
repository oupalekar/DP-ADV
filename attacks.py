import torch
import torch.nn.functional as F
from model import MembershipInferenceAttacker
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch import optim

class PGD():
    def __init__(self) -> None:
        pass

    def fgsm(self, model, x, label, eps):
        #TODO: implement this as an intermediate step of PGD
        # Notes: put the model in eval() mode for this function
        model.eval()
        # x.requires_grad_()
        output = model(x)
        loss = F.cross_entropy(output, label)

        grad = torch.autograd.grad(
                    loss, x, retain_graph=False, create_graph=False
                )[0]

        x_adv = x.detach() + eps * torch.sign(grad)
        return x_adv



    def pgd_untargeted(self, model, x, y, k, eps, eps_step):
        #TODO: implement this 
        # Notes: put the model in eval() mode for this function
        
        model.eval()
        adv_images = x.clone().detach()
        for _ in range(k):
            adv_images.requires_grad = True
            x_adv = self.fgsm(model, adv_images, y, eps_step)
            delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            adv_images = torch.clamp(x + delta, min = 0, max = 1)
        
        return adv_images


def membership_inference_attack(target_model, train_data, test_data, device):
    """
    Perform membership inference attack
    """
    # Prepare feature extraction
    def extract_features(model, data):
        model.eval()
        features = []
        with torch.no_grad():
            for x, _ in data:
                x = x.to(device).flatten()
                # Extract intermediate layer features
                intermediate = model.network[:-1](x)
                features.append(intermediate)
        return torch.cat(features)

    # Extract features
    train_features = extract_features(target_model, train_data)
    test_features = extract_features(target_model, test_data)

    # Prepare attack dataset
    train_labels = torch.ones(train_features.size(0), 1)
    test_labels = torch.zeros(test_features.size(0), 1)

    attack_features = torch.cat([train_features, test_features])
    attack_labels = torch.cat([train_labels, test_labels])

    # Split attack data
    X_train, X_test, y_train, y_test = train_test_split(
        attack_features.numpy(),
        attack_labels.numpy(),
        test_size=0.2
    )

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.FloatTensor(y_train)
    y_test = torch.FloatTensor(y_test)
    print(X_train.size())
    # Membership inference attacker
    attacker = MembershipInferenceAttacker(
        input_dim=X_train.size(-1),
        hidden_dim=64
    )

    # Train attacker
    criterion = nn.BCELoss()
    optimizer = optim.Adam(attacker.parameters(), lr=0.001)

    for epoch in range(50):
        optimizer.zero_grad()
        outputs = attacker(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate attack performance
    with torch.no_grad():
        test_outputs = attacker(X_test)
        predicted = (test_outputs > 0.5).float()
        accuracy = (predicted == y_test).float().mean()

    return accuracy.item()

