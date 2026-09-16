import argparse, json, os, sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from intervention_predictor import InterventionPredictorModel, ActionFeature


def load_data(path):
    with open(path) as f:
        return json.load(f)


def to_action(sample):
    return ActionFeature(
        action_type="submit",
        action_data={"amount": sample["amount"]},
        context={"task_type": "bill_payment", "complexity": 0.5,
                 "risk_level": min(sample["amount"] / 2000.0, 1.0)},
        user_history={},
    )


def train(data_path, epochs=50, lr=1e-3):
    data = load_data(data_path)
    labels = [s["label"] for s in data]
    pos_rate = np.mean(labels)
    print(f"Total samples: {len(data)} | Positive rate: {pos_rate:.1%}")

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42, stratify=labels)
    pos_weight = (1 - pos_rate) / pos_rate
    print(f"Positive class weight: {pos_weight:.2f}")

    model = InterventionPredictorModel()
    criterion = nn.BCELoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        np.random.shuffle(train_data)
        for sample in train_data:
            action_seq = [to_action(sample)]
            label = torch.tensor(float(sample["label"]))
            optimizer.zero_grad()
            out = model(action_seq)
            loss = criterion(out, label) * (pos_weight if sample["label"] == 1 else 1.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs - 1:
            model.eval()
            preds, actuals = [], []
            with torch.no_grad():
                for sample in test_data:
                    p = model([to_action(sample)]).item()
                    preds.append(1 if p > 0.5 else 0)
                    actuals.append(sample["label"])
            print(f"Epoch {epoch}: loss={total_loss/len(train_data):.4f} "
                  f"balanced_acc={balanced_accuracy_score(actuals, preds):.4f} "
                  f"f1={f1_score(actuals, preds, zero_division=0):.4f}")
            model.train()

    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for sample in test_data:
            p = model([to_action(sample)]).item()
            preds.append(1 if p > 0.5 else 0)
            actuals.append(sample["label"])
    print("\n--- FINAL ---")
    print(f"Balanced accuracy: {balanced_accuracy_score(actuals, preds):.4f}")
    print(f"F1: {f1_score(actuals, preds, zero_division=0):.4f}")
    print("Confusion matrix:\n", confusion_matrix(actuals, preds))

    os.makedirs("intervention", exist_ok=True)
    torch.save(model.state_dict(), "intervention/model.pt")
    print("\nSaved -- matches repo's real architecture, will load without shape mismatch.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    train(args.data, args.epochs, args.lr)