import Model as m
import torch
from torch import nn
from tqdm.auto import tqdm
from sklearn.metrics import classification_report
import pickle
from pathlib import Path

class ModelTraining():
    def __init__(self):
        self.model = m.SentimentClassificationModel()

    def save_model(self):
        model_path = Path('Model')
        model_path.mkdir(parents=True, exist_ok=True)
        model_file_name = 'linearRelu.pth'
        file_path = model_path/model_file_name
        torch.save(self.model, file_path)

    def train_model(self, train_data_set, test_data_set):
        loss_fn = nn.CrossEntropyLoss()
        optim_fn = torch.optim.Adam(params=self.model.parameters(), lr=0.01)
        number_of_epochs = 15
        for epoch in tqdm(range(0, number_of_epochs)):
            print(f"\n\nEpoch: {epoch}")
            total_loss = 0
            for batch, (feature, label) in enumerate(train_data_set):
                self.model.train()
                logits = self.model(feature)
                logits = logits.squeeze(dim=1)
                loss = loss_fn(logits, label)
                total_loss = total_loss + loss
                optim_fn.zero_grad()
                loss.backward()
                optim_fn.step()

                if batch % 100 == 0:
                    pred = logits.argmax(1)
                    print(f"\nBatch: {batch}")
                    print(f"\n\nLoss: {total_loss/len(train_data_set)}")
                    print(f"\nTrain classification report \n{classification_report(label.detach().numpy(), pred.detach().numpy())}")

        print("\n\n\nTesting\n\n\n")

        for text_batch, (test_feature, test_label) in enumerate(test_data_set):
            with torch.inference_mode():
                self.model.eval()
                test_logits = self.model(test_feature).squeeze(1)
                test_pred = test_logits.argmax(dim=1)
                if text_batch % 39 == 0:
                    print(f"Test classification report \n{classification_report(test_label.detach().numpy(), test_pred.detach().numpy())}")

        self.save_model()