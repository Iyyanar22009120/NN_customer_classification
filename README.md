# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![image](https://github.com/user-attachments/assets/c3611057-f56d-44fd-935f-3bafd45398f8)


## DESIGN STEPS

### STEP 1:
Data Preprocessing: Clean, normalize, and split data into training, validation, and test sets.

### STEP 2:
Model Design:
 * Input Layer: Number of neurons = features.
 * Hidden Layers: 2 layers with ReLU activation.
 * Output Layer: 4 neurons (segments A, B, C, D) with softmax activation.

### STEP 3:
Model Compilation: Use categorical crossentropy loss, Adam optimizer, and track accuracy.

## STEP 4:
Training: Train with early stopping, batch size (e.g., 32), and suitable epochs.

## STEP 5:
Evaluation: Assess using accuracy, confusion matrix, precision, and recall.

## STEP 6:
Optimization: Tune hyperparameters (layers, neurons, learning rate, batch size).
## PROGRAM

### Name: IYYANAR S
### Register Number: 212222240036

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4)  # 4 classes
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


```
```python
def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()




        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
```
```
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())
```

## Dataset Information

![image](https://github.com/user-attachments/assets/baac52da-3e7f-44f7-8b26-c15f921d0915)


## OUTPUT

### Confusion Matrix

![image](https://github.com/user-attachments/assets/37a63c61-a26b-42a1-839f-06359a405ef1)


### Classification Report

![image](https://github.com/user-attachments/assets/58a3889b-d0ec-4541-b167-12dd0041e7f8)

### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/5dab49b6-597b-4a70-bbae-7286a4c9093c)

## RESULT
Thus a neural network classification model for the given dataset is executed successfully.
