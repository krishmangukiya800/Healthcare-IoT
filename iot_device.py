from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

#Feedforward Neural Network for Binary Classification
class SimpleNN(nn.Module):
    def __init__(self, inputSize: int):
        super().__init__()
        self.fc1 = nn.Linear(inputSize, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Federated Learning Client Device
class HealthMonitorDevice:
    def __init__(self, deviceId: int, inputSize: int, lr: float = 0.01):
        self.deviceId = deviceId
        self.model = SimpleNN(inputSize)
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _trainOneEpoch(self, trainLoader: DataLoader) -> float:
        self.model.train()
        totalLoss = 0.0
        numBatches = 0
        energyEstimate = 0.0

        for xBatch, yBatch in trainLoader:
            self.optimizer.zero_grad()
            logits = self.model(xBatch)
            loss = self.criterion(logits, yBatch)
            loss.backward()
            self.optimizer.step()

            totalLoss += loss.item()
            numBatches += 1
            energyEstimate += xBatch.size(0) * 1e-4

        if numBatches > 0:
            avgLoss = totalLoss / numBatches
        return energyEstimate
   # Perform federated learning local training step
    def trainOnGlobalModel(
        self,
        globalStateDict: dict,
        trainLoader: DataLoader,
        localEpochs: int = 1,
    ) -> Tuple[dict, float]:
        self.model.load_state_dict(globalStateDict)

        totalEnergy = 0.0
        for _ in range(localEpochs):
            totalEnergy += self._trainOneEpoch(trainLoader)

        updatedState = self.model.state_dict()
        return updatedState, totalEnergy