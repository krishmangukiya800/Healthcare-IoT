from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from encryption_util import secureChannelReceive


class EdgeServer:
    def __init__(self, encryptionKey: bytes):
        self.encryptionKey = encryptionKey
        self._receivedUpdates: List[Dict[str, torch.Tensor]] = []
# Receive and decrypt an update sent by a client device
    def receiveEncryptedUpdate(self, deviceId: int, encryptedUpdate: bytes) -> None:
        stateDict = secureChannelReceive(encryptedUpdate, self.encryptionKey)
        self._receivedUpdates.append(stateDict)
# Aggregate updates from all clients using simple FedAvg
    def aggregateUpdates(self) -> Dict[str, torch.Tensor]:
        if not self._receivedUpdates:
            raise ValueError("No updates received to aggregate.")

        numUpdates = len(self._receivedUpdates)
        aggregated = {
            k: v.clone().detach()
            for k, v in self._receivedUpdates[0].items()
        }

        for sd in self._receivedUpdates[1:]:
            for k, v in sd.items():
                aggregated[k] += v

        for k in aggregated:
            aggregated[k] /= numUpdates

        self._receivedUpdates.clear()
        return aggregated
# Evaluate global model accuracy on test dataset
    @staticmethod
    def evaluateGlobalModel(
        model: nn.Module,
        testLoader: DataLoader,
    ) -> float:
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for xBatch, yBatch in testLoader:
                logits = model(xBatch)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                correct += (preds == yBatch).sum().item()
                total += yBatch.numel()

        if total == 0:
            return 0.0
        return 100.0 * correct / total