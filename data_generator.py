import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# simulate random health sensor reading
def _simulateHealthSignals(numSamples: int):
    heartRate = np.random.normal(loc=75, scale=10, size=numSamples)
    bloodGlucose = np.random.normal(loc=110, scale=25, size=numSamples)
    systolicBp = np.random.normal(loc=120, scale=15, size=numSamples)
    diastolicBp = np.random.normal(loc=80, scale=10, size=numSamples)
    bodyTemp = np.random.normal(loc=98.4, scale=0.7, size=numSamples)

    X = np.stack(
        [heartRate, bloodGlucose, systolicBp, diastolicBp, bodyTemp],
        axis=1,
    ).astype(np.float32)

    score = (
        0.015 * (heartRate - 70)
        + 0.02 * (bloodGlucose - 100)
        + 0.01 * (systolicBp - 115)
        + 0.01 * (diastolicBp - 75)
        + 0.5 * (bodyTemp - 98.4)
    )

    y = (score > 0.5).astype(np.float32)

    return X, y

#Generate simulated health data for multiple clients and create train/test loaders for federated learning
def generateAndPartitionHealthData(
    numClients: int = 10,
    dataPointsPerClient: int = 200,
    batchSize: int = 32,
):
    rng = np.random.RandomState(seed=42)

    allX = []
    allY = []
    clientLoaders = {}

    for clientId in range(numClients):
        biasHr = rng.normal(loc=0.0, scale=5.0)
        biasGlucose = rng.normal(loc=0.0, scale=15.0)
        biasSys = rng.normal(loc=0.0, scale=10.0)
        biasDia = rng.normal(loc=0.0, scale=6.0)
        biasTemp = rng.normal(loc=0.0, scale=0.3)

        xClient, yClient = _simulateHealthSignals(dataPointsPerClient)

        xClient[:, 0] += biasHr
        xClient[:, 1] += biasGlucose
        xClient[:, 2] += biasSys
        xClient[:, 3] += biasDia
        xClient[:, 4] += biasTemp

        allX.append(xClient)
        allY.append(yClient)

        tensorX = torch.from_numpy(xClient)
        tensorY = torch.from_numpy(yClient).unsqueeze(1)
        dataset = TensorDataset(tensorX, tensorY)
        loader = DataLoader(dataset, batch_size=batchSize, shuffle=True)
        clientLoaders[clientId] = loader

    allX = np.vstack(allX)
    allY = np.concatenate(allY)
# Create a global test set (20% of all data)
    XTrain, XTest, yTrain, yTest = train_test_split(
        allX,
        allY,
        test_size=0.2,
        random_state=123,
        stratify=allY,
    )

    testTensorX = torch.from_numpy(XTest.astype(np.float32))
    testTensorY = torch.from_numpy(yTest.astype(np.float32)).unsqueeze(1)
    testDataset = TensorDataset(testTensorX, testTensorY)
    testLoader = DataLoader(testDataset, batch_size=batchSize, shuffle=False)

    inputDim = allX.shape[1]
    return clientLoaders, testLoader, inputDim