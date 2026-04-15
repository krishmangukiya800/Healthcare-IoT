import time
from typing import List

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Assuming the following modules exist and are available in the same directory:
from data_generator import generateAndPartitionHealthData
from edge_server import EdgeServer
from encryption_util import generateSymmetricKey, secureChannelSend
from iot_device import SimpleNN, HealthMonitorDevice


# Function to save results to a CSV file
def save_results_to_csv(
        numRounds: int,
        accuracyHistory: List[float],
        avgLatencyHistory: List[float],
        energyHistory: List[float],
        roundTimeHistory: List[float],
        filename: str = "fl_simulation_results.csv",
):
    """Saves the simulation history to a CSV file."""
    data = {
        "Round": list(range(1, numRounds + 1)),
        "Accuracy (%)": accuracyHistory,
        "Avg Latency (ms)": [l * 1000 for l in avgLatencyHistory],
        "Energy (Relative Units)": energyHistory,
        "Round Time (s)": roundTimeHistory,
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")


# Function to generate and save line charts
def plot_results(filename: str = "fl_simulation_results.csv"):
    """Generates line charts for the key FL metrics."""
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        print(f"Error: Could not find {filename}. Run the FL simulation first.")
        return

    rounds = df["Round"]


    # Plot : Accuracy

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, df["Accuracy (%)"], marker='o', linestyle='-', color='tab:blue')
    plt.title('Global Model Accuracy Over FL Rounds')
    plt.xlabel('FL Round')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('fl_accuracy.png')


    # Plot : Latency

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, df["Avg Latency (ms)"], marker='s', linestyle='-', color='tab:red')
    plt.title('Average Secure-Channel Latency Per FL Round')
    plt.xlabel('FL Round')
    plt.ylabel('Avg Latency (ms)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('fl_latency.png')


    # Plot : Energy

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, df["Energy (Relative Units)"], marker='^', linestyle='-', color='tab:green')
    plt.title('Total Local Energy Consumption Per FL Round')
    plt.xlabel('FL Round')
    plt.ylabel('Energy (Relative Units)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('fl_energy.png')


    # Plot : Round Time

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, df["Round Time (s)"], marker='d', linestyle='-', color='tab:purple')
    plt.title('Total Wall-Clock Time Per FL Round')
    plt.xlabel('FL Round')
    plt.ylabel('Round Time (s)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('fl_round_time.png')

    print("\nCharts saved as fl_accuracy.png, fl_latency.png, fl_energy.png, and fl_round_time.png")


def runFederatedLearning(
        numRounds: int = 20,
        numDevices: int = 10,
        localEpochs: int = 1,
        dataPointsPerClient: int = 200,
        batchSize: int = 32,
):
    # Generate local client datasets + global test set
    clientLoaders, testLoader, inputDim = generateAndPartitionHealthData(
        numClients=numDevices,
        dataPointsPerClient=dataPointsPerClient,
        batchSize=batchSize,
    )

    device = torch.device("cpu")
    globalModel = SimpleNN(inputDim).to(device)

    encryptionKey = generateSymmetricKey(keySize=32)
    edgeServer = EdgeServer(encryptionKey=encryptionKey)

    devices: List[HealthMonitorDevice] = []
    for deviceId in range(numDevices):
        devices.append(HealthMonitorDevice(deviceId=deviceId, inputSize=inputDim))

    accuracyHistory: List[float] = []
    avgLatencyHistory: List[float] = []
    energyHistory: List[float] = []
    roundTimeHistory: List[float] = []

    print("--- Federated Learning Simulation: Healthcare IoT ---")
    print(
        f"Devices: {numDevices}, Rounds: {numRounds}, "
        f"Data/client: {dataPointsPerClient}, Local epochs/device: {localEpochs}"
    )

    # Federated Learning Rounds
    for roundIdx in range(1, numRounds + 1):
        roundStart = time.time()
        print(f"\n[Round {roundIdx}] -------------------------------")

        globalState = globalModel.state_dict()
        perRoundLatencies: List[float] = []
        perRoundEnergy = 0.0

        for dev in devices:
            trainLoader = clientLoaders[dev.deviceId]

            localState, localEnergy = dev.trainOnGlobalModel(
                globalStateDict=globalState,
                trainLoader=trainLoader,
                localEpochs=localEpochs,
            )
            perRoundEnergy += localEnergy

            encryptedUpdate, latency = secureChannelSend(
                payloadStateDict=localState,
                key=encryptionKey,
                protocol="MQTT+TLS",
                src=f"device-{dev.deviceId}",
                dst="edge-server",
            )
            perRoundLatencies.append(latency)

            edgeServer.receiveEncryptedUpdate(dev.deviceId, encryptedUpdate)

        aggregatedState = edgeServer.aggregateUpdates()
        globalModel.load_state_dict(aggregatedState)

        accuracy = edgeServer.evaluateGlobalModel(globalModel, testLoader)
        avgLatency = float(np.mean(perRoundLatencies)) if perRoundLatencies else 0.0
        roundTime = time.time() - roundStart

        accuracyHistory.append(accuracy)
        avgLatencyHistory.append(avgLatency)
        energyHistory.append(perRoundEnergy)
        roundTimeHistory.append(roundTime)

        print(
            f"  Accuracy: {accuracy:.2f}% | "
            f"Avg secure-channel latency: {avgLatency * 1000:.2f} ms | "
            f"Energy (relative units): {perRoundEnergy:.4f} | "
            f"Round wall-clock time: {roundTime:.3f} s"
        )

    # FL Simulation Summary
    print("\n--- FL Simulation Finished ---")
    print("Final Accuracy: {:.2f}%".format(accuracyHistory[-1]))
    print("Accuracy History:", [round(a, 2) for a in accuracyHistory])
    print("Avg Latency History (ms):", [round(l * 1000, 2) for l in avgLatencyHistory])
    print("Energy History (relative):", [round(e, 4) for e in energyHistory])
    print("Round Time History (s):", [round(t, 3) for t in roundTimeHistory])

    # Save results to CSV and generate plots
    save_results_to_csv(
        numRounds=numRounds,
        accuracyHistory=accuracyHistory,
        avgLatencyHistory=avgLatencyHistory,
        energyHistory=energyHistory,
        roundTimeHistory=roundTimeHistory,
    )
    plot_results()


# Entry point for standalone script execution
if __name__ == "__main__":
    runFederatedLearning(
        numRounds=20,  
        numDevices=10,
        localEpochs=1,
        dataPointsPerClient=200,
        batchSize=32,
    )