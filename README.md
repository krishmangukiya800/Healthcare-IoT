# Healthcare IoT Federated Learning Simulation

A Python-based simulation of a **Healthcare IoT system** where multiple health-monitoring devices collaboratively train a machine learning model using **Federated Learning (FL)** without sharing raw patient data.

## Overview
This project demonstrates how federated learning can be applied in a healthcare setting to support **privacy-aware distributed training** across IoT devices.

Instead of sending patient data to a central server, each simulated device trains a local model on its own health data and only sends encrypted model updates to an edge server. The edge server aggregates these updates using **Federated Averaging (FedAvg)** to build a better global model over multiple communication rounds.

The simulation also tracks important system and learning metrics such as:
- global model accuracy
- communication latency
- energy consumption
- round execution time

## Features
- Synthetic healthcare sensor data generation
- Multiple IoT clients participating in federated learning
- Local neural network training on each device
- Edge-server aggregation using **FedAvg**
- Simulated secure communication of model updates
- Accuracy evaluation of the global model
- CSV export of round-by-round simulation results
- Automatic generation of performance visualizations

## Tech Stack
- **Python**
- **PyTorch**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Scikit-learn**

## Project Structure
```bash
Healthcare-IoT-main/
├── main_fl_simulation.py      # Main script to run the federated learning workflow
├── iot_device.py              # Local client device logic and neural network model
├── edge_server.py             # Edge server aggregation and evaluation logic
├── data_generator.py          # Synthetic healthcare data generation and partitioning
├── encryption_util.py         # Simulated encryption and secure-channel communication
├── fl_simulation_results.csv  # Saved round-wise simulation metrics
├── fl_accuracy.png            # Plot of global accuracy over rounds
├── fl_latency.png             # Plot of average communication latency per round
├── fl_energy.png              # Plot of total energy usage per round
├── fl_round_time.png          # Plot of round execution time
├── output.png
├── Output2.png
└── Output3.png               
