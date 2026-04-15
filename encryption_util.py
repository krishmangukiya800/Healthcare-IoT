import io
import os
import time
from itertools import cycle
from typing import Dict, Tuple

import numpy as np
import torch

# Generate a random symmetric encryption key 
def generateSymmetricKey(keySize: int = 32) -> bytes:
    return os.urandom(keySize)

#  XOR-based byte-level encryption 
def _xorBytes(data: bytes, key: bytes) -> bytes:
    return bytes(d ^ k for d, k in zip(data, cycle(key)))


def encryptBytes(plaintext: bytes, key: bytes) -> bytes:
    return _xorBytes(plaintext, key)


def decryptBytes(ciphertext: bytes, key: bytes) -> bytes:
    return _xorBytes(ciphertext, key)

# Convert a PyTorch state_dict (parameter tensors) to raw bytes
def tensorDictToBytes(stateDict: Dict[str, torch.Tensor]) -> bytes:
    buffer = io.BytesIO()
    torch.save(stateDict, buffer)
    return buffer.getvalue()

# Convert raw bytes back into a tensor dictionary
def bytesToTensorDict(raw: bytes) -> Dict[str, torch.Tensor]:
    buffer = io.BytesIO(raw)
    buffer.seek(0)
    stateDict = torch.load(buffer, map_location="cpu")
    return stateDict

# Encrypt an entire tensor dictionary update using XOR
def encryptUpdate(stateDict: Dict[str, torch.Tensor], key: bytes) -> bytes:
    raw = tensorDictToBytes(stateDict)
    encrypted = encryptBytes(raw, key)
    return encrypted


def decryptUpdate(encrypted: bytes, key: bytes) -> Dict[str, torch.Tensor]:
    raw = decryptBytes(encrypted, key)
    stateDict = bytesToTensorDict(raw)
    return stateDict

# Simulate secure transmission over a network
def secureChannelSend(
    payloadStateDict: Dict[str, torch.Tensor],
    key: bytes,
    *,
    protocol: str = "MQTT+TLS",
    src: str = "device",
    dst: str = "edge",
) -> Tuple[bytes, float]:
    start = time.time()

    encryptedPayload = encryptUpdate(payloadStateDict, key)

    if "MQTT" in protocol.upper():
        networkDelay = np.random.uniform(0.01, 0.05)
    else:
        networkDelay = np.random.uniform(0.02, 0.06)

    if "TLS" in protocol.upper() or "DTLS" in protocol.upper():
        handshakeDelay = 0.01
    else:
        handshakeDelay = 0.0

    time.sleep(networkDelay + handshakeDelay)
    latency = time.time() - start

    return encryptedPayload, latency

# Server-side decryption after receiving encrypted payload
def secureChannelReceive(encryptedPayload: bytes, key: bytes) -> Dict[str, torch.Tensor]:
    stateDict = decryptUpdate(encryptedPayload, key)
    return stateDict