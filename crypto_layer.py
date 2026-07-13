"""
REAL Diffie-Hellman key exchange (X25519), matching the AA ecosystem's
cryptographic protocol described in the RBI/Sahamati spec. This is a genuine
cryptographic handshake — not a mock — used here to simulate the FIU<->FIP
secure channel establishment before "fetching" the (synthetic) bank data.
"""
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
import os
import json


def run_dhe_handshake():
    """Simulates the FIU (bank) <-> FIP (bank's data source) key exchange.
    Returns a dict describing each real cryptographic step for display."""
    steps = []

    # Step 1: FIU generates ephemeral key pair
    fiu_private = X25519PrivateKey.generate()
    fiu_public = fiu_private.public_key()
    steps.append({
        "step": "FIU generates ephemeral X25519 key pair",
        "detail": f"Public key (hex, first 16 bytes): {fiu_public.public_bytes_raw().hex()[:32]}..."
    })

    # Step 2: FIP generates its own ephemeral key pair
    fip_private = X25519PrivateKey.generate()
    fip_public = fip_private.public_key()
    steps.append({
        "step": "FIP generates its ephemeral X25519 key pair",
        "detail": f"Public key (hex, first 16 bytes): {fip_public.public_bytes_raw().hex()[:32]}..."
    })

    # Step 3: Both sides derive the same shared secret independently
    fiu_shared = fiu_private.exchange(fip_public)
    fip_shared = fip_private.exchange(fiu_public)
    assert fiu_shared == fip_shared, "Handshake failed - shared secrets do not match"

    # Step 4: Derive a symmetric key via HKDF (standard practice, matches DHE spec)
    derived_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=None,
        info=b"cprism-aa-fi-fetch",
    ).derive(fiu_shared)

    steps.append({
        "step": "Shared secret (K_DH) computed independently by both sides",
        "detail": f"Match verified: {fiu_shared == fip_shared}"
    })
    steps.append({
        "step": "Symmetric session key derived via HKDF-SHA256",
        "detail": f"Session key (hex, first 16 bytes): {derived_key.hex()[:32]}..."
    })

    return {"success": True, "steps": steps, "session_key_hex": derived_key.hex()}


def encrypt_mock_fi_payload(session_key_hex: str, payload: dict) -> str:
    """Lightweight XOR-stream demo encryption using the real derived session key,
    just to show the payload is unreadable without the key. (For demo purposes;
    production would use AES-GCM per the AA spec.)"""
    key = bytes.fromhex(session_key_hex)
    data = json.dumps(payload).encode()
    encrypted = bytes([b ^ key[i % len(key)] for i, b in enumerate(data)])
    return encrypted.hex()


def decrypt_fi_payload(session_key_hex: str, encrypted_hex: str) -> dict:
    key = bytes.fromhex(session_key_hex)
    encrypted = bytes.fromhex(encrypted_hex)
    decrypted = bytes([b ^ key[i % len(key)] for i, b in enumerate(encrypted)])
    return json.loads(decrypted.decode())
