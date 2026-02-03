"""
Database utility functions
"""
import hashlib
import secrets
from datetime import datetime
import uuid


def generate_did(prefix='tourist'):
    """
    Generate Decentralized Identifier (DID)
    Format: did:tourist:12345678
    """
    random_id = secrets.token_hex(4)
    return f"did:{prefix}:{random_id}"


def generate_feature_id():
    """Generate unique feature ID"""
    return f"feat_{uuid.uuid4().hex[:12]}"


def generate_session_id(camera_id, tracking_id):
    """Generate tracking session ID"""
    # Include microseconds + short random suffix to avoid collisions in fast loops
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
    rand = uuid.uuid4().hex[:6]
    return f"session_{camera_id}_{tracking_id}_{timestamp}_{rand}"


def generate_incident_id():
    """Generate unique incident ID"""
    return f"incident_{uuid.uuid4().hex[:12]}"


def compute_id_hash(id_number, salt=None):
    """
    Compute SHA256 hash of ID number
    
    Args:
        id_number: Aadhaar/Passport number
        salt: Optional salt (generated if not provided)
    
    Returns:
        (hash_hex, salt)
    """
    if salt is None:
        salt = secrets.token_hex(16)
    
    hash_input = f"{id_number}{salt}".encode('utf-8')
    hash_hex = hashlib.sha256(hash_input).hexdigest()
    
    return hash_hex, salt
