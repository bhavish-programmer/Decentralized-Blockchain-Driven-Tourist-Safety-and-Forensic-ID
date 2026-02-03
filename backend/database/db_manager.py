"""
Database Manager for Smart Tourist Safety System
Handles SQLite and MongoDB operations
"""
import sqlite3
import os
from pathlib import Path
from datetime import datetime
from loguru import logger
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
import secrets
import json


class DatabaseManager:
    """Manages SQLite and MongoDB connections"""
    
    def __init__(self, sqlite_path=None):
        if sqlite_path is None:
            sqlite_path = os.getenv("SQLITE_PATH")
        if not sqlite_path:
            sqlite_path = str((Path(__file__).resolve().parent / 'sqlite' / 'tourist.db'))

        self.sqlite_path = sqlite_path
        self.conn = None
        self.master_key = None
        
        # Create database directory if not exists
        os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
        
        # Initialize databases
        self._init_sqlite()
        self._init_encryption()
    
    def _init_sqlite(self):
        """Initialize SQLite database"""
        logger.info(f"Initializing SQLite database: {self.sqlite_path}")
        
        # Create connection
        self.conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Read and execute schema
        schema_path = Path(__file__).parent / 'sqlite' / 'schema.sql'
        with open(schema_path, 'r') as f:
            schema = f.read()
            self.conn.executescript(schema)
        
        logger.success("SQLite database initialized")
    
    def _init_encryption(self):
        """Initialize encryption keys"""
        # In production, load from secure key management system
        # For demo, generate and store
        key_path = Path(__file__).parent / 'sqlite' / '.master_key'
        
        if key_path.exists():
            with open(key_path, 'rb') as f:
                self.master_key = f.read()
        else:
            self.master_key = AESGCM.generate_key(bit_length=256)
            with open(key_path, 'wb') as f:
                f.write(self.master_key)
            # Secure the file (chmod 600)
            os.chmod(key_path, 0o600)
        
        logger.info("Encryption keys initialized")
    
    def encrypt_data(self, plaintext):
        """Encrypt data using AES-256-GCM"""
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        
        aesgcm = AESGCM(self.master_key)
        nonce = secrets.token_bytes(12)
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        
        # Return nonce + ciphertext
        return nonce + ciphertext
    
    def decrypt_data(self, encrypted_data):
        """Decrypt data"""
        aesgcm = AESGCM(self.master_key)
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:]
        
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode('utf-8')
    
    def register_tourist(self, tourist_data):
        """
        Register a new tourist in the database
        
        Args:
            tourist_data: dict with keys:
                - did: Digital Identity
                - name: Full name
                - id_type: 'aadhaar' or 'passport'
                - id_number: Aadhaar/Passport number
                - phone: Phone number
                - email: Email
                - entry_point: Airport/Station code
                - itinerary: List of travel plans
        
        Returns:
            tourist_id: Integer ID of registered tourist
        """
        import hashlib
        
        # Generate ID hash (use precomputed if provided)
        id_hash = tourist_data.get("id_hash")
        if not id_hash:
            id_hash = hashlib.sha256(
                f"{tourist_data['id_number']}{datetime.now().isoformat()}{secrets.token_hex(16)}".encode()
            ).hexdigest()
        
        # Encrypt sensitive data
        name_encrypted = self.encrypt_data(tourist_data['name'])
        id_number_encrypted = self.encrypt_data(tourist_data['id_number'])
        phone_encrypted = self.encrypt_data(tourist_data.get('phone', ''))
        email_encrypted = self.encrypt_data(tourist_data.get('email', ''))
        
        itinerary_json = json.dumps(tourist_data.get('itinerary', []))
        itinerary_encrypted = self.encrypt_data(itinerary_json)
        
        # Insert into database
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO tourists (
                did, id_hash, name_encrypted, id_type, id_number_encrypted,
                phone_encrypted, email_encrypted, nationality, entry_point,
                expected_exit_timestamp, itinerary_encrypted,
                encryption_key_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            tourist_data['did'],
            id_hash,
            name_encrypted,
            tourist_data['id_type'],
            id_number_encrypted,
            phone_encrypted,
            email_encrypted,
            tourist_data.get('nationality'),
            tourist_data['entry_point'],
            tourist_data.get('expected_exit_timestamp'),
            itinerary_encrypted,
            'master_key_v1'
        ))
        
        self.conn.commit()
        tourist_id = cursor.lastrowid
        
        logger.success(f"Tourist registered: DID={tourist_data['did']}, ID={tourist_id}")
        
        return tourist_id, id_hash
    
    def get_tourist_by_did(self, did):
        """Get tourist information by DID (decrypted)"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM tourists WHERE did = ?', (did,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        # Decrypt sensitive fields
        tourist = dict(row)
        tourist['name'] = self.decrypt_data(row['name_encrypted'])
        tourist['id_number'] = self.decrypt_data(row['id_number_encrypted'])
        tourist['phone'] = self.decrypt_data(row['phone_encrypted'])
        tourist['email'] = self.decrypt_data(row['email_encrypted'])
        
        return tourist

    def update_last_seen(self, did, camera_id):
        """Update last seen camera + timestamp for a tourist."""
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            UPDATE tourists
            SET last_seen_camera = ?, last_seen_timestamp = ?
            WHERE did = ?
            ''',
            (camera_id, datetime.now(), did)
        )
        self.conn.commit()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
