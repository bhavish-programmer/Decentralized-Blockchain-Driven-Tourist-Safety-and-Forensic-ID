"""
MongoDB Manager for MCPT Feature Bank
Handles tourist visual features and tracking sessions
"""
from pymongo import MongoClient, ASCENDING, DESCENDING
from datetime import datetime
import numpy as np
from loguru import logger
import os


class MongoManager:
    """Manages MongoDB operations for MCPT features"""
    
    def __init__(self, connection_string=None):
        """
        Initialize MongoDB connection
        
        Args:
            connection_string: MongoDB URI (default: localhost)
        """
        if connection_string is None:
            # Default connection for local development
            mongo_host = os.getenv('MONGO_HOST', 'localhost')
            mongo_port = os.getenv('MONGO_PORT', '27017')
            mongo_user = os.getenv('MONGO_USER', 'admin')
            mongo_pass = os.getenv('MONGO_PASS', 'Course786')
            
            connection_string = (
                f"mongodb://{mongo_user}:{mongo_pass}@{mongo_host}:{mongo_port}/"
                f"?authSource=admin&authMechanism=SCRAM-SHA-256"
            )
        
        try:
            self.client = MongoClient(connection_string)
            self.db = self.client['tourist_safety']
            
            # Collections
            self.tourist_features = self.db['tourist_features']
            self.tracking_sessions = self.db['tracking_sessions']
            
            # Create indexes
            self._create_indexes()
            
            logger.success(f"MongoDB connected: {connection_string.split('@')[1] if '@' in connection_string else 'localhost'}")
            
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise
    
    def _create_indexes(self):
        """Create indexes for performance"""
        # Tourist features indexes
        self.tourist_features.create_index([("feature_id", ASCENDING)], unique=True)
        self.tourist_features.create_index([("did", ASCENDING)])
        self.tourist_features.create_index([("capture_timestamp", DESCENDING)])
        self.tourist_features.create_index([("camera_id", ASCENDING)])
        
        # Tracking sessions indexes
        self.tracking_sessions.create_index([("session_id", ASCENDING)], unique=True)
        self.tracking_sessions.create_index([("camera_id", ASCENDING), ("tracking_id", ASCENDING)])
        self.tracking_sessions.create_index([("did", ASCENDING)])
        self.tracking_sessions.create_index([("status", ASCENDING)])
        self.tracking_sessions.create_index([("start_timestamp", DESCENDING)])
        # TTL: expire ended sessions after 48 hours to prevent unbounded growth
        try:
            self.tracking_sessions.create_index(
                [("end_timestamp", ASCENDING)],
                expireAfterSeconds=48 * 60 * 60,
                name="end_timestamp_ttl_48h"
            )
        except Exception as e:
            logger.warning(f"TTL index creation skipped: {e}")
        
        logger.info("MongoDB indexes created")
    
    def store_tourist_features(self, feature_data):
        """
        Store tourist visual features
        
        Args:
            feature_data: dict containing:
                - feature_id: Unique identifier
                - did: Digital Identity
                - reid_embedding: 512D numpy array
                - pose_keypoints: List of keypoints
                - appearance: Dict of appearance features
                - camera_id: Camera identifier
                - capture_angles: List of multi-angle captures
        
        Returns:
            inserted_id: MongoDB document ID
        """
        # Convert numpy array to list for JSON serialization
        if isinstance(feature_data.get('reid_embedding'), np.ndarray):
            feature_data['reid_embedding'] = feature_data['reid_embedding'].tolist()
        
        # Add timestamps
        feature_data['capture_timestamp'] = datetime.now()
        feature_data['created_at'] = datetime.now()
        feature_data['updated_at'] = datetime.now()
        
        result = self.tourist_features.insert_one(feature_data)
        
        logger.success(f"Tourist features stored: feature_id={feature_data['feature_id']}, did={feature_data['did']}")
        
        return str(result.inserted_id)
    
    def get_tourist_features(self, did):
        """Get all stored features for a tourist"""
        features = list(self.tourist_features.find({'did': did}).sort('capture_timestamp', DESCENDING))
        
        # Convert ObjectId to string
        for feature in features:
            feature['_id'] = str(feature['_id'])
        
        return features

    def get_all_tourist_features(self):
        """Get all tourist features (for Re-ID gallery)"""
        features = list(self.tourist_features.find({}))
        for feature in features:
            feature['_id'] = str(feature['_id'])
        return features
    
    def start_tracking_session(self, session_data):
        """
        Start a new tracking session
        
        Args:
            session_data: dict containing:
                - session_id: Unique identifier
                - camera_id: Camera identifier
                - tracking_id: ByteTrack ID
                - bbox: Initial bounding box
                - confidence: Detection confidence
        
        Returns:
            session_id: Session identifier
        """
        session = {
            'session_id': session_data['session_id'],
            'camera_id': session_data['camera_id'],
            'tracking_id': session_data['tracking_id'],
            'did': session_data.get('did'),
            'match_confidence': session_data.get('match_confidence', 0.0),
            'tracklets': [{
                'frame_id': 0,
                'timestamp': datetime.now(),
                'bbox': session_data['bbox'],
                'confidence': session_data['confidence']
            }],
            'start_timestamp': datetime.now(),
            'last_seen_timestamp': datetime.now(),
            'status': 'active',
            'total_detections': 1,
            'avg_confidence': session_data['confidence']
        }
        
        self.tracking_sessions.insert_one(session)
        
        logger.info(f"Tracking session started: {session_data['session_id']} (Camera {session_data['camera_id']}, ID #{session_data['tracking_id']})")
        
        return session_data['session_id']
    
    def update_tracking_session(self, session_id, tracklet_data):
        """
        Update tracking session with new detection
        
        Args:
            session_id: Session identifier
            tracklet_data: dict with bbox, confidence, frame_id
        """
        # Add timestamp
        tracklet_data['timestamp'] = datetime.now()
        
        # Update session
        self.tracking_sessions.update_one(
            {'session_id': session_id},
            {
                '$push': {'tracklets': tracklet_data},
                '$set': {'last_seen_timestamp': datetime.now()},
                '$inc': {'total_detections': 1}
            }
        )
    
    def end_tracking_session(self, session_id, status='exited'):
        """
        End a tracking session
        
        Args:
            session_id: Session identifier
            status: Final status (exited/lost/transferred)
        """
        session = self.tracking_sessions.find_one({'session_id': session_id})
        
        if session:
            duration = (datetime.now() - session['start_timestamp']).total_seconds()
            
            self.tracking_sessions.update_one(
                {'session_id': session_id},
                {
                    '$set': {
                        'end_timestamp': datetime.now(),
                        'duration_seconds': int(duration),
                        'status': status
                    }
                }
            )
            
            logger.info(f"Tracking session ended: {session_id} (Status: {status}, Duration: {duration:.1f}s)")
    
    def link_tracking_to_tourist(self, session_id, did, match_confidence):
        """
        Link a tracking session to a registered tourist (after Re-ID match)
        
        Args:
            session_id: Tracking session ID
            did: Tourist DID
            match_confidence: Re-ID matching confidence
        """
        self.tracking_sessions.update_one(
            {'session_id': session_id},
            {
                '$set': {
                    'did': did,
                    'match_confidence': match_confidence,
                    'updated_at': datetime.now()
                }
            }
        )
        
        logger.success(f"Tracking session {session_id} linked to DID {did} (Confidence: {match_confidence:.3f})")

    def update_tracking_best_match(self, session_id, match_data):
        """
        Store best match metadata (crop/embedding) for auditing/debugging.
        """
        self.tracking_sessions.update_one(
            {'session_id': session_id},
            {
                '$set': {
                    'best_match': match_data,
                    'updated_at': datetime.now()
                }
            }
        )
    
    def get_active_sessions(self, camera_id=None):
        """Get all active tracking sessions"""
        query = {'status': 'active'}
        if camera_id:
            query['camera_id'] = camera_id
        
        sessions = list(self.tracking_sessions.find(query).sort('start_timestamp', DESCENDING))
        
        # Convert ObjectId to string
        for session in sessions:
            session['_id'] = str(session['_id'])
        
        return sessions
    
    def get_tourist_trajectory(self, did):
        """
        Get complete trajectory of a tourist across all cameras
        
        Args:
            did: Tourist DID
        
        Returns:
            List of tracking sessions with trajectories
        """
        sessions = list(self.tracking_sessions.find({'did': did}).sort('start_timestamp', ASCENDING))
        
        trajectory = []
        for session in sessions:
            trajectory.append({
                'session_id': session['session_id'],
                'camera_id': session['camera_id'],
                'tracking_id': session['tracking_id'],
                'start_time': session['start_timestamp'],
                'end_time': session.get('end_timestamp'),
                'duration': session.get('duration_seconds'),
                'num_detections': session['total_detections'],
                'tracklets': session['tracklets']
            })
        
        return trajectory
    
    def search_similar_features(self, embedding, threshold=0.7, limit=10):
        """
        Search for similar tourist features using cosine similarity
        (Simplified version - production would use vector search index)
        
        Args:
            embedding: Query embedding (512D vector)
            threshold: Minimum similarity threshold
            limit: Maximum results to return
        
        Returns:
            List of matching tourists with similarity scores
        """
        # In production, use MongoDB Atlas Vector Search or Milvus
        # For demo, we'll do basic comparison
        
        all_features = list(self.tourist_features.find({}))
        results = []
        
        for feature in all_features:
            stored_embedding = np.array(feature['reid_embedding'])
            query_embedding = np.array(embedding)
            
            # Cosine similarity
            similarity = np.dot(query_embedding, stored_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
            )
            
            if similarity >= threshold:
                results.append({
                    'did': feature['did'],
                    'feature_id': feature['feature_id'],
                    'similarity': float(similarity),
                    'capture_timestamp': feature['capture_timestamp']
                })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:limit]
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
