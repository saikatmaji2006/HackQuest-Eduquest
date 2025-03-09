<<<<<<< HEAD
import logging
import json
import numpy as np
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import threading
from collections import defaultdict
import google.generativeai as genai
from flask import Blueprint, jsonify, request, current_app, Flask
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS
from config import APIConfig

# Set up logging
logger = logging.getLogger(__name__)

class RecommendationModel:
    """
    Advanced recommendation model for educational content that combines
    collaborative filtering, content-based filtering, and AI-powered recommendations.
    """

    def __init__(self, gemini_api=None, model_path=None):
        """Initialize the recommendation model."""
        self.gemini_api = gemini_api
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), '../data/models')
        os.makedirs(self.model_path, exist_ok=True)

        # Initialize model components
        self.content_vectors = None
        self.user_vectors = None
        self.content_metadata = {}
        self.user_history = defaultdict(list)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

        # Model state
        self.last_trained = None
        self.is_trained = False
        self.model_lock = threading.RLock()

        # Try to load pre-trained model
        self._load_model()

    def train(self, content_data: List[Dict], user_data: List[Dict], force=False) -> bool:
        """
        Train the recommendation model on content and user data.

        Args:
            content_data: List of content items with metadata
            user_data: List of user interaction data
            force: Force retraining even if recently trained

        Returns:
            bool: True if training was successful
        """
        with self.model_lock:
            # Check if we need to train
            if self.is_trained and not force:
                if self.last_trained and (datetime.now() - self.last_trained) < timedelta(hours=24):
                    logger.info("Skipping training, model was recently trained")
                    return True

            try:
                logger.info(f"Training recommendation model with {len(content_data)} content items and {len(user_data)} user interactions")

                # Process content data
                content_texts = []
                content_ids = []

                for item in content_data:
                    if 'id' not in item or 'description' not in item:
                        continue

                    content_id = item['id']
                    content_text = f"{item.get('title', '')} {item.get('description', '')} {' '.join(item.get('tags', []))}"

                    content_texts.append(content_text)
                    content_ids.append(content_id)
                    self.content_metadata[content_id] = item

                # Create content vectors using TF-IDF
                if content_texts:
                    self.content_vectors = self.vectorizer.fit_transform(content_texts)

                    # Map content IDs to vector indices
                    self.content_id_to_idx = {content_id: idx for idx, content_id in enumerate(content_ids)}
                    self.content_idx_to_id = {idx: content_id for idx, content_id in enumerate(content_ids)}

                # Process user data
                for interaction in user_data:
                    if 'user_id' not in interaction or 'content_id' not in interaction:
                        continue

                    user_id = interaction['user_id']
                    content_id = interaction['content_id']
                    interaction_type = interaction.get('type', 'view')
                    timestamp = interaction.get('timestamp', datetime.now().isoformat())
                    rating = interaction.get('rating', 1.0)

                    # Store in user history
                    self.user_history[user_id].append({
                        'content_id': content_id,
                        'type': interaction_type,
                        'timestamp': timestamp,
                        'rating': rating
                    })

                # Build user vectors based on their content interactions
                self._build_user_vectors()

                # Update model state
                self.is_trained = True
                self.last_trained = datetime.now()

                # Save model
                self._save_model()

                logger.info("Recommendation model training completed successfully")
                return True

            except Exception as e:
                logger.error(f"Error training recommendation model: {str(e)}")
                logger.error(traceback.format_exc())
                return False

    def recommend(self, user_id: str, limit: int = 5,
                  filters: Optional[Dict] = None,
                  strategy: str = 'hybrid') -> List[Dict]:
        """
        Generate personalized recommendations for a user.

        Args:
            user_id: User ID to generate recommendations for
            limit: Maximum number of recommendations to return
            filters: Optional filters to apply (e.g., content type, difficulty)
            strategy: Recommendation strategy ('collaborative', 'content', 'ai', 'hybrid')

        Returns:
            List of recommended content items with scores
        """
        with self.model_lock:
            if not self.is_trained:
                logger.warning("Recommendation model not trained yet")
                return []

            try:
                recommendations = []

                # Get user history
                user_history = self.user_history.get(user_id, [])

                # Get content IDs the user has already interacted with
                user_content_ids = set(item['content_id'] for item in user_history)

                # Choose recommendation strategy
                if strategy == 'collaborative':
                    recommendations = self._collaborative_filtering(user_id, limit, user_content_ids)
                elif strategy == 'content':
                    recommendations = self._content_based_filtering(user_id, limit, user_content_ids)
                elif strategy == 'ai':
                    recommendations = self._ai_recommendations(user_id, limit, user_content_ids)
                else:  # hybrid (default)
                    # Get recommendations from each strategy
                    collab_recs = self._collaborative_filtering(user_id, limit, user_content_ids)
                    content_recs = self._content_based_filtering(user_id, limit, user_content_ids)
                    ai_recs = self._ai_recommendations(user_id, limit, user_content_ids)

                    # Combine and deduplicate recommendations
                    all_recs = {}

                    # Weight different strategies
                    for rec in collab_recs:
                        all_recs[rec['content_id']] = rec
                        all_recs[rec['content_id']]['score'] *= 0.4  # 40% weight

                    for rec in content_recs:
                        if rec['content_id'] in all_recs:
                            all_recs[rec['content_id']]['score'] += rec['score'] * 0.3  # 30% weight
                        else:
                            rec['score'] *= 0.3
                            all_recs[rec['content_id']] = rec

                    for rec in ai_recs:
                        if rec['content_id'] in all_recs:
                            all_recs[rec['content_id']]['score'] += rec['score'] * 0.3  # 30% weight
                        else:
                            rec['score'] *= 0.3
                            all_recs[rec['content_id']] = rec

                    recommendations = list(all_recs.values())

                # Apply filters if provided
                if filters:
                    recommendations = self._apply_filters(recommendations, filters)

                # Sort by score and limit results
                recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)[:limit]

                # Enrich recommendations with metadata
                for rec in recommendations:
                    content_id = rec['content_id']
                    if content_id in self.content_metadata:
                        rec.update(self.content_metadata[content_id])

                return recommendations

            except Exception as e:
                logger.error(f"Error generating recommendations: {str(e)}")
                logger.error(traceback.format_exc())
                return []

    def _collaborative_filtering(self, user_id: str, limit: int,
                                 excluded_content_ids: set) -> List[Dict]:
        """
        Generate recommendations using collaborative filtering.

        Args:
            user_id: User ID to generate recommendations for
            limit: Maximum number of recommendations
            excluded_content_ids: Content IDs to exclude

        Returns:
            List of recommended content items with scores
        """
        recommendations = []

        # Check if we have user vectors
        if self.user_vectors is None or user_id not in self.user_vectors:
            return recommendations

        # Get user vector
        user_vector = self.user_vectors[user_id]

        # Calculate similarity with other users
        user_similarities = {}
        for other_user_id, other_vector in self.user_vectors.items():
            if other_user_id != user_id:
                similarity = cosine_similarity(user_vector.reshape(1, -1),
                                              other_vector.reshape(1, -1))[0][0]
                user_similarities[other_user_id] = similarity

        # Get top similar users
        similar_users = sorted(user_similarities.items(),
                               key=lambda x: x[1], reverse=True)[:10]

        # Get content from similar users
        content_scores = defaultdict(float)
        for similar_user_id, similarity in similar_users:
            if similarity <= 0:
                continue

            for interaction in self.user_history[similar_user_id]:
                content_id = interaction['content_id']
                if content_id in excluded_content_ids:
                    continue

                # Weight by interaction type and similarity
                weight = 1.0
                if interaction['type'] == 'complete':
                    weight = 2.0
                elif interaction['type'] == 'like':
                    weight = 3.0

                content_scores[content_id] += similarity * weight * interaction.get('rating', 1.0)

        # Convert to list of recommendations
        for content_id, score in content_scores.items():
            recommendations.append({
                'content_id': content_id,
                'score': score,
                'method': 'collaborative'
            })

        return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:limit]

    def _content_based_filtering(self, user_id: str, limit: int,
                                 excluded_content_ids: set) -> List[Dict]:
        """
        Generate recommendations using content-based filtering.

        Args:
            user_id: User ID to generate recommendations for
            limit: Maximum number of recommendations
            excluded_content_ids: Content IDs to exclude

        Returns:
            List of recommended content items with scores
        """
        recommendations = []

        # Check if we have content vectors
        if self.content_vectors is None:
            return recommendations

        # Get user history
        user_history = self.user_history.get(user_id, [])
        if not user_history:
            return recommendations

        # Build user profile based on content they've interacted with
        user_profile = np.zeros((1, self.content_vectors.shape[1]))
        profile_weight = 0

        for interaction in user_history:
            content_id = interaction['content_id']
            if content_id not in self.content_id_to_idx:
                continue

            # Get content vector
            content_idx = self.content_id_to_idx[content_id]
            content_vector = self.content_vectors[content_idx]

            # Weight by interaction type and recency
            weight = 1.0
            if interaction['type'] == 'complete':
                weight = 2.0
            elif interaction['type'] == 'like':
                weight = 3.0

            # Add to user profile
            user_profile += weight * content_vector.toarray()
            profile_weight += weight

        # Normalize user profile
        if profile_weight > 0:
            user_profile /= profile_weight

        # Calculate similarity with all content
        content_similarities = cosine_similarity(user_profile, self.content_vectors)[0]

        # Convert to recommendations
        for idx, similarity in enumerate(content_similarities):
            content_id = self.content_idx_to_id[idx]
            if content_id in excluded_content_ids:
                continue

            recommendations.append({
                'content_id': content_id,
                'score': float(similarity),
                'method': 'content'
            })

        return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:limit]

    def _ai_recommendations(self, user_id: str, limit: int,
                            excluded_content_ids: set) -> List[Dict]:
        """
        Generate recommendations using AI (Gemini API).

        Args:
            user_id: User ID to generate recommendations for
            limit: Maximum number of recommendations
            excluded_content_ids: Content IDs to exclude

        Returns:
            List of recommended content items with scores
        """
        recommendations = []

        # Check if Gemini API is available
        if not self.gemini_api:
            logger.warning("Gemini API not available for AI recommendations")
            return recommendations

        try:
            # Get user history
            user_history = self.user_history.get(user_id, [])

            # Prepare user data for Gemini API
            user_data = {
                'user_id': user_id,
                'learning_history': [],
                'interests': [],
                'available_content': []
            }

            # Add learning history
            for interaction in user_history:
                content_id = interaction['content_id']
                if content_id in self.content_metadata:
                    content_info = self.content_metadata[content_id]
                    user_data['learning_history'].append({
                        'content_id': content_id,
                        'title': content_info.get('title', ''),
                        'type': content_info.get('type', ''),
                        'tags': content_info.get('tags', []),
                        'interaction_type': interaction['type'],
                        'timestamp': interaction['timestamp']
                    })

            # Extract interests from user history
            interest_counts = defaultdict(int)
            for interaction in user_history:
                content_id = interaction['content_id']
                if content_id in self.content_metadata:
                    for tag in self.content_metadata[content_id].get('tags', []):
                        interest_counts[tag] += 1

            # Get top interests
            user_data['interests'] = [
                tag for tag, _ in sorted(interest_counts.items(),
                                        key=lambda x: x[1], reverse=True)[:10]
            ]

            # Add available content (excluding already consumed content)
            for content_id, metadata in self.content_metadata.items():
                if content_id not in excluded_content_ids:
                    user_data['available_content'].append({
                        'content_id': content_id,
                        'title': metadata.get('title', ''),
                        'description': metadata.get('description', ''),
                        'type': metadata.get('type', ''),
                        'tags': metadata.get('tags', []),
                        'difficulty': metadata.get('difficulty', 'medium')
                    })

            # Call Gemini API for recommendations
            prompt = f"""
            Based on the user's learning history and interests, recommend the most relevant educational
            content from the available options. Focus on helping the user build a coherent learning
            path that advances their skills progressively. Consider content difficulty, relevance to
            user interests, and potential learning outcomes.

            User data: {json.dumps(user_data, indent=2)}

            Return a JSON array with the recommended content IDs and scores between 0 and 1,
            with 1 being the highest relevance. Example:
            [
                {{"content_id": "content123", "score": 0.95}},
                {{"content_id": "content456", "score": 0.85}}
            ]
            """

            # Get recommendations from Gemini API
            response = self.gemini_api.generate_content(prompt)
            result = response.text

            # Parse results
            try:
                ai_recommendations = json.loads(result)
                for rec in ai_recommendations:
                    if 'content_id' in rec and 'score' in rec:
                        content_id = rec['content_id']
                        if content_id not in excluded_content_ids:
                            recommendations.append({
                                'content_id': content_id,
                                'score': float(rec['score']),
                                'method': 'ai'
                            })

                            # Limit results
                            if len(recommendations) >= limit:
                                break
            except Exception as e:
                logger.error(f"Error parsing Gemini API response: {str(e)}")
                logger.error(f"Raw response: {result}")

        except Exception as e:
            logger.error(f"Error getting AI recommendations: {str(e)}")
            logger.error(traceback.format_exc())

        return recommendations

    def _apply_filters(self, recommendations: List[Dict], filters: Dict) -> List[Dict]:
        """
        Apply filters to recommendations.

        Args:
            recommendations: List of recommendations
            filters: Dictionary of filters to apply

        Returns:
            Filtered list of recommendations
        """
        filtered_recs = []

        for rec in recommendations:
            # Get content ID
            content_id = rec['content_id']
            if content_id not in self.content_metadata:
                continue

            # Get content metadata
            metadata = self.content_metadata[content_id]

            # Check each filter
            matches_all = True

            for filter_key, filter_value in filters.items():
                if filter_key == 'content_type' and filter_value != metadata.get('type'):
                    matches_all = False
                    break
                elif filter_key == 'difficulty' and filter_value != metadata.get('difficulty'):
                    matches_all = False
                    break
                elif filter_key == 'tags' and not set(filter_value).issubset(set(metadata.get('tags', []))):
                    matches_all = False
                    break
                elif filter_key == 'min_duration' and metadata.get('duration', 0) < filter_value:
                    matches_all = False
                    break
                elif filter_key == 'max_duration' and metadata.get('duration', 0) > filter_value:
                    matches_all = False
                    break

            if matches_all:
                filtered_recs.append(rec)

        return filtered_recs

    def _build_user_vectors(self):
        """
        Build user vectors based on their content interactions.
        """
        self.user_vectors = {}

        # Check if we have content vectors
        if self.content_vectors is None:
            return

        # For each user
        for user_id, interactions in self.user_history.items():
            if not interactions:
                continue

            # Initialize user vector
            user_vector = np.zeros((1, self.content_vectors.shape[1]))
            vector_weight = 0

            # Add content vectors to user vector
            for interaction in interactions:
                content_id = interaction['content_id']
                if content_id not in self.content_id_to_idx:
                    continue

                # Get content vector
                content_idx = self.content_id_to_idx[content_id]
                content_vector = self.content_vectors[content_idx]

                # Weight by interaction type and recency
                weight = 1.0
                if interaction['type'] == 'complete':
                    weight = 2.0
                elif interaction['type'] == 'like':
                    weight = 3.0

                # Add to user vector
                user_vector += weight * content_vector.toarray()
                vector_weight += weight

            # Normalize user vector
            if vector_weight > 0:
                user_vector /= vector_weight

                # Store user vector
                self.user_vectors[user_id] = user_vector

    def _save_model(self):
        """
        Save model to disk.
        """
        try:
            # Create model file paths
            vectorizer_path = os.path.join(self.model_path, 'vectorizer.pkl')
            content_vectors_path = os.path.join(self.model_path, 'content_vectors.npz')
            metadata_path = os.path.join(self.model_path, 'metadata.json')

            # Save vectorizer
            joblib.dump(self.vectorizer, vectorizer_path)

            # Save content vectors
            if self.content_vectors is not None:
                np.savez_compressed(
                    content_vectors_path,
                    data=self.content_vectors.data,
                    indices=self.content_vectors.indices,
                    indptr=self.content_vectors.indptr,
                    shape=self.content_vectors.shape
                )

            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump({
                    'last_trained': self.last_trained.isoformat() if self.last_trained else None,
                    'content_id_to_idx': self.content_id_to_idx if hasattr(self, 'content_id_to_idx') else {},
                    'content_metadata': self.content_metadata,
                    'user_history': {k: v for k, v in self.user_history.items()}
                }, f)

            logger.info(f"Model saved to {self.model_path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            logger.error(traceback.format_exc())

    def _load_model(self):
        """
        Load model from disk.
        """
        try:
            # Create model file paths
            vectorizer_path = os.path.join(self.model_path, 'vectorizer.pkl')
            content_vectors_path = os.path.join(self.model_path, 'content_vectors.npz')
            metadata_path = os.path.join(self.model_path, 'metadata.json')

            # Check if files exist
            if not os.path.exists(vectorizer_path) or not os.path.exists(content_vectors_path) or not os.path.exists(metadata_path):
                logger.info("Model files not found, starting with empty model")
                return

            # Load vectorizer
            self.vectorizer = joblib.load(vectorizer_path)

            # Load content vectors
            content_vectors_npz = np.load(content_vectors_path)
            from scipy.sparse import csr_matrix
            self.content_vectors = csr_matrix(
                (content_vectors_npz['data'], content_vectors_npz['indices'], content_vectors_npz['indptr']),
                shape=tuple(content_vectors_npz['shape'])
            )

            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

                self.last_trained = datetime.fromisoformat(metadata['last_trained']) if metadata['last_trained'] else None
                self.content_id_to_idx = metadata['content_id_to_idx']
                self.content_idx_to_id = {int(idx): content_id for content_id, idx in self.content_id_to_idx.items()}
                self.content_metadata = metadata['content_metadata']
                self.user_history = defaultdict(list)
                for user_id, history in metadata['user_history'].items():
                    self.user_history[user_id] = history

            # Build user vectors
            self._build_user_vectors()

            # Update model state
            self.is_trained = True

            logger.info(f"Model loaded from {self.model_path}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())

            # Reset model state
            self.content_vectors = None
            self.user_vectors = None
            self.content_metadata = {}
            self.user_history = defaultdict(list)
            self.is_trained = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert model metadata to dictionary."""
        return {
            'is_trained': self.is_trained,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'num_users': len(self.user_vectors) if self.user_vectors is not None else 0,
            'num_content_items': len(self.content_metadata),
            'model_path': self.model_path
        }

class GeminiAPI:
    """
    Wrapper for Google's Gemini API for AI-powered recommendations and content generation.
    """

    def __init__(self, api_key: str = "YOUR_GEMINI_API_KEY", model: str = "gemini-pro"):
        """
        Initialize the Gemini API wrapper.

        Args:
            api_key: Google API key for Gemini
            model: Gemini model to use
        """
        self.api_key = api_key
        self.model = model

        # Configure the Gemini API
        genai.configure(api_key=api_key)

        # Get the model
        self.genai_model = genai.GenerativeModel(model)

    def generate_content(self, prompt: str, temperature: float = 0.7,
                         max_tokens: Optional[int] = None) -> Any:
        """
        Generate content using the Gemini API.

        Args:
            prompt: Text prompt for content generation
            temperature: Sampling temperature (higher = more creative)
            max_tokens: Maximum number of tokens to generate

        Returns:
            Response from the Gemini API
        """
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]

        response = self.genai_model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        return response

    def analyze_text(self, text: str, analysis_type: str = 'general') -> Dict[str, Any]:
        """
        Analyze text using Gemini API.

        Args:
            text: Text to analyze
            analysis_type: Type of analysis to perform

        Returns:
            Analysis results
        """
        prompts = {
            'general': f"Analyze the following text and provide key insights:\n\n{text}",
            'sentiment': f"Analyze the sentiment of this text:\n\n{text}",
            'topics': f"Extract main topics from this text:\n\n{text}",
            'summary': f"Provide a concise summary of:\n\n{text}"
        }

        prompt = prompts.get(analysis_type, prompts['general'])
        response = self.generate_content(prompt, temperature=0.3)

        return {
            'analysis': response.text,
            'type': analysis_type,
            'timestamp': datetime.now().isoformat()
        }

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
import json

db = SQLAlchemy()

class User(db.Model, UserMixin):
    __tablename__ = 'users'

    id = db.Column(db.String(36), primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    bio = db.Column(db.Text)
    avatar_url = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    role = db.Column(db.String(20), default='learner')  # learner, educator, admin
    xp_points = db.Column(db.Integer, default=0)
    level = db.Column(db.Integer, default=1)
    preferences = db.Column(db.Text)  # JSON string of user preferences

    # Relationships
    enrollments = db.relationship('Enrollment', back_populates='user', cascade='all, delete-orphan')
    achievements = db.relationship('UserAchievement', back_populates='user', cascade='all, delete-orphan')
    badges = db.relationship('UserBadge', back_populates='user', cascade='all, delete-orphan')
    credentials = db.relationship('Credential', back_populates='user', cascade='all, delete-orphan')
    interactions = db.relationship('UserInteraction', back_populates='user', cascade='all, delete-orphan')

    def __init__(self, username, email, password, first_name=None, last_name=None):
        self.id = str(uuid.uuid4())
        self.username = username
        self.email = email
        self.set_password(password)
        self.first_name = first_name
        self.last_name = last_name
        self.created_at = datetime.utcnow()

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def add_xp(self, points):
        self.xp_points += points
        self.update_level()

    def update_level(self):
        # Simple level calculation: level = 1 + floor(xp / 1000)
        new_level = 1 + self.xp_points // 1000
        if new_level > self.level:
            self.level = new_level
            return True
        return False

    def get_preferences(self):
        if self.preferences:
            return json.loads(self.preferences)
        return {}

    def set_preferences(self, preferences):
        self.preferences = json.dumps(preferences)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'bio': self.bio,
            'avatar_url': self.avatar_url,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'role': self.role,
            'xp_points': self.xp_points,
            'level': self.level
        }

class Course(db.Model):
    __tablename__ = 'courses'

    id = db.Column(db.String(36), primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    image_url = db.Column(db.String(255))
    creator_id = db.Column(db.String(36), db.ForeignKey('users.id'))
    category = db.Column(db.String(50))
    difficulty = db.Column(db.String(20))  # beginner, intermediate, advanced
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = db.Column(db.String(20), default='draft')  # draft, published, archived
    tags = db.Column(db.Text)  # JSON array of tags
    prerequisites = db.Column(db.Text)  # JSON array of prerequisite course IDs
    estimated_duration = db.Column(db.Integer)  # in minutes

    # Relationships
    creator = db.relationship('User', backref='created_courses')
    modules = db.relationship('Module', back_populates='course', cascade='all, delete-orphan', order_by='Module.order')
    enrollments = db.relationship('Enrollment', back_populates='course', cascade='all, delete-orphan')

    def __init__(self, title, description=None, creator_id=None):
        self.id = str(uuid.uuid4())
        self.title = title
        self.description = description
        self.creator_id = creator_id
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def set_tags(self, tags_list):
        self.tags = json.dumps(tags_list)

    def get_tags(self):
        if self.tags:
            return json.loads(self.tags)
        return []

    def set_prerequisites(self, prereq_list):
        self.prerequisites = json.dumps(prereq_list)

    def get_prerequisites(self):
        if self.prerequisites:
            return json.loads(self.prerequisites)
        return []

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'image_url': self.image_url,
            'creator_id': self.creator_id,
            'category': self.category,
            'difficulty': self.difficulty,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'status': self.status,
            'tags': self.get_tags(),
            'prerequisites': self.get_prerequisites(),
            'estimated_duration': self.estimated_duration,
            'modules_count': len(self.modules) if self.modules else 0
        }

class Module(db.Model):
    __tablename__ = 'modules'

    id = db.Column(db.String(36), primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    course_id = db.Column(db.String(36), db.ForeignKey('courses.id'), nullable=False)
    order = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    course = db.relationship('Course', back_populates='modules')
    lessons = db.relationship('Lesson', back_populates='module', cascade='all, delete-orphan', order_by='Lesson.order')

    def __init__(self, title, course_id, order, description=None):
        self.id = str(uuid.uuid4())
        self.title = title
        self.course_id = course_id
        self.order = order
        self.description = description
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'course_id': self.course_id,
            'order': self.order,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'lessons_count': len(self.lessons) if self.lessons else 0
        }

class Lesson(db.Model):
    __tablename__ = 'lessons'

    id = db.Column(db.String(36), primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    module_id = db.Column(db.String(36), db.ForeignKey('modules.id'), nullable=False)
    content_type = db.Column(db.String(20))  # video, text, quiz, interactive
    content = db.Column(db.Text)  # Could be HTML content, video URL, or JSON for interactive content
    order = db.Column(db.Integer, nullable=False)
    estimated_duration = db.Column(db.Integer)  # in minutes
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    module = db.relationship('Module', back_populates='lessons')
    quiz_questions = db.relationship('QuizQuestion', back_populates='lesson', cascade='all, delete-orphan')
    resources = db.relationship('LessonResource', back_populates='lesson', cascade='all, delete-orphan')
    interactions = db.relationship('UserInteraction', back_populates='lesson', cascade='all, delete-orphan')

    def __init__(self, title, module_id, content_type, content, order):
        self.id = str(uuid.uuid4())
        self.title = title
        self.module_id = module_id
        self.content_type = content_type
        self.content = content
        self.order = order
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'module_id': self.module_id,
            'content_type': self.content_type,
            'content': self.content,
            'order': self.order,
            'estimated_duration': self.estimated_duration,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Enrollment(db.Model):
    __tablename__ = 'enrollments'

    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    course_id = db.Column(db.String(36), db.ForeignKey('courses.id'), nullable=False)
    enrolled_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_accessed = db.Column(db.DateTime)
    completion_status = db.Column(db.String(20), default='in_progress')  # not_started, in_progress, completed
    progress_data = db.Column(db.Text)  # JSON data tracking lesson completion

    # Relationships
    user = db.relationship('User', back_populates='enrollments')
    course = db.relationship('Course', back_populates='enrollments')

    def __init__(self, user_id, course_id):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.course_id = course_id
        self.enrolled_at = datetime.utcnow()
        self.completion_status = 'not_started'

    def get_progress_data(self):
        if self.progress_data:
            return json.loads(self.progress_data)
        return {}

    def set_progress_data(self, progress_data):
        self.progress_data = json.dumps(progress_data)

    def calculate_progress_percentage(self):
        progress_data = self.get_progress_data()
        if not progress_data or 'completed_lessons' not in progress_data:
            return 0

        total_lessons = progress_data.get('total_lessons', 1)
        completed_lessons = len(progress_data.get('completed_lessons', []))

        if total_lessons == 0:
            return 0

        return min(100, int(completed_lessons / total_lessons * 100))

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'course_id': self.course_id,
            'enrolled_at': self.enrolled_at.isoformat() if self.enrolled_at else None,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'completion_status': self.completion_status,
            'progress_data': self.get_progress_data(),
            'progress_percentage': self.calculate_progress_percentage()
        }

class QuizQuestion(db.Model):
    __tablename__ = 'quiz_questions'

    id = db.Column(db.String(36), primary_key=True)
    lesson_id = db.Column(db.String(36), db.ForeignKey('lessons.id'), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    question_type = db.Column(db.String(20))  # multiple_choice, true_false, fill_blank, essay
    options = db.Column(db.Text)  # JSON array of options (for multiple choice)
    correct_answer = db.Column(db.Text)  # For fill_blank, multiple_choice (correct option index), true_false
    explanation = db.Column(db.Text)  # Explanation of the answer
    points = db.Column(db.Integer, default=1)
    order = db.Column(db.Integer)

    # Relationships
    lesson = db.relationship('Lesson', back_populates='quiz_questions')

    def __init__(self, lesson_id, question_text, question_type, correct_answer, order=0, points=1):
        self.id = str(uuid.uuid4())
        self.lesson_id = lesson_id
        self.question_text = question_text
        self.question_type = question_type
        self.correct_answer = correct_answer
        self.order = order
        self.points = points

    def set_options(self, options_list):
        self.options = json.dumps(options_list)

    def get_options(self):
        if self.options:
            return json.loads(self.options)
        return []

    def to_dict(self):
        return {
            'id': self.id,
            'lesson_id': self.lesson_id,
            'question_text': self.question_text,
            'question_type': self.question_type,
            'options': self.get_options(),
            'explanation': self.explanation,
            'points': self.points,
            'order': self.order
        }

class LessonResource(db.Model):
    __tablename__ = 'lesson_resources'

    id = db.Column(db.String(36), primary_key=True)
    lesson_id = db.Column(db.String(36), db.ForeignKey('lessons.id'), nullable=False)
    title = db.Column(db.String(100), nullable=False)
    resource_type = db.Column(db.String(20))  # pdf, link, image, document
    url = db.Column(db.String(255))
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    lesson = db.relationship('Lesson', back_populates='resources')

    def __init__(self, lesson_id, title, resource_type, url, description=None):
        self.id = str(uuid.uuid4())
        self.lesson_id = lesson_id
        self.title = title
        self.resource_type = resource_type
        self.url = url
        self.description = description
        self.created_at = datetime.utcnow()

    def to_dict(self):
        return {
            'id': self.id,
            'lesson_id': self.lesson_id,
            'title': self.title,
            'resource_type': self.resource_type,
            'url': self.url,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class UserInteraction(db.Model):
    __tablename__ = 'user_interactions'

    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    lesson_id = db.Column(db.String(36), db.ForeignKey('lessons.id'), nullable=False)
    interaction_type = db.Column(db.String(20))  # view, complete, like, bookmark, rate
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    rating = db.Column(db.Float)  # For rate interactions
    metadata = db.Column(db.Text)  # JSON field for additional interaction data

    # Relationships
    user = db.relationship('User', back_populates='interactions')
    lesson = db.relationship('Lesson', back_populates='interactions')

    def __init__(self, user_id, lesson_id, interaction_type):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.lesson_id = lesson_id
        self.interaction_type = interaction_type
        self.timestamp = datetime.utcnow()

    def set_metadata(self, metadata_dict):
        self.metadata = json.dumps(metadata_dict)

    def get_metadata(self):
        if self.metadata:
            return json.loads(self.metadata)
        return {}

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'lesson_id': self.lesson_id,
            'interaction_type': self.interaction_type,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'rating': self.rating,
            'metadata': self.get_metadata()
        }

class Achievement(db.Model):
    __tablename__ = 'achievements'

    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    icon_url = db.Column(db.String(255))
    xp_reward = db.Column(db.Integer, default=0)
    requirement_type = db.Column(db.String(50))  # course_completion, lesson_count, quiz_score, etc.
    requirement_data = db.Column(db.Text)  # JSON data specifying exact requirements

    # Relationships
    user_achievements = db.relationship('UserAchievement', back_populates='achievement')

    def __init__(self, name, description, requirement_type, xp_reward=0):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.requirement_type = requirement_type
        self.xp_reward = xp_reward

    def set_requirement_data(self, requirement_data):
        self.requirement_data = json.dumps(requirement_data)

    def get_requirement_data(self):
        if self.requirement_data:
            return json.loads(self.requirement_data)
        return {}

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'icon_url': self.icon_url,
            'xp_reward': self.xp_reward,
            'requirement_type': self.requirement_type,
            'requirement_data': self.get_requirement_data()
        }

class UserAchievement(db.Model):
    __tablename__ = 'user_achievements'

    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    achievement_id = db.Column(db.String(36), db.ForeignKey('achievements.id'), nullable=False)
    achieved_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    user = db.relationship('User', back_populates='achievements')
    achievement = db.relationship('Achievement', back_populates='user_achievements')

    def __init__(self, user_id, achievement_id):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.achievement_id = achievement_id
        self.achieved_at = datetime.utcnow()

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'achievement_id': self.achievement_id,
            'achieved_at': self.achieved_at.isoformat() if self.achieved_at else None
        }

class Badge(db.Model):
    __tablename__ = 'badges'

    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    icon_url = db.Column(db.String(255))
    badge_type = db.Column(db.String(50))  # skill_badge, course_badge, special_badge
    badge_level = db.Column(db.String(20))  # beginner, intermediate, advanced, expert

    # Relationships
    user_badges = db.relationship('UserBadge', back_populates='badge')

    def __init__(self, name, description, badge_type, badge_level='beginner'):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.badge_type = badge_type
        self.badge_level = badge_level

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'icon_url': self.icon_url,
            'badge_type': self.badge_type,
            'badge_level': self.badge_level
        }

class UserBadge(db.Model):
    __tablename__ = 'user_badges'

    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    badge_id = db.Column(db.String(36), db.ForeignKey('badges.id'), nullable=False)
    awarded_at = db.Column(db.DateTime, default=datetime.utcnow)
    awarded_by = db.Column(db.String(36), db.ForeignKey('users.id'))  # If awarded by another user (e.g., instructor)

    # Relationships
    user = db.relationship('User', back_populates='badges', foreign_keys=[user_id])
    badge = db.relationship('Badge', back_populates='user_badges')
    awarder = db.relationship('User', foreign_keys=[awarded_by])

    def __init__(self, user_id, badge_id, awarded_by=None):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.badge_id = badge_id
        self.awarded_at = datetime.utcnow()
        self.awarded_by = awarded_by

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'badge_id': self.badge_id,
            'awarded_at': self.awarded_at.isoformat() if self.awarded_at else None,
            'awarded_by': self.awarded_by
        }

class Credential(db.Model):
    __tablename__ = 'credentials'

    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(100), nullable=False)
    issuer = db.Column(db.String(100))
    issue_date = db.Column(db.Date)
    expiry_date = db.Column(db.Date)
    credential_type = db.Column(db.String(50))  # certificate, degree, badge, license
    verification_url = db.Column(db.String(255))
    credential_id = db.Column(db.String(100))  # External ID for the credential
    metadata = db.Column(db.Text)  # JSON data with additional credential info

    # Relationships
    user = db.relationship('User', back_populates='credentials')

    def __init__(self, user_id, title, issuer, credential_type, issue_date=None):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.title = title
        self.issuer = issuer
        self.credential_type = credential_type
        self.issue_date = issue_date or datetime.utcnow().date()

    def set_metadata(self, metadata_dict):
        self.metadata = json.dumps(metadata_dict)

    def get_metadata(self):
        if self.metadata:
            return json.loads(self.metadata)
        return {}

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'issuer': self.issuer,
            'issue_date': self.issue_date.isoformat() if self.issue_date else None,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'credential_type': self.credential_type,
            'verification_url': self.verification_url,
            'credential_id': self.credential_id,
            'metadata': self.get_metadata()
        }

class Notification(db.Model):
    __tablename__ = 'notifications'

    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    notification_type = db.Column(db.String(50))  # achievement, reminder, announcement, feedback
    related_id = db.Column(db.String(36))  # ID of related entity (course, achievement, etc.)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_read = db.Column(db.Boolean, default=False)

    # Relationships
    user = db.relationship('User', backref='notifications')

    def __init__(self, user_id, title, message, notification_type, related_id=None):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.title = title
        self.message = message
        self.notification_type = notification_type
        self.related_id = related_id
        self.created_at = datetime.utcnow()
        self.is_read = False

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'message': self.message,
            'notification_type': self.notification_type,
            'related_id': self.related_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_read': self.is_read
        }

# Example usage
if __name__ == "__main__":
    # Initialize Gemini API with your actual API key
    gemini = GeminiAPI(api_key="YOUR_GEMINI_API_KEY")

    # Initialize recommendation model
    model = RecommendationModel(gemini_api=gemini)

    # Test content generation
    response = gemini.generate_content("Create a lesson plan about Python")
    print(response.text)

    # Test text analysis
    analysis = gemini.analyze_text("AI has revolutionized the way we learn.", analysis_type='sentiment')
    print(analysis)

    # Get recommendations
    recommendations = model.recommend(
        user_id="user123",
        limit=5,
        strategy="hybrid"
    )
    print(recommendations)

# Create blueprint
ai_bp = Blueprint('ai', __name__, url_prefix='/api/v1/ai')

from utils.gemini_integration import GeminiAPI

gemini = GeminiAPI(api_key="YOUR_GEMINI_API_KEY")

@ai_bp.route('/')
def index():
    """API root endpoint."""
    return jsonify({
        'name': 'Eduquest AI API',
        'version': current_app.config.get('API_VERSION', 'v1'),
        'status': 'operational',
        'endpoints': {
            'root': '/api/v1/ai/',
            'health': '/api/v1/ai/health',
            'generate': '/api/v1/ai/generate'
        }
    })

@ai_bp.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@ai_bp.route('/generate', methods=['POST'])
def generate():
    """Generate AI content."""
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    data = request.get_json()
    if 'prompt' not in data:
        return jsonify({'error': 'Missing prompt in request body'}), 400
    
    try:
        response = gemini.generate_content(data['prompt'])
        return jsonify({
            'success': True,
            'prompt': data['prompt'],
            'response': response.text,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        current_app.logger.error(f"Error: {str(e)}")
        return jsonify({'error': 'Failed to generate content'}), 500

# Import routes after blueprint creation to avoid circular imports
from . import routes

# Version information
__version__ = '1.0.0'

# Export blueprint
__all__ = ['ai_bp']

def create_app(config_class=None):
    """Initialize Flask application."""
    app = Flask(__name__)
    
    # Configure app
    config_class = config_class or APIConfig
    app.config.from_object(config_class)
    
    # Setup middleware
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Register blueprint
    from api import ai_bp
    app.register_blueprint(ai_bp)
    
    # Add root route
    @app.route('/')
    def index():
        return jsonify({
            'name': 'Eduquest API',
            'version': app.config.get('API_VERSION', 'v1'),
            'endpoints': {
                'ai': '/api/v1/ai/'
            }
        })
    
    return app

# Create app instance
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    port = int(os.environ.get('PORT', 5000))
=======
import logging
import json
import numpy as np
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import threading
from collections import defaultdict
import google.generativeai as genai
from flask import Blueprint, jsonify, request, current_app, Flask
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS
from config import APIConfig

# Set up logging
logger = logging.getLogger(__name__)

class RecommendationModel:
    """
    Advanced recommendation model for educational content that combines
    collaborative filtering, content-based filtering, and AI-powered recommendations.
    """

    def __init__(self, gemini_api=None, model_path=None):
        """Initialize the recommendation model."""
        self.gemini_api = gemini_api
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), '../data/models')
        os.makedirs(self.model_path, exist_ok=True)

        # Initialize model components
        self.content_vectors = None
        self.user_vectors = None
        self.content_metadata = {}
        self.user_history = defaultdict(list)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')

        # Model state
        self.last_trained = None
        self.is_trained = False
        self.model_lock = threading.RLock()

        # Try to load pre-trained model
        self._load_model()

    def train(self, content_data: List[Dict], user_data: List[Dict], force=False) -> bool:
        """
        Train the recommendation model on content and user data.

        Args:
            content_data: List of content items with metadata
            user_data: List of user interaction data
            force: Force retraining even if recently trained

        Returns:
            bool: True if training was successful
        """
        with self.model_lock:
            # Check if we need to train
            if self.is_trained and not force:
                if self.last_trained and (datetime.now() - self.last_trained) < timedelta(hours=24):
                    logger.info("Skipping training, model was recently trained")
                    return True

            try:
                logger.info(f"Training recommendation model with {len(content_data)} content items and {len(user_data)} user interactions")

                # Process content data
                content_texts = []
                content_ids = []

                for item in content_data:
                    if 'id' not in item or 'description' not in item:
                        continue

                    content_id = item['id']
                    content_text = f"{item.get('title', '')} {item.get('description', '')} {' '.join(item.get('tags', []))}"

                    content_texts.append(content_text)
                    content_ids.append(content_id)
                    self.content_metadata[content_id] = item

                # Create content vectors using TF-IDF
                if content_texts:
                    self.content_vectors = self.vectorizer.fit_transform(content_texts)

                    # Map content IDs to vector indices
                    self.content_id_to_idx = {content_id: idx for idx, content_id in enumerate(content_ids)}
                    self.content_idx_to_id = {idx: content_id for idx, content_id in enumerate(content_ids)}

                # Process user data
                for interaction in user_data:
                    if 'user_id' not in interaction or 'content_id' not in interaction:
                        continue

                    user_id = interaction['user_id']
                    content_id = interaction['content_id']
                    interaction_type = interaction.get('type', 'view')
                    timestamp = interaction.get('timestamp', datetime.now().isoformat())
                    rating = interaction.get('rating', 1.0)

                    # Store in user history
                    self.user_history[user_id].append({
                        'content_id': content_id,
                        'type': interaction_type,
                        'timestamp': timestamp,
                        'rating': rating
                    })

                # Build user vectors based on their content interactions
                self._build_user_vectors()

                # Update model state
                self.is_trained = True
                self.last_trained = datetime.now()

                # Save model
                self._save_model()

                logger.info("Recommendation model training completed successfully")
                return True

            except Exception as e:
                logger.error(f"Error training recommendation model: {str(e)}")
                logger.error(traceback.format_exc())
                return False

    def recommend(self, user_id: str, limit: int = 5,
                  filters: Optional[Dict] = None,
                  strategy: str = 'hybrid') -> List[Dict]:
        """
        Generate personalized recommendations for a user.

        Args:
            user_id: User ID to generate recommendations for
            limit: Maximum number of recommendations to return
            filters: Optional filters to apply (e.g., content type, difficulty)
            strategy: Recommendation strategy ('collaborative', 'content', 'ai', 'hybrid')

        Returns:
            List of recommended content items with scores
        """
        with self.model_lock:
            if not self.is_trained:
                logger.warning("Recommendation model not trained yet")
                return []

            try:
                recommendations = []

                # Get user history
                user_history = self.user_history.get(user_id, [])

                # Get content IDs the user has already interacted with
                user_content_ids = set(item['content_id'] for item in user_history)

                # Choose recommendation strategy
                if strategy == 'collaborative':
                    recommendations = self._collaborative_filtering(user_id, limit, user_content_ids)
                elif strategy == 'content':
                    recommendations = self._content_based_filtering(user_id, limit, user_content_ids)
                elif strategy == 'ai':
                    recommendations = self._ai_recommendations(user_id, limit, user_content_ids)
                else:  # hybrid (default)
                    # Get recommendations from each strategy
                    collab_recs = self._collaborative_filtering(user_id, limit, user_content_ids)
                    content_recs = self._content_based_filtering(user_id, limit, user_content_ids)
                    ai_recs = self._ai_recommendations(user_id, limit, user_content_ids)

                    # Combine and deduplicate recommendations
                    all_recs = {}

                    # Weight different strategies
                    for rec in collab_recs:
                        all_recs[rec['content_id']] = rec
                        all_recs[rec['content_id']]['score'] *= 0.4  # 40% weight

                    for rec in content_recs:
                        if rec['content_id'] in all_recs:
                            all_recs[rec['content_id']]['score'] += rec['score'] * 0.3  # 30% weight
                        else:
                            rec['score'] *= 0.3
                            all_recs[rec['content_id']] = rec

                    for rec in ai_recs:
                        if rec['content_id'] in all_recs:
                            all_recs[rec['content_id']]['score'] += rec['score'] * 0.3  # 30% weight
                        else:
                            rec['score'] *= 0.3
                            all_recs[rec['content_id']] = rec

                    recommendations = list(all_recs.values())

                # Apply filters if provided
                if filters:
                    recommendations = self._apply_filters(recommendations, filters)

                # Sort by score and limit results
                recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)[:limit]

                # Enrich recommendations with metadata
                for rec in recommendations:
                    content_id = rec['content_id']
                    if content_id in self.content_metadata:
                        rec.update(self.content_metadata[content_id])

                return recommendations

            except Exception as e:
                logger.error(f"Error generating recommendations: {str(e)}")
                logger.error(traceback.format_exc())
                return []

    def _collaborative_filtering(self, user_id: str, limit: int,
                                 excluded_content_ids: set) -> List[Dict]:
        """
        Generate recommendations using collaborative filtering.

        Args:
            user_id: User ID to generate recommendations for
            limit: Maximum number of recommendations
            excluded_content_ids: Content IDs to exclude

        Returns:
            List of recommended content items with scores
        """
        recommendations = []

        # Check if we have user vectors
        if self.user_vectors is None or user_id not in self.user_vectors:
            return recommendations

        # Get user vector
        user_vector = self.user_vectors[user_id]

        # Calculate similarity with other users
        user_similarities = {}
        for other_user_id, other_vector in self.user_vectors.items():
            if other_user_id != user_id:
                similarity = cosine_similarity(user_vector.reshape(1, -1),
                                              other_vector.reshape(1, -1))[0][0]
                user_similarities[other_user_id] = similarity

        # Get top similar users
        similar_users = sorted(user_similarities.items(),
                               key=lambda x: x[1], reverse=True)[:10]

        # Get content from similar users
        content_scores = defaultdict(float)
        for similar_user_id, similarity in similar_users:
            if similarity <= 0:
                continue

            for interaction in self.user_history[similar_user_id]:
                content_id = interaction['content_id']
                if content_id in excluded_content_ids:
                    continue

                # Weight by interaction type and similarity
                weight = 1.0
                if interaction['type'] == 'complete':
                    weight = 2.0
                elif interaction['type'] == 'like':
                    weight = 3.0

                content_scores[content_id] += similarity * weight * interaction.get('rating', 1.0)

        # Convert to list of recommendations
        for content_id, score in content_scores.items():
            recommendations.append({
                'content_id': content_id,
                'score': score,
                'method': 'collaborative'
            })

        return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:limit]

    def _content_based_filtering(self, user_id: str, limit: int,
                                 excluded_content_ids: set) -> List[Dict]:
        """
        Generate recommendations using content-based filtering.

        Args:
            user_id: User ID to generate recommendations for
            limit: Maximum number of recommendations
            excluded_content_ids: Content IDs to exclude

        Returns:
            List of recommended content items with scores
        """
        recommendations = []

        # Check if we have content vectors
        if self.content_vectors is None:
            return recommendations

        # Get user history
        user_history = self.user_history.get(user_id, [])
        if not user_history:
            return recommendations

        # Build user profile based on content they've interacted with
        user_profile = np.zeros((1, self.content_vectors.shape[1]))
        profile_weight = 0

        for interaction in user_history:
            content_id = interaction['content_id']
            if content_id not in self.content_id_to_idx:
                continue

            # Get content vector
            content_idx = self.content_id_to_idx[content_id]
            content_vector = self.content_vectors[content_idx]

            # Weight by interaction type and recency
            weight = 1.0
            if interaction['type'] == 'complete':
                weight = 2.0
            elif interaction['type'] == 'like':
                weight = 3.0

            # Add to user profile
            user_profile += weight * content_vector.toarray()
            profile_weight += weight

        # Normalize user profile
        if profile_weight > 0:
            user_profile /= profile_weight

        # Calculate similarity with all content
        content_similarities = cosine_similarity(user_profile, self.content_vectors)[0]

        # Convert to recommendations
        for idx, similarity in enumerate(content_similarities):
            content_id = self.content_idx_to_id[idx]
            if content_id in excluded_content_ids:
                continue

            recommendations.append({
                'content_id': content_id,
                'score': float(similarity),
                'method': 'content'
            })

        return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:limit]

    def _ai_recommendations(self, user_id: str, limit: int,
                            excluded_content_ids: set) -> List[Dict]:
        """
        Generate recommendations using AI (Gemini API).

        Args:
            user_id: User ID to generate recommendations for
            limit: Maximum number of recommendations
            excluded_content_ids: Content IDs to exclude

        Returns:
            List of recommended content items with scores
        """
        recommendations = []

        # Check if Gemini API is available
        if not self.gemini_api:
            logger.warning("Gemini API not available for AI recommendations")
            return recommendations

        try:
            # Get user history
            user_history = self.user_history.get(user_id, [])

            # Prepare user data for Gemini API
            user_data = {
                'user_id': user_id,
                'learning_history': [],
                'interests': [],
                'available_content': []
            }

            # Add learning history
            for interaction in user_history:
                content_id = interaction['content_id']
                if content_id in self.content_metadata:
                    content_info = self.content_metadata[content_id]
                    user_data['learning_history'].append({
                        'content_id': content_id,
                        'title': content_info.get('title', ''),
                        'type': content_info.get('type', ''),
                        'tags': content_info.get('tags', []),
                        'interaction_type': interaction['type'],
                        'timestamp': interaction['timestamp']
                    })

            # Extract interests from user history
            interest_counts = defaultdict(int)
            for interaction in user_history:
                content_id = interaction['content_id']
                if content_id in self.content_metadata:
                    for tag in self.content_metadata[content_id].get('tags', []):
                        interest_counts[tag] += 1

            # Get top interests
            user_data['interests'] = [
                tag for tag, _ in sorted(interest_counts.items(),
                                        key=lambda x: x[1], reverse=True)[:10]
            ]

            # Add available content (excluding already consumed content)
            for content_id, metadata in self.content_metadata.items():
                if content_id not in excluded_content_ids:
                    user_data['available_content'].append({
                        'content_id': content_id,
                        'title': metadata.get('title', ''),
                        'description': metadata.get('description', ''),
                        'type': metadata.get('type', ''),
                        'tags': metadata.get('tags', []),
                        'difficulty': metadata.get('difficulty', 'medium')
                    })

            # Call Gemini API for recommendations
            prompt = f"""
            Based on the user's learning history and interests, recommend the most relevant educational
            content from the available options. Focus on helping the user build a coherent learning
            path that advances their skills progressively. Consider content difficulty, relevance to
            user interests, and potential learning outcomes.

            User data: {json.dumps(user_data, indent=2)}

            Return a JSON array with the recommended content IDs and scores between 0 and 1,
            with 1 being the highest relevance. Example:
            [
                {{"content_id": "content123", "score": 0.95}},
                {{"content_id": "content456", "score": 0.85}}
            ]
            """

            # Get recommendations from Gemini API
            response = self.gemini_api.generate_content(prompt)
            result = response.text

            # Parse results
            try:
                ai_recommendations = json.loads(result)
                for rec in ai_recommendations:
                    if 'content_id' in rec and 'score' in rec:
                        content_id = rec['content_id']
                        if content_id not in excluded_content_ids:
                            recommendations.append({
                                'content_id': content_id,
                                'score': float(rec['score']),
                                'method': 'ai'
                            })

                            # Limit results
                            if len(recommendations) >= limit:
                                break
            except Exception as e:
                logger.error(f"Error parsing Gemini API response: {str(e)}")
                logger.error(f"Raw response: {result}")

        except Exception as e:
            logger.error(f"Error getting AI recommendations: {str(e)}")
            logger.error(traceback.format_exc())

        return recommendations

    def _apply_filters(self, recommendations: List[Dict], filters: Dict) -> List[Dict]:
        """
        Apply filters to recommendations.

        Args:
            recommendations: List of recommendations
            filters: Dictionary of filters to apply

        Returns:
            Filtered list of recommendations
        """
        filtered_recs = []

        for rec in recommendations:
            # Get content ID
            content_id = rec['content_id']
            if content_id not in self.content_metadata:
                continue

            # Get content metadata
            metadata = self.content_metadata[content_id]

            # Check each filter
            matches_all = True

            for filter_key, filter_value in filters.items():
                if filter_key == 'content_type' and filter_value != metadata.get('type'):
                    matches_all = False
                    break
                elif filter_key == 'difficulty' and filter_value != metadata.get('difficulty'):
                    matches_all = False
                    break
                elif filter_key == 'tags' and not set(filter_value).issubset(set(metadata.get('tags', []))):
                    matches_all = False
                    break
                elif filter_key == 'min_duration' and metadata.get('duration', 0) < filter_value:
                    matches_all = False
                    break
                elif filter_key == 'max_duration' and metadata.get('duration', 0) > filter_value:
                    matches_all = False
                    break

            if matches_all:
                filtered_recs.append(rec)

        return filtered_recs

    def _build_user_vectors(self):
        """
        Build user vectors based on their content interactions.
        """
        self.user_vectors = {}

        # Check if we have content vectors
        if self.content_vectors is None:
            return

        # For each user
        for user_id, interactions in self.user_history.items():
            if not interactions:
                continue

            # Initialize user vector
            user_vector = np.zeros((1, self.content_vectors.shape[1]))
            vector_weight = 0

            # Add content vectors to user vector
            for interaction in interactions:
                content_id = interaction['content_id']
                if content_id not in self.content_id_to_idx:
                    continue

                # Get content vector
                content_idx = self.content_id_to_idx[content_id]
                content_vector = self.content_vectors[content_idx]

                # Weight by interaction type and recency
                weight = 1.0
                if interaction['type'] == 'complete':
                    weight = 2.0
                elif interaction['type'] == 'like':
                    weight = 3.0

                # Add to user vector
                user_vector += weight * content_vector.toarray()
                vector_weight += weight

            # Normalize user vector
            if vector_weight > 0:
                user_vector /= vector_weight

                # Store user vector
                self.user_vectors[user_id] = user_vector

    def _save_model(self):
        """
        Save model to disk.
        """
        try:
            # Create model file paths
            vectorizer_path = os.path.join(self.model_path, 'vectorizer.pkl')
            content_vectors_path = os.path.join(self.model_path, 'content_vectors.npz')
            metadata_path = os.path.join(self.model_path, 'metadata.json')

            # Save vectorizer
            joblib.dump(self.vectorizer, vectorizer_path)

            # Save content vectors
            if self.content_vectors is not None:
                np.savez_compressed(
                    content_vectors_path,
                    data=self.content_vectors.data,
                    indices=self.content_vectors.indices,
                    indptr=self.content_vectors.indptr,
                    shape=self.content_vectors.shape
                )

            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump({
                    'last_trained': self.last_trained.isoformat() if self.last_trained else None,
                    'content_id_to_idx': self.content_id_to_idx if hasattr(self, 'content_id_to_idx') else {},
                    'content_metadata': self.content_metadata,
                    'user_history': {k: v for k, v in self.user_history.items()}
                }, f)

            logger.info(f"Model saved to {self.model_path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            logger.error(traceback.format_exc())

    def _load_model(self):
        """
        Load model from disk.
        """
        try:
            # Create model file paths
            vectorizer_path = os.path.join(self.model_path, 'vectorizer.pkl')
            content_vectors_path = os.path.join(self.model_path, 'content_vectors.npz')
            metadata_path = os.path.join(self.model_path, 'metadata.json')

            # Check if files exist
            if not os.path.exists(vectorizer_path) or not os.path.exists(content_vectors_path) or not os.path.exists(metadata_path):
                logger.info("Model files not found, starting with empty model")
                return

            # Load vectorizer
            self.vectorizer = joblib.load(vectorizer_path)

            # Load content vectors
            content_vectors_npz = np.load(content_vectors_path)
            from scipy.sparse import csr_matrix
            self.content_vectors = csr_matrix(
                (content_vectors_npz['data'], content_vectors_npz['indices'], content_vectors_npz['indptr']),
                shape=tuple(content_vectors_npz['shape'])
            )

            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

                self.last_trained = datetime.fromisoformat(metadata['last_trained']) if metadata['last_trained'] else None
                self.content_id_to_idx = metadata['content_id_to_idx']
                self.content_idx_to_id = {int(idx): content_id for content_id, idx in self.content_id_to_idx.items()}
                self.content_metadata = metadata['content_metadata']
                self.user_history = defaultdict(list)
                for user_id, history in metadata['user_history'].items():
                    self.user_history[user_id] = history

            # Build user vectors
            self._build_user_vectors()

            # Update model state
            self.is_trained = True

            logger.info(f"Model loaded from {self.model_path}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())

            # Reset model state
            self.content_vectors = None
            self.user_vectors = None
            self.content_metadata = {}
            self.user_history = defaultdict(list)
            self.is_trained = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert model metadata to dictionary."""
        return {
            'is_trained': self.is_trained,
            'last_trained': self.last_trained.isoformat() if self.last_trained else None,
            'num_users': len(self.user_vectors) if self.user_vectors is not None else 0,
            'num_content_items': len(self.content_metadata),
            'model_path': self.model_path
        }

class GeminiAPI:
    """
    Wrapper for Google's Gemini API for AI-powered recommendations and content generation.
    """

    def __init__(self, api_key: str = "YOUR_GEMINI_API_KEY", model: str = "gemini-pro"):
        """
        Initialize the Gemini API wrapper.

        Args:
            api_key: Google API key for Gemini
            model: Gemini model to use
        """
        self.api_key = api_key
        self.model = model

        # Configure the Gemini API
        genai.configure(api_key=api_key)

        # Get the model
        self.genai_model = genai.GenerativeModel(model)

    def generate_content(self, prompt: str, temperature: float = 0.7,
                         max_tokens: Optional[int] = None) -> Any:
        """
        Generate content using the Gemini API.

        Args:
            prompt: Text prompt for content generation
            temperature: Sampling temperature (higher = more creative)
            max_tokens: Maximum number of tokens to generate

        Returns:
            Response from the Gemini API
        """
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]

        response = self.genai_model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        return response

    def analyze_text(self, text: str, analysis_type: str = 'general') -> Dict[str, Any]:
        """
        Analyze text using Gemini API.

        Args:
            text: Text to analyze
            analysis_type: Type of analysis to perform

        Returns:
            Analysis results
        """
        prompts = {
            'general': f"Analyze the following text and provide key insights:\n\n{text}",
            'sentiment': f"Analyze the sentiment of this text:\n\n{text}",
            'topics': f"Extract main topics from this text:\n\n{text}",
            'summary': f"Provide a concise summary of:\n\n{text}"
        }

        prompt = prompts.get(analysis_type, prompts['general'])
        response = self.generate_content(prompt, temperature=0.3)

        return {
            'analysis': response.text,
            'type': analysis_type,
            'timestamp': datetime.now().isoformat()
        }

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
import json

db = SQLAlchemy()

class User(db.Model, UserMixin):
    __tablename__ = 'users'

    id = db.Column(db.String(36), primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    bio = db.Column(db.Text)
    avatar_url = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    role = db.Column(db.String(20), default='learner')  # learner, educator, admin
    xp_points = db.Column(db.Integer, default=0)
    level = db.Column(db.Integer, default=1)
    preferences = db.Column(db.Text)  # JSON string of user preferences

    # Relationships
    enrollments = db.relationship('Enrollment', back_populates='user', cascade='all, delete-orphan')
    achievements = db.relationship('UserAchievement', back_populates='user', cascade='all, delete-orphan')
    badges = db.relationship('UserBadge', back_populates='user', cascade='all, delete-orphan')
    credentials = db.relationship('Credential', back_populates='user', cascade='all, delete-orphan')
    interactions = db.relationship('UserInteraction', back_populates='user', cascade='all, delete-orphan')

    def __init__(self, username, email, password, first_name=None, last_name=None):
        self.id = str(uuid.uuid4())
        self.username = username
        self.email = email
        self.set_password(password)
        self.first_name = first_name
        self.last_name = last_name
        self.created_at = datetime.utcnow()

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def add_xp(self, points):
        self.xp_points += points
        self.update_level()

    def update_level(self):
        # Simple level calculation: level = 1 + floor(xp / 1000)
        new_level = 1 + self.xp_points // 1000
        if new_level > self.level:
            self.level = new_level
            return True
        return False

    def get_preferences(self):
        if self.preferences:
            return json.loads(self.preferences)
        return {}

    def set_preferences(self, preferences):
        self.preferences = json.dumps(preferences)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'bio': self.bio,
            'avatar_url': self.avatar_url,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'role': self.role,
            'xp_points': self.xp_points,
            'level': self.level
        }

class Course(db.Model):
    __tablename__ = 'courses'

    id = db.Column(db.String(36), primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    image_url = db.Column(db.String(255))
    creator_id = db.Column(db.String(36), db.ForeignKey('users.id'))
    category = db.Column(db.String(50))
    difficulty = db.Column(db.String(20))  # beginner, intermediate, advanced
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    status = db.Column(db.String(20), default='draft')  # draft, published, archived
    tags = db.Column(db.Text)  # JSON array of tags
    prerequisites = db.Column(db.Text)  # JSON array of prerequisite course IDs
    estimated_duration = db.Column(db.Integer)  # in minutes

    # Relationships
    creator = db.relationship('User', backref='created_courses')
    modules = db.relationship('Module', back_populates='course', cascade='all, delete-orphan', order_by='Module.order')
    enrollments = db.relationship('Enrollment', back_populates='course', cascade='all, delete-orphan')

    def __init__(self, title, description=None, creator_id=None):
        self.id = str(uuid.uuid4())
        self.title = title
        self.description = description
        self.creator_id = creator_id
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def set_tags(self, tags_list):
        self.tags = json.dumps(tags_list)

    def get_tags(self):
        if self.tags:
            return json.loads(self.tags)
        return []

    def set_prerequisites(self, prereq_list):
        self.prerequisites = json.dumps(prereq_list)

    def get_prerequisites(self):
        if self.prerequisites:
            return json.loads(self.prerequisites)
        return []

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'image_url': self.image_url,
            'creator_id': self.creator_id,
            'category': self.category,
            'difficulty': self.difficulty,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'status': self.status,
            'tags': self.get_tags(),
            'prerequisites': self.get_prerequisites(),
            'estimated_duration': self.estimated_duration,
            'modules_count': len(self.modules) if self.modules else 0
        }

class Module(db.Model):
    __tablename__ = 'modules'

    id = db.Column(db.String(36), primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    course_id = db.Column(db.String(36), db.ForeignKey('courses.id'), nullable=False)
    order = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    course = db.relationship('Course', back_populates='modules')
    lessons = db.relationship('Lesson', back_populates='module', cascade='all, delete-orphan', order_by='Lesson.order')

    def __init__(self, title, course_id, order, description=None):
        self.id = str(uuid.uuid4())
        self.title = title
        self.course_id = course_id
        self.order = order
        self.description = description
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'course_id': self.course_id,
            'order': self.order,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'lessons_count': len(self.lessons) if self.lessons else 0
        }

class Lesson(db.Model):
    __tablename__ = 'lessons'

    id = db.Column(db.String(36), primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    module_id = db.Column(db.String(36), db.ForeignKey('modules.id'), nullable=False)
    content_type = db.Column(db.String(20))  # video, text, quiz, interactive
    content = db.Column(db.Text)  # Could be HTML content, video URL, or JSON for interactive content
    order = db.Column(db.Integer, nullable=False)
    estimated_duration = db.Column(db.Integer)  # in minutes
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    module = db.relationship('Module', back_populates='lessons')
    quiz_questions = db.relationship('QuizQuestion', back_populates='lesson', cascade='all, delete-orphan')
    resources = db.relationship('LessonResource', back_populates='lesson', cascade='all, delete-orphan')
    interactions = db.relationship('UserInteraction', back_populates='lesson', cascade='all, delete-orphan')

    def __init__(self, title, module_id, content_type, content, order):
        self.id = str(uuid.uuid4())
        self.title = title
        self.module_id = module_id
        self.content_type = content_type
        self.content = content
        self.order = order
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'module_id': self.module_id,
            'content_type': self.content_type,
            'content': self.content,
            'order': self.order,
            'estimated_duration': self.estimated_duration,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Enrollment(db.Model):
    __tablename__ = 'enrollments'

    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    course_id = db.Column(db.String(36), db.ForeignKey('courses.id'), nullable=False)
    enrolled_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_accessed = db.Column(db.DateTime)
    completion_status = db.Column(db.String(20), default='in_progress')  # not_started, in_progress, completed
    progress_data = db.Column(db.Text)  # JSON data tracking lesson completion

    # Relationships
    user = db.relationship('User', back_populates='enrollments')
    course = db.relationship('Course', back_populates='enrollments')

    def __init__(self, user_id, course_id):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.course_id = course_id
        self.enrolled_at = datetime.utcnow()
        self.completion_status = 'not_started'

    def get_progress_data(self):
        if self.progress_data:
            return json.loads(self.progress_data)
        return {}

    def set_progress_data(self, progress_data):
        self.progress_data = json.dumps(progress_data)

    def calculate_progress_percentage(self):
        progress_data = self.get_progress_data()
        if not progress_data or 'completed_lessons' not in progress_data:
            return 0

        total_lessons = progress_data.get('total_lessons', 1)
        completed_lessons = len(progress_data.get('completed_lessons', []))

        if total_lessons == 0:
            return 0

        return min(100, int(completed_lessons / total_lessons * 100))

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'course_id': self.course_id,
            'enrolled_at': self.enrolled_at.isoformat() if self.enrolled_at else None,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'completion_status': self.completion_status,
            'progress_data': self.get_progress_data(),
            'progress_percentage': self.calculate_progress_percentage()
        }

class QuizQuestion(db.Model):
    __tablename__ = 'quiz_questions'

    id = db.Column(db.String(36), primary_key=True)
    lesson_id = db.Column(db.String(36), db.ForeignKey('lessons.id'), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    question_type = db.Column(db.String(20))  # multiple_choice, true_false, fill_blank, essay
    options = db.Column(db.Text)  # JSON array of options (for multiple choice)
    correct_answer = db.Column(db.Text)  # For fill_blank, multiple_choice (correct option index), true_false
    explanation = db.Column(db.Text)  # Explanation of the answer
    points = db.Column(db.Integer, default=1)
    order = db.Column(db.Integer)

    # Relationships
    lesson = db.relationship('Lesson', back_populates='quiz_questions')

    def __init__(self, lesson_id, question_text, question_type, correct_answer, order=0, points=1):
        self.id = str(uuid.uuid4())
        self.lesson_id = lesson_id
        self.question_text = question_text
        self.question_type = question_type
        self.correct_answer = correct_answer
        self.order = order
        self.points = points

    def set_options(self, options_list):
        self.options = json.dumps(options_list)

    def get_options(self):
        if self.options:
            return json.loads(self.options)
        return []

    def to_dict(self):
        return {
            'id': self.id,
            'lesson_id': self.lesson_id,
            'question_text': self.question_text,
            'question_type': self.question_type,
            'options': self.get_options(),
            'explanation': self.explanation,
            'points': self.points,
            'order': self.order
        }

class LessonResource(db.Model):
    __tablename__ = 'lesson_resources'

    id = db.Column(db.String(36), primary_key=True)
    lesson_id = db.Column(db.String(36), db.ForeignKey('lessons.id'), nullable=False)
    title = db.Column(db.String(100), nullable=False)
    resource_type = db.Column(db.String(20))  # pdf, link, image, document
    url = db.Column(db.String(255))
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    lesson = db.relationship('Lesson', back_populates='resources')

    def __init__(self, lesson_id, title, resource_type, url, description=None):
        self.id = str(uuid.uuid4())
        self.lesson_id = lesson_id
        self.title = title
        self.resource_type = resource_type
        self.url = url
        self.description = description
        self.created_at = datetime.utcnow()

    def to_dict(self):
        return {
            'id': self.id,
            'lesson_id': self.lesson_id,
            'title': self.title,
            'resource_type': self.resource_type,
            'url': self.url,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class UserInteraction(db.Model):
    __tablename__ = 'user_interactions'

    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    lesson_id = db.Column(db.String(36), db.ForeignKey('lessons.id'), nullable=False)
    interaction_type = db.Column(db.String(20))  # view, complete, like, bookmark, rate
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    rating = db.Column(db.Float)  # For rate interactions
    metadata = db.Column(db.Text)  # JSON field for additional interaction data

    # Relationships
    user = db.relationship('User', back_populates='interactions')
    lesson = db.relationship('Lesson', back_populates='interactions')

    def __init__(self, user_id, lesson_id, interaction_type):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.lesson_id = lesson_id
        self.interaction_type = interaction_type
        self.timestamp = datetime.utcnow()

    def set_metadata(self, metadata_dict):
        self.metadata = json.dumps(metadata_dict)

    def get_metadata(self):
        if self.metadata:
            return json.loads(self.metadata)
        return {}

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'lesson_id': self.lesson_id,
            'interaction_type': self.interaction_type,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'rating': self.rating,
            'metadata': self.get_metadata()
        }

class Achievement(db.Model):
    __tablename__ = 'achievements'

    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    icon_url = db.Column(db.String(255))
    xp_reward = db.Column(db.Integer, default=0)
    requirement_type = db.Column(db.String(50))  # course_completion, lesson_count, quiz_score, etc.
    requirement_data = db.Column(db.Text)  # JSON data specifying exact requirements

    # Relationships
    user_achievements = db.relationship('UserAchievement', back_populates='achievement')

    def __init__(self, name, description, requirement_type, xp_reward=0):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.requirement_type = requirement_type
        self.xp_reward = xp_reward

    def set_requirement_data(self, requirement_data):
        self.requirement_data = json.dumps(requirement_data)

    def get_requirement_data(self):
        if self.requirement_data:
            return json.loads(self.requirement_data)
        return {}

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'icon_url': self.icon_url,
            'xp_reward': self.xp_reward,
            'requirement_type': self.requirement_type,
            'requirement_data': self.get_requirement_data()
        }

class UserAchievement(db.Model):
    __tablename__ = 'user_achievements'

    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    achievement_id = db.Column(db.String(36), db.ForeignKey('achievements.id'), nullable=False)
    achieved_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    user = db.relationship('User', back_populates='achievements')
    achievement = db.relationship('Achievement', back_populates='user_achievements')

    def __init__(self, user_id, achievement_id):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.achievement_id = achievement_id
        self.achieved_at = datetime.utcnow()

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'achievement_id': self.achievement_id,
            'achieved_at': self.achieved_at.isoformat() if self.achieved_at else None
        }

class Badge(db.Model):
    __tablename__ = 'badges'

    id = db.Column(db.String(36), primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    icon_url = db.Column(db.String(255))
    badge_type = db.Column(db.String(50))  # skill_badge, course_badge, special_badge
    badge_level = db.Column(db.String(20))  # beginner, intermediate, advanced, expert

    # Relationships
    user_badges = db.relationship('UserBadge', back_populates='badge')

    def __init__(self, name, description, badge_type, badge_level='beginner'):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.badge_type = badge_type
        self.badge_level = badge_level

    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'icon_url': self.icon_url,
            'badge_type': self.badge_type,
            'badge_level': self.badge_level
        }

class UserBadge(db.Model):
    __tablename__ = 'user_badges'

    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    badge_id = db.Column(db.String(36), db.ForeignKey('badges.id'), nullable=False)
    awarded_at = db.Column(db.DateTime, default=datetime.utcnow)
    awarded_by = db.Column(db.String(36), db.ForeignKey('users.id'))  # If awarded by another user (e.g., instructor)

    # Relationships
    user = db.relationship('User', back_populates='badges', foreign_keys=[user_id])
    badge = db.relationship('Badge', back_populates='user_badges')
    awarder = db.relationship('User', foreign_keys=[awarded_by])

    def __init__(self, user_id, badge_id, awarded_by=None):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.badge_id = badge_id
        self.awarded_at = datetime.utcnow()
        self.awarded_by = awarded_by

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'badge_id': self.badge_id,
            'awarded_at': self.awarded_at.isoformat() if self.awarded_at else None,
            'awarded_by': self.awarded_by
        }

class Credential(db.Model):
    __tablename__ = 'credentials'

    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(100), nullable=False)
    issuer = db.Column(db.String(100))
    issue_date = db.Column(db.Date)
    expiry_date = db.Column(db.Date)
    credential_type = db.Column(db.String(50))  # certificate, degree, badge, license
    verification_url = db.Column(db.String(255))
    credential_id = db.Column(db.String(100))  # External ID for the credential
    metadata = db.Column(db.Text)  # JSON data with additional credential info

    # Relationships
    user = db.relationship('User', back_populates='credentials')

    def __init__(self, user_id, title, issuer, credential_type, issue_date=None):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.title = title
        self.issuer = issuer
        self.credential_type = credential_type
        self.issue_date = issue_date or datetime.utcnow().date()

    def set_metadata(self, metadata_dict):
        self.metadata = json.dumps(metadata_dict)

    def get_metadata(self):
        if self.metadata:
            return json.loads(self.metadata)
        return {}

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'issuer': self.issuer,
            'issue_date': self.issue_date.isoformat() if self.issue_date else None,
            'expiry_date': self.expiry_date.isoformat() if self.expiry_date else None,
            'credential_type': self.credential_type,
            'verification_url': self.verification_url,
            'credential_id': self.credential_id,
            'metadata': self.get_metadata()
        }

class Notification(db.Model):
    __tablename__ = 'notifications'

    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    title = db.Column(db.String(100), nullable=False)
    message = db.Column(db.Text, nullable=False)
    notification_type = db.Column(db.String(50))  # achievement, reminder, announcement, feedback
    related_id = db.Column(db.String(36))  # ID of related entity (course, achievement, etc.)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_read = db.Column(db.Boolean, default=False)

    # Relationships
    user = db.relationship('User', backref='notifications')

    def __init__(self, user_id, title, message, notification_type, related_id=None):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.title = title
        self.message = message
        self.notification_type = notification_type
        self.related_id = related_id
        self.created_at = datetime.utcnow()
        self.is_read = False

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'message': self.message,
            'notification_type': self.notification_type,
            'related_id': self.related_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_read': self.is_read
        }

# Example usage
if __name__ == "__main__":
    # Initialize Gemini API with your actual API key
    gemini = GeminiAPI(api_key="YOUR_GEMINI_API_KEY")

    # Initialize recommendation model
    model = RecommendationModel(gemini_api=gemini)

    # Test content generation
    response = gemini.generate_content("Create a lesson plan about Python")
    print(response.text)

    # Test text analysis
    analysis = gemini.analyze_text("AI has revolutionized the way we learn.", analysis_type='sentiment')
    print(analysis)

    # Get recommendations
    recommendations = model.recommend(
        user_id="user123",
        limit=5,
        strategy="hybrid"
    )
    print(recommendations)

# Create blueprint
ai_bp = Blueprint('ai', __name__, url_prefix='/api/v1/ai')

from utils.gemini_integration import GeminiAPI

gemini = GeminiAPI(api_key="YOUR_GEMINI_API_KEY")

@ai_bp.route('/')
def index():
    """API root endpoint."""
    return jsonify({
        'name': 'Eduquest AI API',
        'version': current_app.config.get('API_VERSION', 'v1'),
        'status': 'operational',
        'endpoints': {
            'root': '/api/v1/ai/',
            'health': '/api/v1/ai/health',
            'generate': '/api/v1/ai/generate'
        }
    })

@ai_bp.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })

@ai_bp.route('/generate', methods=['POST'])
def generate():
    """Generate AI content."""
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    data = request.get_json()
    if 'prompt' not in data:
        return jsonify({'error': 'Missing prompt in request body'}), 400
    
    try:
        response = gemini.generate_content(data['prompt'])
        return jsonify({
            'success': True,
            'prompt': data['prompt'],
            'response': response.text,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        current_app.logger.error(f"Error: {str(e)}")
        return jsonify({'error': 'Failed to generate content'}), 500

# Import routes after blueprint creation to avoid circular imports
from . import routes

# Version information
__version__ = '1.0.0'

# Export blueprint
__all__ = ['ai_bp']

def create_app(config_class=None):
    """Initialize Flask application."""
    app = Flask(__name__)
    
    # Configure app
    config_class = config_class or APIConfig
    app.config.from_object(config_class)
    
    # Setup middleware
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Register blueprint
    from api import ai_bp
    app.register_blueprint(ai_bp)
    
    # Add root route
    @app.route('/')
    def index():
        return jsonify({
            'name': 'Eduquest API',
            'version': app.config.get('API_VERSION', 'v1'),
            'endpoints': {
                'ai': '/api/v1/ai/'
            }
        })
    
    return app

# Create app instance
app = create_app()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    port = int(os.environ.get('PORT', 5000))
>>>>>>> f41f548 (frontend)
    app.run(host='0.0.0.0', port=port, debug=True)