<<<<<<< HEAD
from flask import Blueprint, request, jsonify, current_app, g
import logging
import time
import json
import asyncio
import threading
import hashlib
import requests
import traceback
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List, Any, Optional, Union, Callable
import uuid
import re
import random

# Set up logger
logger = logging.getLogger(__name__)

# Create blueprint only - no app creation here
ai_bp = Blueprint('ai', __name__, url_prefix='/api/ai')

# Import utilities after blueprint creation
from utils.cache import SimpleCache 
from utils.rate_limiter import RateLimiter
from utils.auth import APIKeyAuth

# Initialize components (moved to blueprint setup function)
cache = SimpleCache(default_ttl=300)
rate_limiter = RateLimiter()
api_key_auth = APIKeyAuth()

class GeminiAPI:
    """Advanced client for interacting with Google's Gemini API with key rotation."""

    def __init__(self, api_keys=None):
        """Initialize the Gemini API client with multiple API keys."""
        self.api_keys = api_keys or []
        self.base_url = "https://generativelanguage.googleapis.com/v1"
        self.default_model = "gemini-pro"
        self.session = requests.Session()

        # Key usage tracking
        self.key_usage = {key: 0 for key in self.api_keys}
        self.key_lock = threading.RLock()

        # Configure retry strategy
        retry_strategy = requests.adapters.Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)

    def _get_next_api_key(self, purpose=None):
        """
        Get the next available API key using round-robin with purpose-specific selection.

        Args:
            purpose: Optional purpose for key selection ('recommendation', 'analysis',
                    'content_generation', 'assessment')
        """
        with self.key_lock:
            if not self.api_keys:
                logger.warning("No Gemini API keys available")
                return None

            # If purpose is specified, use a dedicated key if available
            purpose_key_map = {
                'recommendation': 0,
                'analysis': 1,
                'content_generation': 2,
                'assessment': 3
            }

            if purpose and purpose in purpose_key_map and purpose_key_map[purpose] < len(self.api_keys):
                # Use the dedicated key for this purpose
                key_index = purpose_key_map[purpose]
                key = self.api_keys[key_index]
                self.key_usage[key] += 1
                return key

            # Otherwise, use the least used key (load balancing)
            key = min(self.key_usage, key=self.key_usage.get)
            self.key_usage[key] += 1
            return key

    def generate_content(self, prompt, model=None, temperature=0.7, max_tokens=None,
                         top_p=None, top_k=None, safety_settings=None, purpose='content_generation'):
        """Generate content using the Gemini API."""
        api_key = self._get_next_api_key(purpose=purpose)

        if not api_key:
            # Return mock response if no API key
            return self._mock_generate_response(prompt)

        model = model or self.default_model
        url = f"{self.base_url}/models/{model}:generateContent?key={api_key}"

        # Prepare request payload
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
            }
        }

        # Add optional parameters
        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        if top_p:
            payload["generationConfig"]["topP"] = top_p
        if top_k:
            payload["generationConfig"]["topK"] = top_k
        if safety_settings:
            payload["safetySettings"] = safety_settings

        try:
            start_time = time.time()
            response = self.session.post(url, json=payload)
            response.raise_for_status()

            # Log latency
            latency = time.time() - start_time
            logger.debug(f"Gemini API request completed in {latency:.2f}s")

            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise

    def analyze_text(self, text, analysis_type="sentiment", model=None):
        """Analyze text using the Gemini API."""
        api_key = self._get_next_api_key(purpose='analysis')

        if not api_key:
            # Return mock response if no API key
            return self._mock_analyze_response(text, analysis_type)

        model = model or self.default_model

        # Create prompt based on analysis type
        if analysis_type == "sentiment":
            prompt = (
                "Analyze the sentiment of this text and respond with a JSON object "
                "containing 'sentiment' (positive, negative, or neutral), "
                "'confidence' (0-1), and 'explanation' (brief reason): "
                f"{text}"
            )
        elif analysis_type == "entities":
            prompt = (
                "Extract the key entities from this text and respond with a JSON object "
                "containing 'entities' (array of objects with 'name', 'type', and 'relevance' fields): "
                f"{text}"
            )
        elif analysis_type == "keywords":
            prompt = (
                "Extract the main keywords from this text and respond with a JSON object "
                "containing 'keywords' (array of objects with 'keyword' and 'relevance' fields): "
                f"{text}"
            )
        elif analysis_type == "summary":
            prompt = (
                "Summarize this text and respond with a JSON object "
                "containing 'summary' (string) and 'key_points' (array of strings): "
                f"{text}"
            )
        elif analysis_type == "classification":
            prompt = (
                "Classify this text into categories and respond with a JSON object "
                "containing 'categories' (array of objects with 'category' and 'confidence' fields): "
                f"{text}"
            )
        else:
            prompt = (
                f"Analyze this text for {analysis_type} and respond with a JSON object: "
                f"{text}"
            )

        # Generate content with low temperature for more deterministic results
        response = self.generate_content(prompt, model=model, temperature=0.1, purpose='analysis')

        # Extract and parse JSON from response
        try:
            text_response = self._extract_text(response)
            # Try to find JSON in the response
            json_match = re.search(r'```json\s*(.*?)\s*```', text_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without markdown
                json_match = re.search(r'(\{.*\})', text_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = text_response

            analysis_result = json.loads(json_str)
            return analysis_result
        except Exception as e:
            logger.error(f"Error parsing analysis response: {str(e)}")
            logger.error(f"Raw response: {text_response}")
            raise

    def generate_recommendations(self, user_data, model=None):
        """Generate personalized learning recommendations based on user data."""
        prompt = (
            "Based on the following user learning data, generate personalized educational "
            "recommendations for what they should learn next. Format your response as a JSON "
            "object with the arrays: 'recommended_topics', 'recommended_resources', and "
            "'learning_path'. For each recommendation, include a 'title', 'description', "
            "'difficulty', and 'relevance_score' (0-1). User data: "
            f"{json.dumps(user_data)}"
        )
        return self.generate_content(prompt, temperature=0.3, purpose='recommendation')

    def assess_learning(self, submission, criteria, model=None):
        """Assess a learning submission against given criteria."""
        prompt = (
            "Evaluate the following learning submission against the assessment criteria. "
            "Provide a detailed analysis, feedback, and scoring. Format your response as a JSON "
            "object with 'score' (0-100), 'feedback' (string), 'strengths' (array), 'areas_for_improvement' (array), "
            "and 'next_steps' (array). Submission: "
            f"{submission}\n\n"
            "Assessment Criteria: "
            f"{json.dumps(criteria)}"
        )
        return self.generate_content(prompt, temperature=0.1, purpose='assessment')

    def _extract_text(self, response):
        """Extract text from Gemini API response."""
        try:
            # Handle different response formats
            if 'candidates' in response and response['candidates']:
                if 'content' in response['candidates'][0]:
                    parts = response['candidates'][0]['content'].get('parts', [])
                    if parts and 'text' in parts[0]:
                        return parts[0]['text']

            # Fallback to returning the raw response
            return str(response)
        except Exception as e:
            logger.error(f"Error extracting text from response: {str(e)}")
            return str(response)

    def _mock_generate_response(self, prompt):
        """Generate a mock response for testing purposes."""
        logger.warning("Using mock Gemini API response")

        # Create a simple mock response
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "This is a mock response from the Gemini API. "
                                        "Please configure a valid API key for actual responses."
                            }
                        ]
                    }
                }
            ]
        }

    def _mock_analyze_response(self, text, analysis_type):
        """Generate a mock analysis response for testing purposes."""
        logger.warning(f"Using mock Gemini API analysis response for {analysis_type}")

        # Create appropriate mock responses based on analysis type
        if analysis_type == "sentiment":
            return {
                "sentiment": random.choice(["positive", "negative", "neutral"]),
                "confidence": round(random.random(), 2),
                "explanation": "This is a mock sentiment analysis."
            }
        elif analysis_type == "entities":
            return {
                "entities": [
                    {
                        "name": "Example Entity",
                        "type": "ORGANIZATION",
                        "relevance": round(random.random(), 2)
                    }
                ]
            }
        elif analysis_type == "keywords":
            return {
                "keywords": [
                    {
                        "keyword": "example",
                        "relevance": round(random.random(), 2)
                    },
                    {
                        "keyword": "test",
                        "relevance": round(random.random(), 2)
                    }
                ]
            }
        else:
            return {
                "analysis": f"Mock {analysis_type} analysis",
                "confidence": round(random.random(), 2)
            }

def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')

        if not api_key:
            return jsonify({'error': 'API key is required'}), 401

        if not api_key_auth.validate_key(api_key):
            return jsonify({'error': 'Invalid API key'}), 401

        # Store key owner in Flask's g object for access in route handlers
        g.api_key_owner = api_key_auth.get_key_owner(api_key)
        g.api_key_permissions = api_key_auth.get_key_permissions(api_key)

        # Check permissions if endpoint requires specific ones
        endpoint = request.endpoint
        if endpoint and endpoint.startswith('ai.'):
            # Extract route name from endpoint (e.g., 'ai.recommend' -> 'recommend')
            route_name = endpoint.split('.')[-1]
            required_permission = f"ai:{route_name}"

            if required_permission not in g.api_key_permissions and '*' not in g.api_key_permissions:
                return jsonify({'error': f'Permission denied: {required_permission} required'}), 403

        return f(*args, **kwargs)
    return decorated_function

def rate_limit(limit=100, period=60, key_func=None):
    """
    Decorator to apply rate limiting.

    Args:
        limit: Maximum number of requests allowed in the period
        period: Time period in seconds
        key_func: Function to determine the rate limit key (default: IP address)
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Determine rate limit key
            if key_func:
                key = key_func()
            else:
                # Default: use IP address or API key if available
                if hasattr(g, 'api_key_owner'):
                    key = f"api_key:{g.api_key_owner}"
                else:
                    key = request.remote_addr

            # Check rate limit
            is_limited, remaining, reset_time = rate_limiter.is_rate_limited(
                key, limit, period, strategy='sliding_window'
            )

            # Set rate limit headers
            response_headers = {
                'X-RateLimit-Limit': str(limit),
                'X-RateLimit-Remaining': str(remaining),
                'X-RateLimit-Reset': str(int(reset_time))
            }

            if is_limited:
                response = jsonify({'error': 'Rate limit exceeded'})
                response.headers.extend(response_headers)
                response.status_code = 429
                return response

            # Execute the route function
            response = f(*args, **kwargs)

            # If response is a tuple (response, status_code), get just the response
            if isinstance(response, tuple):
                resp_obj = response[0]
                # Add headers to the response object
                for key, value in response_headers.items():
                    resp_obj.headers[key] = value
                return response
            else:
                # Add headers to the response object
                for key, value in response_headers.items():
                    response.headers[key] = value
                return response

        return decorated_function
    return decorator

def cache_response(ttl=None, key_prefix=None, unless=None):
    """
    Decorator to cache API responses.

    Args:
        ttl: Cache TTL in seconds (default: None, uses cache default)
        key_prefix: Prefix for cache keys (default: None)
        unless: Function that returns True if caching should be skipped
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Skip caching if condition is met
            if unless and unless():
                return f(*args, **kwargs)

            # Create cache key
            key_parts = [key_prefix or request.endpoint]

            # Add request path and query parameters to key
            key_parts.append(request.path)
            key_parts.append(hashlib.md5(request.query_string).hexdigest())

            # Add request body to key if it's a JSON request
            if request.is_json:
                key_parts.append(hashlib.md5(json.dumps(request.json, sort_keys=True).encode()).hexdigest())

            cache_key = ":".join(key_parts)

            # Try to get from cache
            cached_response = cache.get(cache_key)
            if cached_response:
                return cached_response

            # Execute the route function
            response = f(*args, **kwargs)

            # Cache the response
            if isinstance(response, tuple):
                # Only cache successful responses
                if response[1] == 200:
                    cache.set(cache_key, response, ttl)
            else:
                cache.set(cache_key, response, ttl)

            return response
        return decorated_function
    return decorator

def log_request(f):
    """Decorator to log API requests and responses."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Log request
        request_id = str(uuid.uuid4())
        g.request_id = request_id

        logger.info(f"Request {request_id}: {request.method} {request.path}")
        if request.is_json:
            # Log request body but mask sensitive data
            masked_data = mask_sensitive_data(request.json)
            logger.debug(f"Request {request_id} body: {json.dumps(masked_data)}")

        start_time = time.time()

        try:
            # Execute the route function
            response = f(*args, **kwargs)

            # Log response
            end_time = time.time()
            duration = end_time - start_time

            if isinstance(response, tuple):
                status_code = response[1]
            else:
                status_code = 200

            logger.info(f"Response {request_id}: {status_code} in {duration:.2f}s")

            return response
        except Exception as e:
            # Log exception
            end_time = time.time()
            duration = end_time - start_time

            logger.error(f"Exception {request_id}: {str(e)} in {duration:.2f}s")
            logger.error(traceback.format_exc())

            # Return error response
            return jsonify({
                'error': 'Internal server error',
                'request_id': request_id
            }), 500

    return decorated_function

def mask_sensitive_data(data, sensitive_keys=None):
    """Mask sensitive data for logging purposes."""
    if sensitive_keys is None:
        sensitive_keys = ['api_key', 'password', 'token', 'secret']

    if isinstance(data, dict):
        masked_data = {}
        for key, value in data.items():
            if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                masked_data[key] = '********'
            elif isinstance(value, (dict, list)):
                masked_data[key] = mask_sensitive_data(value, sensitive_keys)
            else:
                masked_data[key] = value
        return masked_data
    elif isinstance(data, list):
        return [mask_sensitive_data(item, sensitive_keys) for item in data]
    else:
        return data

# Move initialization to blueprint setup
@ai_bp.record
def setup_blueprint(state):
    """Initialize blueprint with application context."""
    app = state.app
    
    # Load API keys from application config
    api_key_auth.load_keys_from_config(app.config)

    # Load Gemini API keys from config
    gemini_keys = app.config.get('GEMINI_API_KEYS', [])

    # Create Gemini API client and attach to app
    if not hasattr(app, 'gemini_api'):
        app.gemini_api = GeminiAPI(api_keys=gemini_keys)

    logger.info("AI Blueprint initialized successfully")

@ai_bp.route('/recommend', methods=['POST'])
@log_request
@require_api_key
@rate_limit(limit=10, period=60)
@cache_response(ttl=300, key_prefix='recommend', unless=lambda: request.args.get('nocache') == '1')
def recommend():
    """Generate personalized learning recommendations based on user data."""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    user_data = request.json

    # Validate required fields
    required_fields = ['user_id', 'learning_history', 'interests']
    missing_fields = [field for field in required_fields if field not in user_data]
    if missing_fields:
        return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

    try:
        # Get Gemini API client
        gemini_api = current_app.gemini_api

        # Generate recommendations
        response = gemini_api.generate_recommendations(user_data)
        recommendations = gemini_api._extract_text(response)

        # Try to parse as JSON
        try:
            recommendations_json = json.loads(recommendations)
            return jsonify(recommendations_json)
        except json.JSONDecodeError:
            # Return as text if not valid JSON
            return jsonify({'recommendations': recommendations})

    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to generate recommendations', 'details': str(e)}), 500

@ai_bp.route('/analyze', methods=['POST'])
@log_request
@require_api_key
@rate_limit(limit=20, period=60)
def analyze():
    """Analyze text for various purposes (sentiment, entities, etc.)."""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.json

    # Validate required fields
    if 'text' not in data:
        return jsonify({'error': 'Missing required field: text'}), 400

    # Get analysis type
    analysis_type = data.get('analysis_type', 'sentiment')

    try:
        # Get Gemini API client
        gemini_api = current_app.gemini_api

        # Analyze text
        analysis_result = gemini_api.analyze_text(data['text'], analysis_type)

        return jsonify(analysis_result)
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to analyze text', 'details': str(e)}), 500

@ai_bp.route('/assess', methods=['POST'])
@log_request
@require_api_key
@rate_limit(limit=5, period=60)
def assess():
    """Assess a learning submission against given criteria."""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.json

    # Validate required fields
    required_fields = ['submission', 'criteria']
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

    try:
        # Get Gemini API client
        gemini_api = current_app.gemini_api

        # Assess submission
        response = gemini_api.assess_learning(data['submission'], data['criteria'])
        assessment_text = gemini_api._extract_text(response)

        # Try to parse as JSON
        try:
            assessment_json = json.loads(assessment_text)
            return jsonify(assessment_json)
        except json.JSONDecodeError:
            # Return as text if not valid JSON
            return jsonify({'assessment': assessment_text})

    except Exception as e:
        logger.error(f"Error assessing submission: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to assess submission', 'details': str(e)}), 500

@ai_bp.route('/generate', methods=['POST'])
@log_request
@require_api_key
@rate_limit(limit=10, period=60)
def generate():
    """Generate content based on a prompt."""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.json

    # Validate required fields
    if 'prompt' not in data:
        return jsonify({'error': 'Missing required field: prompt'}), 400

    # Get optional parameters
    model = data.get('model')
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens')

    try:
        # Get Gemini API client
        gemini_api = current_app.gemini_api

        # Generate content
        response = gemini_api.generate_content(
            data['prompt'],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        generated_text = gemini_api._extract_text(response)

        return jsonify({'generated_text': generated_text})
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to generate content', 'details': str(e)}), 500

@ai_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for the AI API."""
    # Check Gemini API
    try:
        gemini_api = current_app.gemini_api
        test_prompt = "Hello, this is a health check. Please respond with 'ok'."
        response = gemini_api.generate_content(test_prompt, temperature=0.1)
        gemini_status = "ok" if response else "error"
    except Exception as e:
        gemini_status = f"error: {str(e)}"

    # Check cache
    try:
        cache_stats = cache.get_stats()
        cache_status = "ok"
    except Exception as e:
        cache_stats = {}
        cache_status = f"error: {str(e)}"

    # Return health status
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'gemini_api': gemini_status,
            'cache': cache_status,
            'cache_stats': cache_stats
        }
    })

@ai_bp.route('/metrics', methods=['GET'])
@require_api_key
@log_request
def metrics():
    """Metrics endpoint for the AI API."""
    # Check if caller has admin permissions
    if 'admin' not in g.api_key_permissions and '*' not in g.api_key_permissions:
        return jsonify({'error': 'Permission denied: admin permission required'}), 403

    # Get cache statistics
    try:
        cache_stats = cache.get_stats()
    except Exception as e:
        cache_stats = {}
        logger.error(f"Error retrieving cache stats: {str(e)}")

    # Get API key usage statistics
    gemini_api = current_app.gemini_api
    key_usage = gemini_api.key_usage

    # Prepare the metrics response
    metrics_response = {
        'cache': cache_stats,
        'key_usage': {key[-8:]: usage for key, usage in key_usage.items()}  # Only show last 8 chars of keys
    }

    return jsonify(metrics_response)

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
        return jsonify({
            'success': True,
            'prompt': data['prompt'],
            'response': 'Test response',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        return jsonify({'error': 'Failed to generate content'}), 500
=======
from flask import Blueprint, request, jsonify, current_app, g
import logging
import time
import json
import asyncio
import threading
import hashlib
import requests
import traceback
from datetime import datetime, timedelta
from functools import wraps
from typing import Dict, List, Any, Optional, Union, Callable
import uuid
import re
import random

# Set up logger
logger = logging.getLogger(__name__)

# Create blueprint only - no app creation here
ai_bp = Blueprint('ai', __name__, url_prefix='/api/ai')

# Import utilities after blueprint creation
from utils.cache import SimpleCache 
from utils.rate_limiter import RateLimiter
from utils.auth import APIKeyAuth

# Initialize components (moved to blueprint setup function)
cache = SimpleCache(default_ttl=300)
rate_limiter = RateLimiter()
api_key_auth = APIKeyAuth()

class GeminiAPI:
    """Advanced client for interacting with Google's Gemini API with key rotation."""

    def __init__(self, api_keys=None):
        """Initialize the Gemini API client with multiple API keys."""
        self.api_keys = api_keys or []
        self.base_url = "https://generativelanguage.googleapis.com/v1"
        self.default_model = "gemini-pro"
        self.session = requests.Session()

        # Key usage tracking
        self.key_usage = {key: 0 for key in self.api_keys}
        self.key_lock = threading.RLock()

        # Configure retry strategy
        retry_strategy = requests.adapters.Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)

    def _get_next_api_key(self, purpose=None):
        """
        Get the next available API key using round-robin with purpose-specific selection.

        Args:
            purpose: Optional purpose for key selection ('recommendation', 'analysis',
                    'content_generation', 'assessment')
        """
        with self.key_lock:
            if not self.api_keys:
                logger.warning("No Gemini API keys available")
                return None

            # If purpose is specified, use a dedicated key if available
            purpose_key_map = {
                'recommendation': 0,
                'analysis': 1,
                'content_generation': 2,
                'assessment': 3
            }

            if purpose and purpose in purpose_key_map and purpose_key_map[purpose] < len(self.api_keys):
                # Use the dedicated key for this purpose
                key_index = purpose_key_map[purpose]
                key = self.api_keys[key_index]
                self.key_usage[key] += 1
                return key

            # Otherwise, use the least used key (load balancing)
            key = min(self.key_usage, key=self.key_usage.get)
            self.key_usage[key] += 1
            return key

    def generate_content(self, prompt, model=None, temperature=0.7, max_tokens=None,
                         top_p=None, top_k=None, safety_settings=None, purpose='content_generation'):
        """Generate content using the Gemini API."""
        api_key = self._get_next_api_key(purpose=purpose)

        if not api_key:
            # Return mock response if no API key
            return self._mock_generate_response(prompt)

        model = model or self.default_model
        url = f"{self.base_url}/models/{model}:generateContent?key={api_key}"

        # Prepare request payload
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
            }
        }

        # Add optional parameters
        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens
        if top_p:
            payload["generationConfig"]["topP"] = top_p
        if top_k:
            payload["generationConfig"]["topK"] = top_k
        if safety_settings:
            payload["safetySettings"] = safety_settings

        try:
            start_time = time.time()
            response = self.session.post(url, json=payload)
            response.raise_for_status()

            # Log latency
            latency = time.time() - start_time
            logger.debug(f"Gemini API request completed in {latency:.2f}s")

            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Gemini API: {str(e)}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response body: {e.response.text}")
            raise

    def analyze_text(self, text, analysis_type="sentiment", model=None):
        """Analyze text using the Gemini API."""
        api_key = self._get_next_api_key(purpose='analysis')

        if not api_key:
            # Return mock response if no API key
            return self._mock_analyze_response(text, analysis_type)

        model = model or self.default_model

        # Create prompt based on analysis type
        if analysis_type == "sentiment":
            prompt = (
                "Analyze the sentiment of this text and respond with a JSON object "
                "containing 'sentiment' (positive, negative, or neutral), "
                "'confidence' (0-1), and 'explanation' (brief reason): "
                f"{text}"
            )
        elif analysis_type == "entities":
            prompt = (
                "Extract the key entities from this text and respond with a JSON object "
                "containing 'entities' (array of objects with 'name', 'type', and 'relevance' fields): "
                f"{text}"
            )
        elif analysis_type == "keywords":
            prompt = (
                "Extract the main keywords from this text and respond with a JSON object "
                "containing 'keywords' (array of objects with 'keyword' and 'relevance' fields): "
                f"{text}"
            )
        elif analysis_type == "summary":
            prompt = (
                "Summarize this text and respond with a JSON object "
                "containing 'summary' (string) and 'key_points' (array of strings): "
                f"{text}"
            )
        elif analysis_type == "classification":
            prompt = (
                "Classify this text into categories and respond with a JSON object "
                "containing 'categories' (array of objects with 'category' and 'confidence' fields): "
                f"{text}"
            )
        else:
            prompt = (
                f"Analyze this text for {analysis_type} and respond with a JSON object: "
                f"{text}"
            )

        # Generate content with low temperature for more deterministic results
        response = self.generate_content(prompt, model=model, temperature=0.1, purpose='analysis')

        # Extract and parse JSON from response
        try:
            text_response = self._extract_text(response)
            # Try to find JSON in the response
            json_match = re.search(r'```json\s*(.*?)\s*```', text_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without markdown
                json_match = re.search(r'(\{.*\})', text_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = text_response

            analysis_result = json.loads(json_str)
            return analysis_result
        except Exception as e:
            logger.error(f"Error parsing analysis response: {str(e)}")
            logger.error(f"Raw response: {text_response}")
            raise

    def generate_recommendations(self, user_data, model=None):
        """Generate personalized learning recommendations based on user data."""
        prompt = (
            "Based on the following user learning data, generate personalized educational "
            "recommendations for what they should learn next. Format your response as a JSON "
            "object with the arrays: 'recommended_topics', 'recommended_resources', and "
            "'learning_path'. For each recommendation, include a 'title', 'description', "
            "'difficulty', and 'relevance_score' (0-1). User data: "
            f"{json.dumps(user_data)}"
        )
        return self.generate_content(prompt, temperature=0.3, purpose='recommendation')

    def assess_learning(self, submission, criteria, model=None):
        """Assess a learning submission against given criteria."""
        prompt = (
            "Evaluate the following learning submission against the assessment criteria. "
            "Provide a detailed analysis, feedback, and scoring. Format your response as a JSON "
            "object with 'score' (0-100), 'feedback' (string), 'strengths' (array), 'areas_for_improvement' (array), "
            "and 'next_steps' (array). Submission: "
            f"{submission}\n\n"
            "Assessment Criteria: "
            f"{json.dumps(criteria)}"
        )
        return self.generate_content(prompt, temperature=0.1, purpose='assessment')

    def _extract_text(self, response):
        """Extract text from Gemini API response."""
        try:
            # Handle different response formats
            if 'candidates' in response and response['candidates']:
                if 'content' in response['candidates'][0]:
                    parts = response['candidates'][0]['content'].get('parts', [])
                    if parts and 'text' in parts[0]:
                        return parts[0]['text']

            # Fallback to returning the raw response
            return str(response)
        except Exception as e:
            logger.error(f"Error extracting text from response: {str(e)}")
            return str(response)

    def _mock_generate_response(self, prompt):
        """Generate a mock response for testing purposes."""
        logger.warning("Using mock Gemini API response")

        # Create a simple mock response
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": "This is a mock response from the Gemini API. "
                                        "Please configure a valid API key for actual responses."
                            }
                        ]
                    }
                }
            ]
        }

    def _mock_analyze_response(self, text, analysis_type):
        """Generate a mock analysis response for testing purposes."""
        logger.warning(f"Using mock Gemini API analysis response for {analysis_type}")

        # Create appropriate mock responses based on analysis type
        if analysis_type == "sentiment":
            return {
                "sentiment": random.choice(["positive", "negative", "neutral"]),
                "confidence": round(random.random(), 2),
                "explanation": "This is a mock sentiment analysis."
            }
        elif analysis_type == "entities":
            return {
                "entities": [
                    {
                        "name": "Example Entity",
                        "type": "ORGANIZATION",
                        "relevance": round(random.random(), 2)
                    }
                ]
            }
        elif analysis_type == "keywords":
            return {
                "keywords": [
                    {
                        "keyword": "example",
                        "relevance": round(random.random(), 2)
                    },
                    {
                        "keyword": "test",
                        "relevance": round(random.random(), 2)
                    }
                ]
            }
        else:
            return {
                "analysis": f"Mock {analysis_type} analysis",
                "confidence": round(random.random(), 2)
            }

def require_api_key(f):
    """Decorator to require API key authentication."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')

        if not api_key:
            return jsonify({'error': 'API key is required'}), 401

        if not api_key_auth.validate_key(api_key):
            return jsonify({'error': 'Invalid API key'}), 401

        # Store key owner in Flask's g object for access in route handlers
        g.api_key_owner = api_key_auth.get_key_owner(api_key)
        g.api_key_permissions = api_key_auth.get_key_permissions(api_key)

        # Check permissions if endpoint requires specific ones
        endpoint = request.endpoint
        if endpoint and endpoint.startswith('ai.'):
            # Extract route name from endpoint (e.g., 'ai.recommend' -> 'recommend')
            route_name = endpoint.split('.')[-1]
            required_permission = f"ai:{route_name}"

            if required_permission not in g.api_key_permissions and '*' not in g.api_key_permissions:
                return jsonify({'error': f'Permission denied: {required_permission} required'}), 403

        return f(*args, **kwargs)
    return decorated_function

def rate_limit(limit=100, period=60, key_func=None):
    """
    Decorator to apply rate limiting.

    Args:
        limit: Maximum number of requests allowed in the period
        period: Time period in seconds
        key_func: Function to determine the rate limit key (default: IP address)
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Determine rate limit key
            if key_func:
                key = key_func()
            else:
                # Default: use IP address or API key if available
                if hasattr(g, 'api_key_owner'):
                    key = f"api_key:{g.api_key_owner}"
                else:
                    key = request.remote_addr

            # Check rate limit
            is_limited, remaining, reset_time = rate_limiter.is_rate_limited(
                key, limit, period, strategy='sliding_window'
            )

            # Set rate limit headers
            response_headers = {
                'X-RateLimit-Limit': str(limit),
                'X-RateLimit-Remaining': str(remaining),
                'X-RateLimit-Reset': str(int(reset_time))
            }

            if is_limited:
                response = jsonify({'error': 'Rate limit exceeded'})
                response.headers.extend(response_headers)
                response.status_code = 429
                return response

            # Execute the route function
            response = f(*args, **kwargs)

            # If response is a tuple (response, status_code), get just the response
            if isinstance(response, tuple):
                resp_obj = response[0]
                # Add headers to the response object
                for key, value in response_headers.items():
                    resp_obj.headers[key] = value
                return response
            else:
                # Add headers to the response object
                for key, value in response_headers.items():
                    response.headers[key] = value
                return response

        return decorated_function
    return decorator

def cache_response(ttl=None, key_prefix=None, unless=None):
    """
    Decorator to cache API responses.

    Args:
        ttl: Cache TTL in seconds (default: None, uses cache default)
        key_prefix: Prefix for cache keys (default: None)
        unless: Function that returns True if caching should be skipped
    """
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Skip caching if condition is met
            if unless and unless():
                return f(*args, **kwargs)

            # Create cache key
            key_parts = [key_prefix or request.endpoint]

            # Add request path and query parameters to key
            key_parts.append(request.path)
            key_parts.append(hashlib.md5(request.query_string).hexdigest())

            # Add request body to key if it's a JSON request
            if request.is_json:
                key_parts.append(hashlib.md5(json.dumps(request.json, sort_keys=True).encode()).hexdigest())

            cache_key = ":".join(key_parts)

            # Try to get from cache
            cached_response = cache.get(cache_key)
            if cached_response:
                return cached_response

            # Execute the route function
            response = f(*args, **kwargs)

            # Cache the response
            if isinstance(response, tuple):
                # Only cache successful responses
                if response[1] == 200:
                    cache.set(cache_key, response, ttl)
            else:
                cache.set(cache_key, response, ttl)

            return response
        return decorated_function
    return decorator

def log_request(f):
    """Decorator to log API requests and responses."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Log request
        request_id = str(uuid.uuid4())
        g.request_id = request_id

        logger.info(f"Request {request_id}: {request.method} {request.path}")
        if request.is_json:
            # Log request body but mask sensitive data
            masked_data = mask_sensitive_data(request.json)
            logger.debug(f"Request {request_id} body: {json.dumps(masked_data)}")

        start_time = time.time()

        try:
            # Execute the route function
            response = f(*args, **kwargs)

            # Log response
            end_time = time.time()
            duration = end_time - start_time

            if isinstance(response, tuple):
                status_code = response[1]
            else:
                status_code = 200

            logger.info(f"Response {request_id}: {status_code} in {duration:.2f}s")

            return response
        except Exception as e:
            # Log exception
            end_time = time.time()
            duration = end_time - start_time

            logger.error(f"Exception {request_id}: {str(e)} in {duration:.2f}s")
            logger.error(traceback.format_exc())

            # Return error response
            return jsonify({
                'error': 'Internal server error',
                'request_id': request_id
            }), 500

    return decorated_function

def mask_sensitive_data(data, sensitive_keys=None):
    """Mask sensitive data for logging purposes."""
    if sensitive_keys is None:
        sensitive_keys = ['api_key', 'password', 'token', 'secret']

    if isinstance(data, dict):
        masked_data = {}
        for key, value in data.items():
            if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                masked_data[key] = '********'
            elif isinstance(value, (dict, list)):
                masked_data[key] = mask_sensitive_data(value, sensitive_keys)
            else:
                masked_data[key] = value
        return masked_data
    elif isinstance(data, list):
        return [mask_sensitive_data(item, sensitive_keys) for item in data]
    else:
        return data

# Move initialization to blueprint setup
@ai_bp.record
def setup_blueprint(state):
    """Initialize blueprint with application context."""
    app = state.app
    
    # Load API keys from application config
    api_key_auth.load_keys_from_config(app.config)

    # Load Gemini API keys from config
    gemini_keys = app.config.get('GEMINI_API_KEYS', [])

    # Create Gemini API client and attach to app
    if not hasattr(app, 'gemini_api'):
        app.gemini_api = GeminiAPI(api_keys=gemini_keys)

    logger.info("AI Blueprint initialized successfully")

@ai_bp.route('/recommend', methods=['POST'])
@log_request
@require_api_key
@rate_limit(limit=10, period=60)
@cache_response(ttl=300, key_prefix='recommend', unless=lambda: request.args.get('nocache') == '1')
def recommend():
    """Generate personalized learning recommendations based on user data."""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    user_data = request.json

    # Validate required fields
    required_fields = ['user_id', 'learning_history', 'interests']
    missing_fields = [field for field in required_fields if field not in user_data]
    if missing_fields:
        return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

    try:
        # Get Gemini API client
        gemini_api = current_app.gemini_api

        # Generate recommendations
        response = gemini_api.generate_recommendations(user_data)
        recommendations = gemini_api._extract_text(response)

        # Try to parse as JSON
        try:
            recommendations_json = json.loads(recommendations)
            return jsonify(recommendations_json)
        except json.JSONDecodeError:
            # Return as text if not valid JSON
            return jsonify({'recommendations': recommendations})

    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to generate recommendations', 'details': str(e)}), 500

@ai_bp.route('/analyze', methods=['POST'])
@log_request
@require_api_key
@rate_limit(limit=20, period=60)
def analyze():
    """Analyze text for various purposes (sentiment, entities, etc.)."""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.json

    # Validate required fields
    if 'text' not in data:
        return jsonify({'error': 'Missing required field: text'}), 400

    # Get analysis type
    analysis_type = data.get('analysis_type', 'sentiment')

    try:
        # Get Gemini API client
        gemini_api = current_app.gemini_api

        # Analyze text
        analysis_result = gemini_api.analyze_text(data['text'], analysis_type)

        return jsonify(analysis_result)
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to analyze text', 'details': str(e)}), 500

@ai_bp.route('/assess', methods=['POST'])
@log_request
@require_api_key
@rate_limit(limit=5, period=60)
def assess():
    """Assess a learning submission against given criteria."""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.json

    # Validate required fields
    required_fields = ['submission', 'criteria']
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

    try:
        # Get Gemini API client
        gemini_api = current_app.gemini_api

        # Assess submission
        response = gemini_api.assess_learning(data['submission'], data['criteria'])
        assessment_text = gemini_api._extract_text(response)

        # Try to parse as JSON
        try:
            assessment_json = json.loads(assessment_text)
            return jsonify(assessment_json)
        except json.JSONDecodeError:
            # Return as text if not valid JSON
            return jsonify({'assessment': assessment_text})

    except Exception as e:
        logger.error(f"Error assessing submission: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to assess submission', 'details': str(e)}), 500

@ai_bp.route('/generate', methods=['POST'])
@log_request
@require_api_key
@rate_limit(limit=10, period=60)
def generate():
    """Generate content based on a prompt."""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON'}), 400

    data = request.json

    # Validate required fields
    if 'prompt' not in data:
        return jsonify({'error': 'Missing required field: prompt'}), 400

    # Get optional parameters
    model = data.get('model')
    temperature = data.get('temperature', 0.7)
    max_tokens = data.get('max_tokens')

    try:
        # Get Gemini API client
        gemini_api = current_app.gemini_api

        # Generate content
        response = gemini_api.generate_content(
            data['prompt'],
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        generated_text = gemini_api._extract_text(response)

        return jsonify({'generated_text': generated_text})
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Failed to generate content', 'details': str(e)}), 500

@ai_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for the AI API."""
    # Check Gemini API
    try:
        gemini_api = current_app.gemini_api
        test_prompt = "Hello, this is a health check. Please respond with 'ok'."
        response = gemini_api.generate_content(test_prompt, temperature=0.1)
        gemini_status = "ok" if response else "error"
    except Exception as e:
        gemini_status = f"error: {str(e)}"

    # Check cache
    try:
        cache_stats = cache.get_stats()
        cache_status = "ok"
    except Exception as e:
        cache_stats = {}
        cache_status = f"error: {str(e)}"

    # Return health status
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'gemini_api': gemini_status,
            'cache': cache_status,
            'cache_stats': cache_stats
        }
    })

@ai_bp.route('/metrics', methods=['GET'])
@require_api_key
@log_request
def metrics():
    """Metrics endpoint for the AI API."""
    # Check if caller has admin permissions
    if 'admin' not in g.api_key_permissions and '*' not in g.api_key_permissions:
        return jsonify({'error': 'Permission denied: admin permission required'}), 403

    # Get cache statistics
    try:
        cache_stats = cache.get_stats()
    except Exception as e:
        cache_stats = {}
        logger.error(f"Error retrieving cache stats: {str(e)}")

    # Get API key usage statistics
    gemini_api = current_app.gemini_api
    key_usage = gemini_api.key_usage

    # Prepare the metrics response
    metrics_response = {
        'cache': cache_stats,
        'key_usage': {key[-8:]: usage for key, usage in key_usage.items()}  # Only show last 8 chars of keys
    }

    return jsonify(metrics_response)

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
        return jsonify({
            'success': True,
            'prompt': data['prompt'],
            'response': 'Test response',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        return jsonify({'error': 'Failed to generate content'}), 500
>>>>>>> f41f548 (frontend)
