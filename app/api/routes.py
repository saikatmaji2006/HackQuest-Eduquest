<<<<<<< HEAD
from flask import Blueprint, jsonify, request, current_app, g
from datetime import datetime
import logging
from functools import wraps
from models.recommendation_models import GeminiAPI, RecommendationModel
from typing import Callable, Any

# Set up logger
logger = logging.getLogger(__name__)

# Create blueprints
ai_bp = Blueprint('ai', __name__, url_prefix='/api/v1/ai')

def require_api_key(f: Callable) -> Callable:
    """Decorator to require API key for routes."""
    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'error': 'No API key provided'}), 401
        if api_key not in current_app.config['API_KEYS'].values():
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

# Configure Gemini API for the blueprint
@ai_bp.record
def on_blueprint_init(state: Any) -> None:
    """Initialize Gemini API when blueprint is registered."""
    app = state.app
    api_key = app.config.get('GEMINI_API_KEY')
    if not hasattr(app, 'gemini_api'):
        app.gemini_api = GeminiAPI(api_key=api_key)
        app.recommendation_model = RecommendationModel(gemini_api=app.gemini_api)
        logger.info("Initialized Gemini API and Recommendation Model")

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
            'generate': '/api/v1/ai/generate',
            'recommend': '/api/v1/ai/recommend'
        }
    })

@ai_bp.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'gemini_api': hasattr(current_app, 'gemini_api'),
        'recommendation_model': hasattr(current_app, 'recommendation_model')
    })

@ai_bp.route('/generate', methods=['POST'])
@require_api_key
def generate():
    """Generate AI content."""
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    data = request.get_json()
    if 'prompt' not in data:
        return jsonify({'error': 'Missing prompt in request body'}), 400
    
    try:
        response = current_app.gemini_api.generate_content(
            prompt=data['prompt'],
            temperature=data.get('temperature', 0.7),
            max_tokens=data.get('max_tokens')
        )
        return jsonify({
            'success': True,
            'prompt': data['prompt'],
            'response': response.text,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        return jsonify({'error': 'Failed to generate content'}), 500

@ai_bp.route('/recommend', methods=['POST'])
@require_api_key
def recommend():
    """Get AI-powered recommendations."""
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    data = request.get_json()
    if 'user_id' not in data:
        return jsonify({'error': 'Missing user_id in request body'}), 400
    
    try:
        recommendations = current_app.recommendation_model.recommend(
            user_id=data['user_id'],
            limit=data.get('limit', 5),
            strategy=data.get('strategy', 'hybrid')
        )
        return jsonify({
            'success': True,
            'user_id': data['user_id'],
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
=======
from flask import Blueprint, jsonify, request, current_app, g
from datetime import datetime
import logging
from functools import wraps
from models.recommendation_models import GeminiAPI, RecommendationModel
from typing import Callable, Any

# Set up logger
logger = logging.getLogger(__name__)

# Create blueprints
ai_bp = Blueprint('ai', __name__, url_prefix='/api/v1/ai')

def require_api_key(f: Callable) -> Callable:
    """Decorator to require API key for routes."""
    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return jsonify({'error': 'No API key provided'}), 401
        if api_key not in current_app.config['API_KEYS'].values():
            return jsonify({'error': 'Invalid API key'}), 401
        return f(*args, **kwargs)
    return decorated_function

# Configure Gemini API for the blueprint
@ai_bp.record
def on_blueprint_init(state: Any) -> None:
    """Initialize Gemini API when blueprint is registered."""
    app = state.app
    api_key = app.config.get('GEMINI_API_KEY')
    if not hasattr(app, 'gemini_api'):
        app.gemini_api = GeminiAPI(api_key=api_key)
        app.recommendation_model = RecommendationModel(gemini_api=app.gemini_api)
        logger.info("Initialized Gemini API and Recommendation Model")

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
            'generate': '/api/v1/ai/generate',
            'recommend': '/api/v1/ai/recommend'
        }
    })

@ai_bp.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'gemini_api': hasattr(current_app, 'gemini_api'),
        'recommendation_model': hasattr(current_app, 'recommendation_model')
    })

@ai_bp.route('/generate', methods=['POST'])
@require_api_key
def generate():
    """Generate AI content."""
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    data = request.get_json()
    if 'prompt' not in data:
        return jsonify({'error': 'Missing prompt in request body'}), 400
    
    try:
        response = current_app.gemini_api.generate_content(
            prompt=data['prompt'],
            temperature=data.get('temperature', 0.7),
            max_tokens=data.get('max_tokens')
        )
        return jsonify({
            'success': True,
            'prompt': data['prompt'],
            'response': response.text,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error generating content: {str(e)}")
        return jsonify({'error': 'Failed to generate content'}), 500

@ai_bp.route('/recommend', methods=['POST'])
@require_api_key
def recommend():
    """Get AI-powered recommendations."""
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    data = request.get_json()
    if 'user_id' not in data:
        return jsonify({'error': 'Missing user_id in request body'}), 400
    
    try:
        recommendations = current_app.recommendation_model.recommend(
            user_id=data['user_id'],
            limit=data.get('limit', 5),
            strategy=data.get('strategy', 'hybrid')
        )
        return jsonify({
            'success': True,
            'user_id': data['user_id'],
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
>>>>>>> f41f548 (frontend)
        return jsonify({'error': 'Failed to get recommendations'}), 500