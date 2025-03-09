<<<<<<< HEAD
from flask import Flask, jsonify, request, g, current_app
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS
import logging
import os
import time
import uuid
from logging.handlers import RotatingFileHandler
import traceback
from functools import wraps

# Import blueprints and config
from api.ai_routes import ai_bp
from api.leetcode_routes import leetcode_bp
from config import Config, get_config

def create_app(config_name="default"):
    """Initialize Flask application."""
    app = Flask(__name__)
    
    # Configure app using get_config
    config = get_config(config_name)
    app.config.from_object(config)
    
    # Setup middleware
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Register blueprints
    app.register_blueprint(ai_bp, url_prefix='/api/v1/ai')
    app.register_blueprint(leetcode_bp, url_prefix='/api/v1/leetcode')
    
    # Configure logging
    configure_logging(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    @app.route('/')
    def index():
        return jsonify({
            'name': 'Eduquest API',
            'version': app.config['API_VERSION'],
            'endpoints': {
                'ai': f"{app.config['API_PREFIX']}/ai",
                'leetcode': f"{app.config['API_PREFIX']}/leetcode"
            }
        })
    
    return app

def configure_logging(app):
    """Configure API logging."""
    if not app.debug and not app.testing:
        if not os.path.exists('logs'):
            os.mkdir('logs')
            
        log_file = os.path.join('logs', 'eduquest.log')
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=1024 * 1024,  # 1MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Eduquest API startup')

def api_error_response(error, status_code, message=None):
    """Generate standardized API error response."""
    error_id = str(uuid.uuid4())
    current_app.logger.error(f'Error {error_id}: {str(error)}')
    
    response = jsonify({
        "error": {
            "id": error_id,
            "code": status_code,
            "type": type(error).__name__,
            "message": message or str(error)
        }
    })
    response.status_code = status_code
    return response

def register_error_handlers(app):
    """Register API error handlers."""
    
    @app.errorhandler(400)
    def bad_request(error):
        return api_error_response(error, 400)

    @app.errorhandler(401)
    def unauthorized(error):
        return api_error_response(error, 401, "Authentication required")

    @app.errorhandler(403)
    def forbidden(error):
        return api_error_response(error, 403, "Permission denied")

    @app.errorhandler(404)
    def not_found(error):
        return api_error_response(error, 404, "Resource not found")

    @app.errorhandler(429)
    def too_many_requests(error):
        response = api_error_response(error, 429, "Rate limit exceeded")
        response.headers['Retry-After'] = '60'
        return response

    @app.errorhandler(Exception)
    def handle_error(error):
        return api_error_response(error, 500, "Internal server error")

def api_key_required(f):
    """API key authentication decorator."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return api_error_response(ValueError("Missing API key"), 401)
            
        if api_key not in current_app.config['API_KEYS'].values():
            return api_error_response(ValueError("Invalid API key"), 401)
            
        g.api_client = {
            'key': api_key,
            'request_id': str(uuid.uuid4())
        }
        
        return f(*args, **kwargs)
    return decorated_function

# Create the API application
app = create_app()

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5000))
        
        # Enable development server settings
        app.run(
            host='0.0.0.0',  # Allow external connections
            port=port,
            debug=True,      # Enable debug mode
            use_reloader=True,  # Enable hot reloading
            threaded=True    # Enable threading
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
=======
from flask import Flask, jsonify, request, g, current_app
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS
import logging
import os
import time
import uuid
from logging.handlers import RotatingFileHandler
import traceback
from functools import wraps

# Import blueprints and config
from api.ai_routes import ai_bp
from api.leetcode_routes import leetcode_bp
from config import Config, get_config

def create_app(config_name="default"):
    """Initialize Flask application."""
    app = Flask(__name__)
    
    # Configure app using get_config
    config = get_config(config_name)
    app.config.from_object(config)
    
    # Setup middleware
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Register blueprints
    app.register_blueprint(ai_bp, url_prefix='/api/v1/ai')
    app.register_blueprint(leetcode_bp, url_prefix='/api/v1/leetcode')
    
    # Configure logging
    configure_logging(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    @app.route('/')
    def index():
        return jsonify({
            'name': 'Eduquest API',
            'version': app.config['API_VERSION'],
            'endpoints': {
                'ai': f"{app.config['API_PREFIX']}/ai",
                'leetcode': f"{app.config['API_PREFIX']}/leetcode"
            }
        })
    
    return app

def configure_logging(app):
    """Configure API logging."""
    if not app.debug and not app.testing:
        if not os.path.exists('logs'):
            os.mkdir('logs')
            
        log_file = os.path.join('logs', 'eduquest.log')
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=1024 * 1024,  # 1MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Eduquest API startup')

def api_error_response(error, status_code, message=None):
    """Generate standardized API error response."""
    error_id = str(uuid.uuid4())
    current_app.logger.error(f'Error {error_id}: {str(error)}')
    
    response = jsonify({
        "error": {
            "id": error_id,
            "code": status_code,
            "type": type(error).__name__,
            "message": message or str(error)
        }
    })
    response.status_code = status_code
    return response

def register_error_handlers(app):
    """Register API error handlers."""
    
    @app.errorhandler(400)
    def bad_request(error):
        return api_error_response(error, 400)

    @app.errorhandler(401)
    def unauthorized(error):
        return api_error_response(error, 401, "Authentication required")

    @app.errorhandler(403)
    def forbidden(error):
        return api_error_response(error, 403, "Permission denied")

    @app.errorhandler(404)
    def not_found(error):
        return api_error_response(error, 404, "Resource not found")

    @app.errorhandler(429)
    def too_many_requests(error):
        response = api_error_response(error, 429, "Rate limit exceeded")
        response.headers['Retry-After'] = '60'
        return response

    @app.errorhandler(Exception)
    def handle_error(error):
        return api_error_response(error, 500, "Internal server error")

def api_key_required(f):
    """API key authentication decorator."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return api_error_response(ValueError("Missing API key"), 401)
            
        if api_key not in current_app.config['API_KEYS'].values():
            return api_error_response(ValueError("Invalid API key"), 401)
            
        g.api_client = {
            'key': api_key,
            'request_id': str(uuid.uuid4())
        }
        
        return f(*args, **kwargs)
    return decorated_function

# Create the API application
app = create_app()

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5000))
        
        # Enable development server settings
        app.run(
            host='0.0.0.0',  # Allow external connections
            port=port,
            debug=True,      # Enable debug mode
            use_reloader=True,  # Enable hot reloading
            threaded=True    # Enable threading
        )
    except Exception as e:
        print(f"Failed to start server: {e}")
>>>>>>> f41f548 (frontend)
        traceback.print_exc()