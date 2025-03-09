<<<<<<< HEAD
from flask import Blueprint, jsonify, request, current_app, abort
from datetime import datetime, timedelta
import logging
import os
import json
import time
import hashlib
from functools import wraps
from cachetools import TTLCache, LRUCache
from redis import Redis
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import asyncio
import concurrent.futures
from leetcode_api import LeetCode # type: ignore

# Set up logger with more detailed configuration
logger = logging.getLogger(__name__)

# Create blueprint
leetcode_bp = Blueprint('leetcode', __name__)

# Initialize more sophisticated caching system
problems_cache = TTLCache(maxsize=200, ttl=3600)  # 1 hour cache
problem_cache = TTLCache(maxsize=1000, ttl=1800)  # 30 minutes cache
submissions_cache = TTLCache(maxsize=500, ttl=900)  # 15 minutes cache
user_stats_cache = TTLCache(maxsize=100, ttl=1200)  # 20 minutes cache

# Try to use Redis for distributed caching if available
try:
    redis_url = os.getenv('REDIS_URL')
    if redis_url:
        redis_client = Redis.from_url(redis_url)
        redis_available = True
    else:
        redis_available = False
except ImportError:
    redis_available = False

# Set up rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Performance metrics
request_times = LRUCache(maxsize=1000)

def timed_execution(f):
    """Decorator to measure execution time of functions."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Store metrics
        endpoint = request.endpoint if request else f.__name__
        request_times[f"{endpoint}:{time.time()}"] = execution_time
        
        # Add timing header to response if it's a Flask response
        if hasattr(result, 'headers'):
            result.headers['X-Execution-Time'] = f"{execution_time:.4f}s"
        
        logger.debug(f"Execution time for {f.__name__}: {execution_time:.4f}s")
        return result
    return wrapper

def advanced_cache(cache_obj, key_prefix, ttl=None):
    """Advanced caching decorator with support for Redis and local cache."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Generate a more robust cache key including query parameters
            request_args = request.args.to_dict() if request else {}
            cache_data = {
                'args': args,
                'kwargs': kwargs,
                'query_params': request_args
            }
            cache_key = f"{key_prefix}:{hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()}"
            
            # Try Redis first if available
            if redis_available:
                try:
                    cached_data = redis_client.get(cache_key)
                    if cached_data:
                        logger.debug(f"Redis cache hit for {cache_key}")
                        return jsonify(json.loads(cached_data))
                except Exception as e:
                    logger.warning(f"Redis error: {str(e)}")
            
            # Fall back to local cache
            result = cache_obj.get(cache_key)
            if result is not None:
                logger.debug(f"Local cache hit for {cache_key}")
                return jsonify(json.loads(result))
            
            # Execute the function if no cache hit
            result = f(*args, **kwargs)
            
            # Only cache successful responses
            if hasattr(result, 'status_code') and 200 <= result.status_code < 300:
                result_json = json.dumps(result.get_json())
                cache_obj[cache_key] = result_json
                
                # Also cache in Redis if available
                if redis_available:
                    try:
                        redis_ttl = ttl if ttl else 3600  # Default 1 hour
                        redis_client.setex(cache_key, redis_ttl, result_json)
                    except Exception as e:
                        logger.warning(f"Redis caching error: {str(e)}")
            
            return result
        return wrapper
    return decorator

def error_handler(f):
    """Advanced error handling decorator."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            error_id = hashlib.md5(f"{time.time()}{str(e)}".encode()).hexdigest()[:8]
            logger.error(f"Error ID {error_id}: {str(e)}", exc_info=True)
            
            # Determine appropriate status code
            status_code = 500
            if "authentication failed" in str(e).lower():
                status_code = 401
            elif "not found" in str(e).lower():
                status_code = 404
            elif "rate limit" in str(e).lower():
                status_code = 429
            
            return jsonify({
                'success': False,
                'error': str(e),
                'error_id': error_id,
                'timestamp': datetime.now().isoformat()
            }), status_code
    return wrapper

class LeetCodeAPI:
    def __init__(self):
        self.client = LeetCode()
        self.session = os.getenv('LEETCODE_SESSION', '')
        self.csrf_token = os.getenv('CSRF_TOKEN', '')
        self.last_auth = None
        self.auth_ttl = 3600  # Re-authenticate every hour
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        
    def authenticate(self):
        """Authenticate with LeetCode with caching."""
        current_time = time.time()
        
        # Return cached authentication if still valid
        if self.last_auth and current_time - self.last_auth < self.auth_ttl:
            return True
            
        try:
            self.client.auth(self.session, self.csrf_token)
            self.last_auth = current_time
            return True
        except Exception as e:
            logger.error(f"LeetCode authentication failed: {str(e)}")
            self.last_auth = None
            return False
    
    def run_async(self, func, *args, **kwargs):
        """Run a function asynchronously using thread pool."""
        return self.executor.submit(func, *args, **kwargs)

leetcode_api = LeetCodeAPI()

@leetcode_bp.route('/problems')
@limiter.limit("30 per minute")
@timed_execution
@advanced_cache(problems_cache, 'problems', ttl=3600)
@error_handler
def get_problems():
    """Get list of LeetCode problems with advanced filtering."""
    if not leetcode_api.authenticate():
        abort(401, description="LeetCode authentication failed")
    
    # Get query parameters with defaults
    difficulty = request.args.get('difficulty')
    topic = request.args.get('topic')
    status = request.args.get('status')
    search = request.args.get('search')
    limit = min(int(request.args.get('limit', 100)), 500)  # Cap at 500
    offset = int(request.args.get('offset', 0))
    sort_by = request.args.get('sort_by', 'id')
    sort_order = request.args.get('sort_order', 'asc')
    
    problems = leetcode_api.client.problems()
    
    # Apply filters
    if difficulty:
        problems = [p for p in problems if p['difficulty'].lower() == difficulty.lower()]
    if topic:
        problems = [p for p in problems if topic.lower() in [t.lower() for t in p.get('topics', [])]]
    if status:
        problems = [p for p in problems if p['status'].lower() == status.lower()]
    if search:
        problems = [p for p in problems if search.lower() in p.get('title', '').lower() or 
                                           search.lower() in p.get('titleSlug', '').lower()]
    
    # Sort problems
    reverse_sort = sort_order.lower() == 'desc'
    if sort_by == 'difficulty':
        difficulty_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
        problems.sort(key=lambda p: difficulty_map.get(p.get('difficulty'), 0), reverse=reverse_sort)
    elif sort_by == 'acceptance':
        problems.sort(key=lambda p: float(p.get('stats', {}).get('acRate', '0').rstrip('%') or 0), reverse=reverse_sort)
    elif sort_by == 'frequency':
        problems.sort(key=lambda p: p.get('frequency', 0), reverse=reverse_sort)
    else:  # Default to ID
        problems.sort(key=lambda p: p.get('frontendQuestionId', 0), reverse=reverse_sort)
    
    # Apply pagination
    total_count = len(problems)
    problems = problems[offset:offset+limit]
    
    return jsonify({
        'success': True,
        'count': len(problems),
        'total_count': total_count,
        'offset': offset,
        'limit': limit,
        'data': problems,
        'timestamp': datetime.now().isoformat()
    })

@leetcode_bp.route('/daily')
@limiter.limit("60 per hour")
@timed_execution
@advanced_cache(problem_cache, 'daily', ttl=3600)
@error_handler
def get_daily_problem():
    """Get daily LeetCode challenge with enhanced details."""
    if not leetcode_api.authenticate():
        abort(401, description="LeetCode authentication failed")
        
    daily = leetcode_api.client.daily()
    
    # Enhance with problem details if available
    if daily and 'titleSlug' in daily:
        try:
            problem_details = leetcode_api.client.problem(daily['titleSlug'])
            daily.update({
                'difficulty_level': problem_details.get('difficulty'),
                'acceptance_rate': problem_details.get('stats', {}).get('acceptanceRate'),
                'topics': problem_details.get('topicTags', []),
                'similar_problems': problem_details.get('similarQuestions', [])
            })
        except Exception as e:
            logger.warning(f"Could not fetch additional details for daily challenge: {str(e)}")
    
    return jsonify({
        'success': True,
        'data': daily,
        'timestamp': datetime.now().isoformat()
    })

@leetcode_bp.route('/problem/<title_slug>')
@limiter.limit("60 per hour")
@timed_execution
@advanced_cache(problem_cache, 'problem', ttl=1800)
@error_handler
def get_problem(title_slug):
    """Get comprehensive details for a specific LeetCode problem."""
    if not leetcode_api.authenticate():
        abort(401, description="LeetCode authentication failed")
        
    problem = leetcode_api.client.problem(title_slug)
    
    if not problem:
        abort(404, description=f"Problem {title_slug} not found")
    
    # Enhance problem data with more structured information
    enhanced_problem = {
        **problem,
        'difficulty_level': problem.get('difficulty'),
        'acceptance_rate': problem.get('stats', {}).get('acceptanceRate'),
        'total_accepted': problem.get('stats', {}).get('totalAccepted'),
        'total_submissions': problem.get('stats', {}).get('totalSubmissions'),
        'similar_problems': problem.get('similarQuestions', []),
        'topics': problem.get('topicTags', []),
        'companies': problem.get('companyTags', []),
        'hints': problem.get('hints', []),
        'solution': {
            'available': problem.get('solution') is not None,
            'url': f"https://leetcode.com/problems/{title_slug}/solution/" if problem.get('solution') else None
        }
    }
    
    # Add code snippets in a more structured format
    if 'codeSnippets' in problem:
        enhanced_problem['code_templates'] = {
            snippet.get('langSlug'): {
                'code': snippet.get('code'),
                'language': snippet.get('lang')
            } for snippet in problem.get('codeSnippets', [])
        }
    
    return jsonify({
        'success': True,
        'data': enhanced_problem,
        'timestamp': datetime.now().isoformat()
    })

@leetcode_bp.route('/submissions/<title_slug>')
@limiter.limit("30 per minute")
@timed_execution
@advanced_cache(submissions_cache, 'submissions', ttl=900)
@error_handler
def get_submissions(title_slug):
    """Get detailed submissions for a specific problem."""
    if not leetcode_api.authenticate():
        abort(401, description="LeetCode authentication failed")
    
    # Get query parameters
    limit = min(int(request.args.get('limit', 20)), 100)  # Cap at 100
    status = request.args.get('status')  # Accepted, Wrong Answer, etc.
    
    submissions = leetcode_api.client.submissions(title_slug)
    
    # Filter by status if specified
    if status:
        submissions = [s for s in submissions if s.get('statusDisplay', '').lower() == status.lower()]
    
    # Limit the number of submissions returned
    submissions = submissions[:limit]
    
    # Enhance submission data
    enhanced_submissions = []
    for submission in submissions:
        enhanced_submission = {
            **submission,
            'runtime_percentile': submission.get('runtimePercentile'),
            'memory_percentile': submission.get('memoryPercentile'),
            'submission_timestamp': submission.get('timestamp'),
            'formatted_time': datetime.fromtimestamp(submission.get('timestamp', 0)).isoformat() if submission.get('timestamp') else None
        }
        enhanced_submissions.append(enhanced_submission)
    
    return jsonify({
        'success': True,
        'count': len(enhanced_submissions),
        'data': enhanced_submissions,
        'timestamp': datetime.now().isoformat()
    })

@leetcode_bp.route('/user/profile')
@limiter.limit("20 per minute")
@timed_execution
@advanced_cache(user_stats_cache, 'user_profile', ttl=1200)
@error_handler
def get_user_profile():
    """Get authenticated user's LeetCode profile and statistics."""
    if not leetcode_api.authenticate():
        abort(401, description="LeetCode authentication failed")
    
    profile = leetcode_api.client.user()
    
    if not profile:
        abort(404, description="User profile not found")
    
    # Enhance with calculated statistics
    solved_problems = profile.get('submitStats', {}).get('acSubmissionNum', [])
    total_problems = profile.get('submitStats', {}).get('totalSubmissionNum', [])
    
    # Calculate solve rates by difficulty
    solve_rates = {}
    for difficulty in ['Easy', 'Medium', 'Hard']:
        solved = next((item['count'] for item in solved_problems if item['difficulty'] == difficulty), 0)
        total = next((item['count'] for item in total_problems if item['difficulty'] == difficulty), 0)
        solve_rates[difficulty.lower()] = {
            'solved': solved,
            'total': total,
            'rate': f"{(solved/total*100):.1f}%" if total > 0 else "0%"
        }

    enhanced_profile = {
        'username': profile.get('username'),
        'real_name': profile.get('realName'),
        'location': profile.get('profile', {}).get('countryName'),
        'ranking': profile.get('profile', {}).get('ranking'),
        'reputation': profile.get('profile', {}).get('reputation'),
        'website': profile.get('profile', {}).get('websites', []),
        'github': profile.get('profile', {}).get('githubUrl'),
        'linkedin': profile.get('profile', {}).get('linkedinUrl'),
        'twitter': profile.get('profile', {}).get('twitterUrl'),
        'skills': profile.get('profile', {}).get('skillTags', []),
        'company': profile.get('profile', {}).get('company'),
        'school': profile.get('profile', {}).get('school'),
        'stats': {
            'total_solved': sum(item['count'] for item in solved_problems),
            'total_questions': sum(item['count'] for item in total_problems),
            'acceptance_rate': profile.get('submitStats', {}).get('acRate', '0%'),
            'solve_rates': solve_rates,
            'contribution_points': profile.get('contributions', {}).get('points', 0),
            'reputation': profile.get('profile', {}).get('reputation', 0),
            'solution_count': profile.get('solutionCount', 0)
        },
        'badges': profile.get('badges', []),
        'recent_submissions': profile.get('recentSubmissionList', [])[:5]
    }

    return jsonify({
        'success': True,
        'data': enhanced_profile,
        'timestamp': datetime.now().isoformat()
    })

@leetcode_bp.route('/contest/upcoming')
@limiter.limit("20 per minute")
@timed_execution
@advanced_cache(problem_cache, 'upcoming_contests', ttl=1800)
@error_handler
def get_upcoming_contests():
    """Get information about upcoming LeetCode contests."""
    if not leetcode_api.authenticate():
        abort(401, description="LeetCode authentication failed")

    contests = leetcode_api.client.upcoming_contests()
    
    enhanced_contests = []
    for contest in contests:
        enhanced_contest = {
            'title': contest.get('title'),
            'start_time': datetime.fromtimestamp(contest.get('startTime', 0)).isoformat(),
            'end_time': datetime.fromtimestamp(contest.get('endTime', 0)).isoformat(),
            'duration': contest.get('duration'),
            'contest_type': contest.get('contestType'),
            'registration_url': f"https://leetcode.com/contest/{contest.get('titleSlug')}",
            'registered_count': contest.get('registeredCount', 0)
        }
        enhanced_contests.append(enhanced_contest)

    return jsonify({
        'success': True,
        'count': len(enhanced_contests),
        'data': enhanced_contests,
        'timestamp': datetime.now().isoformat()
    })

@leetcode_bp.route('/metrics')
@limiter.limit("10 per minute")
@error_handler
def get_api_metrics():
    """Get API performance metrics."""
    if not request.headers.get('X-Admin-Key') == os.getenv('ADMIN_API_KEY'):
        abort(403, description="Admin access required")

    # Calculate average response times
    response_times = list(request_times.values())
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0

    # Cache statistics
    cache_stats = {
        'problems': len(problems_cache),
        'problem_details': len(problem_cache),
        'submissions': len(submissions_cache),
        'user_stats': len(user_stats_cache)
    }

    return jsonify({
        'success': True,
        'data': {
            'average_response_time': f"{avg_response_time:.4f}s",
            'total_requests': len(response_times),
            'cache_stats': cache_stats,
            'redis_available': redis_available
        },
        'timestamp': datetime.now().isoformat()
    })

# Error handlers
@leetcode_bp.errorhandler(401)
def unauthorized_error(error):
    return jsonify({
        'success': False,
        'error': 'Unauthorized',
        'message': str(error.description),
        'timestamp': datetime.now().isoformat()
    }), 401

@leetcode_bp.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'success': False,
        'error': 'Not Found',
        'message': str(error.description),
        'timestamp': datetime.now().isoformat()
    }), 404

@leetcode_bp.errorhandler(429)
def ratelimit_error(error):
    return jsonify({
        'success': False,
        'error': 'Rate Limit Exceeded',
        'message': str(error.description),
        'timestamp': datetime.now().isoformat()
=======
from flask import Blueprint, jsonify, request, current_app, abort
from datetime import datetime, timedelta
import logging
import os
import json
import time
import hashlib
from functools import wraps
from cachetools import TTLCache, LRUCache
from redis import Redis
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import asyncio
import concurrent.futures
from leetcode_api import LeetCode # type: ignore

# Set up logger with more detailed configuration
logger = logging.getLogger(__name__)

# Create blueprint
leetcode_bp = Blueprint('leetcode', __name__)

# Initialize more sophisticated caching system
problems_cache = TTLCache(maxsize=200, ttl=3600)  # 1 hour cache
problem_cache = TTLCache(maxsize=1000, ttl=1800)  # 30 minutes cache
submissions_cache = TTLCache(maxsize=500, ttl=900)  # 15 minutes cache
user_stats_cache = TTLCache(maxsize=100, ttl=1200)  # 20 minutes cache

# Try to use Redis for distributed caching if available
try:
    redis_url = os.getenv('REDIS_URL')
    if redis_url:
        redis_client = Redis.from_url(redis_url)
        redis_available = True
    else:
        redis_available = False
except ImportError:
    redis_available = False

# Set up rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Performance metrics
request_times = LRUCache(maxsize=1000)

def timed_execution(f):
    """Decorator to measure execution time of functions."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # Store metrics
        endpoint = request.endpoint if request else f.__name__
        request_times[f"{endpoint}:{time.time()}"] = execution_time
        
        # Add timing header to response if it's a Flask response
        if hasattr(result, 'headers'):
            result.headers['X-Execution-Time'] = f"{execution_time:.4f}s"
        
        logger.debug(f"Execution time for {f.__name__}: {execution_time:.4f}s")
        return result
    return wrapper

def advanced_cache(cache_obj, key_prefix, ttl=None):
    """Advanced caching decorator with support for Redis and local cache."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            # Generate a more robust cache key including query parameters
            request_args = request.args.to_dict() if request else {}
            cache_data = {
                'args': args,
                'kwargs': kwargs,
                'query_params': request_args
            }
            cache_key = f"{key_prefix}:{hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()}"
            
            # Try Redis first if available
            if redis_available:
                try:
                    cached_data = redis_client.get(cache_key)
                    if cached_data:
                        logger.debug(f"Redis cache hit for {cache_key}")
                        return jsonify(json.loads(cached_data))
                except Exception as e:
                    logger.warning(f"Redis error: {str(e)}")
            
            # Fall back to local cache
            result = cache_obj.get(cache_key)
            if result is not None:
                logger.debug(f"Local cache hit for {cache_key}")
                return jsonify(json.loads(result))
            
            # Execute the function if no cache hit
            result = f(*args, **kwargs)
            
            # Only cache successful responses
            if hasattr(result, 'status_code') and 200 <= result.status_code < 300:
                result_json = json.dumps(result.get_json())
                cache_obj[cache_key] = result_json
                
                # Also cache in Redis if available
                if redis_available:
                    try:
                        redis_ttl = ttl if ttl else 3600  # Default 1 hour
                        redis_client.setex(cache_key, redis_ttl, result_json)
                    except Exception as e:
                        logger.warning(f"Redis caching error: {str(e)}")
            
            return result
        return wrapper
    return decorator

def error_handler(f):
    """Advanced error handling decorator."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            error_id = hashlib.md5(f"{time.time()}{str(e)}".encode()).hexdigest()[:8]
            logger.error(f"Error ID {error_id}: {str(e)}", exc_info=True)
            
            # Determine appropriate status code
            status_code = 500
            if "authentication failed" in str(e).lower():
                status_code = 401
            elif "not found" in str(e).lower():
                status_code = 404
            elif "rate limit" in str(e).lower():
                status_code = 429
            
            return jsonify({
                'success': False,
                'error': str(e),
                'error_id': error_id,
                'timestamp': datetime.now().isoformat()
            }), status_code
    return wrapper

class LeetCodeAPI:
    def __init__(self):
        self.client = LeetCode()
        self.session = os.getenv('LEETCODE_SESSION', '')
        self.csrf_token = os.getenv('CSRF_TOKEN', '')
        self.last_auth = None
        self.auth_ttl = 3600  # Re-authenticate every hour
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
        
    def authenticate(self):
        """Authenticate with LeetCode with caching."""
        current_time = time.time()
        
        # Return cached authentication if still valid
        if self.last_auth and current_time - self.last_auth < self.auth_ttl:
            return True
            
        try:
            self.client.auth(self.session, self.csrf_token)
            self.last_auth = current_time
            return True
        except Exception as e:
            logger.error(f"LeetCode authentication failed: {str(e)}")
            self.last_auth = None
            return False
    
    def run_async(self, func, *args, **kwargs):
        """Run a function asynchronously using thread pool."""
        return self.executor.submit(func, *args, **kwargs)

leetcode_api = LeetCodeAPI()

@leetcode_bp.route('/problems')
@limiter.limit("30 per minute")
@timed_execution
@advanced_cache(problems_cache, 'problems', ttl=3600)
@error_handler
def get_problems():
    """Get list of LeetCode problems with advanced filtering."""
    if not leetcode_api.authenticate():
        abort(401, description="LeetCode authentication failed")
    
    # Get query parameters with defaults
    difficulty = request.args.get('difficulty')
    topic = request.args.get('topic')
    status = request.args.get('status')
    search = request.args.get('search')
    limit = min(int(request.args.get('limit', 100)), 500)  # Cap at 500
    offset = int(request.args.get('offset', 0))
    sort_by = request.args.get('sort_by', 'id')
    sort_order = request.args.get('sort_order', 'asc')
    
    problems = leetcode_api.client.problems()
    
    # Apply filters
    if difficulty:
        problems = [p for p in problems if p['difficulty'].lower() == difficulty.lower()]
    if topic:
        problems = [p for p in problems if topic.lower() in [t.lower() for t in p.get('topics', [])]]
    if status:
        problems = [p for p in problems if p['status'].lower() == status.lower()]
    if search:
        problems = [p for p in problems if search.lower() in p.get('title', '').lower() or 
                                           search.lower() in p.get('titleSlug', '').lower()]
    
    # Sort problems
    reverse_sort = sort_order.lower() == 'desc'
    if sort_by == 'difficulty':
        difficulty_map = {'Easy': 1, 'Medium': 2, 'Hard': 3}
        problems.sort(key=lambda p: difficulty_map.get(p.get('difficulty'), 0), reverse=reverse_sort)
    elif sort_by == 'acceptance':
        problems.sort(key=lambda p: float(p.get('stats', {}).get('acRate', '0').rstrip('%') or 0), reverse=reverse_sort)
    elif sort_by == 'frequency':
        problems.sort(key=lambda p: p.get('frequency', 0), reverse=reverse_sort)
    else:  # Default to ID
        problems.sort(key=lambda p: p.get('frontendQuestionId', 0), reverse=reverse_sort)
    
    # Apply pagination
    total_count = len(problems)
    problems = problems[offset:offset+limit]
    
    return jsonify({
        'success': True,
        'count': len(problems),
        'total_count': total_count,
        'offset': offset,
        'limit': limit,
        'data': problems,
        'timestamp': datetime.now().isoformat()
    })

@leetcode_bp.route('/daily')
@limiter.limit("60 per hour")
@timed_execution
@advanced_cache(problem_cache, 'daily', ttl=3600)
@error_handler
def get_daily_problem():
    """Get daily LeetCode challenge with enhanced details."""
    if not leetcode_api.authenticate():
        abort(401, description="LeetCode authentication failed")
        
    daily = leetcode_api.client.daily()
    
    # Enhance with problem details if available
    if daily and 'titleSlug' in daily:
        try:
            problem_details = leetcode_api.client.problem(daily['titleSlug'])
            daily.update({
                'difficulty_level': problem_details.get('difficulty'),
                'acceptance_rate': problem_details.get('stats', {}).get('acceptanceRate'),
                'topics': problem_details.get('topicTags', []),
                'similar_problems': problem_details.get('similarQuestions', [])
            })
        except Exception as e:
            logger.warning(f"Could not fetch additional details for daily challenge: {str(e)}")
    
    return jsonify({
        'success': True,
        'data': daily,
        'timestamp': datetime.now().isoformat()
    })

@leetcode_bp.route('/problem/<title_slug>')
@limiter.limit("60 per hour")
@timed_execution
@advanced_cache(problem_cache, 'problem', ttl=1800)
@error_handler
def get_problem(title_slug):
    """Get comprehensive details for a specific LeetCode problem."""
    if not leetcode_api.authenticate():
        abort(401, description="LeetCode authentication failed")
        
    problem = leetcode_api.client.problem(title_slug)
    
    if not problem:
        abort(404, description=f"Problem {title_slug} not found")
    
    # Enhance problem data with more structured information
    enhanced_problem = {
        **problem,
        'difficulty_level': problem.get('difficulty'),
        'acceptance_rate': problem.get('stats', {}).get('acceptanceRate'),
        'total_accepted': problem.get('stats', {}).get('totalAccepted'),
        'total_submissions': problem.get('stats', {}).get('totalSubmissions'),
        'similar_problems': problem.get('similarQuestions', []),
        'topics': problem.get('topicTags', []),
        'companies': problem.get('companyTags', []),
        'hints': problem.get('hints', []),
        'solution': {
            'available': problem.get('solution') is not None,
            'url': f"https://leetcode.com/problems/{title_slug}/solution/" if problem.get('solution') else None
        }
    }
    
    # Add code snippets in a more structured format
    if 'codeSnippets' in problem:
        enhanced_problem['code_templates'] = {
            snippet.get('langSlug'): {
                'code': snippet.get('code'),
                'language': snippet.get('lang')
            } for snippet in problem.get('codeSnippets', [])
        }
    
    return jsonify({
        'success': True,
        'data': enhanced_problem,
        'timestamp': datetime.now().isoformat()
    })

@leetcode_bp.route('/submissions/<title_slug>')
@limiter.limit("30 per minute")
@timed_execution
@advanced_cache(submissions_cache, 'submissions', ttl=900)
@error_handler
def get_submissions(title_slug):
    """Get detailed submissions for a specific problem."""
    if not leetcode_api.authenticate():
        abort(401, description="LeetCode authentication failed")
    
    # Get query parameters
    limit = min(int(request.args.get('limit', 20)), 100)  # Cap at 100
    status = request.args.get('status')  # Accepted, Wrong Answer, etc.
    
    submissions = leetcode_api.client.submissions(title_slug)
    
    # Filter by status if specified
    if status:
        submissions = [s for s in submissions if s.get('statusDisplay', '').lower() == status.lower()]
    
    # Limit the number of submissions returned
    submissions = submissions[:limit]
    
    # Enhance submission data
    enhanced_submissions = []
    for submission in submissions:
        enhanced_submission = {
            **submission,
            'runtime_percentile': submission.get('runtimePercentile'),
            'memory_percentile': submission.get('memoryPercentile'),
            'submission_timestamp': submission.get('timestamp'),
            'formatted_time': datetime.fromtimestamp(submission.get('timestamp', 0)).isoformat() if submission.get('timestamp') else None
        }
        enhanced_submissions.append(enhanced_submission)
    
    return jsonify({
        'success': True,
        'count': len(enhanced_submissions),
        'data': enhanced_submissions,
        'timestamp': datetime.now().isoformat()
    })

@leetcode_bp.route('/user/profile')
@limiter.limit("20 per minute")
@timed_execution
@advanced_cache(user_stats_cache, 'user_profile', ttl=1200)
@error_handler
def get_user_profile():
    """Get authenticated user's LeetCode profile and statistics."""
    if not leetcode_api.authenticate():
        abort(401, description="LeetCode authentication failed")
    
    profile = leetcode_api.client.user()
    
    if not profile:
        abort(404, description="User profile not found")
    
    # Enhance with calculated statistics
    solved_problems = profile.get('submitStats', {}).get('acSubmissionNum', [])
    total_problems = profile.get('submitStats', {}).get('totalSubmissionNum', [])
    
    # Calculate solve rates by difficulty
    solve_rates = {}
    for difficulty in ['Easy', 'Medium', 'Hard']:
        solved = next((item['count'] for item in solved_problems if item['difficulty'] == difficulty), 0)
        total = next((item['count'] for item in total_problems if item['difficulty'] == difficulty), 0)
        solve_rates[difficulty.lower()] = {
            'solved': solved,
            'total': total,
            'rate': f"{(solved/total*100):.1f}%" if total > 0 else "0%"
        }

    enhanced_profile = {
        'username': profile.get('username'),
        'real_name': profile.get('realName'),
        'location': profile.get('profile', {}).get('countryName'),
        'ranking': profile.get('profile', {}).get('ranking'),
        'reputation': profile.get('profile', {}).get('reputation'),
        'website': profile.get('profile', {}).get('websites', []),
        'github': profile.get('profile', {}).get('githubUrl'),
        'linkedin': profile.get('profile', {}).get('linkedinUrl'),
        'twitter': profile.get('profile', {}).get('twitterUrl'),
        'skills': profile.get('profile', {}).get('skillTags', []),
        'company': profile.get('profile', {}).get('company'),
        'school': profile.get('profile', {}).get('school'),
        'stats': {
            'total_solved': sum(item['count'] for item in solved_problems),
            'total_questions': sum(item['count'] for item in total_problems),
            'acceptance_rate': profile.get('submitStats', {}).get('acRate', '0%'),
            'solve_rates': solve_rates,
            'contribution_points': profile.get('contributions', {}).get('points', 0),
            'reputation': profile.get('profile', {}).get('reputation', 0),
            'solution_count': profile.get('solutionCount', 0)
        },
        'badges': profile.get('badges', []),
        'recent_submissions': profile.get('recentSubmissionList', [])[:5]
    }

    return jsonify({
        'success': True,
        'data': enhanced_profile,
        'timestamp': datetime.now().isoformat()
    })

@leetcode_bp.route('/contest/upcoming')
@limiter.limit("20 per minute")
@timed_execution
@advanced_cache(problem_cache, 'upcoming_contests', ttl=1800)
@error_handler
def get_upcoming_contests():
    """Get information about upcoming LeetCode contests."""
    if not leetcode_api.authenticate():
        abort(401, description="LeetCode authentication failed")

    contests = leetcode_api.client.upcoming_contests()
    
    enhanced_contests = []
    for contest in contests:
        enhanced_contest = {
            'title': contest.get('title'),
            'start_time': datetime.fromtimestamp(contest.get('startTime', 0)).isoformat(),
            'end_time': datetime.fromtimestamp(contest.get('endTime', 0)).isoformat(),
            'duration': contest.get('duration'),
            'contest_type': contest.get('contestType'),
            'registration_url': f"https://leetcode.com/contest/{contest.get('titleSlug')}",
            'registered_count': contest.get('registeredCount', 0)
        }
        enhanced_contests.append(enhanced_contest)

    return jsonify({
        'success': True,
        'count': len(enhanced_contests),
        'data': enhanced_contests,
        'timestamp': datetime.now().isoformat()
    })

@leetcode_bp.route('/metrics')
@limiter.limit("10 per minute")
@error_handler
def get_api_metrics():
    """Get API performance metrics."""
    if not request.headers.get('X-Admin-Key') == os.getenv('ADMIN_API_KEY'):
        abort(403, description="Admin access required")

    # Calculate average response times
    response_times = list(request_times.values())
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0

    # Cache statistics
    cache_stats = {
        'problems': len(problems_cache),
        'problem_details': len(problem_cache),
        'submissions': len(submissions_cache),
        'user_stats': len(user_stats_cache)
    }

    return jsonify({
        'success': True,
        'data': {
            'average_response_time': f"{avg_response_time:.4f}s",
            'total_requests': len(response_times),
            'cache_stats': cache_stats,
            'redis_available': redis_available
        },
        'timestamp': datetime.now().isoformat()
    })

# Error handlers
@leetcode_bp.errorhandler(401)
def unauthorized_error(error):
    return jsonify({
        'success': False,
        'error': 'Unauthorized',
        'message': str(error.description),
        'timestamp': datetime.now().isoformat()
    }), 401

@leetcode_bp.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'success': False,
        'error': 'Not Found',
        'message': str(error.description),
        'timestamp': datetime.now().isoformat()
    }), 404

@leetcode_bp.errorhandler(429)
def ratelimit_error(error):
    return jsonify({
        'success': False,
        'error': 'Rate Limit Exceeded',
        'message': str(error.description),
        'timestamp': datetime.now().isoformat()
>>>>>>> f41f548 (frontend)
    }), 429