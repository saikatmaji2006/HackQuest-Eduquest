<<<<<<< HEAD
import os
import time
import json
import logging
import requests
import random
import re
import threading
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import google.generativeai as genai
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle
import backoff
import tenacity
from dataclasses import dataclass, field
import numpy as np
from tqdm.auto import tqdm

def setup_logger(name=None, level=logging.INFO, log_file=None, log_format=None):
    logger = logging.getLogger(name or __name__)
    logger.setLevel(level)
    log_format = log_format or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

logger = setup_logger(level=logging.INFO)

@dataclass
class APIKeyMetadata:
    usage: int = 0
    success: int = 0
    error_count: int = 0
    last_used: Optional[datetime] = None
    last_error: Optional[str] = None
    total_tokens: int = 0
    success_rate: float = 1.0
    avg_latency: float = 0.0
    latency_samples: List[float] = field(default_factory=list)
    error_history: List[Tuple[datetime, str]] = field(default_factory=list)
    rate_limited: bool = False
    rate_limit_until: Optional[datetime] = None

    def add_latency(self, latency: float) -> None:
        self.latency_samples.append(latency)
        if len(self.latency_samples) > 100:
            self.latency_samples.pop(0)
        self.avg_latency = sum(self.latency_samples) / len(self.latency_samples)

    def record_error(self, error_message: str) -> None:
        self.error_count += 1
        self.last_error = error_message
        self.error_history.append((datetime.now(), error_message))
        if len(self.error_history) > 20:
            self.error_history.pop(0)
        total_attempts = self.success + self.error_count
        self.success_rate = self.success / total_attempts if total_attempts > 0 else 1.0

    def record_success(self, tokens_used: int = 0) -> None:
        self.success += 1
        self.total_tokens += tokens_used
        total_attempts = self.success + self.error_count
        self.success_rate = self.success / total_attempts if total_attempts > 0 else 1.0

    def set_rate_limited(self, duration_seconds: int = 60) -> None:
        self.rate_limited = True
        self.rate_limit_until = datetime.now() + timedelta(seconds=duration_seconds)

    def is_rate_limited(self) -> bool:
        if not self.rate_limited:
            return False
        if datetime.now() > self.rate_limit_until:
            self.rate_limited = False
            return False
        return True

class PersistentCache:
    def __init__(self, max_size=1000, ttl=3600, cache_dir=".gemini_cache", use_compression=True):
        self.memory_cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.disk_hits = 0
        self.cache_dir = cache_dir
        self.use_compression = use_compression
        os.makedirs(cache_dir, exist_ok=True)
        self.access_times_file = os.path.join(cache_dir, "_access_times.json")
        self.access_times = {}
        self._load_access_times()

    def _load_access_times(self):
        if os.path.exists(self.access_times_file):
            try:
                with open(self.access_times_file, 'r') as f:
                    self.access_times = json.load(f)
                    for key, timestamp in self.access_times.items():
                        self.access_times[key] = datetime.fromisoformat(timestamp)
            except (json.JSONDecodeError, IOError):
                self.access_times = {}

    def _save_access_times(self):
        try:
            serializable_times = {k: v.isoformat() for k, v in self.access_times.items()}
            with open(self.access_times_file, 'w') as f:
                json.dump(serializable_times, f)
        except IOError:
            pass

    def _get_cache_file_path(self, key):
        hashed_key = hashlib.md5(str(key).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed_key}.cache")

    def get(self, key):
        with self.lock:
            if key in self.memory_cache:
                item = self.memory_cache[key]
                if datetime.now() < item['expires_at']:
                    self.access_times[str(key)] = datetime.now()
                    self._save_access_times()
                    self.hits += 1
                    return item['value']
                else:
                    del self.memory_cache[key]
                    if str(key) in self.access_times:
                        del self.access_times[str(key)]
            cache_file = self._get_cache_file_path(key)
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        if self.use_compression:
                            import gzip
                            cached_data = pickle.loads(gzip.decompress(f.read()))
                        else:
                            cached_data = pickle.load(f)
                    if datetime.now() < cached_data['expires_at']:
                        self.memory_cache[key] = cached_data
                        self.access_times[str(key)] = datetime.now()
                        self._save_access_times()
                        self.disk_hits += 1
                        return cached_data['value']
                    else:
                        os.remove(cache_file)
                        if str(key) in self.access_times:
                            del self.access_times[str(key)]
                except (pickle.PickleError, IOError, gzip.error):
                    pass
            self.misses += 1
            return None

    def set(self, key, value, ttl=None):
        ttl = ttl or self.ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        with self.lock:
            cache_item = {
                'value': value,
                'expires_at': expires_at
            }
            if len(self.memory_cache) >= self.max_size:
                self._evict_lru_items()
            self.memory_cache[key] = cache_item
            self.access_times[str(key)] = datetime.now()
            self._save_access_times()
            cache_file = self._get_cache_file_path(key)
            try:
                with open(cache_file, 'wb') as f:
                    if self.use_compression:
                        import gzip
                        f.write(gzip.compress(pickle.dumps(cache_item)))
                    else:
                        pickle.dump(cache_item, f)
            except (pickle.PickleError, IOError):
                pass

    def _evict_lru_items(self, count=1):
        if not self.access_times:
            for _ in range(min(count, len(self.memory_cache))):
                if self.memory_cache:
                    key = next(iter(self.memory_cache))
                    del self.memory_cache[key]
            return
        items_with_times = [(k, self.access_times.get(str(k), datetime.min))
                            for k in self.memory_cache.keys()]
        items_with_times.sort(key=lambda x: x[1])
        for i in range(min(count, len(items_with_times))):
            key_to_remove = items_with_times[i][0]
            if key_to_remove in self.memory_cache:
                del self.memory_cache[key_to_remove]

    def clear(self, memory_only=False):
        with self.lock:
            self.memory_cache.clear()
            if not memory_only:
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.cache'):
                        try:
                            os.remove(os.path.join(self.cache_dir, filename))
                        except IOError:
                            pass
                self.access_times = {}
                self._save_access_times()

    def get_stats(self):
        with self.lock:
            total_requests = self.hits + self.disk_hits + self.misses
            hit_rate = (self.hits + self.disk_hits) / total_requests if total_requests > 0 else 0
            disk_cache_size = 0
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    try:
                        file_path = os.path.join(self.cache_dir, filename)
                        disk_cache_size += os.path.getsize(file_path)
                    except OSError:
                        pass
            return {
                'memory_cache_size': len(self.memory_cache),
                'memory_hits': self.hits,
                'disk_hits': self.disk_hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'disk_cache_size_bytes': disk_cache_size,
                'disk_cache_size_mb': disk_cache_size / (1024 * 1024)
            }

    def cleanup_expired(self):
        with self.lock:
            now = datetime.now()
            removed_count = 0
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    file_path = os.path.join(self.cache_dir, filename)
                    try:
                        with open(file_path, 'rb') as f:
                            if self.use_compression:
                                import gzip
                                cached_data = pickle.loads(gzip.decompress(f.read()))
                            else:
                                cached_data = pickle.load(f)
                        if now > cached_data['expires_at']:
                            os.remove(file_path)
                            key = filename[:-6]
                            if key in self.access_times:
                                del self.access_times[key]
                            removed_count += 1
                    except (pickle.PickleError, IOError, gzip.error):
                        try:
                            os.remove(file_path)
                            removed_count += 1
                        except IOError:
                            pass
            self._save_access_times()
            return removed_count

class SmartKeyManager:
    STRATEGIES = {
        'round_robin': 'Simple rotation through keys',
        'least_used': 'Select key with least usage',
        'best_performance': 'Select key with best performance (success rate and latency)',
        'adaptive': 'Adaptively select based on key performance metrics'
    }

    def __init__(self, api_keys=None, default_strategy='adaptive'):
        self.keys = []
        self.metadata = {}
        self.current_index = 0
        self.lock = threading.RLock()
        self.strategy = default_strategy
        self.cooldown_periods = {
            'short': 60,
            'medium': 300,
            'long': 1800
        }
        if api_keys:
            if isinstance(api_keys, list):
                for key in api_keys:
                    self.add_key(key)
            elif isinstance(api_keys, dict):
                for key, metadata in api_keys.items():
                    self.add_key(key, metadata)

    def add_key(self, key, metadata=None):
        with self.lock:
            if key not in self.keys:
                self.keys.append(key)
                self.metadata[key] = APIKeyMetadata(**(metadata or {}))

    def remove_key(self, key):
        with self.lock:
            if key in self.keys:
                self.keys.remove(key)
                if key in self.metadata:
                    del self.metadata[key]
                if self.current_index >= len(self.keys) and self.keys:
                    self.current_index = 0

    def get_next_key(self, purpose=None, strategy=None):
        with self.lock:
            if not self.keys:
                return None
            strategy = strategy or self.strategy
            available_keys = [k for k in self.keys if not self.metadata[k].is_rate_limited()]
            if not available_keys:
                available_keys = self.keys
            if strategy == 'round_robin':
                key = available_keys[self.current_index % len(available_keys)]
                self.current_index = (self.current_index + 1) % len(available_keys)
            elif strategy == 'least_used':
                key = min(available_keys, key=lambda k: self.metadata[k].usage)
            elif strategy == 'best_performance':
                def score_key(k):
                    meta = self.metadata[k]
                    success_factor = meta.success_rate
                    latency_factor = 1.0 - min(1.0, meta.avg_latency / 10.0) if meta.avg_latency > 0 else 0.5
                    return (success_factor * 0.7) + (latency_factor * 0.3)
                key = max(available_keys, key=score_key)
            elif strategy == 'adaptive':
                if random.random() < 0.1:
                    key = random.choice(available_keys)
                else:
                    def adaptive_score(k):
                        meta = self.metadata[k]
                        score = meta.success_rate
                        if meta.error_count > 5:
                            score *= 0.8
                        if meta.latency_samples:
                            recent_latency = sum(meta.latency_samples[-5:]) / min(5, len(meta.latency_samples))
                            latency_factor = 1.0 - min(1.0, recent_latency / 10.0)
                            score = (score * 0.7) + (latency_factor * 0.3)
                        return score
                    key = max(available_keys, key=adaptive_score)
            else:
                key = available_keys[self.current_index % len(available_keys)]
                self.current_index = (self.current_index + 1) % len(available_keys)
            self.metadata[key].usage += 1
            self.metadata[key].last_used = datetime.now()
            return key

    def handle_success(self, key, latency=None, tokens_used=0):
        with self.lock:
            if key in self.metadata:
                self.metadata[key].record_success(tokens_used)
                if latency is not None:
                    self.metadata[key].add_latency(latency)

    def handle_error(self, key, error):
        with self.lock:
            if key not in self.metadata:
                return False
            error_str = str(error)
            self.metadata[key].record_error(error_str)
            if "rate limit" in error_str.lower() or "quota" in error_str.lower() or "429" in error_str:
                self.metadata[key].set_rate_limited(self.cooldown_periods['medium'])
            elif "invalid" in error_str.lower() and "key" in error_str.lower() or "401" in error_str or "403" in error_str:
                self.remove_key(key)
                return True
            if self.metadata[key].error_count > 5 and self.metadata[key].success_rate < 0.5:
                self.metadata[key].set_rate_limited(self.cooldown_periods['long'])
            return False

    def get_key_status(self):
        with self.lock:
            return {
                'total_keys': len(self.keys),
                'available_keys': sum(1 for k in self.keys if not self.metadata[k].is_rate_limited()),
                'strategy': self.strategy,
                'key_status': {k: {
                    'usage': self.metadata[k].usage,
                    'success_rate': self.metadata[k].success_rate,
                    'avg_latency': self.metadata[k].avg_latency,
                    'rate_limited': self.metadata[k].is_rate_limited(),
                    'rate_limit_until': self.metadata[k].rate_limit_until.isoformat()
                        if self.metadata[k].rate_limit_until else None,
                    'total_tokens': self.metadata[k].total_tokens,
                    'last_error': self.metadata[k].last_error
                } for k in self.keys}
            }

    def set_strategy(self, strategy):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Invalid strategy: {strategy}. Valid options: {', '.join(self.STRATEGIES.keys())}")
        with self.lock:
            self.strategy = strategy

class TokenCounter:
    AVG_CHARS_PER_TOKEN = 4.0

    @staticmethod
    def estimate_tokens(text):
        if not text:
            return 0
        return int(len(text) / TokenCounter.AVG_CHARS_PER_TOKEN)

    @staticmethod
    def estimate_prompt_tokens(prompt):
        if isinstance(prompt, str):
            return TokenCounter.estimate_tokens(prompt)
        elif isinstance(prompt, dict):
            return sum(TokenCounter.estimate_prompt_tokens(v) for v in prompt.values())
        elif isinstance(prompt, list):
            return sum(TokenCounter.estimate_prompt_tokens(item) for item in prompt)
        else:
            return 0

class GeminiAPI:
    def __init__(self, api_keys=None, default_model="gemini-pro",
                 cache_enabled=True, cache_dir=".gemini_cache",
                 key_strategy="adaptive", request_timeout=30):
        self.key_manager = SmartKeyManager(api_keys, default_strategy=key_strategy)
        self.default_model = default_model
        self.base_url = "https://generativelanguage.googleapis.com/v1"
        self.request_timeout = request_timeout
        self.session = self._create_session()
        self.async_session = None
        self.cache_enabled = cache_enabled
        self.cache = PersistentCache(cache_dir=cache_dir) if cache_enabled else None
        self.genai_models = {}
        for key in self.key_manager.keys:
            self._init_genai_model(key)
        self.stats = {
            'requests': 0,
            'successes': 0,
            'errors': 0,
            'cache_hits': 0,
            'estimated_tokens': 0,
            'start_time': datetime.now()
        }
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.model_capabilities = {
            "gemini-pro": {
                "max_tokens": 30720,
                "supports_functions": True,
                "input_cost_per_1k": 0.00035,
                "output_cost_per_1k": 0.00120
            },
            "gemini-ultra": {
                "max_tokens": 32768,
                "supports_functions": True,
                "input_cost_per_1k": 0.00175,
                "output_cost_per_1k": 0.00580
            }
        }

    def _create_session(self):
        session = requests.Session()
        retry_strategy = requests.adapters.Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        return session

    async def _get_async_session(self):
        if self.async_session is None:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            self.async_session = aiohttp.ClientSession(timeout=timeout)
        return self.async_session

    def _init_genai_model(self, key):
        try:
            genai.configure(api_key=key)
            self.genai_models[key] = genai.GenerativeModel(self.default_model)
            return True
        except Exception:
            return False

    def _cache_key(self, model, prompt, temperature, max_tokens, **kwargs):
        key_parts = [
            model,
            str(prompt)[:200] if isinstance(prompt, str) else str(hash(str(prompt)))[:50],
            str(temperature),
            str(max_tokens)
        ]
        for param_name in sorted(kwargs.keys()):
            if param_name in ['top_p', 'top_k', 'safety_settings']:
                key_parts.append(f"{param_name}:{str(kwargs[param_name])}")
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _extract_text(self, response):
        if hasattr(response, 'text'):
            return response.text
        try:
            if isinstance(response, dict):
                if 'candidates' in response:
                    candidate = response['candidates'][0]
                    if 'content' in candidate:
                        parts = candidate['content'].get('parts', [])
                        for part in parts:
                            if 'text' in part:
                                return part['text']
        except Exception:
            pass
        return ""

    def _estimate_token_usage(self, prompt, response_text):
        prompt_tokens = TokenCounter.estimate_tokens(prompt if isinstance(prompt, str) else str(prompt))
        response_tokens = TokenCounter.estimate_tokens(response_text)
        return {
            'prompt_tokens': prompt_tokens,
            'response_tokens': response_tokens,
            'total_tokens': prompt_tokens + response_tokens
        }

    def _extract_error_details(self, error):
        error_info = {
            'type': type(error).__name__,
            'message': str(error)
        }
        if isinstance(error, requests.exceptions.RequestException) and hasattr(error, 'response'):
            resp = error.response
            if resp:
                error_info['status_code'] = resp.status_code
                try:
                    error_info['response_body'] = resp.json()
                except:
                    error_info['response_text'] = resp.text[:500]
        return error_info

    def _handle_request_error(self, api_key, error, retry_count=0, max_retries=3):
        error_details = self._extract_error_details(error)
        self.stats['errors'] += 1
        key_disabled = self.key_manager.handle_error(api_key, error)
        should_retry = False
        if retry_count < max_retries:
            if isinstance(error, (requests.exceptions.ConnectionError,
                                requests.exceptions.Timeout)):
                should_retry = True
            elif (isinstance(error, requests.exceptions.HTTPError) and
                  hasattr(error, 'response') and
                  error.response and error.response.status_code == 429):
                should_retry = True
            elif (isinstance(error, requests.exceptions.HTTPError) and
                  hasattr(error, 'response') and
                  error.response and 500 <= error.response.status_code < 600):
                should_retry = True
            elif key_disabled == False:
                should_retry = True
        if should_retry:
            backoff_time = min(2 ** retry_count, 32)
            time.sleep(backoff_time)
        return should_retry

    def generate_content(self, prompt, model=None, temperature=0.7, max_tokens=None,
                       use_cache=True, retry_count=0, **kwargs):
        model = model or self.default_model
        start_time = time.time()
        api_key = None
        cache_key = None
        if self.cache_enabled and use_cache:
            cache_key = self._cache_key(model, prompt, temperature, max_tokens, **kwargs)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.stats['cache_hits'] += 1
                return cached_result
        api_key = self.key_manager.get_next_key()
        if not api_key:
            raise ValueError("No API keys available")
        try:
            self.stats['requests'] += 1
            if api_key not in self.genai_models:
                self._init_genai_model(api_key)
            genai.configure(api_key=api_key)
            generation_config = {
                "temperature": temperature,
            }
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens
            for param_name in ['top_p', 'top_k']:
                if param_name in kwargs:
                    generation_config[param_name] = kwargs[param_name]
            safety_settings = kwargs.get('safety_settings', None)
            model_obj = genai.GenerativeModel(model)
            if safety_settings:
                model_obj = genai.GenerativeModel(model, safety_settings=safety_settings)
            if isinstance(prompt, str):
                response = model_obj.generate_content(prompt, generation_config=generation_config)
            else:
                response = model_obj.generate_content(prompt, generation_config=generation_config)
            response_text = self._extract_text(response)
            token_usage = self._estimate_token_usage(prompt, response_text)
            elapsed_time = time.time() - start_time
            self.stats['successes'] += 1
            self.stats['estimated_tokens'] += token_usage['total_tokens']
            self.key_manager.handle_success(
                api_key,
                latency=elapsed_time,
                tokens_used=token_usage['total_tokens']
            )
            if self.cache_enabled and use_cache and cache_key:
                self.cache.set(cache_key, response)
            return response
        except Exception as e:
            should_retry = self._handle_request_error(api_key, e, retry_count=retry_count)
            if should_retry and retry_count < 3:
                return self.generate_content(
                    prompt, model, temperature, max_tokens,
                    use_cache, retry_count + 1, **kwargs
                )
            raise

    async def async_generate_content(self, prompt, model=None, temperature=0.7, max_tokens=None,
                                  use_cache=True, retry_count=0, **kwargs):
        model = model or self.default_model
        start_time = time.time()
        api_key = None
        cache_key = None
        if self.cache_enabled and use_cache:
            cache_key = self._cache_key(model, prompt, temperature, max_tokens, **kwargs)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.stats['cache_hits'] += 1
                return cached_result
        api_key = self.key_manager.get_next_key()
        if not api_key:
            raise ValueError("No API keys available")
        try:
            self.stats['requests'] += 1
            session = await self._get_async_session()
            url = f"{self.base_url}/models/{model}:generateContent"
            request_data = {
                "contents": [{"parts": [{"text": prompt}] if isinstance(prompt, str) else prompt}],
                "generationConfig": {
                    "temperature": temperature
                }
            }
            if max_tokens:
                request_data["generationConfig"]["maxOutputTokens"] = max_tokens
            for param_name in ['top_p', 'top_k']:
                if param_name in kwargs:
                    request_data["generationConfig"][param_name] = kwargs[param_name]
            if 'safety_settings' in kwargs:
                request_data["safetySettings"] = kwargs['safety_settings']
            params = {"key": api_key}
            async with session.post(url, json=request_data, params=params, timeout=self.request_timeout) as response:
                response.raise_for_status()
                result = await response.json()
            response_text = self._extract_text(result)
            token_usage = self._estimate_token_usage(prompt, response_text)
            elapsed_time = time.time() - start_time
            self.stats['successes'] += 1
            self.stats['estimated_tokens'] += token_usage['total_tokens']
            self.key_manager.handle_success(
                api_key,
                latency=elapsed_time,
                tokens_used=token_usage['total_tokens']
            )
            if self.cache_enabled and use_cache and cache_key:
                self.cache.set(cache_key, result)
            return result
        except Exception as e:
            error_details = self._extract_error_details(e)
            self.stats['errors'] += 1
            key_disabled = self.key_manager.handle_error(api_key, e)
            if retry_count < 3:
                await asyncio.sleep(min(2 ** retry_count, 32))
                return await self.async_generate_content(
                    prompt, model, temperature, max_tokens,
                    use_cache, retry_count + 1, **kwargs
                )
            raise

    def batch_generate(self, prompts, model=None, temperature=0.7, max_tokens=None,
                      parallel=True, max_workers=None, use_cache=True, **kwargs):
        if not prompts:
            return []
        if not parallel:
            return [self.generate_content(
                p, model, temperature, max_tokens, use_cache, **kwargs
            ) for p in prompts]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.generate_content,
                    p, model, temperature, max_tokens, use_cache, **kwargs
                )
                for p in prompts
            ]
            results = []
            for future in tqdm(futures, total=len(futures), desc="Processing prompts"):
                try:
                    results.append(future.result())
                except Exception:
                    results.append(None)
            return results

    async def async_batch_generate(self, prompts, model=None, temperature=0.7, max_tokens=None,
                                use_cache=True, concurrency_limit=5, **kwargs):
        if not prompts:
            return []
        semaphore = asyncio.Semaphore(concurrency_limit)
        async def process_with_semaphore(prompt):
            async with semaphore:
                return await self.async_generate_content(
                    prompt, model, temperature, max_tokens, use_cache, **kwargs
                )
        tasks = [process_with_semaphore(p) for p in prompts]
        results = []
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
            except Exception:
                results.append(None)
        return results

    def estimate_cost(self, input_tokens, output_tokens, model=None):
        model = model or self.default_model
        model_info = self.model_capabilities.get(model, {
            "input_cost_per_1k": 0.00035,
            "output_cost_per_1k": 0.00120
        })
        input_cost = (input_tokens / 1000) * model_info["input_cost_per_1k"]
        output_cost = (output_tokens / 1000) * model_info["output_cost_per_1k"]
        return input_cost + output_cost

    def get_stats(self):
        stats = dict(self.stats)
        runtime = datetime.now() - stats.pop('start_time')
        stats['runtime_seconds'] = runtime.total_seconds()
        stats['runtime_formatted'] = str(runtime).split('.')[0]
        total_requests = stats['requests']
        stats['success_rate'] = (stats['successes'] / total_requests) if total_requests > 0 else 0
        if self.cache_enabled:
            stats['cache'] = self.cache.get_stats()
        stats['keys'] = self.key_manager.get_key_status()
        stats['estimated_cost'] = self.estimate_cost(
            stats['estimated_tokens'] // 2,
            stats['estimated_tokens'] // 2
        )
        return stats

    def limit_tokens(self, text, max_tokens):
        if not text:
            return text
        current_tokens = TokenCounter.estimate_tokens(text)
        if current_tokens <= max_tokens:
            return text
        approx_chars = int(max_tokens * TokenCounter.AVG_CHARS_PER_TOKEN)
        truncated = text[:approx_chars]
        last_period = truncated.rfind('.')
        if last_period > approx_chars * 0.8:
            truncated = truncated[:last_period+1]
        return truncated

    def close(self):
        self.session.close()
        if self.async_session:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.async_session.close())
            else:
                loop.run_until_complete(self.async_session.close())
        self.thread_pool.shutdown(wait=False)

    async def aclose(self):
        self.session.close()
        if self.async_session:
            await self.async_session.close()
        self.thread_pool.shutdown(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

if __name__ == "__main__":
    api_keys = [
        "AIzaSyAvdYKVRzTkbI-TxdmgcpQt3z4nHis1Ifg",
        "AIzaSyBxespyCjGQ1M73KQRDk0om_khk1M_z2pc",
        "AIzaSyCEP6y4h-kHFXRtsLHWcoLbYW7X1RZFe74",
        "AIzaSyBczbG7_7JJEEfrbh8QbGHTgGYpMKTzgA4"
    ]
    gemini = GeminiAPI(
        api_keys=api_keys,
        default_model="gemini-pro",
        cache_enabled=True,
        key_strategy="adaptive"
    )
    try:
        response = gemini.generate_content(
            "Write a short poem about artificial intelligence.",
            temperature=0.7
        )
        print("Generated text:")
        print(response.text)
        prompts = [
            "Summarize the benefits of renewable energy.",
            "Explain quantum computing in simple terms.",
            "What are the main challenges in modern healthcare?"
        ]
        batch_results = gemini.batch_generate(prompts)
        print("\nBatch results:")
        for i, result in enumerate(batch_results):
            if result:
                print(f"\nPrompt {i+1}:")
                print(result.text)
        async def async_example():
            async with GeminiAPI(api_keys=api_keys) as async_gemini:
                async_response = await async_gemini.async_generate_content(
                    "What are three trending technologies in 2024?"
                )
                print("\nAsync result:")
                print(async_gemini._extract_text(async_response))
                async_batch = await async_gemini.async_batch_generate(prompts[:2])
                print("\nAsync batch results:")
                for i, result in enumerate(async_batch):
                    if result:
                        print(f"\nAsync Prompt {i+1}:")
                        print(async_gemini._extract_text(result))
        asyncio.run(async_example())
        print("\nAPI client statistics:")
        stats = gemini.get_stats()
        print(json.dumps(stats, indent=2, default=str))
    finally:
        gemini.close()
=======
import os
import time
import json
import logging
import requests
import random
import re
import threading
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import google.generativeai as genai
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle
import backoff
import tenacity
from dataclasses import dataclass, field
import numpy as np
from tqdm.auto import tqdm

def setup_logger(name=None, level=logging.INFO, log_file=None, log_format=None):
    logger = logging.getLogger(name or __name__)
    logger.setLevel(level)
    log_format = log_format or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger

logger = setup_logger(level=logging.INFO)

@dataclass
class APIKeyMetadata:
    usage: int = 0
    success: int = 0
    error_count: int = 0
    last_used: Optional[datetime] = None
    last_error: Optional[str] = None
    total_tokens: int = 0
    success_rate: float = 1.0
    avg_latency: float = 0.0
    latency_samples: List[float] = field(default_factory=list)
    error_history: List[Tuple[datetime, str]] = field(default_factory=list)
    rate_limited: bool = False
    rate_limit_until: Optional[datetime] = None

    def add_latency(self, latency: float) -> None:
        self.latency_samples.append(latency)
        if len(self.latency_samples) > 100:
            self.latency_samples.pop(0)
        self.avg_latency = sum(self.latency_samples) / len(self.latency_samples)

    def record_error(self, error_message: str) -> None:
        self.error_count += 1
        self.last_error = error_message
        self.error_history.append((datetime.now(), error_message))
        if len(self.error_history) > 20:
            self.error_history.pop(0)
        total_attempts = self.success + self.error_count
        self.success_rate = self.success / total_attempts if total_attempts > 0 else 1.0

    def record_success(self, tokens_used: int = 0) -> None:
        self.success += 1
        self.total_tokens += tokens_used
        total_attempts = self.success + self.error_count
        self.success_rate = self.success / total_attempts if total_attempts > 0 else 1.0

    def set_rate_limited(self, duration_seconds: int = 60) -> None:
        self.rate_limited = True
        self.rate_limit_until = datetime.now() + timedelta(seconds=duration_seconds)

    def is_rate_limited(self) -> bool:
        if not self.rate_limited:
            return False
        if datetime.now() > self.rate_limit_until:
            self.rate_limited = False
            return False
        return True

class PersistentCache:
    def __init__(self, max_size=1000, ttl=3600, cache_dir=".gemini_cache", use_compression=True):
        self.memory_cache = {}
        self.max_size = max_size
        self.ttl = ttl
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.disk_hits = 0
        self.cache_dir = cache_dir
        self.use_compression = use_compression
        os.makedirs(cache_dir, exist_ok=True)
        self.access_times_file = os.path.join(cache_dir, "_access_times.json")
        self.access_times = {}
        self._load_access_times()

    def _load_access_times(self):
        if os.path.exists(self.access_times_file):
            try:
                with open(self.access_times_file, 'r') as f:
                    self.access_times = json.load(f)
                    for key, timestamp in self.access_times.items():
                        self.access_times[key] = datetime.fromisoformat(timestamp)
            except (json.JSONDecodeError, IOError):
                self.access_times = {}

    def _save_access_times(self):
        try:
            serializable_times = {k: v.isoformat() for k, v in self.access_times.items()}
            with open(self.access_times_file, 'w') as f:
                json.dump(serializable_times, f)
        except IOError:
            pass

    def _get_cache_file_path(self, key):
        hashed_key = hashlib.md5(str(key).encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed_key}.cache")

    def get(self, key):
        with self.lock:
            if key in self.memory_cache:
                item = self.memory_cache[key]
                if datetime.now() < item['expires_at']:
                    self.access_times[str(key)] = datetime.now()
                    self._save_access_times()
                    self.hits += 1
                    return item['value']
                else:
                    del self.memory_cache[key]
                    if str(key) in self.access_times:
                        del self.access_times[str(key)]
            cache_file = self._get_cache_file_path(key)
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        if self.use_compression:
                            import gzip
                            cached_data = pickle.loads(gzip.decompress(f.read()))
                        else:
                            cached_data = pickle.load(f)
                    if datetime.now() < cached_data['expires_at']:
                        self.memory_cache[key] = cached_data
                        self.access_times[str(key)] = datetime.now()
                        self._save_access_times()
                        self.disk_hits += 1
                        return cached_data['value']
                    else:
                        os.remove(cache_file)
                        if str(key) in self.access_times:
                            del self.access_times[str(key)]
                except (pickle.PickleError, IOError, gzip.error):
                    pass
            self.misses += 1
            return None

    def set(self, key, value, ttl=None):
        ttl = ttl or self.ttl
        expires_at = datetime.now() + timedelta(seconds=ttl)
        with self.lock:
            cache_item = {
                'value': value,
                'expires_at': expires_at
            }
            if len(self.memory_cache) >= self.max_size:
                self._evict_lru_items()
            self.memory_cache[key] = cache_item
            self.access_times[str(key)] = datetime.now()
            self._save_access_times()
            cache_file = self._get_cache_file_path(key)
            try:
                with open(cache_file, 'wb') as f:
                    if self.use_compression:
                        import gzip
                        f.write(gzip.compress(pickle.dumps(cache_item)))
                    else:
                        pickle.dump(cache_item, f)
            except (pickle.PickleError, IOError):
                pass

    def _evict_lru_items(self, count=1):
        if not self.access_times:
            for _ in range(min(count, len(self.memory_cache))):
                if self.memory_cache:
                    key = next(iter(self.memory_cache))
                    del self.memory_cache[key]
            return
        items_with_times = [(k, self.access_times.get(str(k), datetime.min))
                            for k in self.memory_cache.keys()]
        items_with_times.sort(key=lambda x: x[1])
        for i in range(min(count, len(items_with_times))):
            key_to_remove = items_with_times[i][0]
            if key_to_remove in self.memory_cache:
                del self.memory_cache[key_to_remove]

    def clear(self, memory_only=False):
        with self.lock:
            self.memory_cache.clear()
            if not memory_only:
                for filename in os.listdir(self.cache_dir):
                    if filename.endswith('.cache'):
                        try:
                            os.remove(os.path.join(self.cache_dir, filename))
                        except IOError:
                            pass
                self.access_times = {}
                self._save_access_times()

    def get_stats(self):
        with self.lock:
            total_requests = self.hits + self.disk_hits + self.misses
            hit_rate = (self.hits + self.disk_hits) / total_requests if total_requests > 0 else 0
            disk_cache_size = 0
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    try:
                        file_path = os.path.join(self.cache_dir, filename)
                        disk_cache_size += os.path.getsize(file_path)
                    except OSError:
                        pass
            return {
                'memory_cache_size': len(self.memory_cache),
                'memory_hits': self.hits,
                'disk_hits': self.disk_hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'disk_cache_size_bytes': disk_cache_size,
                'disk_cache_size_mb': disk_cache_size / (1024 * 1024)
            }

    def cleanup_expired(self):
        with self.lock:
            now = datetime.now()
            removed_count = 0
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.cache'):
                    file_path = os.path.join(self.cache_dir, filename)
                    try:
                        with open(file_path, 'rb') as f:
                            if self.use_compression:
                                import gzip
                                cached_data = pickle.loads(gzip.decompress(f.read()))
                            else:
                                cached_data = pickle.load(f)
                        if now > cached_data['expires_at']:
                            os.remove(file_path)
                            key = filename[:-6]
                            if key in self.access_times:
                                del self.access_times[key]
                            removed_count += 1
                    except (pickle.PickleError, IOError, gzip.error):
                        try:
                            os.remove(file_path)
                            removed_count += 1
                        except IOError:
                            pass
            self._save_access_times()
            return removed_count

class SmartKeyManager:
    STRATEGIES = {
        'round_robin': 'Simple rotation through keys',
        'least_used': 'Select key with least usage',
        'best_performance': 'Select key with best performance (success rate and latency)',
        'adaptive': 'Adaptively select based on key performance metrics'
    }

    def __init__(self, api_keys=None, default_strategy='adaptive'):
        self.keys = []
        self.metadata = {}
        self.current_index = 0
        self.lock = threading.RLock()
        self.strategy = default_strategy
        self.cooldown_periods = {
            'short': 60,
            'medium': 300,
            'long': 1800
        }
        if api_keys:
            if isinstance(api_keys, list):
                for key in api_keys:
                    self.add_key(key)
            elif isinstance(api_keys, dict):
                for key, metadata in api_keys.items():
                    self.add_key(key, metadata)

    def add_key(self, key, metadata=None):
        with self.lock:
            if key not in self.keys:
                self.keys.append(key)
                self.metadata[key] = APIKeyMetadata(**(metadata or {}))

    def remove_key(self, key):
        with self.lock:
            if key in self.keys:
                self.keys.remove(key)
                if key in self.metadata:
                    del self.metadata[key]
                if self.current_index >= len(self.keys) and self.keys:
                    self.current_index = 0

    def get_next_key(self, purpose=None, strategy=None):
        with self.lock:
            if not self.keys:
                return None
            strategy = strategy or self.strategy
            available_keys = [k for k in self.keys if not self.metadata[k].is_rate_limited()]
            if not available_keys:
                available_keys = self.keys
            if strategy == 'round_robin':
                key = available_keys[self.current_index % len(available_keys)]
                self.current_index = (self.current_index + 1) % len(available_keys)
            elif strategy == 'least_used':
                key = min(available_keys, key=lambda k: self.metadata[k].usage)
            elif strategy == 'best_performance':
                def score_key(k):
                    meta = self.metadata[k]
                    success_factor = meta.success_rate
                    latency_factor = 1.0 - min(1.0, meta.avg_latency / 10.0) if meta.avg_latency > 0 else 0.5
                    return (success_factor * 0.7) + (latency_factor * 0.3)
                key = max(available_keys, key=score_key)
            elif strategy == 'adaptive':
                if random.random() < 0.1:
                    key = random.choice(available_keys)
                else:
                    def adaptive_score(k):
                        meta = self.metadata[k]
                        score = meta.success_rate
                        if meta.error_count > 5:
                            score *= 0.8
                        if meta.latency_samples:
                            recent_latency = sum(meta.latency_samples[-5:]) / min(5, len(meta.latency_samples))
                            latency_factor = 1.0 - min(1.0, recent_latency / 10.0)
                            score = (score * 0.7) + (latency_factor * 0.3)
                        return score
                    key = max(available_keys, key=adaptive_score)
            else:
                key = available_keys[self.current_index % len(available_keys)]
                self.current_index = (self.current_index + 1) % len(available_keys)
            self.metadata[key].usage += 1
            self.metadata[key].last_used = datetime.now()
            return key

    def handle_success(self, key, latency=None, tokens_used=0):
        with self.lock:
            if key in self.metadata:
                self.metadata[key].record_success(tokens_used)
                if latency is not None:
                    self.metadata[key].add_latency(latency)

    def handle_error(self, key, error):
        with self.lock:
            if key not in self.metadata:
                return False
            error_str = str(error)
            self.metadata[key].record_error(error_str)
            if "rate limit" in error_str.lower() or "quota" in error_str.lower() or "429" in error_str:
                self.metadata[key].set_rate_limited(self.cooldown_periods['medium'])
            elif "invalid" in error_str.lower() and "key" in error_str.lower() or "401" in error_str or "403" in error_str:
                self.remove_key(key)
                return True
            if self.metadata[key].error_count > 5 and self.metadata[key].success_rate < 0.5:
                self.metadata[key].set_rate_limited(self.cooldown_periods['long'])
            return False

    def get_key_status(self):
        with self.lock:
            return {
                'total_keys': len(self.keys),
                'available_keys': sum(1 for k in self.keys if not self.metadata[k].is_rate_limited()),
                'strategy': self.strategy,
                'key_status': {k: {
                    'usage': self.metadata[k].usage,
                    'success_rate': self.metadata[k].success_rate,
                    'avg_latency': self.metadata[k].avg_latency,
                    'rate_limited': self.metadata[k].is_rate_limited(),
                    'rate_limit_until': self.metadata[k].rate_limit_until.isoformat()
                        if self.metadata[k].rate_limit_until else None,
                    'total_tokens': self.metadata[k].total_tokens,
                    'last_error': self.metadata[k].last_error
                } for k in self.keys}
            }

    def set_strategy(self, strategy):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Invalid strategy: {strategy}. Valid options: {', '.join(self.STRATEGIES.keys())}")
        with self.lock:
            self.strategy = strategy

class TokenCounter:
    AVG_CHARS_PER_TOKEN = 4.0

    @staticmethod
    def estimate_tokens(text):
        if not text:
            return 0
        return int(len(text) / TokenCounter.AVG_CHARS_PER_TOKEN)

    @staticmethod
    def estimate_prompt_tokens(prompt):
        if isinstance(prompt, str):
            return TokenCounter.estimate_tokens(prompt)
        elif isinstance(prompt, dict):
            return sum(TokenCounter.estimate_prompt_tokens(v) for v in prompt.values())
        elif isinstance(prompt, list):
            return sum(TokenCounter.estimate_prompt_tokens(item) for item in prompt)
        else:
            return 0

class GeminiAPI:
    def __init__(self, api_keys=None, default_model="gemini-pro",
                 cache_enabled=True, cache_dir=".gemini_cache",
                 key_strategy="adaptive", request_timeout=30):
        self.key_manager = SmartKeyManager(api_keys, default_strategy=key_strategy)
        self.default_model = default_model
        self.base_url = "https://generativelanguage.googleapis.com/v1"
        self.request_timeout = request_timeout
        self.session = self._create_session()
        self.async_session = None
        self.cache_enabled = cache_enabled
        self.cache = PersistentCache(cache_dir=cache_dir) if cache_enabled else None
        self.genai_models = {}
        for key in self.key_manager.keys:
            self._init_genai_model(key)
        self.stats = {
            'requests': 0,
            'successes': 0,
            'errors': 0,
            'cache_hits': 0,
            'estimated_tokens': 0,
            'start_time': datetime.now()
        }
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.model_capabilities = {
            "gemini-pro": {
                "max_tokens": 30720,
                "supports_functions": True,
                "input_cost_per_1k": 0.00035,
                "output_cost_per_1k": 0.00120
            },
            "gemini-ultra": {
                "max_tokens": 32768,
                "supports_functions": True,
                "input_cost_per_1k": 0.00175,
                "output_cost_per_1k": 0.00580
            }
        }

    def _create_session(self):
        session = requests.Session()
        retry_strategy = requests.adapters.Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        return session

    async def _get_async_session(self):
        if self.async_session is None:
            timeout = aiohttp.ClientTimeout(total=self.request_timeout)
            self.async_session = aiohttp.ClientSession(timeout=timeout)
        return self.async_session

    def _init_genai_model(self, key):
        try:
            genai.configure(api_key=key)
            self.genai_models[key] = genai.GenerativeModel(self.default_model)
            return True
        except Exception:
            return False

    def _cache_key(self, model, prompt, temperature, max_tokens, **kwargs):
        key_parts = [
            model,
            str(prompt)[:200] if isinstance(prompt, str) else str(hash(str(prompt)))[:50],
            str(temperature),
            str(max_tokens)
        ]
        for param_name in sorted(kwargs.keys()):
            if param_name in ['top_p', 'top_k', 'safety_settings']:
                key_parts.append(f"{param_name}:{str(kwargs[param_name])}")
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _extract_text(self, response):
        if hasattr(response, 'text'):
            return response.text
        try:
            if isinstance(response, dict):
                if 'candidates' in response:
                    candidate = response['candidates'][0]
                    if 'content' in candidate:
                        parts = candidate['content'].get('parts', [])
                        for part in parts:
                            if 'text' in part:
                                return part['text']
        except Exception:
            pass
        return ""

    def _estimate_token_usage(self, prompt, response_text):
        prompt_tokens = TokenCounter.estimate_tokens(prompt if isinstance(prompt, str) else str(prompt))
        response_tokens = TokenCounter.estimate_tokens(response_text)
        return {
            'prompt_tokens': prompt_tokens,
            'response_tokens': response_tokens,
            'total_tokens': prompt_tokens + response_tokens
        }

    def _extract_error_details(self, error):
        error_info = {
            'type': type(error).__name__,
            'message': str(error)
        }
        if isinstance(error, requests.exceptions.RequestException) and hasattr(error, 'response'):
            resp = error.response
            if resp:
                error_info['status_code'] = resp.status_code
                try:
                    error_info['response_body'] = resp.json()
                except:
                    error_info['response_text'] = resp.text[:500]
        return error_info

    def _handle_request_error(self, api_key, error, retry_count=0, max_retries=3):
        error_details = self._extract_error_details(error)
        self.stats['errors'] += 1
        key_disabled = self.key_manager.handle_error(api_key, error)
        should_retry = False
        if retry_count < max_retries:
            if isinstance(error, (requests.exceptions.ConnectionError,
                                requests.exceptions.Timeout)):
                should_retry = True
            elif (isinstance(error, requests.exceptions.HTTPError) and
                  hasattr(error, 'response') and
                  error.response and error.response.status_code == 429):
                should_retry = True
            elif (isinstance(error, requests.exceptions.HTTPError) and
                  hasattr(error, 'response') and
                  error.response and 500 <= error.response.status_code < 600):
                should_retry = True
            elif key_disabled == False:
                should_retry = True
        if should_retry:
            backoff_time = min(2 ** retry_count, 32)
            time.sleep(backoff_time)
        return should_retry

    def generate_content(self, prompt, model=None, temperature=0.7, max_tokens=None,
                       use_cache=True, retry_count=0, **kwargs):
        model = model or self.default_model
        start_time = time.time()
        api_key = None
        cache_key = None
        if self.cache_enabled and use_cache:
            cache_key = self._cache_key(model, prompt, temperature, max_tokens, **kwargs)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.stats['cache_hits'] += 1
                return cached_result
        api_key = self.key_manager.get_next_key()
        if not api_key:
            raise ValueError("No API keys available")
        try:
            self.stats['requests'] += 1
            if api_key not in self.genai_models:
                self._init_genai_model(api_key)
            genai.configure(api_key=api_key)
            generation_config = {
                "temperature": temperature,
            }
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens
            for param_name in ['top_p', 'top_k']:
                if param_name in kwargs:
                    generation_config[param_name] = kwargs[param_name]
            safety_settings = kwargs.get('safety_settings', None)
            model_obj = genai.GenerativeModel(model)
            if safety_settings:
                model_obj = genai.GenerativeModel(model, safety_settings=safety_settings)
            if isinstance(prompt, str):
                response = model_obj.generate_content(prompt, generation_config=generation_config)
            else:
                response = model_obj.generate_content(prompt, generation_config=generation_config)
            response_text = self._extract_text(response)
            token_usage = self._estimate_token_usage(prompt, response_text)
            elapsed_time = time.time() - start_time
            self.stats['successes'] += 1
            self.stats['estimated_tokens'] += token_usage['total_tokens']
            self.key_manager.handle_success(
                api_key,
                latency=elapsed_time,
                tokens_used=token_usage['total_tokens']
            )
            if self.cache_enabled and use_cache and cache_key:
                self.cache.set(cache_key, response)
            return response
        except Exception as e:
            should_retry = self._handle_request_error(api_key, e, retry_count=retry_count)
            if should_retry and retry_count < 3:
                return self.generate_content(
                    prompt, model, temperature, max_tokens,
                    use_cache, retry_count + 1, **kwargs
                )
            raise

    async def async_generate_content(self, prompt, model=None, temperature=0.7, max_tokens=None,
                                  use_cache=True, retry_count=0, **kwargs):
        model = model or self.default_model
        start_time = time.time()
        api_key = None
        cache_key = None
        if self.cache_enabled and use_cache:
            cache_key = self._cache_key(model, prompt, temperature, max_tokens, **kwargs)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.stats['cache_hits'] += 1
                return cached_result
        api_key = self.key_manager.get_next_key()
        if not api_key:
            raise ValueError("No API keys available")
        try:
            self.stats['requests'] += 1
            session = await self._get_async_session()
            url = f"{self.base_url}/models/{model}:generateContent"
            request_data = {
                "contents": [{"parts": [{"text": prompt}] if isinstance(prompt, str) else prompt}],
                "generationConfig": {
                    "temperature": temperature
                }
            }
            if max_tokens:
                request_data["generationConfig"]["maxOutputTokens"] = max_tokens
            for param_name in ['top_p', 'top_k']:
                if param_name in kwargs:
                    request_data["generationConfig"][param_name] = kwargs[param_name]
            if 'safety_settings' in kwargs:
                request_data["safetySettings"] = kwargs['safety_settings']
            params = {"key": api_key}
            async with session.post(url, json=request_data, params=params, timeout=self.request_timeout) as response:
                response.raise_for_status()
                result = await response.json()
            response_text = self._extract_text(result)
            token_usage = self._estimate_token_usage(prompt, response_text)
            elapsed_time = time.time() - start_time
            self.stats['successes'] += 1
            self.stats['estimated_tokens'] += token_usage['total_tokens']
            self.key_manager.handle_success(
                api_key,
                latency=elapsed_time,
                tokens_used=token_usage['total_tokens']
            )
            if self.cache_enabled and use_cache and cache_key:
                self.cache.set(cache_key, result)
            return result
        except Exception as e:
            error_details = self._extract_error_details(e)
            self.stats['errors'] += 1
            key_disabled = self.key_manager.handle_error(api_key, e)
            if retry_count < 3:
                await asyncio.sleep(min(2 ** retry_count, 32))
                return await self.async_generate_content(
                    prompt, model, temperature, max_tokens,
                    use_cache, retry_count + 1, **kwargs
                )
            raise

    def batch_generate(self, prompts, model=None, temperature=0.7, max_tokens=None,
                      parallel=True, max_workers=None, use_cache=True, **kwargs):
        if not prompts:
            return []
        if not parallel:
            return [self.generate_content(
                p, model, temperature, max_tokens, use_cache, **kwargs
            ) for p in prompts]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self.generate_content,
                    p, model, temperature, max_tokens, use_cache, **kwargs
                )
                for p in prompts
            ]
            results = []
            for future in tqdm(futures, total=len(futures), desc="Processing prompts"):
                try:
                    results.append(future.result())
                except Exception:
                    results.append(None)
            return results

    async def async_batch_generate(self, prompts, model=None, temperature=0.7, max_tokens=None,
                                use_cache=True, concurrency_limit=5, **kwargs):
        if not prompts:
            return []
        semaphore = asyncio.Semaphore(concurrency_limit)
        async def process_with_semaphore(prompt):
            async with semaphore:
                return await self.async_generate_content(
                    prompt, model, temperature, max_tokens, use_cache, **kwargs
                )
        tasks = [process_with_semaphore(p) for p in prompts]
        results = []
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
            except Exception:
                results.append(None)
        return results

    def estimate_cost(self, input_tokens, output_tokens, model=None):
        model = model or self.default_model
        model_info = self.model_capabilities.get(model, {
            "input_cost_per_1k": 0.00035,
            "output_cost_per_1k": 0.00120
        })
        input_cost = (input_tokens / 1000) * model_info["input_cost_per_1k"]
        output_cost = (output_tokens / 1000) * model_info["output_cost_per_1k"]
        return input_cost + output_cost

    def get_stats(self):
        stats = dict(self.stats)
        runtime = datetime.now() - stats.pop('start_time')
        stats['runtime_seconds'] = runtime.total_seconds()
        stats['runtime_formatted'] = str(runtime).split('.')[0]
        total_requests = stats['requests']
        stats['success_rate'] = (stats['successes'] / total_requests) if total_requests > 0 else 0
        if self.cache_enabled:
            stats['cache'] = self.cache.get_stats()
        stats['keys'] = self.key_manager.get_key_status()
        stats['estimated_cost'] = self.estimate_cost(
            stats['estimated_tokens'] // 2,
            stats['estimated_tokens'] // 2
        )
        return stats

    def limit_tokens(self, text, max_tokens):
        if not text:
            return text
        current_tokens = TokenCounter.estimate_tokens(text)
        if current_tokens <= max_tokens:
            return text
        approx_chars = int(max_tokens * TokenCounter.AVG_CHARS_PER_TOKEN)
        truncated = text[:approx_chars]
        last_period = truncated.rfind('.')
        if last_period > approx_chars * 0.8:
            truncated = truncated[:last_period+1]
        return truncated

    def close(self):
        self.session.close()
        if self.async_session:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.async_session.close())
            else:
                loop.run_until_complete(self.async_session.close())
        self.thread_pool.shutdown(wait=False)

    async def aclose(self):
        self.session.close()
        if self.async_session:
            await self.async_session.close()
        self.thread_pool.shutdown(wait=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

if __name__ == "__main__":
    api_keys = [
        "AIzaSyAvdYKVRzTkbI-TxdmgcpQt3z4nHis1Ifg",
        "AIzaSyBxespyCjGQ1M73KQRDk0om_khk1M_z2pc",
        "AIzaSyCEP6y4h-kHFXRtsLHWcoLbYW7X1RZFe74",
        "AIzaSyBczbG7_7JJEEfrbh8QbGHTgGYpMKTzgA4"
    ]
    gemini = GeminiAPI(
        api_keys=api_keys,
        default_model="gemini-pro",
        cache_enabled=True,
        key_strategy="adaptive"
    )
    try:
        response = gemini.generate_content(
            "Write a short poem about artificial intelligence.",
            temperature=0.7
        )
        print("Generated text:")
        print(response.text)
        prompts = [
            "Summarize the benefits of renewable energy.",
            "Explain quantum computing in simple terms.",
            "What are the main challenges in modern healthcare?"
        ]
        batch_results = gemini.batch_generate(prompts)
        print("\nBatch results:")
        for i, result in enumerate(batch_results):
            if result:
                print(f"\nPrompt {i+1}:")
                print(result.text)
        async def async_example():
            async with GeminiAPI(api_keys=api_keys) as async_gemini:
                async_response = await async_gemini.async_generate_content(
                    "What are three trending technologies in 2024?"
                )
                print("\nAsync result:")
                print(async_gemini._extract_text(async_response))
                async_batch = await async_gemini.async_batch_generate(prompts[:2])
                print("\nAsync batch results:")
                for i, result in enumerate(async_batch):
                    if result:
                        print(f"\nAsync Prompt {i+1}:")
                        print(async_gemini._extract_text(result))
        asyncio.run(async_example())
        print("\nAPI client statistics:")
        stats = gemini.get_stats()
        print(json.dumps(stats, indent=2, default=str))
    finally:
        gemini.close()
>>>>>>> f41f548 (frontend)
