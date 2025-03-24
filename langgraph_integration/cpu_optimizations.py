import time
from typing import Dict, Any, Optional, List, Tuple, Callable, TypeVar
import threading
from functools import lru_cache, wraps
import os

# Type variable for generic function types
T = TypeVar('T')
R = TypeVar('R')

class CPUOptimizer:
    """Optimize performance for CPU-only environments"""

    def __init__(self, max_cache_size: int = 1000, cache_ttl: int = 3600):
        self.max_cache_size = max_cache_size
        self.cache_ttl = cache_ttl  # Time-to-live in seconds
        self.cache = {}
        self.cache_lock = threading.Lock()
        self._initialize_environment()

    def _initialize_environment(self) -> None:
        """Set optimal environment variables for CPU performance"""
        # Set thread count for pytorch/tensorflow if installed
        # This prevents overloading the CPU with too many threads
        cpu_count = os.cpu_count() or 4
        optimal_threads = max(1, min(cpu_count - 1, 4))  # Leave at least 1 CPU for OS
        
        # Set environment variables that control threading in common libraries
        os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
        os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)
        os.environ['NUMEXPR_MAX_THREADS'] = str(optimal_threads)
        
        # PyTorch specific (if installed)
        try:
            import torch
            torch.set_num_threads(optimal_threads)
        except ImportError:
            pass
        
        # Set environment variables for CPU performance over GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA

    @lru_cache(maxsize=1024)
    def cached_text_analysis(self, text: str) -> Dict[str, Any]:
        """
        Cached text analysis for frequently repeated queries
        Using lru_cache for simplicity - adequate for CPU usage
        """
        # This would be replaced with actual analysis logic
        time.sleep(0.01)  # Simulate processing time
        return {"result": "cached_analysis", "text_length": len(text)}

    def batch_process(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple texts in a single batch
        More efficient than individual processing
        """
        results = []
        for text in texts:
            # Use cache when possible
            result = self.cached_text_analysis(text)
            results.append(result)
        return results

    def timed_with_timeout(self, func: Callable[..., T], *args, timeout: float = 1.0, **kwargs) -> Tuple[T, float]:
        """
        Run a function with a timeout
        Returns result and execution time
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time

        # For CPU optimization, we don't actually stop execution
        # but we can mark it as taking too long
        if execution_time > timeout:
            print(f"Warning: Function took {execution_time:.4f}s, exceeding timeout of {timeout}s")

        return result, execution_time

    def clean_cache(self) -> None:
        """Clean expired cache entries"""
        with self.cache_lock:
            current_time = time.time()
            expired_keys = [
                key for key, (value, timestamp) in self.cache.items()
                if current_time - timestamp > self.cache_ttl
            ]

            for key in expired_keys:
                del self.cache[key]
    
    def memoize(self, func: Callable[..., R]) -> Callable[..., R]:
        """
        Custom memoization decorator with TTL
        More flexible than lru_cache for complex objects
        """
        cache = {}
        lock = threading.Lock()
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from the arguments
            key_parts = [repr(arg) for arg in args]
            key_parts.extend(f"{k}:{repr(v)}" for k, v in sorted(kwargs.items()))
            key = "::".join(key_parts)
            
            with lock:
                # Check if result exists in cache and is not expired
                if key in cache:
                    result, timestamp = cache[key]
                    if time.time() - timestamp <= self.cache_ttl:
                        return result
            
            # Compute result if not in cache or expired
            result = func(*args, **kwargs)
            
            with lock:
                # Store result in cache with timestamp
                cache[key] = (result, time.time())
                
                # Clean cache if it gets too large
                if len(cache) > self.max_cache_size:
                    # Simple approach: remove the oldest entries
                    sorted_cache = sorted(cache.items(), key=lambda x: x[1][1])
                    # Keep only the newest entries
                    cache.clear()
                    for k, v in sorted_cache[-self.max_cache_size:]:
                        cache[k] = v
            
            return result
        
        return wrapper

    def parallelize(self, func: Callable[[Any], R], items: List[Any], max_workers: Optional[int] = None) -> List[R]:
        """
        Process items in parallel using threads
        Optimized for I/O bound tasks
        """
        if not items:
            return []
        
        cpu_count = os.cpu_count() or 4
        # Default to number of CPUs (or 4) for max workers
        if max_workers is None:
            max_workers = cpu_count
        
        # Adjust to reasonable limits
        max_workers = min(max_workers, len(items), cpu_count * 2)
        
        # For very small lists, don't bother with parallelization
        if len(items) <= 2 or max_workers <= 1:
            return [func(item) for item in items]
            
        results = [None] * len(items)
        threads = []
        
        def process_item(index, item):
            try:
                results[index] = func(item)
            except Exception as e:
                print(f"Error processing item {index}: {str(e)}")
                results[index] = None
                
        # Create and start threads
        for i, item in enumerate(items):
            thread = threading.Thread(target=process_item, args=(i, item))
            threads.append(thread)
            thread.start()
            
            # Limit concurrency to max_workers
            if len(threads) >= max_workers:
                for t in threads:
                    t.join()
                threads = []
                
        # Wait for any remaining threads
        for thread in threads:
            thread.join()
            
        return results 