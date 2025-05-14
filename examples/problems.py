"""
Collection of example problems to test the system prompt learning framework.
Each problem includes a description and a solution to test the feedback loop.
"""

PROBLEMS = [
    {
        "name": "List Filtering",
        "description": """
        Write a function that takes a list of numbers and returns a new list
        containing only the even numbers, maintaining their original order.
        """,
        "solution": """
        def get_even_numbers(numbers):
            return [num for num in numbers if num % 2 == 0]
        """
    },
    {
        "name": "String Palindrome",
        "description": """
        Write a function that checks if a given string is a palindrome,
        ignoring case and non-alphanumeric characters.
        """,
        "solution": """
        def is_palindrome(text):
            # Clean the string: remove non-alphanumeric and convert to lowercase
            cleaned = ''.join(c.lower() for c in text if c.isalnum())
            return cleaned == cleaned[::-1]
        """
    },
    {
        "name": "Tree Traversal",
        "description": """
        Implement a function that performs an in-order traversal of a binary tree
        and returns the values in a list.
        """,
        "solution": """
        class TreeNode:
            def __init__(self, val=0, left=None, right=None):
                self.val = val
                self.left = left
                self.right = right

        def inorder_traversal(root):
            result = []
            def traverse(node):
                if not node:
                    return
                traverse(node.left)
                result.append(node.val)
                traverse(node.right)
            traverse(root)
            return result
        """
    },
    {
        "name": "Concurrent Task Processing",
        "description": """
        Write a function that processes a list of tasks concurrently using
        asyncio, with a maximum number of concurrent tasks.
        """,
        "solution": """
        import asyncio
        from typing import List, Callable, Any

        async def process_tasks(
            tasks: List[Callable],
            max_concurrent: int
        ) -> List[Any]:
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def process_with_semaphore(task):
                async with semaphore:
                    return await task()
            
            return await asyncio.gather(
                *[process_with_semaphore(task) for task in tasks]
            )
        """
    },
    {
        "name": "Database Connection Pool",
        "description": """
        Implement a connection pool for database connections that manages
        a fixed number of connections and handles connection reuse.
        """,
        "solution": """
        from queue import Queue
        from typing import Optional
        import threading

        class ConnectionPool:
            def __init__(self, max_connections: int):
                self.max_connections = max_connections
                self.connections = Queue(maxsize=max_connections)
                self.lock = threading.Lock()
                self.active_connections = 0
            
            def get_connection(self) -> Optional[object]:
                with self.lock:
                    if self.active_connections < self.max_connections:
                        # Create new connection
                        conn = self._create_connection()
                        self.active_connections += 1
                        return conn
                    return self.connections.get()
            
            def release_connection(self, connection: object) -> None:
                self.connections.put(connection)
            
            def _create_connection(self) -> object:
                # Simulate connection creation
                return object()
        """
    },
    {
        "name": "Caching Decorator",
        "description": """
        Create a decorator that caches function results based on input
        parameters, with a maximum cache size and TTL (time-to-live).
        """,
        "solution": """
        from functools import wraps
        from datetime import datetime, timedelta
        from typing import Any, Callable, Dict, Tuple

        def cache(ttl_seconds: int = 300, max_size: int = 100):
            cache_data: Dict[Tuple, Tuple[Any, datetime]] = {}
            
            def decorator(func: Callable):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    key = (*args, *sorted(kwargs.items()))
                    now = datetime.now()
                    
                    # Check cache
                    if key in cache_data:
                        result, timestamp = cache_data[key]
                        if now - timestamp < timedelta(seconds=ttl_seconds):
                            return result
                    
                    # Calculate result
                    result = func(*args, **kwargs)
                    
                    # Update cache
                    if len(cache_data) >= max_size:
                        # Remove oldest entry
                        oldest_key = min(
                            cache_data.keys(),
                            key=lambda k: cache_data[k][1]
                        )
                        del cache_data[oldest_key]
                    
                    cache_data[key] = (result, now)
                    return result
                return wrapper
            return decorator
        """
    },
    {
        "name": "Error Handling Middleware",
        "description": """
        Create a middleware function that handles exceptions in a web
        application and returns appropriate error responses.
        """,
        "solution": """
        from typing import Callable, Dict, Any
        from functools import wraps

        def error_handler(
            error_mapping: Dict[type, Dict[str, Any]] = None
        ) -> Callable:
            if error_mapping is None:
                error_mapping = {
                    ValueError: {"status": 400, "message": "Bad Request"},
                    KeyError: {"status": 404, "message": "Not Found"},
                    Exception: {"status": 500, "message": "Internal Server Error"}
                }
            
            def decorator(func: Callable) -> Callable:
                @wraps(func)
                def wrapper(*args, **kwargs) -> Dict[str, Any]:
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        error_type = type(e)
                        error_info = error_mapping.get(
                            error_type,
                            error_mapping[Exception]
                        )
                        return {
                            "error": error_info["message"],
                            "status": error_info["status"],
                            "details": str(e)
                        }
                return wrapper
            return decorator
        """
    }
]

def get_problem(index: int = 0) -> dict:
    """Get a problem by index."""
    return PROBLEMS[index % len(PROBLEMS)]

def get_all_problems() -> list:
    """Get all problems."""
    return PROBLEMS 