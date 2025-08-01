# performance_optimizer.py
"""
Performance optimization system for TaxoConserv
Implements caching, lazy loading, and memory optimization strategies
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import functools
import hashlib
import pickle
import os
from typing import Any, Callable, Dict, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaxoConservCache:
    """
    Advanced caching system for TaxoConserv operations
    """
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        self.ensure_cache_dir()
        
        # Memory cache for frequently accessed data
        self.memory_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'disk_saves': 0,
            'disk_loads': 0
        }
    
    def ensure_cache_dir(self):
        """Ensure cache directory exists"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def get_cache_key(self, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, pd.DataFrame):
            # Use DataFrame shape and column hash for key
            cols_hash = hashlib.md5(str(sorted(data.columns)).encode()).hexdigest()[:8]
            return f"df_{data.shape[0]}x{data.shape[1]}_{cols_hash}"
        elif isinstance(data, (str, int, float, bool)):
            return hashlib.md5(str(data).encode()).hexdigest()[:12]
        else:
            return hashlib.md5(str(data).encode()).hexdigest()[:12]
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        # Check memory cache first
        if key in self.memory_cache:
            self.cache_stats['hits'] += 1
            logger.debug(f"Cache HIT (memory): {key}")
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    value = pickle.load(f)
                    # Store in memory for faster future access
                    self.memory_cache[key] = value
                    self.cache_stats['hits'] += 1
                    self.cache_stats['disk_loads'] += 1
                    logger.debug(f"Cache HIT (disk): {key}")
                    return value
            except Exception as e:
                logger.warning(f"Failed to load cache {key}: {e}")
        
        self.cache_stats['misses'] += 1
        logger.debug(f"Cache MISS: {key}")
        return None
    
    def set(self, key: str, value: Any, persist_to_disk: bool = True):
        """Set cached value"""
        # Store in memory
        self.memory_cache[key] = value
        
        # Optionally persist to disk
        if persist_to_disk:
            try:
                cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
                with open(cache_file, 'wb') as f:
                    pickle.dump(value, f)
                self.cache_stats['disk_saves'] += 1
                logger.debug(f"Cache SAVE: {key}")
            except Exception as e:
                logger.warning(f"Failed to save cache {key}: {e}")
    
    def clear(self):
        """Clear all caches"""
        self.memory_cache.clear()
        # Clear disk cache
        for file in os.listdir(self.cache_dir):
            if file.endswith('.pkl'):
                os.remove(os.path.join(self.cache_dir, file))
        logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.cache_stats,
            'total_requests': total_requests,
            'hit_rate': round(hit_rate, 2)
        }


# Global cache instance
cache = TaxoConservCache()


def performance_timer(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        func_name = func.__name__
        if execution_time > 0.1:  # Only log slow operations
            logger.info(f"⏱️ {func_name}: {execution_time:.3f}s")
        
        return result
    return wrapper


def streamlit_cache_data(ttl: Optional[int] = None):
    """Enhanced Streamlit cache decorator with custom TTL"""
    def decorator(func: Callable) -> Callable:
        if ttl:
            return st.cache_data(ttl=ttl)(func)
        else:
            return st.cache_data(func)
    return decorator


@streamlit_cache_data(ttl=300)  # Cache for 5 minutes
@performance_timer
def cached_statistical_analysis(data_hash: str, score_column: str, group_column: str) -> Dict:
    """Cached statistical analysis"""
    # This will be called by the main analysis function
    from src.analysis import perform_statistical_analysis
    
    # Get original data from session state (temporary solution)
    data = st.session_state.get('current_data')
    if data is None:
        raise ValueError("Data not available for analysis")
    
    return perform_statistical_analysis(data, score_column, group_column)


@streamlit_cache_data(ttl=600)  # Cache for 10 minutes
@performance_timer
def cached_data_processing(file_content: bytes, file_type: str) -> pd.DataFrame:
    """Cached data loading and processing"""
    import io
    
    try:
        if file_type == 'csv':
            data = pd.read_csv(io.BytesIO(file_content))
        elif file_type == 'tsv':
            data = pd.read_csv(io.BytesIO(file_content), sep='\t')
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Basic data cleaning
        data = data.dropna(axis=1, how='all')  # Remove empty columns
        
        return data
        
    except Exception as e:
        logger.error(f"Error processing file: {e}")
        raise


@streamlit_cache_data(ttl=1800)  # Cache for 30 minutes
@performance_timer
def cached_conservation_score_detection(data_hash: str, columns: list) -> Dict:
    """Cached conservation score detection"""
    from src.score_utils import detect_conservation_scores
    
    # Create a dummy DataFrame with the columns for detection
    dummy_df = pd.DataFrame(columns=columns)
    return detect_conservation_scores(dummy_df)


class DataOptimizer:
    """
    Data optimization utilities for better performance
    """
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        optimized_df = df.copy()
        
        # Optimize numeric columns
        for col in optimized_df.select_dtypes(include=[np.number]).columns:
            col_min = optimized_df[col].min()
            col_max = optimized_df[col].max()
            
            # Optimize integers
            if optimized_df[col].dtype in ['int64', 'int32']:
                if col_min >= -128 and col_max <= 127:
                    optimized_df[col] = optimized_df[col].astype(np.int8)
                elif col_min >= -32768 and col_max <= 32767:
                    optimized_df[col] = optimized_df[col].astype(np.int16)
                elif col_min >= -2147483648 and col_max <= 2147483647:
                    optimized_df[col] = optimized_df[col].astype(np.int32)
            
            # Optimize floats
            elif optimized_df[col].dtype in ['float64']:
                optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        # Optimize string columns
        for col in optimized_df.select_dtypes(include=['object']).columns:
            if optimized_df[col].nunique() / len(optimized_df) < 0.5:  # If less than 50% unique
                optimized_df[col] = optimized_df[col].astype('category')
        
        return optimized_df
    
    @staticmethod
    def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
        """Get memory usage statistics"""
        memory_usage = df.memory_usage(deep=True)
        
        return {
            'total_mb': memory_usage.sum() / 1024 / 1024,
            'index_mb': memory_usage.iloc[0] / 1024 / 1024,
            'data_mb': memory_usage.iloc[1:].sum() / 1024 / 1024,
            'columns': {col: usage / 1024 / 1024 for col, usage in memory_usage.items()}
        }


class PerformanceMonitor:
    """
    Performance monitoring and reporting
    """
    
    def __init__(self):
        self.metrics = {
            'data_load_time': [],
            'analysis_time': [],
            'plot_generation_time': [],
            'memory_usage': []
        }
    
    def record_metric(self, metric_name: str, value: float):
        """Record a performance metric"""
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'avg': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'last': values[-1] if values else 0,
                    'count': len(values)
                }
            else:
                summary[metric_name] = {
                    'avg': 0, 'min': 0, 'max': 0, 'last': 0, 'count': 0
                }
        
        return summary
    
    def clear_metrics(self):
        """Clear all recorded metrics"""
        for key in self.metrics:
            self.metrics[key].clear()


# Global performance monitor
performance_monitor = PerformanceMonitor()


def optimize_streamlit_config():
    """Optimize Streamlit configuration for better performance"""
    
    # Add custom CSS for performance
    st.markdown("""
        <style>
        /* Optimize rendering performance */
        .stDataFrame {
            max-height: 400px;
            overflow-y: auto;
        }
        
        /* Reduce animation overhead */
        .stProgress > div > div > div > div {
            transition: none !important;
        }
        
        /* Optimize sidebar */
        .css-1d391kg {
            padding-top: 1rem;
        }
        
        /* Faster scrolling */
        .main .block-container {
            padding-top: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)


class LazyDataLoader:
    """
    Lazy loading system for large datasets
    """
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        self.loaded_chunks = {}
    
    def load_chunk(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        """Load a specific chunk of data"""
        chunk_key = f"{start_idx}_{end_idx}"
        
        if chunk_key not in self.loaded_chunks:
            self.loaded_chunks[chunk_key] = data.iloc[start_idx:end_idx].copy()
        
        return self.loaded_chunks[chunk_key]
    
    def get_paginated_data(self, data: pd.DataFrame, page: int, page_size: int = 100) -> Tuple[pd.DataFrame, Dict]:
        """Get paginated data for display"""
        total_rows = len(data)
        total_pages = (total_rows + page_size - 1) // page_size
        
        start_idx = page * page_size
        end_idx = min(start_idx + page_size, total_rows)
        
        chunk_data = self.load_chunk(data, start_idx, end_idx)
        
        pagination_info = {
            'current_page': page,
            'total_pages': total_pages,
            'total_rows': total_rows,
            'page_size': page_size,
            'start_idx': start_idx,
            'end_idx': end_idx
        }
        
        return chunk_data, pagination_info


# Global lazy loader
lazy_loader = LazyDataLoader()


# Export key functions
__all__ = [
    'cache',
    'performance_timer',
    'streamlit_cache_data',
    'cached_statistical_analysis',
    'cached_data_processing',
    'cached_conservation_score_detection',
    'DataOptimizer',
    'PerformanceMonitor',
    'performance_monitor',
    'optimize_streamlit_config',
    'LazyDataLoader',
    'lazy_loader'
]
