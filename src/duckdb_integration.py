# duckdb_integration.py
"""
DuckDB integration for high-performance data processing in TaxoConserv
Optimizes large dataset operations with in-memory columnar database
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Try to import DuckDB (optional dependency)
try:
    import duckdb  # type: ignore
    DUCKDB_AVAILABLE = True
    logger.info("DuckDB available for high-performance data processing")
except ImportError:
    duckdb = None  # type: ignore
    DUCKDB_AVAILABLE = False
    logger.warning("DuckDB not available - falling back to pandas operations")


class DuckDBProcessor:
    """
    High-performance data processor using DuckDB
    """
    
    def __init__(self):
        self.connection = None
        self.tables = {}
        
        if DUCKDB_AVAILABLE and duckdb is not None:
            try:
                self.connection = duckdb.connect(':memory:')  # type: ignore
                logger.info("DuckDB in-memory database initialized")
            except Exception as e:
                logger.error(f"Failed to initialize DuckDB: {e}")
                self.connection = None
    
    def is_available(self) -> bool:
        """Check if DuckDB is available and initialized"""
        return DUCKDB_AVAILABLE and self.connection is not None
    
    def register_dataframe(self, df: pd.DataFrame, table_name: str = 'main_data') -> bool:
        """Register pandas DataFrame as DuckDB table"""
        if not self.is_available() or self.connection is None:
            return False
        
        try:
            self.connection.register(table_name, df)  # type: ignore
            self.tables[table_name] = {
                'rows': len(df),
                'columns': list(df.columns),
                'registered_at': pd.Timestamp.now()
            }
            logger.info(f"Registered DataFrame as table '{table_name}' ({len(df):,} rows)")
            return True
        except Exception as e:
            logger.error(f"Failed to register DataFrame: {e}")
            return False
    
    def execute_query(self, query: str) -> Optional[pd.DataFrame]:
        """Execute SQL query and return results as DataFrame"""
        if not self.is_available() or self.connection is None:
            return None
        
        try:
            result = self.connection.execute(query).df()  # type: ignore
            logger.debug(f"Query executed successfully: {len(result):,} rows returned")
            return result
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return None
    
    def get_group_statistics(self, table_name: str, score_column: str, group_column: str) -> Optional[pd.DataFrame]:
        """Get group statistics using SQL aggregation"""
        query = f"""
        SELECT 
            {group_column} as group_name,
            COUNT(*) as count,
            AVG({score_column}) as mean_score,
            MEDIAN({score_column}) as median_score,
            STDDEV({score_column}) as std_score,
            MIN({score_column}) as min_score,
            MAX({score_column}) as max_score,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {score_column}) as q25,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {score_column}) as q75
        FROM {table_name}
        WHERE {score_column} IS NOT NULL
          AND {group_column} IS NOT NULL
        GROUP BY {group_column}
        ORDER BY mean_score DESC
        """
        
        return self.execute_query(query)
    
    def get_filtered_data(self, table_name: str, filters: Dict[str, Any], 
                         limit: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Get filtered data with optional limit"""
        where_clauses = []
        
        for column, value in filters.items():
            if isinstance(value, str):
                where_clauses.append(f"{column} = '{value}'")
            elif isinstance(value, (list, tuple)):
                if len(value) == 2:  # Range filter
                    where_clauses.append(f"{column} BETWEEN {value[0]} AND {value[1]}")
                else:  # IN filter
                    value_str = "', '".join(map(str, value))
                    where_clauses.append(f"{column} IN ('{value_str}')")
            else:
                where_clauses.append(f"{column} = {value}")
        
        where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
        limit_clause = f" LIMIT {limit}" if limit else ""
        
        query = f"""
        SELECT * FROM {table_name}
        WHERE {where_clause}
        {limit_clause}
        """
        
        return self.execute_query(query)
    
    def get_correlation_matrix(self, table_name: str, numeric_columns: List[str]) -> Optional[pd.DataFrame]:
        """Calculate correlation matrix for numeric columns"""
        if len(numeric_columns) < 2:
            return None
        
        # Build correlation query
        correlations = []
        for i, col1 in enumerate(numeric_columns):
            for j, col2 in enumerate(numeric_columns):
                if i <= j:  # Only calculate upper triangle + diagonal
                    correlations.append(f"CORR({col1}, {col2}) as {col1}_{col2}")
        
        query = f"""
        SELECT {', '.join(correlations)}
        FROM {table_name}
        WHERE {' AND '.join([f'{col} IS NOT NULL' for col in numeric_columns])}
        """
        
        result = self.execute_query(query)
        if result is None or len(result) == 0:
            return None
        
        # Convert to proper correlation matrix format
        corr_matrix = pd.DataFrame(index=numeric_columns, columns=numeric_columns)
        
        for col in result.columns:
            if '_' in col:
                col1, col2 = col.split('_', 1)
                value = result.iloc[0][col]
                corr_matrix.loc[col1, col2] = value
                corr_matrix.loc[col2, col1] = value  # Symmetric
        
        return corr_matrix.astype(float)
    
    def detect_outliers_iqr(self, table_name: str, score_column: str, 
                           group_column: str) -> Optional[pd.DataFrame]:
        """Detect outliers using IQR method"""
        query = f"""
        WITH group_stats AS (
            SELECT 
                {group_column},
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {score_column}) as q1,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {score_column}) as q3,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {score_column}) - 
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {score_column}) as iqr
            FROM {table_name}
            WHERE {score_column} IS NOT NULL AND {group_column} IS NOT NULL
            GROUP BY {group_column}
        ),
        outliers AS (
            SELECT 
                t.*,
                s.q1,
                s.q3,
                s.iqr,
                s.q1 - 1.5 * s.iqr as lower_bound,
                s.q3 + 1.5 * s.iqr as upper_bound
            FROM {table_name} t
            JOIN group_stats s ON t.{group_column} = s.{group_column}
            WHERE t.{score_column} < s.q1 - 1.5 * s.iqr 
               OR t.{score_column} > s.q3 + 1.5 * s.iqr
        )
        SELECT 
            {group_column},
            COUNT(*) as outlier_count,
            ARRAY_AGG({score_column}) as outlier_values
        FROM outliers
        GROUP BY {group_column}
        ORDER BY outlier_count DESC
        """
        
        return self.execute_query(query)
    
    def sample_data(self, table_name: str, sample_size: int = 1000, 
                   stratified_by: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get stratified or random sample of data"""
        if stratified_by:
            # Stratified sampling
            query = f"""
            WITH group_counts AS (
                SELECT {stratified_by}, COUNT(*) as group_size
                FROM {table_name}
                GROUP BY {stratified_by}
            ),
            sample_sizes AS (
                SELECT 
                    {stratified_by},
                    GREATEST(1, CAST({sample_size} * group_size / SUM(group_size) OVER () AS INTEGER)) as sample_count
                FROM group_counts
            )
            SELECT t.*
            FROM {table_name} t
            JOIN sample_sizes s ON t.{stratified_by} = s.{stratified_by}
            QUALIFY ROW_NUMBER() OVER (PARTITION BY t.{stratified_by} ORDER BY RANDOM()) <= s.sample_count
            """
        else:
            # Simple random sampling
            query = f"""
            SELECT *
            FROM {table_name}
            ORDER BY RANDOM()
            LIMIT {sample_size}
            """
        
        return self.execute_query(query)
    
    def get_table_info(self, table_name: str) -> Optional[Dict]:
        """Get information about registered table"""
        if table_name not in self.tables:
            return None
        
        # Get additional stats from DuckDB
        stats_query = f"""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT *) as unique_rows
        FROM {table_name}
        """
        
        stats_result = self.execute_query(stats_query)
        info = self.tables[table_name].copy()
        
        if stats_result is not None and len(stats_result) > 0:
            info.update({
                'total_rows': stats_result.iloc[0]['total_rows'],
                'unique_rows': stats_result.iloc[0]['unique_rows']
            })
        
        return info
    
    def close(self):
        """Close DuckDB connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.tables.clear()
            logger.info("DuckDB connection closed")


# Global DuckDB processor instance
duckdb_processor = DuckDBProcessor()


@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_fast_group_statistics(data: pd.DataFrame, score_column: str, 
                             group_column: str) -> pd.DataFrame:
    """Get group statistics using DuckDB if available, otherwise pandas"""
    
    if duckdb_processor.is_available() and len(data) > 5000:
        # Use DuckDB for large datasets
        table_name = 'temp_stats_data'
        if duckdb_processor.register_dataframe(data, table_name):
            result = duckdb_processor.get_group_statistics(table_name, score_column, group_column)
            if result is not None:
                logger.info(f"Group statistics calculated using DuckDB ({len(data):,} rows)")
                return result
    
    # Fallback to pandas
    logger.info(f"Group statistics calculated using pandas ({len(data):,} rows)")
    return data.groupby(group_column)[score_column].agg([
        'count', 'mean', 'median', 'std', 'min', 'max',
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75)
    ]).round(4)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_fast_outlier_detection(data: pd.DataFrame, score_column: str, 
                              group_column: str) -> pd.DataFrame:
    """Fast outlier detection using DuckDB if available"""
    
    if duckdb_processor.is_available() and len(data) > 2000:
        table_name = 'temp_outlier_data'
        if duckdb_processor.register_dataframe(data, table_name):
            result = duckdb_processor.detect_outliers_iqr(table_name, score_column, group_column)
            if result is not None:
                logger.info(f"Outlier detection using DuckDB ({len(data):,} rows)")
                return result
    
    # Fallback to pandas
    logger.info(f"Outlier detection using pandas ({len(data):,} rows)")
    outlier_info = {}
    
    for group in data[group_column].unique():
        group_data = data[data[group_column] == group][score_column].dropna()
        if len(group_data) > 3:  # Need at least 4 points for IQR
            q1 = group_data.quantile(0.25)
            q3 = group_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = group_data[(group_data < lower_bound) | (group_data > upper_bound)]
            outlier_info[group] = {
                'outlier_count': len(outliers),
                'outlier_values': list(outliers)
            }
    
    return pd.DataFrame(outlier_info).T


def show_duckdb_status():
    """Show DuckDB integration status in Streamlit"""
    if DUCKDB_AVAILABLE:
        if duckdb_processor.is_available():
            st.success("🦆 DuckDB integration active - High-performance mode enabled")
            
            # Show registered tables
            if duckdb_processor.tables:
                with st.expander("📊 DuckDB Tables", expanded=False):
                    for table_name, info in duckdb_processor.tables.items():
                        st.write(f"**{table_name}**: {info['rows']:,} rows, {len(info['columns'])} columns")
        else:
            st.warning("🦆 DuckDB available but not initialized")
    else:
        st.info("🦆 DuckDB not installed - Using standard pandas operations")
        st.info("💡 Install DuckDB for better performance: `pip install duckdb`")


def optimize_large_dataset(data: pd.DataFrame, threshold: int = 10000) -> pd.DataFrame:
    """Optimize large dataset processing (silent operation)"""
    if len(data) > threshold and duckdb_processor.is_available():
        # Register with DuckDB for fast operations (silent)
        table_name = f"large_dataset_{len(data)}"
        duckdb_processor.register_dataframe(data, table_name)
    
    return data


# Export key functions
__all__ = [
    'DuckDBProcessor',
    'duckdb_processor',
    'get_fast_group_statistics',
    'get_fast_outlier_detection',
    'show_duckdb_status',
    'optimize_large_dataset',
    'DUCKDB_AVAILABLE'
]
