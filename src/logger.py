"""
Logging utility for PyDoc AI
Tracks queries, retrievals, and performance.
"""

import logging
import json
from pathlib import Path
from datetime import datetime
import config


class PyDocLogger:
    """Centralized logging for PyDoc AI."""
    
    def __init__(self):
        """Initialize logger."""
        self.logs_dir = config.BASE_DIR / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create loggers
        self.query_logger = self._setup_logger('query', 'queries.log')
        self.error_logger = self._setup_logger('error', 'errors.log')
        self.performance_logger = self._setup_logger('performance', 'performance.log')
    
    def _setup_logger(self, name, filename):
        """Set up a logger with file and console handlers."""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # File handler
        log_file = self.logs_dir / filename
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler (only for errors)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_query(self, query, num_results, response_length, sources):
        """
        Log a user query and its results.
        
        Args:
            query: User question
            num_results: Number of chunks retrieved
            response_length: Length of LLM response
            sources: List of source metadata
        """
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'num_results': num_results,
            'response_length': response_length,
            'sources': [
                {
                    'source': s['source'],
                    'module': s.get('module', 'N/A'),
                    'score': round(float(s['score']), 2)
                }
                for s in sources
            ]
        }
        
        self.query_logger.info(json.dumps(log_data))
    
    def log_retrieval(self, query, stage, num_candidates, top_scores):
        """
        Log retrieval stage details.
        
        Args:
            query: User question
            stage: Stage name (hybrid_search, rerank)
            num_candidates: Number of results
            top_scores: Top 3 scores
        """
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'stage': stage,
            'num_candidates': num_candidates,
            'top_3_scores': [round(float(s), 2) for s in top_scores[:3]]
        }
        
        self.performance_logger.info(json.dumps(log_data))
    
    def log_error(self, error_type, error_message, query=None):
        """
        Log an error.
        
        Args:
            error_type: Type of error
            error_message: Error description
            query: User query that caused error (if applicable)
        """
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': str(error_message),
            'query': query
        }
        
        self.error_logger.error(json.dumps(log_data))
    
    def log_performance(self, operation, duration_seconds):
        """
        Log performance metrics.
        
        Args:
            operation: Name of operation
            duration_seconds: Time taken in seconds
        """
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'operation': operation,
            'duration_seconds': round(duration_seconds, 3)
        }
        
        self.performance_logger.info(json.dumps(log_data))


# Global logger instance
logger = PyDocLogger()