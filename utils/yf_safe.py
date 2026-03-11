"""
Thread-safe yfinance wrapper.

yfinance has a race condition issue when called from multiple threads simultaneously.
This module provides a shared lock to serialize yfinance calls during parallel agent execution.
"""
import threading

# Global lock for thread-safe yfinance access
_yf_lock = threading.Lock()


def yf_download(*args, **kwargs):
    """Thread-safe wrapper for yfinance.download()"""
    import yfinance as yf
    with _yf_lock:
        return yf.download(*args, **kwargs)