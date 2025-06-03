"""Enhanced logger class for metal-ligand complex analysis."""

import datetime
import time
from rich.console import Console
import inspect
import os
from pathlib import Path

# Global flag to track if we've shown the initialization message
_init_message_shown = False


class MLXTBLogger:
    """
    A logger class using rich for colored console output.
    Formats messages as: [Date Time dt] (file:line) MESSAGE_TYPE message
    - info messages are green
    - error messages are red
    - debug messages are yellow
    - warning messages are orange
    """

    def __init__(self, show_init: bool = True, debug_enabled: bool = True) -> None:
        global _init_message_shown
        
        self.console = Console()
        self.start_time = time.time()
        self.start_datetime = datetime.datetime.now()
        self.debug_enabled = debug_enabled
        
        # Only show initialization message in main process
        # Check if we're in a multiprocessing worker process
        import multiprocessing
        is_main_process = multiprocessing.current_process().name == 'MainProcess'
        
        if show_init and not _init_message_shown and is_main_process:
            init_msg = (
                f"[bold cyan]ML-XTB Logger initialized at "
                f"{self.start_datetime.strftime('%Y-%m-%d %H:%M:%S')}[/bold cyan]"
            )
            self.console.print(init_msg)
            _init_message_shown = True

    def _log(self, message: str, style: str, log_type: str) -> None:
        """Internal logging method to format and print messages.
        
        Args:
            message (str): The message content to log
            style (str): The Rich style string to apply to the log type
            log_type (str): The type of log (e.g., 'INFO', 'ERROR')
        """
        current_time = time.time()
        dt = current_time - self.start_time
        now = datetime.datetime.now()
        now_str_full = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # Get caller information
        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back if frame and frame.f_back else None
        filename_repr = "unknown_file"
        lineno = "?"
        
        if caller_frame:
            try:
                abs_filename_path = Path(inspect.getfile(caller_frame)).resolve()
                lineno = str(caller_frame.f_lineno)
                
                # Try to get relative path from project root
                logger_path = Path(__file__).resolve()
                src_root = logger_path.parent.parent.parent
                
                if abs_filename_path.is_relative_to(src_root):
                    rel_path = abs_filename_path.relative_to(src_root)
                    module_parts = list(rel_path.parent.parts)
                    base_filename = rel_path.stem  # Remove .py extension
                    
                    if module_parts:
                        filename_repr = f"{'.'.join(module_parts)}.{base_filename}"
                    else:
                        filename_repr = base_filename
                else:
                    filename_repr = Path(os.path.relpath(abs_filename_path)).stem
                    
            except Exception:
                try:
                    filename_repr = Path(
                        os.path.relpath(inspect.getfile(caller_frame))
                    ).stem
                except Exception:
                    filename_repr = "unknown_file"
        
        # Format the log message
        padded_log_type = f"{log_type.upper():<5}"
        log_prefix = (
            f"[#637e96][{now_str_full}] ({dt:.2f}s) "
            f"({filename_repr}:{lineno})[/#637e96]"
        )
        log_line = (
            f"{log_prefix} [{style}]{padded_log_type}[/{style}] "
            f"[white]{message}[/white]"
        )
        self.console.print(log_line, highlight=False)

    def info(self, message: str) -> None:
        """Logs an informational message."""
        self._log(message, style="green", log_type="INFO")

    def error(self, message: str) -> None:
        """Logs an error message."""
        self._log(message, style="bold red", log_type="ERROR")

    def debug(self, message: str) -> None:
        """Logs a debug message."""
        if self.debug_enabled:
            self._log(message, style="yellow", log_type="DEBUG")

    def warning(self, message: str) -> None:
        """Logs a warning message."""
        self._log(message, style="orange1", log_type="WARN")

    def exception(self, message: str) -> None:
        """Logs an exception message."""
        self._log(message, style="bold red", log_type="EXCPT")
    
    def set_debug(self, enabled: bool) -> None:
        """Enable or disable debug logging."""
        self.debug_enabled = enabled
        if enabled:
            self.info("Debug logging enabled")
        else:
            self.info("Debug logging disabled")
    
    def enable_debug(self) -> None:
        """Enable debug logging."""
        self.set_debug(True)
    
    def disable_debug(self) -> None:
        """Disable debug logging."""
        self.set_debug(False)


# Global logger instance
# Debug can be controlled by environment variable ML_XTB_DEBUG (0 or 1)
debug_enabled = os.environ.get('ML_XTB_DEBUG', '1') == '1'
logger = MLXTBLogger(debug_enabled=debug_enabled)


if __name__ == "__main__":
    # Test the logger
    test_logger = MLXTBLogger()
    time.sleep(0.5)
    test_logger.info("This is an informational message.")
    time.sleep(0.3)

    def some_function():
        test_logger.error("This is an error message from a function.")
        test_logger.debug("This is a debug message from a function.")
        test_logger.warning("This is a warning message from a function.")
        test_logger.exception("This is an exception message from a function.")

    some_function()