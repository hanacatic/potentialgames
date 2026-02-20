import logging


class ColoredFormatter(logging.Formatter):
    """Logging formatter that adds colors based on log level."""

    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[95m', # Magenta
    }
    RESET = '\033[0m'

    def format(self, record):
        
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        
        return f"{color}{message}{self.RESET}"

class ColoredLogger(logging.Logger):
    """Custom logger that uses ColoredFormatter for colored output in the console."""
    
    def __init__(self, name, level=logging.DEBUG):
        super().__init__(name, level)
        
        handler = logging.StreamHandler()
        formatter = ColoredFormatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        self.addHandler(handler)
        self.propagate = False

logger = ColoredLogger("Colored Logger") # Create a logger instance with colored console output