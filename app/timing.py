
import time
from datetime import datetime
import csv
import os


class Timer:
    """
    Context manager per misurare il tempo di esecuzione di un blocco di codice.

    Usage:
        with Timer('Nome operazione'):
            # codice da misurare
    """
    def __init__(self, block_name, log_file='logs/timing_log.csv'):
        self.block_name = block_name
        self.log_file = log_file
        self.start = None

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        duration = time.time() - self.start
        print(f"{self.block_name}: {duration:.2f} secondi")
        self._log_to_csv(duration)

    def _log_to_csv(self, duration):
        """Salva il tempo su file CSV."""
        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_exists = os.path.isfile(self.log_file)
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Timestamp', 'Block', 'Duration (s)'])
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([timestamp, self.block_name, f"{duration:.2f}"])


# Funzioni deprecate - mantenute per compatibilità
def start_timer():
    """Deprecated: usare Timer context manager."""
    return time.time()

def log_execution_time(start_time, block_name, log_file='logs/timing_log.csv'):
    """Deprecated: usare Timer context manager."""
    end_time = time.time()
    duration = end_time - start_time
    print(f"{block_name} Execution Time: {duration:.2f} seconds")
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Timestamp', 'Block', 'Duration (s)'])
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, block_name, f"{duration:.2f}"])
