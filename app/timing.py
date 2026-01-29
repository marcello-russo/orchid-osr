
import time
from datetime import datetime
import csv
import os

def start_timer():
    """Starts the timer and returns the current time."""
    return time.time()

def log_execution_time(start_time, block_name, log_file='logs/timing_log.csv'):
    """
    Calculates duration, prints it, and logs it to a CSV file.
    
    Args:
        start_time (float): The start time timestamp.
        block_name (str): The name of the code block being timed.
        log_file (str): Path to the CSV log file. Defaults to 'logs/timing_log.csv'.
    """
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"{block_name} Execution Time: {duration:.2f} seconds")
    
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        # Write header if new file
        if not file_exists:
            writer.writerow(['Timestamp', 'Block', 'Duration (s)'])
        
        # Write record
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([timestamp, block_name, f"{duration:.2f}"])
