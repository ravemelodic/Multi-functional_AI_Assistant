"""
Celery worker entry point
"""
import os
import sys

if __name__ == '__main__':
    worker_type = os.getenv('WORKER_TYPE', 'all')
    
    print(f"Starting Celery worker (type: {worker_type})")
    
    # Build celery worker command arguments
    argv = ['celery', '-A', 'tasks', 'worker', '--loglevel=INFO']
    
    # Configure which tasks this worker should handle
    if worker_type == 'video':
        # Video generation: mostly I/O wait on SiliconFlow API → can overlap 2 per process
        argv.extend(['--queues=video', '--concurrency=2'])
    elif worker_type == 'ocr':
        # Document analysis: ~1s CPU (PyMuPDF) + ~3s I/O (LLM call)
        # Concurrency=4 allows overlapping LLM waits for higher throughput
        argv.extend(['--queues=ocr', '--concurrency=4'])
    
    # Replace current process with celery worker
    os.execvp('celery', argv)
