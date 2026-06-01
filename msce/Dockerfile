FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY engine.py product_engine.py frontier_benchmark.py calibration_benchmark.py benchmark_questions.py ./

# Expose API port (if running msce_api.py)
EXPOSE 8766

# Default command: run the benchmark
CMD ["python", "run_benchmark.py"]
