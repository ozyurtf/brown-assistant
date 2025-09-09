#!/bin/bash

# RAG Application Startup Script
# This script handles the complete workflow for the RAG application

set -e  # Exit on any error

echo "🚀 Starting RAG Application Workflow..."

# Function to check if a file exists and is not empty
check_file() {
    if [[ -f "$1" && -s "$1" ]]; then
        echo "$1 exists and is not empty"
        return 0
    else
        echo "$1 does not exist or is empty"
        return 1
    fi
}

# Function to run a Python script with error handling
run_python_script() {
    local script_name=$1
    local description=$2
    
    echo "🔄 Running $description..."
    if python "$script_name"; then
        echo "$description completed successfully"
    else
        echo "$description failed"
        exit 1
    fi
}

# Check if data extraction is needed
NEED_EXTRACTION=false

if ! check_file "files/bulletin.json"; then
    echo "Bulletin data not found, will extract..."
    NEED_EXTRACTION=true
fi

if ! check_file "files/cab.json"; then
    echo "CAB data not found, will extract..."
    NEED_EXTRACTION=true
fi

# Step 1: Extract data if needed
if [ "$NEED_EXTRACTION" = true ]; then
    echo "Starting data extraction phase..."
    
    # Extract bulletin data
    if ! check_file "files/bulletin.json"; then
        run_python_script "bulletin.py" "Bulletin data extraction"
    fi
    
    # Extract CAB data
    if ! check_file "files/cab.json"; then
        run_python_script "cab.py" "CAB data extraction"
    fi
else
    echo "Data files already exist, skipping extraction"
fi

# Step 2: Check if vector store needs to be created
NEED_INDEXING=false

if ! check_file "vector_store.pkl"; then
    echo "Vector store not found, will create..."
    NEED_INDEXING=true
fi

if [[ ! -d "vector_store" || -z "$(ls -A vector_store 2>/dev/null)" ]]; then
    echo "Vector store directory empty, will create..."
    NEED_INDEXING=true
fi

# Step 3: Create vector store if needed
if [ "$NEED_INDEXING" = true ]; then
    echo "Creating vector store and embeddings..."
    run_python_script "indexing.py" "Vector store creation and indexing"
else
    echo "Vector store already exists, skipping indexing"
fi

# Step 4: Verify all required files exist
echo "Verifying all required files..."
REQUIRED_FILES=("files/bulletin.json" "files/cab.json" "vector_store.pkl")

for file in "${REQUIRED_FILES[@]}"; do
    if ! check_file "$file"; then
        echo "Critical file missing: $file"
        exit 1
    fi
done

echo "All required files verified"

# Step 5: Start the services
echo "Starting API and UI services..."

# Start FastAPI in the background
echo "Starting FastAPI server on port 8000..."
uvicorn api:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to start (with retries)
echo "Waiting for FastAPI to initialize (this may take 30-60 seconds for model loading)..."
for i in {1..30}; do
    if curl -f http://localhost:8000/docs > /dev/null 2>&1; then
        echo "FastAPI server started successfully"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "FastAPI server failed to start after 30 attempts"
        kill $API_PID 2>/dev/null || true
        exit 1
    fi
    echo "Attempt $i/30: API not ready yet, waiting 2 seconds..."
    sleep 2
done

# Start Streamlit
echo "Starting Streamlit UI on port 8501..."
streamlit run ui.py --server.address 0.0.0.0 --server.port 8501 &
UI_PID=$!

# Function to cleanup on exit
cleanup() {
    echo "Shutting down services..."
    kill $API_PID $UI_PID 2>/dev/null || true
    wait $API_PID $UI_PID 2>/dev/null || true
    echo "Goodbye!"
}

# Set trap for cleanup
trap cleanup SIGTERM SIGINT

echo "RAG Application is ready!"
echo "API Documentation: http://localhost:8000/docs"
echo "Streamlit UI: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for services
wait
