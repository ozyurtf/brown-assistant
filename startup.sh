#!/bin/bash

# RAG Application Startup Script
# This script handles the complete workflow for the RAG application

set -e  # Exit on any error

echo "Starting RAG Application Workflow..."

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
    
    echo "Running $description..."
    if python "$script_name"; then
        echo "$description completed successfully"
    else
        echo "$description failed"
        exit 1
    fi
}

# Check if data extraction is needed
NEED_EXTRACTION=false

if ! check_file "files/concentration.json"; then
    echo "Concentration data not found, will extract..."
    NEED_EXTRACTION=true
fi

if ! check_file "files/cab.json"; then
    echo "CAB data not found, will extract..."
    NEED_EXTRACTION=true
fi

# Step 1: Extract data if needed
if [ "$NEED_EXTRACTION" = true ]; then
    echo "Starting data extraction phase..."
    
    # Extract concentration data
    if ! check_file "files/concentration.json"; then
        run_python_script "bulletin.py" "Concentration data extraction"
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
REQUIRED_FILES=("files/concentration.json" "files/cab.json" "vector_store.pkl")

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
API_PORT_VAL=${API_PORT:-8000}
echo "Starting FastAPI server on port ${API_PORT_VAL}..."
uvicorn api:app --host 0.0.0.0 --port ${API_PORT_VAL} &
API_PID=$!

# Wait for API to start (with retries)
echo "Waiting for FastAPI to initialize (this may take 30-60 seconds for model loading)..."
for i in {1..30}; do
    if curl -f http://localhost:${API_PORT_VAL}/docs > /dev/null 2>&1; then
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
STREAMLIT_ADDR_VAL=${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}
STREAMLIT_PORT_VAL=${STREAMLIT_SERVER_PORT:-8501}
echo "Starting Streamlit UI on ${STREAMLIT_ADDR_VAL}:${STREAMLIT_PORT_VAL}..."
streamlit run ui.py --server.address ${STREAMLIT_ADDR_VAL} --server.port ${STREAMLIT_PORT_VAL} &
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
echo "API Documentation: http://localhost:${API_PORT_VAL}/docs"
echo "Streamlit UI: http://localhost:${STREAMLIT_PORT_VAL}"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for services
wait
