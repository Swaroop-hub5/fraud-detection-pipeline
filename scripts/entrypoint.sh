#!/bin/bash
# Generic Docker entrypoint
# Start the server on port 8080 (Standard container port)
uvicorn src.api:app --host 0.0.0.0 --port 8080