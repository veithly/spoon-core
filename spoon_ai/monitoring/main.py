#!/usr/bin/env python
# spoon_ai/monitoring/standalone.py

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import sys

# Add parent directory to sys.path to ensure the spoon_ai package can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("monitoring-service")

# Create FastAPI application
app = FastAPI(
    title="Crypto Monitoring Service",
    description="A service for monitoring cryptocurrency metrics and sending alerts",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, should be restricted in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and register API routes
from spoon_ai.monitoring.api.routes import router as monitoring_router
app.include_router(monitoring_router)

# Add health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok", "service": "monitoring"}

# Start task scheduler
from spoon_ai.monitoring.core.tasks import MonitoringTaskManager
task_manager = MonitoringTaskManager()

@app.on_event("startup")
async def startup_event():
    """Event handler for service startup"""
    logger.info("Starting monitoring service...")
    # Scheduler is already started when MonitoringTaskManager is initialized

@app.on_event("shutdown")
async def shutdown_event():
    """Event handler for service shutdown"""
    logger.info("Shutting down monitoring service...")
    # Stop scheduler
    task_manager.scheduler.stop()

# If this file is run directly, start the service
if __name__ == "__main__":
    # Get configuration parameters, can be read from environment variables
    host = os.getenv("MONITORING_HOST", "0.0.0.0")
    port = int(os.getenv("MONITORING_PORT", "8888"))
    
    logger.info(f"Starting monitoring service on {host}:{port}")
    uvicorn.run("spoon_ai.monitoring.main:app", host=host, port=port, reload=True)