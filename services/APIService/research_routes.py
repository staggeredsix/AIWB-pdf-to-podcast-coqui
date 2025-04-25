"""
Research-to-Podcast API routes for the PDF-to-Podcast system.

This module adds routes to the existing APIService to handle research-based
podcast generation requests.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, Form, Query
from fastapi.responses import Response
import logging
import os
import json
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import httpx
import asyncio
import time

# Import from shared modules
from shared.api_types import ServiceType, JobStatus
from shared.connection import ConnectionManager
from shared.job import JobStatusManager
from shared.otel import OpenTelemetryInstrumentation

# Service URL
RESEARCH_SERVICE_URL = os.getenv("RESEARCH_SERVICE_URL", "http://research-service:8005")

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchPodcastRequest(BaseModel):
    """Model for research-to-podcast request parameters."""
    topic: str
    style: str = "conversational"  # conversational or monologue
    duration: int = 5  # minutes
    depth: str = "medium"  # shallow, medium, deep
    focus: Optional[str] = None
    speaker_1_name: str = "Host"
    speaker_2_name: Optional[str] = "Guest"
    userId: str = "default"


def create_research_router(
    telemetry: OpenTelemetryInstrumentation,
    job_manager: JobStatusManager,
    connection_manager: ConnectionManager
) -> APIRouter:
    """
    Create and configure the research API router.
    
    Args:
        telemetry: OpenTelemetry instrumentation
        job_manager: Job status manager
        connection_manager: WebSocket connection manager
        
    Returns:
        Configured FastAPI router
    """
    router = APIRouter(prefix="/research", tags=["research"])
    
    @router.post("/generate", status_code=202)
    async def generate_research_podcast(
        request: ResearchPodcastRequest,
        background_tasks: BackgroundTasks
    ):
        """
        Generate a podcast from web research on a topic.
        
        This endpoint initiates the research-to-podcast process by:
        1. Creating a new job
        2. Submitting a research request
        3. Tracking the job status
        
        Args:
            request: Research podcast request parameters
            background_tasks: FastAPI background tasks handler
            
        Returns:
            Dict containing job_id for tracking
        """
        with telemetry.tracer.start_as_current_span("api.research.generate") as span:
            # Generate job ID
            job_id = f"research_{int(time.time())}"
            span.set_attribute("job_id", job_id)
            span.set_attribute("topic", request.topic)
            
            # Initialize job status
            job_manager.create_job(job_id)
            
            # Start processing in background
            background_tasks.add_task(
                process_research_request,
                job_id,
                request,
                telemetry,
                job_manager,
                connection_manager
            )
            
            return {"job_id": job_id}
    
    @router.websocket("/ws/status/{job_id}")
    async def websocket_status(websocket: WebSocket, job_id: str):
        """
        WebSocket endpoint for real-time status updates on research jobs.
        
        This endpoint works the same way as the PDF processing status endpoint,
        providing real-time updates on research and podcast generation progress.
        
        Args:
            websocket: WebSocket connection
            job_id: Job ID to track
        """
        try:
            # Accept the connection
            await connection_manager.connect(websocket, job_id)
            logger.info(f"WebSocket connection established for research job {job_id}")
            
            # Send ready check
            await websocket.send_json({"type": "ready_check"})
            
            # Wait for client acknowledgment
            try:
                response = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
                if response != "ready":
                    logger.warning(f"Client {job_id} sent invalid ready response: {response}")
                    return
                logger.info(f"Client {job_id} acknowledged ready state")
            except asyncio.TimeoutError:
                logger.warning(f"Client {job_id} ready check timeout")
                return
            except Exception as e:
                logger.error(f"Error during ready check for {job_id}: {e}")
                return
            
            # Send initial status for all services
            for service in ServiceType:
                hget_key = f"status:{job_id}:{str(service)}"
                
                status_data = job_manager.redis.hgetall(hget_key)
                if status_data:
                    status_msg = {
                        "service": service.value,
                        "status": status_data.get(b"status", b"").decode(),
                        "message": status_data.get(b"message", b"").decode(),
                    }
                    await websocket.send_json(status_msg)
            
            # Keep connection alive and handle client messages
            while True:
                try:
                    data = await websocket.receive_text()
                    if data == "ping":
                        await websocket.send_text("pong")
                except Exception:
                    break
                
                await asyncio.sleep(0.1)
                
        except Exception as e:
            logger.error(f"WebSocket error for research job {job_id}: {e}")
        finally:
            connection_manager.disconnect(websocket, job_id)
    
    @router.get("/status/{job_id}")
    async def get_research_status(job_id: str):
        """
        Get the current status of a research-to-podcast job.
        
        Args:
            job_id: Job ID to check
            
        Returns:
            Dict containing status information
        """
        with telemetry.tracer.start_as_current_span("api.research.status") as span:
            span.set_attribute("job_id", job_id)
            
            try:
                # Get aggregated status from Redis
                status = {}
                for service in ServiceType:
                    hget_key = f"status:{job_id}:{str(service)}"
                    service_status = job_manager.redis.hgetall(hget_key)
                    if service_status:
                        status[service.value] = {
                            k.decode(): v.decode() for k, v in service_status.items()
                        }
                
                # If we have status info, return it
                if status:
                    return status
                
                # If not, check with the research service directly
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{RESEARCH_SERVICE_URL}/status/{job_id}")
                    
                    if response.status_code == 404:
                        raise HTTPException(status_code=404, detail="Job not found")
                    
                    response.raise_for_status()
                    return response.json()
                    
            except httpx.HTTPError as e:
                logger.error(f"HTTP error getting research status: {e}")
                raise HTTPException(status_code=500, detail=f"Error connecting to research service: {str(e)}")
            except Exception as e:
                logger.error(f"Error getting research status: {e}")
                raise HTTPException(status_code=500, detail=f"Error getting job status: {str(e)}")
    
    @router.get("/output/{job_id}")
    async def get_research_output(job_id: str):
        """
        Get the output of a completed research-to-podcast job.
        
        Args:
            job_id: Job ID to get output for
            
        Returns:
            Audio file response if job is complete
        """
        with telemetry.tracer.start_as_current_span("api.research.output") as span:
            span.set_attribute("job_id", job_id)
            
            # Check if TTS completed
            tts_status_key = f"status:{job_id}:{str(ServiceType.TTS)}"
            tts_status = job_manager.redis.hgetall(tts_status_key)
            
            # If TTS is not completed, check if research is still in progress
            if not tts_status or tts_status.get(b"status", b"").decode() != str(JobStatus.COMPLETED):
                # Check with research service
                try:
                    async with httpx.AsyncClient() as client:
                        status_response = await client.get(f"{RESEARCH_SERVICE_URL}/status/{job_id}")
                        if status_response.status_code != 200:
                            raise HTTPException(status_code=404, detail="Research job not found or not completed")
                        
                        status_data = status_response.json()
                        if status_data.get("status") != str(JobStatus.COMPLETED):
                            raise HTTPException(
                                status_code=202, 
                                detail="Research still in progress"
                            )
                except httpx.HTTPError:
                    raise HTTPException(status_code=500, detail="Error connecting to research service")
            
            # Get audio from TTS service via main API endpoint
            try:
                # Forward to main output endpoint
                return Response(
                    status_code=307,  # Temporary redirect
                    headers={"Location": f"/output/{job_id}"}
                )
            except Exception as e:
                logger.error(f"Error getting research output: {e}")
                raise HTTPException(status_code=500, detail=f"Error getting research output: {str(e)}")
                
async def process_research_request(
    job_id: str,
    request: ResearchPodcastRequest,
    telemetry: OpenTelemetryInstrumentation,
    job_manager: JobStatusManager,
    connection_manager: ConnectionManager
):
    """
    Process a research request by forwarding it to the research service.
    
    This background task submits the request to the Research Service and tracks its progress.
    
    Args:
        job_id: Unique job identifier
        request: Research podcast request parameters
        telemetry: OpenTelemetry instrumentation
        job_manager: Job status manager
        connection_manager: WebSocket connection manager
    """
    with telemetry.tracer.start_as_current_span("api.process_research_request") as span:
        try:
            # Prepare request for research service
            research_request = {
                "topic": request.topic,
                "job_id": job_id,
                "style": request.style,
                "duration": request.duration,
                "depth": request.depth,
                "focus": request.focus,
                "speaker_1_name": request.speaker_1_name,
                "speaker_2_name": request.speaker_2_name,
                "userId": request.userId
            }
            
            # Submit to research service
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{RESEARCH_SERVICE_URL}/research",
                    json=research_request
                )
                
                if response.status_code != 202:
                    error_msg = f"Research service returned status {response.status_code}: {response.text}"
                    logger.error(error_msg)
                    job_manager.update_status(job_id, JobStatus.FAILED, error_msg)
                    return
                
                # Update status
                job_manager.update_status(
                    job_id,
                    JobStatus.PROCESSING,
                    f"Research request submitted for topic: {request.topic}"
                )
                
                # The research service will handle the rest of the process
                logger.info(f"Research request for job {job_id} forwarded to research service")
                
        except Exception as e:
            error_msg = f"Error processing research request: {str(e)}"
            logger.error(error_msg)
            job_manager.update_status(job_id, JobStatus.FAILED, error_msg)
            
    return router
