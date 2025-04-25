

import httpx
import logging
import os
from typing import Dict, Any, List, Optional
import json
import asyncio

# Constants
RESEARCH_SERVICE_URL = os.getenv("RESEARCH_SERVICE_URL", "http://research-service:8005")
API_SERVICE_URL = os.getenv("API_SERVICE_URL", "http://api-service:8002")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResearchIntegration:
    """
    Integration class to connect research capabilities with the PDF-to-Podcast workflow.
    
    This class provides methods to:
    1. Submit research requests
    2. Track research job status
    3. Convert research results to podcast format
    """
    
    def __init__(self, timeout: int = 60):
        """Initialize the integration with configurable timeout."""
        self.timeout = timeout
    
    async def submit_research_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a request to research a topic and generate podcast content.
        
        Args:
            request_data: Dictionary containing research parameters
            
        Returns:
            Dict with job_id and status information
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{RESEARCH_SERVICE_URL}/research",
                    json=request_data
                )
                response.raise_for_status()
                return response.json()
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error submitting research request: {e}")
            raise
        except Exception as e:
            logger.error(f"Error submitting research request: {e}")
            raise
    
    async def get_research_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the current status of a research job.
        
        Args:
            job_id: The unique identifier for the job
            
        Returns:
            Dict containing status information
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{RESEARCH_SERVICE_URL}/status/{job_id}")
            response.raise_for_status()
            return response.json()
    
    async def get_research_output(self, job_id: str) -> Dict[str, Any]:
        """
        Get the output of a completed research job.
        
        Args:
            job_id: The unique identifier for the job
            
        Returns:
            Dict containing the research results
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{RESEARCH_SERVICE_URL}/output/{job_id}")
            response.raise_for_status()
            return response.json()
    
    async def poll_until_complete(self, job_id: str, max_retries: int = 30, delay: int = 5) -> Dict[str, Any]:
        """
        Poll the research job status until it completes or fails.
        
        Args:
            job_id: The unique identifier for the job
            max_retries: Maximum number of status checks
            delay: Seconds to wait between checks
            
        Returns:
            Dict containing the final status
            
        Raises:
            TimeoutError if max_retries is reached without completion
        """
        for _ in range(max_retries):
            status = await self.get_research_status(job_id)
            
            if "status" in status:
                if status["status"] == "completed" or status["status"] == "JobStatus.COMPLETED":
                    return status
                elif status["status"] == "failed" or status["status"] == "JobStatus.FAILED":
                    raise Exception(f"Research job failed: {status.get('message', 'No error message')}")
            
            await asyncio.sleep(delay)
        
        raise TimeoutError(f"Research job {job_id} did not complete within the allotted time")
    
    async def get_podcast_output(self, job_id: str) -> bytes:
        """
        Get the final podcast audio output for a job.
        
        Args:
            job_id: The unique identifier for the job
            
        Returns:
            Bytes containing the audio file
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{API_SERVICE_URL}/output/{job_id}")
            response.raise_for_status()
            return response.content


async def process_research_to_podcast(
    topic: str,
    style: str = "conversational",
    duration: int = 5,
    depth: str = "medium",
    focus: Optional[str] = None,
    speaker_1_name: str = "Host",
    speaker_2_name: str = "Guest",
    user_id: str = "default"
) -> Dict[str, Any]:
    """
    Process a research topic into a podcast from start to finish.
    
    This function orchestrates the entire flow:
    1. Submit research request
    2. Wait for research completion
    3. Wait for podcast generation
    4. Return result information
    
    Args:
        topic: The research topic
        style: Podcast style ("conversational" or "monologue")
        duration: Target duration in minutes
        depth: Research depth ("shallow", "medium", "deep")
        focus: Optional focus area
        speaker_1_name: Name for speaker 1
        speaker_2_name: Name for speaker 2 (ignored if monologue)
        user_id: User identifier for storage and tracking
        
    Returns:
        Dict with information about the completed podcast
    """
    integration = ResearchIntegration()
    
    # Generate a job ID (in a real implementation this would be UUID)
    import time
    job_id = f"research_{int(time.time())}"
    
    # Step 1: Submit research request
    request_data = {
        "topic": topic,
        "job_id": job_id,
        "style": style,
        "duration": duration,
        "depth": depth,
        "focus": focus,
        "speaker_1_name": speaker_1_name,
        "speaker_2_name": speaker_2_name if style != "monologue" else None,
        "userId": user_id
    }
    
    submit_result = await integration.submit_research_request(request_data)
    logger.info(f"Research request submitted with job ID: {job_id}")
    
    # Step 2: Poll until research is complete
    try:
        await integration.poll_until_complete(job_id)
        logger.info(f"Research completed for job ID: {job_id}")
    except Exception as e:
        logger.error(f"Research failed: {e}")
        raise
    
    # Step 3: Wait for podcast generation to complete
    # Note: At this point, the research service has forwarded to the agent service,
    # so we need to poll the API service for the final result
    
    try:
        # Wait a bit for the research results to be processed
        await asyncio.sleep(5)
        
        # Get podcast audio
        audio_data = await integration.get_podcast_output(job_id)
        
        # Get research results for metadata
        research_data = await integration.get_research_output(job_id)
        
        return {
            "job_id": job_id,
            "topic": topic,
            "audio_size": len(audio_data),
            "research_metadata": research_data,
            "completed": True
        }
        
    except Exception as e:
        logger.error(f"Failed to get podcast output: {e}")
        raise
