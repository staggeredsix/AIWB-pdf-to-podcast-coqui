

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import os
import asyncio
import json
import time

from app.backend.services.web_scraper import WebScraper
from app.backend.research_agent.source_gathering import SourceGatherer
from app.backend.research_agent.analysis import ResearchAnalyzer
from app.backend.research_agent.agent import ResearchAgent
from app.backend.rag.rag_database import RAGDatabase

# Import shared components from PDF-to-Podcast
from shared.api_types import JobStatus, ServiceType
from shared.job import JobStatusManager
from shared.otel import OpenTelemetryInstrumentation, OpenTelemetryConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(debug=True)

# Set up OpenTelemetry instrumentation
telemetry = OpenTelemetryInstrumentation()
config = OpenTelemetryConfig(
    service_name="research-service",
    otlp_endpoint=os.getenv("OTLP_ENDPOINT", "http://jaeger:4317"),
    enable_redis=True,
    enable_requests=True,
)
telemetry.initialize(config, app)

# Initialize job manager
job_manager = JobStatusManager(ServiceType.PDF, telemetry=telemetry)

# Service configuration
MAX_SOURCES = int(os.getenv("MAX_SOURCES", "15"))
MAX_RESEARCH_TIME = int(os.getenv("MAX_RESEARCH_TIME", "120"))  # seconds
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://agent-service:8964")


class ResearchRequest(BaseModel):
    """Research request parameters."""
    topic: str
    job_id: str
    style: str = "conversational"  # conversational, monologue
    duration: int = 5  # minutes
    depth: str = "medium"  # shallow, medium, deep
    focus: Optional[str] = None
    speaker_1_name: str = "Host"
    speaker_2_name: Optional[str] = "Guest"
    userId: str = Field(..., description="User ID for storage and tracking")


class ResearchSource(BaseModel):
    """Structured research source information."""
    url: str
    title: str
    content: str
    timestamp: Optional[str] = None


@app.post("/research", status_code=202)
async def research_topic(request: ResearchRequest, background_tasks: BackgroundTasks):
    """
    Research a topic and convert findings to podcast content.
    
    This endpoint initiates a background task to research the requested topic,
    process the findings, and prepare them for podcast generation.
    """
    with telemetry.tracer.start_as_current_span("research.initiate") as span:
        span.set_attribute("topic", request.topic)
        span.set_attribute("job_id", request.job_id)
        span.set_attribute("style", request.style)
        span.set_attribute("duration", request.duration)
        
        job_manager.create_job(request.job_id)
        job_manager.update_status(
            request.job_id, 
            JobStatus.PROCESSING, 
            f"Starting research on topic: {request.topic}"
        )
        
        background_tasks.add_task(
            process_research_task,
            request.job_id,
            request.topic,
            request.style,
            request.duration,
            request.depth,
            request.focus,
            request.speaker_1_name,
            request.speaker_2_name,
            request.userId
        )
        
        return {"job_id": request.job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get the current status of a research job."""
    with telemetry.tracer.start_as_current_span("research.status") as span:
        span.set_attribute("job_id", job_id)
        
        status = job_manager.get_status(job_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Job not found")
            
        return status


@app.get("/output/{job_id}")
async def get_output(job_id: str):
    """Get the research results for a completed job."""
    with telemetry.tracer.start_as_current_span("research.output") as span:
        span.set_attribute("job_id", job_id)
        
        result = job_manager.get_result(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Result not found")
            
        return json.loads(result.decode())


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


async def process_research_task(
    job_id: str,
    topic: str,
    style: str,
    duration: int,
    depth: str,
    focus: Optional[str],
    speaker_1_name: str,
    speaker_2_name: Optional[str],
    user_id: str
):
    """
    Process a research task and prepare content for podcast generation.
    
    This function performs the following steps:
    1. Research the topic using web sources
    2. Analyze and synthesize findings
    3. Generate a structured document format compatible with podcast creation
    4. Submit the document for podcast generation
    """
    with telemetry.tracer.start_as_current_span("research.process") as span:
        try:
            span.set_attribute("job_id", job_id)
            span.set_attribute("topic", topic)
            
            # Initialize components
            web_scraper = WebScraper()
            source_gatherer = SourceGatherer(web_scraper)
            
            # We'd normally use a real LLM client, but for now we'll use a simplified version
            llm_client = SimpleLLMClient()
            
            # Configure research depth
            max_sources = MAX_SOURCES
            if depth == "shallow":
                max_sources = 5
            elif depth == "deep":
                max_sources = 20
            
            # Step 1: Update status and determine search queries
            job_manager.update_status(
                job_id, 
                JobStatus.PROCESSING, 
                "Generating search queries"
            )
            
            # TODO: Replace this with actual LLM-generated queries once integrated
            search_queries = await generate_search_queries(topic, focus)
            logger.info(f"Generated search queries: {search_queries}")
            
            # Step 2: Gather sources
            job_manager.update_status(
                job_id, 
                JobStatus.PROCESSING, 
                f"Gathering sources on {topic}"
            )
            
            sources = await source_gatherer.gather_sources(search_queries)
            # Limit sources to max_sources
            sources = sources[:max_sources]
            logger.info(f"Gathered {len(sources)} sources")
            
            # Step 3: Analyze findings
            job_manager.update_status(
                job_id, 
                JobStatus.PROCESSING, 
                "Analyzing research findings"
            )
            
            # Create research agent if we have a rag_db or use just the analyzer
            try:
                rag_db = RAGDatabase()  # Initialize with default settings
                research_agent = ResearchAgent(rag_db, llm_client, web_scraper)
                research_results = await research_agent.research(
                    query=topic,
                    mode="both",  # Use both RAG and web search
                    conversation_history=[]  # No conversation history for now
                )
                analysis = research_results.get("analysis", "")
                technical_summary = research_results.get("technical_summary", "")
                accessible_summary = research_results.get("accessible_summary", "")
            except Exception as e:
                logger.error(f"Error with research agent: {e}")
                # Fallback to just the analyzer
                analyzer = ResearchAnalyzer(llm_client)
                
                research_state = {
                    "query": topic,
                    "findings": format_sources_as_findings(sources),
                    "focus": focus,
                    "depth": depth
                }
                
                analysis = await analyzer.synthesize_findings(research_state)
                summaries = await analyzer.generate_summaries(
                    analysis=analysis,
                    context=""
                )
                technical_summary = summaries.get("technical", "")
                accessible_summary = summaries.get("accessible", "")
            
            # Step 4: Generate podcast-ready content
            job_manager.update_status(
                job_id, 
                JobStatus.PROCESSING, 
                "Generating podcast content from research"
            )
            
            markdown_content = generate_markdown_content(
                topic=topic,
                analysis=analysis,
                technical_summary=technical_summary,
                accessible_summary=accessible_summary,
                focus=focus
            )
            
            # Step 5: Create a "PDF-like" document for processing
            title = f"Research on {topic}"
            
            # Create a PDF metadata structure compatible with PDF-to-Podcast
            pdf_metadata = [{
                "filename": f"{job_id}_research.md",
                "markdown": markdown_content,
                "type": "target",
                "status": "success",
                "error": None
            }]
            
            # Step 6: Forward to transcription service
            job_manager.update_status(
                job_id, 
                JobStatus.PROCESSING, 
                "Forwarding to podcast generation"
            )
            
            # Create transcription request parameters for the Agent Service
            transcription_params = {
                "pdf_metadata": pdf_metadata,
                "job_id": job_id,
                "name": title,
                "duration": duration,
                "speaker_1_name": speaker_1_name,
                "monologue": style == "monologue",
                "guide": focus,
                "userId": user_id
            }
            
            # Add speaker 2 if not monologue
            if style != "monologue" and speaker_2_name:
                transcription_params["speaker_2_name"] = speaker_2_name
                
            # Add default voice mapping
            transcription_params["voice_mapping"] = {
                "speaker-1": "iP95p4xoKVk53GoZ742B"  # Default voice ID
            }
            
            if style != "monologue":
                transcription_params["voice_mapping"]["speaker-2"] = "9BWtsMINqrJLrRacOk9x"  # Default voice ID for speaker 2
            
            # Forward to agent service for podcast generation
            import httpx
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{AGENT_SERVICE_URL}/transcribe",
                    json=transcription_params
                )
                
                if response.status_code != 202:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Agent service error: {response.text}"
                    )
            
            # Store research results for reference
            job_manager.set_result_with_expiration(
                job_id,
                json.dumps({
                    "topic": topic,
                    "markdown": markdown_content,
                    "sources": [s.get("url") for s in sources],
                    "transcription_params": transcription_params
                }).encode(),
                ex=3600  # 1 hour expiration
            )
            
            # Final status update
            job_manager.update_status(
                job_id, 
                JobStatus.COMPLETED, 
                "Research completed and forwarded to podcast generation"
            )
            
        except Exception as e:
            logger.error(f"Research processing error: {str(e)}")
            job_manager.update_status(job_id, JobStatus.FAILED, str(e))


async def generate_search_queries(topic: str, focus: Optional[str] = None) -> List[str]:
    """Generate effective search queries for the research topic."""
    # This would normally use your LLM client, but we'll use a simpler approach
    base_queries = [topic]
    
    if focus:
        base_queries.append(f"{topic} {focus}")
    
    # Add some variations
    variations = [
        f"{topic} research",
        f"{topic} analysis",
        f"{topic} explained"
    ]
    
    return base_queries + variations


def format_sources_as_findings(sources: List[Dict]) -> List[Dict]:
    """Format gathered sources as research findings."""
    findings = []
    
    for i, source in enumerate(sources):
        findings.append({
            "id": str(i),
            "content": source.get("content", ""),
            "source": "web",
            "url": source.get("url", ""),
            "title": source.get("title", ""),
            "timestamp": source.get("timestamp", ""),
        })
    
    return findings


def generate_markdown_content(
    topic: str,
    analysis: str,
    technical_summary: str,
    accessible_summary: str,
    focus: Optional[str] = None
) -> str:
    """
    Generate structured markdown content from research results.
    
    This content will be passed to the podcast generation pipeline as if it
    were a PDF document, so it should be well-structured and comprehensive.
    """
    content_sections = [
        f"# Research on {topic}",
        "",
        f"## Overview",
        "",
        f"{accessible_summary or 'This document contains research findings on ' + topic + ', synthesized from multiple web sources.'}",
        ""
    ]
    
    # Add focus section if provided
    if focus:
        content_sections.extend([
            f"## Focus Area: {focus}",
            "",
            f"This research focuses specifically on {focus} as it relates to {topic}.",
            ""
        ])
    
    # Add technical summary if available
    if technical_summary:
        content_sections.extend([
            f"## Technical Summary",
            "",
            technical_summary,
            ""
        ])
    
    # Add main analysis
    content_sections.extend([
        f"## Analysis",
        "",
        analysis,
        ""
    ])
    
    # Add conclusion
    content_sections.extend([
        "## Conclusion",
        "",
        f"This research on {topic} has examined multiple perspectives and sources.",
        "The findings represent a synthesis of available information at the time of research.",
        ""
    ])
    
    return "\n".join(content_sections)


class SimpleLLMClient:
    """A simplified LLM client for placeholder purposes."""
    
    async def generate_search_queries(self, topic: str, current_findings: List = None, conversation_history: List = None) -> List[str]:
        """Generate search queries for the topic."""
        base_queries = [topic]
        variations = [
            f"{topic} research",
            f"{topic} analysis",
            f"{topic} explained",
            f"{topic} latest developments",
            f"{topic} industry trends"
        ]
        return base_queries + variations
    
    async def generate_response(self, prompt: str, response_format: str = None) -> str:
        """Generate a response to a prompt."""
        # In a real implementation, this would call an LLM API
        return f"This is a placeholder response for: {prompt[:50]}..."


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
