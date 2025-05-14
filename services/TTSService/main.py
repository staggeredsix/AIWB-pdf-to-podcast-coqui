from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import logging
import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import io

import torch
from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, get_voices
import torchaudio
import soundfile as sf

from shared.api_types import ServiceType, JobStatus
from shared.job import JobStatusManager
from shared.otel import OpenTelemetryInstrumentation, OpenTelemetryConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tortoise TTS Service", debug=True)

MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "3"))

# Voice mappings for Tortoise
# You can add custom voices by placing audio samples in tortoise/voices/ directory
VOICE_MAPPING = {
    "speaker-1": "male_voice",  # Built-in male voice
    "speaker-2": "female_voice",  # Built-in female voice
    "bob": "male_voice",
    "kate": "female_voice",
    "speaker1": "male_voice",
    "speaker2": "female_voice",
}

telemetry = OpenTelemetryInstrumentation()
telemetry.initialize(
    OpenTelemetryConfig(
        service_name="tortoise-tts-service",
        otlp_endpoint=os.getenv("OTLP_ENDPOINT", "http://jaeger:4317"),
        enable_redis=True,
        enable_requests=True,
    ),
    app,
)

job_manager = JobStatusManager(ServiceType.TTS, telemetry=telemetry)


class DialogueEntry(BaseModel):
    text: str
    speaker: str
    voice_id: Optional[str] = None


class TTSRequest(BaseModel):
    dialogue: List[DialogueEntry]
    job_id: str
    scratchpad: Optional[str] = ""
    voice_mapping: Optional[Dict[str, str]] = VOICE_MAPPING


class VoiceInfo(BaseModel):
    voice_id: str
    name: str
    description: Optional[str] = None


class TortoiseService:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)
        
        # Initialize Tortoise
        logger.info("Initializing Tortoise TTS model...")
        self.tts = TextToSpeech(
            use_deepspeed=True if torch.cuda.is_available() else False,
            kv_cache=True,
            half=True if torch.cuda.is_available() else False
        )
        
        # Get available voices
        self.available_voices = get_voices()
        
        # Set default voices if they don't exist
        if "male_voice" not in self.available_voices:
            self.default_male_voice = self.available_voices[0] if self.available_voices else "random"
        else:
            self.default_male_voice = "male_voice"
            
        if "female_voice" not in self.available_voices:
            # Try to find a female-sounding voice or use random
            female_voices = [v for v in self.available_voices if any(word in v.lower() for word in ['female', 'woman', 'lady', 'girl'])]
            self.default_female_voice = female_voices[0] if female_voices else "random"
        else:
            self.default_female_voice = "female_voice"
            
        logger.info(f"Available voices: {self.available_voices}")
        logger.info(f"Default male voice: {self.default_male_voice}")
        logger.info(f"Default female voice: {self.default_female_voice}")

    @lru_cache(maxsize=1)
    def get_available_voices(self) -> List[VoiceInfo]:
        """Get list of available voices with descriptions."""
        voices = []
        for voice in self.available_voices:
            description = "Custom voice" if voice != "random" else "Random generated voice"
            voices.append(VoiceInfo(
                voice_id=voice,
                name=voice.replace("_", " ").title(),
                description=description
            ))
        return voices

    def _map_speaker_to_voice(self, speaker: str, voice_mapping: Dict[str, str]) -> str:
        """Map speaker to voice, handling fallbacks."""
        raw_speaker = speaker.strip().lower()
        
        # First check the voice mapping
        voice = voice_mapping.get(raw_speaker)
        if voice and voice in self.available_voices:
            return voice
            
        # Fallback to default voices
        if "speaker-1" in raw_speaker or "male" in raw_speaker or "bob" in raw_speaker:
            return self.default_male_voice
        elif "speaker-2" in raw_speaker or "female" in raw_speaker or "kate" in raw_speaker:
            return self.default_female_voice
        else:
            # Default fallback
            return "random"

    async def process_job(self, job_id: str, request: TTSRequest):
        """Process TTS job with Tortoise."""
        with telemetry.tracer.start_as_current_span("tortoise.process_job") as span:
            try:
                job_manager.create_job(job_id)
                
                # Process dialogue
                combined_audio = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool, self._process_dialogue, request.dialogue, request.voice_mapping
                )
                
                if not isinstance(combined_audio, bytes):
                    raise Exception(f"TTS output is not bytes: {type(combined_audio)}")
                    
                job_manager.set_result(job_id, combined_audio)
                job_manager.update_status(job_id, JobStatus.COMPLETED, "Tortoise TTS generation completed")
                
            except Exception as e:
                logger.error(f"[Tortoise TTS ERROR] Job {job_id}: {e}")
                job_manager.update_status(job_id, JobStatus.FAILED, str(e))

    def _process_dialogue(self, dialogue: List[DialogueEntry], voice_mapping: Dict[str, str]) -> bytes:
        """Process dialogue entries using Tortoise TTS."""
        audio_segments = []
        
        for idx, entry in enumerate(dialogue):
            logger.info(f"[Tortoise TTS] Processing entry {idx+1}/{len(dialogue)}: {entry.speaker}")
            
            # Map speaker to voice
            voice = self._map_speaker_to_voice(entry.speaker, voice_mapping)
            
            # Clean text
            text = entry.text.strip()
            if not text:
                continue
                
            # Generate audio with Tortoise
            try:
                # Split long texts into smaller chunks if needed
                if len(text) > 500:
                    chunks = self._split_text(text, 500)
                    chunk_audios = []
                    for chunk in chunks:
                        audio = self._generate_audio_chunk(chunk, voice)
                        chunk_audios.append(audio)
                    # Concatenate chunks
                    combined_chunk = torch.cat(chunk_audios, dim=-1)
                    audio_segments.append(combined_chunk)
                else:
                    audio = self._generate_audio_chunk(text, voice)
                    audio_segments.append(audio)
                    
            except Exception as e:
                logger.error(f"Error generating audio for entry {idx}: {e}")
                # Create a small silence as fallback
                silence = torch.zeros(1, int(22050 * 0.5))  # 0.5 seconds of silence
                audio_segments.append(silence)
        
        # Concatenate all audio segments
        if audio_segments:
            full_audio = torch.cat(audio_segments, dim=-1)
            
            # Convert to numpy and save as WAV
            audio_np = full_audio.squeeze().cpu().numpy()
            
            # Normalize audio
            if audio_np.max() > 1.0 or audio_np.min() < -1.0:
                audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))
            
            # Convert to bytes
            buffer = io.BytesIO()
            sf.write(buffer, audio_np, 22050, format='WAV')
            buffer.seek(0)
            return buffer.read()
        else:
            # Return empty audio file
            buffer = io.BytesIO()
            sf.write(buffer, np.zeros(22050), 22050, format='WAV')
            buffer.seek(0)
            return buffer.read()

    def _generate_audio_chunk(self, text: str, voice: str) -> torch.Tensor:
        """Generate audio for a single text chunk."""
        if voice == "random":
            # Generate with random voice
            audio = self.tts.tts_with_preset(
                text,
                preset="fast",  # Can be 'ultra_fast', 'fast', 'standard', 'high_quality'
                voice_samples=None,
                conditioning_latents=None
            )
        else:
            # Use specific voice
            audio = self.tts.tts_with_preset(
                text,
                voice_samples=voice,
                preset="fast"
            )
        
        return audio

    def _split_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks at sentence boundaries."""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + sentence) < max_length:
                current_chunk += sentence + ". " if not sentence.endswith('.') else sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". " if not sentence.endswith('.') else sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks


# Initialize service
tts_service = TortoiseService()


@app.get("/voices")
async def list_voices() -> List[VoiceInfo]:
    """Get list of available voices."""
    return tts_service.get_available_voices()


@app.post("/generate_tts", status_code=202)
async def generate_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    """Start TTS generation job."""
    background_tasks.add_task(tts_service.process_job, request.job_id, request)
    return {"job_id": request.job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get job status."""
    status = job_manager.get_status(job_id)
    if not status:
        raise HTTPException(404, "Job not found")
    return status


@app.get("/output/{job_id}")
async def get_output(job_id: str):
    """Get the generated audio file."""
    data = job_manager.get_result(job_id)
    if not data:
        raise HTTPException(404, "Result not found")
    return Response(
        content=data,
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


@app.post("/cleanup")
async def cleanup_jobs():
    """Clean up old jobs."""
    removed = job_manager.cleanup_old_jobs()
    return {"removed": removed}


@app.get("/health")
async def health():
    """Health check endpoint."""
    voices = tts_service.get_available_voices()
    return {
        "status": "healthy", 
        "voices": len(voices), 
        "max_concurrent": MAX_CONCURRENT_REQUESTS,
        "model": "Tortoise TTS"
    }
