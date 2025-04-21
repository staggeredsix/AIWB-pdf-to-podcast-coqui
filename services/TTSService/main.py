# services/TTSService/main.py

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

import torch
from TTS.api import TTS
import soundfile as sf
from io import BytesIO

from shared.api_types import ServiceType, JobStatus
from shared.job import JobStatusManager
from shared.otel import OpenTelemetryInstrumentation, OpenTelemetryConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Coqui TTS Service", debug=True)

MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))

# Accept flexible speaker roles
SPEAKER_MAPPING = {
    "bob": "p230",
    "kate": "p225",
    "speaker-1": "p230",
    "speaker-2": "p225",
    "speaker1": "p230",
    "speaker2": "p225",
}

telemetry = OpenTelemetryInstrumentation()
telemetry.initialize(
    OpenTelemetryConfig(
        service_name="tts-service",
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
    voice_mapping: Optional[Dict[str, str]] = SPEAKER_MAPPING


class VoiceInfo(BaseModel):
    voice_id: str
    name: str
    description: Optional[str] = None


class TTSService:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tts_model = TTS(
            model_name="tts_models/en/vctk/vits",
            progress_bar=False,
        ).to(device)
        self.valid_speakers = self.tts_model.speakers
        logger.info(f"[Init] Available speakers: {self.valid_speakers}")

    @lru_cache(maxsize=1)
    def get_available_voices(self) -> List[VoiceInfo]:
        return [
            VoiceInfo(voice_id="p230", name="Bob", description="Male speaker"),
            VoiceInfo(voice_id="p225", name="Kate", description="Female speaker"),
        ]

    async def process_job(self, job_id: str, request: TTSRequest):
        with telemetry.tracer.start_as_current_span("tts.process_job") as span:
            try:
                job_manager.create_job(job_id)
                combined = await asyncio.get_event_loop().run_in_executor(
                    None, self._process_dialogue, request.dialogue
                )
                if not isinstance(combined, (bytes, bytearray)):
                    raise Exception(f"TTS output is not bytes: {type(combined)}, value: {combined}")
                job_manager.set_result(job_id, combined)
                job_manager.update_status(job_id, JobStatus.COMPLETED, "Done")
            except Exception as e:
                logger.error(f"[TTS ERROR] Job {job_id}: {e}")
                job_manager.update_status(job_id, JobStatus.FAILED, str(e))

    def _process_dialogue(self, dialogue: List[DialogueEntry]) -> bytes:
        sample_rate = self.tts_model.synthesizer.output_sample_rate
        all_audio = []

        for idx, entry in enumerate(dialogue):
            raw_speaker = entry.speaker.strip().lower()
            speaker_id = SPEAKER_MAPPING.get(raw_speaker)

            if speaker_id is None or speaker_id not in self.valid_speakers:
                raise Exception(f"Invalid speaker ID '{speaker_id}' for role '{entry.speaker}'")

            cleaned_text = bytes(entry.text, "utf-8").decode("unicode_escape").replace("\n", " ").strip()

            logger.info(f"[TTS] Line {idx+1} → {entry.speaker} ({speaker_id}): {cleaned_text}")

            wav = self.tts_model.tts(cleaned_text, speaker=speaker_id)

            if not isinstance(wav, (list, np.ndarray)):
                raise Exception(f"TTS output invalid for {entry.speaker}: {wav}")

            logger.info(f"[TTS] Line {idx+1} → generated {len(wav)} samples")
            all_audio.append(wav)

        full_audio = np.concatenate(all_audio)
        buf = BytesIO()
        sf.write(buf, full_audio, sample_rate, format="WAV")
        buf.seek(0)
        return buf.read()


tts_service = TTSService()


@app.get("/voices")
async def list_voices() -> List[VoiceInfo]:
    return tts_service.get_available_voices()


@app.post("/generate_tts", status_code=202)
async def generate_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(tts_service.process_job, request.job_id, request)
    return {"job_id": request.job_id}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    status = job_manager.get_status(job_id)
    if not status:
        raise HTTPException(404, "Job not found")
    return status


@app.get("/output/{job_id}")
async def get_output(job_id: str):
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
    removed = job_manager.cleanup_old_jobs()
    return {"removed": removed}


@app.get("/health")
async def health():
    voices = tts_service.get_available_voices()
    return {"status": "healthy", "voices": len(voices), "max": MAX_CONCURRENT_REQUESTS}

