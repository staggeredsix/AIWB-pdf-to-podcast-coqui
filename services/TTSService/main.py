from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import List, Dict, Optional
import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import torch
import torchaudio
import soundfile as sf
from speechbrain.pretrained import Tacotron2, HIFIGAN, SpeakerRecognition

from shared.api_types import ServiceType, JobStatus
from shared.job import JobStatusManager
from shared.otel import OpenTelemetryInstrumentation, OpenTelemetryConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SpeechBrain TTS Service", debug=True)
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))

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

device = "cuda" if torch.cuda.is_available() else "cpu"


class YourTTSWrapper:
    def __init__(self, tts, vocoder, speaker_encoder):
        self.tts = tts
        self.vocoder = vocoder
        self.speaker_encoder = speaker_encoder

    def encode_speaker(self, audio_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if audio_tensor.dim() == 2 and audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, 16000)
        if audio_tensor.shape[1] > 16000 * 20:
            audio_tensor = audio_tensor[:, :16000 * 20]
        embedding = self.speaker_encoder.encode_batch(audio_tensor.to(device))
        return embedding.squeeze(0)

    def generate(self, text: str, speaker_embedding: torch.Tensor) -> torch.Tensor:
        mel, _, _ = self.tts.encode_text(text, speaker_embedding.unsqueeze(0))
        waveform = self.vocoder.decode_batch(mel)
        return waveform.squeeze(0)


tts = Tacotron2.from_hparams(
    source="speechbrain/tts-tacotron2-ljspeech",
    savedir="models/tts",
    run_opts={"device": device},
)
vocoder = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech",
    savedir="models/vocoder",
    run_opts={"device": device},
)
speaker_encoder = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="models/encoder",
    run_opts={"device": device},
)

model = YourTTSWrapper(tts, vocoder, speaker_encoder)


class DialogueEntry(BaseModel):
    text: str
    speaker: str
    voice_id: Optional[str] = None


class TTSRequest(BaseModel):
    dialogue: List[DialogueEntry]
    job_id: str
    voice_mapping: Dict[str, str]


class TTSService:
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS)

    def _process_dialogue(self, dialogue: List[DialogueEntry], mapping: Dict[str, str]) -> bytes:
        audio_segments = []
        sample_rate = 22050
        for entry in dialogue:
            emb_path = mapping.get(entry.speaker)
            if not emb_path or not os.path.exists(emb_path):
                raise Exception(f"Embedding file for {entry.speaker} not found")
            embedding = torch.load(emb_path, map_location=device)
            wav = model.generate(entry.text, embedding)
            audio_segments.append(wav.cpu())
        full_audio = torch.cat(audio_segments, dim=-1)
        buf = BytesIO()
        sf.write(buf, full_audio.numpy(), sample_rate, format="WAV")
        buf.seek(0)
        return buf.read()

    async def process_job(self, job_id: str, request: TTSRequest):
        with telemetry.tracer.start_as_current_span("tts.process_job"):
            try:
                job_manager.create_job(job_id)
                combined = await asyncio.get_event_loop().run_in_executor(
                    None, self._process_dialogue, request.dialogue, request.voice_mapping
                )
                job_manager.set_result(job_id, combined)
                job_manager.update_status(job_id, JobStatus.COMPLETED, "Done")
            except Exception as e:
                logger.error(f"[TTS ERROR] Job {job_id}: {e}")
                job_manager.update_status(job_id, JobStatus.FAILED, str(e))


tts_service = TTSService()


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
    return Response(content=data, media_type="audio/wav")


@app.get("/health")
async def health():
    return {"status": "healthy"}
