"""ASR engine implementations."""

import asyncio
import structlog
from typing import Optional

from app.core.interfaces import ASRInterface
from app.core.registry import register_component, ComponentRegistry

logger = structlog.get_logger()


@register_component(ComponentRegistry.ASR, "whisper")
class WhisperASR(ASRInterface):

    def __init__(self, model_size: str = "base", **kwargs):
        self._model = None
        self._model_size = model_size

    def _load_model(self):
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self._model_size)
            logger.info("whisper_loaded", size=self._model_size)

    async def transcribe(self, audio_path: str) -> Optional[str]:
        try:
            self._load_model()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._model.transcribe(audio_path)
            )
            text = result.get("text", "").strip() if result else None
            if not text:
                logger.warning("asr_returned_empty", path=audio_path)
                return None
            return text
        except Exception as e:
            logger.error("asr_failed", path=audio_path, error=str(e))
            return None


@register_component(ComponentRegistry.ASR, "api")
class APIASR(ASRInterface):
    """OpenAI-compatible Whisper API."""

    def __init__(self, base_url: str = "https://api.openai.com/v1", api_key: str = "", **kwargs):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key or "no-key")

    async def transcribe(self, audio_path: str) -> Optional[str]:
        try:
            with open(audio_path, "rb") as f:
                response = await self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                )
            return response.text if response.text else None
        except Exception as e:
            logger.error("api_asr_failed", path=audio_path, error=str(e))
            return None