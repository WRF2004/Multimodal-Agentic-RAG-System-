"""OCR engine implementations."""

import asyncio
import structlog
from typing import Optional

from app.core.interfaces import OCRInterface
from app.core.registry import register_component, ComponentRegistry

logger = structlog.get_logger()


@register_component(ComponentRegistry.OCR, "tesseract")
class TesseractOCR(OCRInterface):

    def __init__(self, lang: str = "chi_sim+eng", **kwargs):
        self.lang = lang

    async def extract_text(self, image_path: str) -> Optional[str]:
        try:
            import pytesseract
            from PIL import Image
            loop = asyncio.get_event_loop()
            image = await loop.run_in_executor(None, Image.open, image_path)
            text = await loop.run_in_executor(
                None,
                lambda: pytesseract.image_to_string(image, lang=self.lang)
            )
            result = text.strip() if text else None
            if not result:
                logger.warning("ocr_returned_empty", path=image_path)
                return None
            return result
        except Exception as e:
            logger.error("ocr_failed", path=image_path, error=str(e))
            return None


@register_component(ComponentRegistry.OCR, "paddle")
class PaddleOCR_Engine(OCRInterface):

    def __init__(self, lang: str = "ch", **kwargs):
        self._ocr = None
        self.lang = lang

    def _init_ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)

    async def extract_text(self, image_path: str) -> Optional[str]:
        try:
            self._init_ocr()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, lambda: self._ocr.ocr(image_path, cls=True)
            )
            if not result or not result[0]:
                logger.warning("paddle_ocr_empty", path=image_path)
                return None
            texts = []
            for line in result[0]:
                if line and len(line) >= 2 and line[1]:
                    text_content = line[1][0] if isinstance(line[1], (list, tuple)) else str(line[1])
                    texts.append(text_content)
            return "\n".join(texts) if texts else None
        except Exception as e:
            logger.error("paddle_ocr_failed", path=image_path, error=str(e))
            return None