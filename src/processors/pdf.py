from typing import Dict, Any
import pdfplumber
from ..interfaces.processor import BaseProcessor

class PDFProcessor(BaseProcessor):
    def __init__(self):
        """PDF 프로세서 초기화"""
        pass

    def process(self, file_path: str) -> str:
        """PDF 파일에서 텍스트 추출"""
        try:
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"PDF 처리 중 오류 발생: {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        """추출된 텍스트 정제"""
        if not text:
            return ""
        
        # 불필요한 공백 제거
        text = " ".join(text.split())
        
        # 기본적인 텍스트 정제
        text = text.replace("\x0c", "\n")  # 페이지 구분자 처리
        
        return text.strip()
