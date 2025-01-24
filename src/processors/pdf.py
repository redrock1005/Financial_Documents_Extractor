from typing import Dict, Any
import pdfplumber
import re
from interfaces.processor import BaseProcessor
import os
import json
import logging

class PDFProcessor(BaseProcessor):
    def __init__(self):
        """PDF 프로세서 초기화"""
        # 기본 디렉토리 설정
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(self.base_dir, "processed_data")
        self.text_dir = os.path.join(self.output_dir, "extracted_text")
        self.chunks_dir = os.path.join(self.output_dir, "chunks")
        
        # 필요한 디렉토리 생성
        for dir_path in [self.output_dir, self.text_dir, self.chunks_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
        self.processed_files_path = os.path.join(self.output_dir, "processed_files.json")
        self.processed_files = self._load_processed_files()

    def process(self, file_path: str) -> Dict[str, Any]:
        """PDF 파일 처리 및 결과 반환"""
        try:
            file_name = os.path.basename(file_path)
            text_path = os.path.join(self.text_dir, f"{file_name}.txt")
            chunks_path = os.path.join(self.chunks_dir, f"{file_name}.json")

            # 이미 처리된 파일인지 확인
            if file_path in self.processed_files:
                print(f"이미 처리된 파일입니다: {file_path}")
                # 저장된 결과 반환
                with open(text_path, 'r', encoding='utf-8') as f:
                    extracted_text = f.read()
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                return {
                    "text": extracted_text,
                    "chunks": chunks_data,
                    "is_cached": True
                }

            # 새로운 파일 처리
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                
                cleaned_text = self._clean_text(text)
                
                # 텍스트 저장
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)
                
                # 처리된 파일 목록에 추가
                self.processed_files.append(file_path)
                self._save_processed_files()
                
                return {
                    "text": cleaned_text,
                    "chunks": None,  # ChunkProcessor에서 처리 예정
                    "is_cached": False
                }

        except Exception as e:
            print(f"PDF 처리 중 오류 발생: {e}")
            return {
                "text": "",
                "chunks": None,
                "is_cached": False,
                "error": str(e)
            }

    def _clean_text(self, text: str) -> str:
        """추출된 텍스트 정제"""
        if not text:
            return ""
        
        # 불필요한 공백 제거
        text = " ".join(text.split())
        
        # 특수문자 처리
        text = re.sub(r'[^\w\s\.,\-\(\)\/]', '', text)
        
        # 연속된 줄바꿈 정리
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # 불필요한 공백 제거
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def _load_processed_files(self):
        """이미 처리된 파일 목록을 로드"""
        if os.path.exists(self.processed_files_path):
            with open(self.processed_files_path, 'r') as f:
                return json.load(f)
        else:
            return []

    def _save_processed_files(self):
        """처리된 파일 목록 저장"""
        try:
            with open(self.processed_files_path, 'w', encoding='utf-8') as f:
                json.dump(self.processed_files, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"처리된 파일 목록 저장 중 오류 발생: {e}")
