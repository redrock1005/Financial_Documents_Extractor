# src/utils/tokenizer.py
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer
import re

class TokenizerUtils:
    def __init__(self, model_path: str = "/Users/jungseokoh/Desktop/Cursor_prac_2024/product_factory/KB-Albert"):
        """토크나이저 유틸리티 초기화"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_length = 512

    def tokenize(self, text: str) -> Dict[str, Any]:
        """텍스트를 토큰화하고 관련 정보 반환"""
        if not text.strip():
            return {
                "tokens": [],
                "token_ids": [],
                "num_tokens": 0
            }

        # 토큰화
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )

        return {
            "tokens": self.tokenizer.tokenize(text),
            "token_ids": encoding["input_ids"],
            "num_tokens": len(encoding["input_ids"])
        }

    def batch_tokenize(self, texts: List[str], batch_size: int = 32) -> List[Dict[str, Any]]:
        """여러 텍스트를 배치로 토큰화"""
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            results.extend([self.tokenize(text) for text in batch])
        return results

    def get_token_length(self, text: str) -> int:
        """텍스트의 토큰 수 반환"""
        return self.tokenize(text)["num_tokens"]

    def split_by_token_length(self, text: str, max_tokens: int = 512) -> List[str]:
        """텍스트를 토큰 수 기준으로 분할"""
        if not text.strip():
            return []

        sentences = self._split_into_sentences(text)
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self.get_token_length(sentence)
            
            if sentence_tokens > max_tokens:
                # 긴 문장은 단어 단위로 분할
                words = sentence.split()
                temp_chunk = ""
                temp_tokens = 0
                
                for word in words:
                    word_tokens = self.get_token_length(word)
                    if temp_tokens + word_tokens <= max_tokens:
                        temp_chunk = (temp_chunk + " " + word).strip()
                        temp_tokens += word_tokens
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = word
                        temp_tokens = word_tokens
                
                if temp_chunk:
                    chunks.append(temp_chunk)
                continue

            if current_tokens + sentence_tokens <= max_tokens:
                current_chunk = (current_chunk + " " + sentence).strip()
                current_tokens += sentence_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
                current_tokens = sentence_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분할"""
        # 문장 구분자 패턴
        pattern = r'[.!?]\s+'
        
        # 문장 분할
        sentences = re.split(pattern, text)
        
        # 빈 문장 제거 및 정리
        return [s.strip() for s in sentences if s.strip()]