from typing import List, Dict, Any
from ..interfaces.processor import BaseProcessor
import re
from transformers import AutoTokenizer

class ChunkProcessor(BaseProcessor):
    """텍스트 청킹 처리 클래스"""

    def __init__(self, min_tokens: int = 100, max_tokens: int = 512):
        """
        Args:
            min_tokens: 청크당 최소 토큰 수
            max_tokens: 청크당 최대 토큰 수 (KB-ALBERT 제한)
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        # 로컬 KB-ALBERT 토크나이저 초기화
        tokenizer_path = "/Users/jungseokoh/Desktop/Cursor_prac_2024/product_factory/KB-Albert"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.section_patterns = [
            r'^\s*\d+[.]\s+',
            r'\n\s*\d+[.]\s+',
            r'^\s*[①②③④⑤⑥⑦⑧⑨]\s+',
            r'\n\s*[①②③④⑤⑥⑦⑧⑨]\s+',
            r'^\s*[-•★]\s+',
            r'\n\s*[-•★]\s+',
            r'\n{2,}'
        ]
        print("Tokenizer initialized successfully")  # 디버깅용

    def _get_token_length(self, text: str) -> int:
        """텍스트의 토큰 수를 안전하게 계산"""
        try:
            # 토큰화 시 max_length 제한 적용
            tokens = self.tokenizer.encode(
                text,
                max_length=self.max_tokens,
                truncation=True
            )
            return len(tokens) - 2  # [CLS], [SEP] 토큰 제외
        except Exception as e:
            print(f"Tokenization error: {e}")
            return self.max_tokens + 1  # 최대 토큰 수 초과 표시

    def process(self, text: str) -> Dict[str, Any]:
        """텍스트를 청크로 분할하고 메타데이터 생성"""
        if not text.strip():
            return {"chunks": [], "metadata": {"num_chunks": 0, "avg_chunk_size": 0}}

        # 전체 텍스트가 최소 크기보다 작더라도 의미 있는 섹션이면 처리
        chunks = self.create_chunks(text)
        return {
            "chunks": chunks,
            "metadata": {
                "num_chunks": len(chunks),
                "avg_chunk_size": sum(chunk["tokens"] for chunk in chunks) / len(chunks) if chunks else 0
            }
        }

    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        if not text.strip():
            return []

        # 전체 텍스트를 먼저 줄 단위로 분리
        lines = text.split('\n')
        chunks = []
        current_text = ""
        current_tokens = 0
        current_start = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 현재 줄의 토큰 수 계산
            line_tokens = self._get_token_length(line)
            
            # 현재 청크와 새 줄을 합쳤을 때의 토큰 수가 max_tokens를 초과하는 경우
            if current_tokens + line_tokens > self.max_tokens:
                if current_text:
                    chunks.append({
                        "text": current_text,
                        "start_pos": current_start,
                        "length": len(current_text),
                        "tokens": current_tokens
                    })
                current_text = line
                current_tokens = line_tokens
                current_start = text.find(line)
            else:
                if current_text:
                    current_text += "\n"
                current_text += line
                current_tokens += line_tokens

        # 마지막 청크 처리
        if current_text:
            chunks.append({
                "text": current_text,
                "start_pos": current_start,
                "length": len(current_text),
                "tokens": current_tokens
            })

        return chunks

    def _split_into_sections(self, text: str) -> List[Dict[str, Any]]:
        """텍스트를 섹션 단위로 분할"""
        # 시작 위치를 포함하여 모든 매칭 찾기
        matches = []
        for pattern in self.section_patterns:
            for match in re.finditer(pattern, text):
                matches.append(match.start())
        
        # 분할 위치 정렬
        split_points = sorted(set([0] + matches + [len(text)]))
        
        sections = []
        for i in range(len(split_points) - 1):
            start = split_points[i]
            end = split_points[i + 1]
            section_text = text[start:end].strip()
            if section_text:  # 빈 섹션 제외
                sections.append({
                    "text": section_text,
                    "start": start,
                    "end": end
                })
        
        return sections

    def _split_into_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분할"""
        sentences = re.split(r'[.!?]\s+', text)
        return [s.strip() for s in sentences if s.strip()]
