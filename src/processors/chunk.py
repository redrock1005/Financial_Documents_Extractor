from typing import List, Dict, Any
from interfaces.processor import BaseProcessor
from openai import OpenAI
import os
from dotenv import load_dotenv
import tiktoken
import json

class ChunkProcessor(BaseProcessor):
    def __init__(self, max_tokens: int = 512):
        """초기화"""
        # OpenAI 클라이언트 초기화
        load_dotenv('/Users/jungseokoh/Desktop/Cursor_prac_2024/product_factory/.env')
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        self.client = OpenAI(api_key=api_key)
        
        # GPT 토크나이저 초기화
        self.gpt_tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
        self.max_tokens = max_tokens
        print("ChunkProcessor 초기화 완료")

    def _get_token_length(self, text: str) -> int:
        """GPT 토크나이저로 토큰 수 계산"""
        return len(self.gpt_tokenizer.encode(text))

    def _split_long_chunk(self, chunk: str) -> List[str]:
        """긴 청크를 토큰 제한에 맞게 분할"""
        sub_chunks = []
        current_chunk = ""
        current_tokens = 0
        
        # 1. 문장 단위로 1차 분할
        sentences = [s.strip() + '.' for s in chunk.split('.') if s.strip()]
        
        for sentence in sentences:
            sentence_tokens = self._get_token_length(sentence)
            
            # 단일 문장이 토큰 제한을 초과하는 경우
            if sentence_tokens > self.max_tokens:
                if current_chunk:
                    sub_chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0
                
                # 문장을 더 작은 단위로 분할 (구두점, 접속사 등 기준)
                delimiters = [',', '그리고', '또한', '하지만', '따라서', '그러나']
                temp_parts = [sentence]
                
                for delimiter in delimiters:
                    new_parts = []
                    for part in temp_parts:
                        if self._get_token_length(part) > self.max_tokens:
                            split_parts = [p.strip() for p in part.split(delimiter) if p.strip()]
                            new_parts.extend(split_parts)
                        else:
                            new_parts.append(part)
                    temp_parts = new_parts
                
                # 여전히 긴 부분이 있다면 강제 분할
                for part in temp_parts:
                    part_tokens = self._get_token_length(part)
                    if part_tokens > self.max_tokens:
                        words = part.split()
                        temp_chunk = ""
                        temp_tokens = 0
                        
                        for word in words:
                            word_tokens = self._get_token_length(word)
                            if temp_tokens + word_tokens <= self.max_tokens:
                                temp_chunk += word + " "
                                temp_tokens += word_tokens
                            else:
                                if temp_chunk:
                                    sub_chunks.append(temp_chunk.strip())
                                temp_chunk = word + " "
                                temp_tokens = word_tokens
                        
                        if temp_chunk:
                            sub_chunks.append(temp_chunk.strip())
                    else:
                        sub_chunks.append(part)
                    
            # 현재 청크에 문장을 추가할 수 있는 경우
            elif current_tokens + sentence_tokens <= self.max_tokens:
                current_chunk += sentence + " "
                current_tokens += sentence_tokens
            
            # 새로운 청크 시작
            else:
                if current_chunk:
                    sub_chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
                current_tokens = sentence_tokens
        
        # 마지막 청크 처리
        if current_chunk:
            sub_chunks.append(current_chunk.strip())
        
        return sub_chunks

    def _create_semantic_chunks(self, text: str) -> List[str]:
        """GPT를 사용하여 의미 단위로 텍스트 분할"""
        try:
            prompt = """
            주어진 텍스트를 의미 단위로 분할해주세요. 다음 규칙을 따라주세요:
            1. 각 청크는 하나의 완결된 의미 단위여야 합니다.
            2. 문맥의 연속성을 최대한 유지해주세요.
            3. 각 청크는 구분자 '###'로 분리해주세요.
            4. 원문의 내용만 사용하고 새로운 내용을 추가하지 마세요.
            5. 모든 청크가 의미적으로 독립적으로 이해될 수 있어야 합니다.

            텍스트:
            {text}

            위 텍스트를 의미 단위로 분할하여 '###' 구분자로 구분된 형태로 출력해주세요.
            """

            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{
                    "role": "system",
                    "content": "당신은 은행 상품 설명서를 분석하고 의미 단위로 분류하는 전문가입니다."
                },
                {
                    "role": "user",
                    "content": prompt.format(text=text)
                }],
                temperature=0.3
            )

            # GPT 응답에서 청크 추출
            chunks_text = response.choices[0].message.content.strip()
            chunks = [chunk.strip() for chunk in chunks_text.split('###') if chunk.strip()]
            
            print(f"GPT가 생성한 청크 수: {len(chunks)}")
            return chunks

        except Exception as e:
            print(f"GPT 청킹 중 오류 발생: {e}")
            # 오류 발생 시 단순 문장 분할로 폴백
            return [s.strip() + '.' for s in text.split('.') if s.strip()]

    def process(self, text: str, file_name: str) -> Dict[str, Any]:
        """텍스트를 의미 단위로 청킹"""
        try:
            # 1. 의미 단위로 청크 생성
            semantic_chunks = self._create_semantic_chunks(text)
            
            # 2. 토큰 제한을 초과하는 청크 재분할
            final_chunks = []
            for chunk in semantic_chunks:
                if self._get_token_length(chunk) > self.max_tokens:
                    # 토큰 제한 초과 시 더 작은 단위로 분할
                    sub_chunks = self._split_long_chunk(chunk)
                    final_chunks.extend(sub_chunks)
                else:
                    final_chunks.append(chunk)
            
            # 3. 결과 구조화
            result = {
                "chunks": [
                    {
                        "text": chunk,
                        "tokens": self._get_token_length(chunk)
                    } for chunk in final_chunks
                ],
                "metadata": {
                    "num_chunks": len(final_chunks),
                    "max_tokens": self.max_tokens
                }
            }
            
            # 4. 결과 저장
            chunks_path = os.path.join(self.chunks_dir, f"{file_name}.json")
            with open(chunks_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            return result

        except Exception as e:
            print(f"청킹 중 오류 발생: {e}")
            raise
