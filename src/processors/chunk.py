from typing import List, Dict, Any
from interfaces.processor import BaseProcessor
from openai import OpenAI
import os
from dotenv import load_dotenv
import tiktoken

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
        """긴 청크를 의미 단위로 분할"""
        # 문장 단위로 분할
        sentences = [s.strip() + '.' for s in chunk.split('.') if s.strip()]
        sub_chunks = []
        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._get_token_length(sentence)
            
            # 단일 문장이 토큰 제한을 초과하는 경우
            if sentence_tokens > self.max_tokens:
                if current_chunk:
                    sub_chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0
                
                # 문장을 더 작은 단위(구두점 기준)로 분할
                sub_sentences = [s.strip() + ',' for s in sentence.split(',') if s.strip()]
                temp_chunk = ""
                temp_tokens = 0
                
                for sub_sent in sub_sentences:
                    sub_tokens = self._get_token_length(sub_sent)
                    if temp_tokens + sub_tokens <= self.max_tokens:
                        temp_chunk += sub_sent + " "
                        temp_tokens += sub_tokens
                    else:
                        if temp_chunk:
                            sub_chunks.append(temp_chunk.strip())
                        temp_chunk = sub_sent + " "
                        temp_tokens = sub_tokens
                
                if temp_chunk:
                    sub_chunks.append(temp_chunk.strip())
                    
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

    def process(self, text: str) -> Dict[str, Any]:
        """텍스트를 의미 단위로 청킹"""
        if not text.strip():
            return {"chunks": [], "metadata": {"num_chunks": 0}}

        print("\n=== 청킹 프로세스 시작 ===")
        
        # 1. GPT를 사용하여 의미 단위로 청킹
        print("1. GPT 의미 단위 분할 시작")
        initial_chunks = self._create_semantic_chunks(text)
        print(f"   의미 단위 분할 완료: {len(initial_chunks)}개의 청크 생성")
        
        # 2. 토큰 수 확인 및 필요시 재분할
        print("\n2. 토큰 수 검증 및 재분할 시작")
        final_chunks = []
        for i, chunk in enumerate(initial_chunks, 1):
            chunk_tokens = self._get_token_length(chunk)
            print(f"   청크 {i}/{len(initial_chunks)} 검사 중 (토큰 수: {chunk_tokens})")
            
            if chunk_tokens <= self.max_tokens:
                final_chunks.append(chunk)
            else:
                print(f"   청크 {i} 재분할 필요 (토큰 수: {chunk_tokens} > {self.max_tokens})")
                sub_chunks = self._split_long_chunk(chunk)
                print(f"   재분할 결과: {len(sub_chunks)}개의 서브 청크 생성")
                final_chunks.extend(sub_chunks)
        
        # 3. 최종 검증
        print(f"\n3. 최종 검증")
        validated_chunks = []
        for i, chunk in enumerate(final_chunks, 1):
            tokens = self._get_token_length(chunk)
            if tokens <= self.max_tokens:
                validated_chunks.append(chunk)
            else:
                print(f"   경고: 청크 {i}가 여전히 토큰 제한 초과 ({tokens} > {self.max_tokens})")
                sub_chunks = self._split_long_chunk(chunk)
                validated_chunks.extend(sub_chunks)
        
        print(f"\n=== 청킹 완료 ===")
        print(f"최종 청크 수: {len(validated_chunks)}")
        
        return {
            "chunks": [{"text": chunk, "tokens": self._get_token_length(chunk)} 
                      for chunk in validated_chunks],
            "metadata": {"num_chunks": len(validated_chunks)}
        }
