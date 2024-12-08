import os
from langchain_community.document_loaders import PyPDFLoader
import logging
from datetime import datetime
import re
import unicodedata
from agentic_chunker import AgenticChunker
from langchain.chains import create_extraction_chain
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Optional
import json

class PDFPreprocessor:
    def __init__(self):
        # 로깅 설정
        self._setup_logging()
        
        # 파일 경로 설정
        self.base_dir = "/Users/jungseokoh/Desktop/Cursor_prac_2024/product_factory"
        self.pdf_path = os.path.join(self.base_dir, "doc", "토스뱅크_가계대출_상품설명서.pdf")
        self.target_page = 1

        # AgenticChunker 초기화 - 프롬프트 수정
        self.chunker = AgenticChunker(openai_api_key=os.getenv('OPENAI_API_KEY'))
        
        # 청킹 조건 수정을 위한 프롬프트 오버라이드
        self.chunker._find_relevant_chunk = self._custom_find_relevant_chunk
        self.chunker.print_logging = True
        self.chunker.generate_new_metadata_ind = True

    def _custom_find_relevant_chunk(self, proposition):
        """커스텀 청킹 로직"""
        current_chunk_outline = self.chunker.get_chunk_outline()

        CUSTOM_PROMPT = ChatPromptTemplate.from_messages([
            (
                "system",
                """
                금융 상품 설명서의 텍스트를 의미 단위로 청킹하는 전문가입니다.
                다음 규칙에 따라 청크를 생성하거나 기존 청크에 추가하세요:

                1. 동일한 주제나 조건에 대한 설명은 반드시 같은 청크로 묶어야 합니다.
                2. 대출 한도, 금리, 상환 방식과 같은 핵심 조건들은 관련 내용을 모두 포함하여 하나의 청크로 만드세요.
                3. 체크박스(□) 항목이 연속되는 경우, 같은 주제의 체크박스들은 하나의 청크로 묶으세요.
                4. 제목과 그에 해당하는 상세 설명은 반드시 같은 청크에 포함해야 합니다.
                5. 날짜, 기간, 금액 정보는 반드시 관련 조건이나 설명과 함께 묶어야 합니다.
                6. 하나의 청크는 반드시 문장으로 구성되어야 합니다. 

                입력된 텍스트가 기존 청크의 주제나 조건과 관련이 있다면 해당 청크 ID를 반환하고,
                새로운 주제나 조건이라면 "No chunks"를 반환하세요.

                응답 형식:
                {
                    "chunk_id": "청크ID 또는 No chunks"
                }
                """,
            ),
            ("user", "Current Chunks:\n--Start of current chunks--\n{current_chunk_outline}\n--End of current chunks--"),
            ("user", "Determine if the following statement should belong to one of the chunks outlined:\n{proposition}"),
        ])

        # 구조화된 출력을 위한 함수 정의
        def get_chunk_id(text: str) -> dict:
            return {
                "chunk_id": str
            }

        try:
            # 구조화된 출력 사용
            response = self.chunker.llm.with_structured_output(get_chunk_id).invoke(
                CUSTOM_PROMPT.format_messages(
                    current_chunk_outline=current_chunk_outline,
                    proposition=proposition
                )
            )

            chunk_found = response.get("chunk_id", "No chunks")

            if chunk_found == "No chunks" or len(chunk_found) != self.chunker.id_truncate_limit:
                return None

            return chunk_found

        except Exception as e:
            self.logger.error(f"청크 ID 추출 중 오류 발생: {str(e)}")
            return None

    def _clean_text(self, text: str) -> list:
        """텍스트 정제 및 초기 분할"""
        self.logger.debug("원본 텍스트:\n%s", text)
        
        # 1. 기본 정제
        text = unicodedata.normalize('NFKC', text)
        
        # 2. 줄바꿈 처리 개선
        # 2.1 단어 중간 줄바꿈 수정
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        # 2.2 문장 중간 줄바꿈 수정
        text = re.sub(r'([가-힣])\s*\n\s*([가-힣])', r'\1\2', text)
        # 2.3 특수문자 주변 줄바꿈 수정
        text = re.sub(r'([가-힣])\s*\n\s*([.!?※])', r'\1\2', text)
        
        # 3. 의미 단위로 분할
        segments = []
        
        # 3.1 체크박스 항목 분리 (체크박스는 독립된 단위로 유지)
        checkbox_pattern = r'(□[^□\n]+?(?:예|아니오|동의|미동의|대상|비대상)[^□\n]*)'
        parts = re.split(checkbox_pattern, text)
        
        for part in parts:
            if re.match(checkbox_pattern, part):
                # 체크박스 항목은 그대로 보존
                segments.append(part.strip())
            else:
                # 나머지 텍스트는 추가 분할
                # 실제 문장 끝을 나타내는 구두점으로 분할
                sentences = re.split(r'([.!?])\s+(?=[가-힣])', part)
                current_sentence = ''
                for i in range(0, len(sentences)-1, 2):
                    if sentences[i].strip():
                        current_sentence = sentences[i].strip() + sentences[i+1]
                        if len(current_sentence) > 10:  # 최소 길이 체크
                            segments.append(current_sentence)
        
        # 4. 정제된 세그먼트 필터링 및 최종 정리
        segments = [
            re.sub(r'\s+', ' ', seg).strip()  # 연속된 공백을 하나로
            for seg in segments 
            if len(seg.strip()) > 10  # 너무 짧은 세그먼트 제외
        ]
        
        self.logger.debug("분할된 세그먼트:\n%s", '\n'.join(segments))
        return segments

    def process_pdf(self):
        """PDF 전처리 및 청킹 실행"""
        try:
            self.logger.info(f"PDF 처리 시작: {self.pdf_path}")
            
            # PDF 로딩
            loader = PyPDFLoader(self.pdf_path)
            pages = loader.load_and_split()
            
            if self.target_page > len(pages):
                raise ValueError(f"지정한 페이지({self.target_page})가 PDF 총 페이지 수({len(pages)})를 초과합니다.")
            
            # 1페이지만 처리
            page = pages[self.target_page - 1]
            self.logger.info(f"페이지 {self.target_page} 처리 중...")
            
            # 텍스트 전처리 및 세그먼트 분할
            segments = self._clean_text(page.page_content)
            self.logger.info(f"전처리 완료. {len(segments)}개의 세그먼트 생성")
            
            # 청킹 수행
            self.logger.info("청킹 시작...")
            self.chunker.add_propositions(segments)
            
            # 청킹 결과 출력
            self.logger.info("\n=== 청킹 결과 ===")
            self.chunker.pretty_print_chunks()
            
            # 결과 저장
            output_dir = os.path.join(self.base_dir, "chunks")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(
                output_dir, 
                f"chunks_page_{self.target_page}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'metadata': {
                        'source_file': self.pdf_path,
                        'page_number': self.target_page,
                        'processing_date': datetime.now().isoformat()
                    },
                    'segments': segments,
                    'chunks': self.chunker.get_chunks(get_type='dict')
                }, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"결과가 {output_file}에 저장되었습니다.")
            
            return self.chunker.get_chunks(get_type='dict')

        except Exception as e:
            self.logger.error(f"처리 중 오류 발생: {str(e)}", exc_info=True)
            raise

    def _setup_logging(self):
        """로깅 설정"""
        self.logger = logging.getLogger('PDFPreprocessor')
        self.logger.setLevel(logging.DEBUG)
        
        # 로그 파일 핸들
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(log_dir, f"preprocessing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 포맷터
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

def main():
    preprocessor = PDFPreprocessor()
    preprocessor.process_pdf()

if __name__ == "__main__":
    main() 