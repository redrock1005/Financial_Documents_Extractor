import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 명시적으로 병렬 처리 비활성화

import sys
from dotenv import load_dotenv
from processors.pdf import PDFProcessor
from processors.chunk import ChunkProcessor
from embedders.kb_albert import KBAlbertEmbedder
from sklearn.cluster import KMeans
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
from openai import OpenAI
from typing import List, Tuple, Optional, Dict, Any
import pandas as pd
import logging
from tiktoken import get_encoding
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import json
from agentic_chunker import AgenticChunker
from langchain.schema import Document

# 새로 분리한 클래스들 import
from extractors.financial_product_extractor import FinancialProductExtractor
from storage.sheets_manager import GoogleSheetsManager

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

class DocumentProcessor:
    def __init__(self):
        try:
            load_dotenv('/Users/jungseokoh/Desktop/Cursor_prac_2024/product_factory/.env')
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
            if not self.openai_api_key:
                raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
            
            # 기존 프로세서들
            self.pdf_processor = PDFProcessor()
            self.chunk_processor = ChunkProcessor()
            self.embedder = KBAlbertEmbedder()
            
            # 새로운 컴포넌트들 추가
            self.financial_extractor = FinancialProductExtractor(api_key=self.openai_api_key)
            self.sheets_manager = GoogleSheetsManager()
            
            print("DocumentProcessor 초기화 완료")
            
        except Exception as e:
            print(f"초기화 중 오류 발생: {e}")
            raise

    def process_document(self, pdf_path: str) -> Dict[str, Any]:
        """개선된 문서 처리 파이프라인"""
        try:
            # 1. PDF 존재 확인
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            
            # 2. PDF 텍스트 추출 및 벡터 저장소 생성
            vector_store = self.pdf_processor.process(pdf_path)
            print("PDF 처리 및 벡터 저장소 생성 완료")

            # 3. 금융 상품 정보 추출
            classified_df, unclassified_df = self.financial_extractor.extract_factors(vector_store)
            print("금융 상품 정보 추출 완료")

            # 4. 구글 시트에 저장
            sheet_id = self.sheets_manager.save_to_sheets(
                classified_df, 
                unclassified_df
            )
            print(f"구글 시트 저장 완료: {sheet_id}")

            return {
                "sheet_id": sheet_id,
                "sheet_url": f"https://docs.google.com/spreadsheets/d/{sheet_id}",
                "classified_data": classified_df.to_dict(),
                "unclassified_data": unclassified_df.to_dict()
            }

        except Exception as e:
            print(f"문서 처리 중 오류 발생: {str(e)}")
            raise

class PDFQuestionAnswering:
    def __init__(self, doc_dir: str, index_path: str = None):
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
            
        self.doc_dir = doc_dir
        self.index_path = index_path
        
        # Google Sheets Manager 초기화 - 경로 수정
        credentials_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # src의 상위 디렉토리로 이동
            'credentials',
            'google_sheets_credentials.json'
        )
        print(f"인증 파일 경로: {credentials_path}")  # 디버깅용
        self.sheets_manager = GoogleSheetsManager(credentials_path)
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        
        # 각 컴포넌트 초기화
        self.pdf_processor = PDFProcessor()
        self.extractor = FinancialProductExtractor(api_key=self.api_key)

    def process_pdf(self) -> None:
        """PDF 문서 처리 및 벡터 저장소 생성/업데이트"""
        try:
            loader = PyPDFLoader(self.doc_dir)
            self.documents = loader.load()  # documents 저장
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_documents(self.documents)
            
            self.vector_store = FAISS.from_documents(
                texts,
                self.embeddings
            )
            print("\n✅ 벡터 저장소가 성공적으로 생성/업데이트되었습니다.")
                
        except Exception as e:
            logging.error(f"PDF 처리 중 오류 발생: {str(e)}")
            raise
    
    def setup_qa_system(self, 
                       temperature: float = 0,
                       search_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """QA 시스템 설정"""
        if not hasattr(self, 'vector_store'):
            raise ValueError("벡터 저장소가 초기화되지 않았습니다. process_pdf()를 먼저 실행해주세요.")
            
        if search_kwargs is None:
            search_kwargs = {
                'k': 10,
                'score_threshold': 0.5
            }
            
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model="gpt-3.5-turbo-16k",
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=temperature
        )
        
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs=search_kwargs
            ),
            return_source_documents=True
        )

    def analyze_and_save(self):
        """문서 분석 및 결과 저장"""
        try:
            logging.info("\n=== PDF 분석 및 구조화 시작 ===")
            self.setup_qa_system()
            
            # 벡터 저장소에서 문서 내용 가져오기
            documents = self.vector_store.similarity_search("", k=100)
            doc_contents = [doc.page_content for doc in documents]
            logging.info(f"총 {len(doc_contents)}개의 문서 청크 검색됨")
            
            # FinancialProductExtractor를 통해 정보 추출 및 DataFrame 생성
            classified_df = self.extractor.extract_factors(doc_contents)
            logging.info(f"추출된 DataFrame 크기: {classified_df.shape}")
            
            if classified_df.empty:
                logging.warning("추출된 DataFrame이 비어있습니다")
            
            logging.info("\n=== 구조화된 데이터 구글 시트 저장 중 ===")
            sheet_id = self.sheets_manager.save_to_sheets(classified_df, pd.DataFrame())
            logging.info(f"\n✅ 분석 완료! 구글 시트 ID: {sheet_id}")
            logging.info(f"https://docs.google.com/spreadsheets/d/{sheet_id}")
            
        except Exception as e:
            logging.error(f"분석 및 저장 중 오류 발생: {str(e)}")
            raise

    def extract_loan_factors(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        try:
            # ... (기존 코드 유지) ...
            
            return classified_df, pd.DataFrame()

        except Exception as e:
            logging.error(f"결과 처리 중 오류 발생: {str(e)}")
            print(f"\n상세 오류 정보: {str(e)}")
            # 오류 발생 시 빈 DataFrame 반환
            return pd.DataFrame(columns=["구분", "대분류", "중분류", "소분류", "상세"]), pd.DataFrame()

    def save_to_sheets(self, classified_df: pd.DataFrame, unclassified_df: pd.DataFrame):
        """결과를 구글 시트에 저장"""
        try:
            sheet_id = os.getenv('GOOGLE_SHEETS_ID')
            print(f"읽어온 GOOGLE_SHEETS_ID: {sheet_id}")
            
            if not sheet_id:
                raise ValueError("GOOGLE_SHEETS_ID가 .env 파일에 설정되지 않았습니다.")
            
            # 분류된 항목 저장
            if isinstance(classified_df, pd.DataFrame) and not classified_df.empty:
                values = [classified_df.columns.tolist()] + classified_df.values.tolist()
                self.sheets_manager.update_values(sheet_id, '시트1!A1', values)
                
            # 미분류 항목 저장
            if isinstance(unclassified_df, pd.DataFrame) and not unclassified_df.empty:
                values = [unclassified_df.columns.tolist()] + unclassified_df.values.tolist()
                self.sheets_manager.update_values(sheet_id, '시트1!H1', values)
                
            return sheet_id
            
        except Exception as e:
            logging.error(f"구글 시트 저장 중 오류 발생: {str(e)}")
            raise

def main():
    # 절대 경로 설정
    doc_dir = "/Users/jungseokoh/Desktop/Cursor_prac_2024/product_factory/doc/카카오뱅크_가계대출_상품설명서.pdf"
    
    # doc 디렉토리가 없으면 생성
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)
        print(f"doc 디렉토리가 생성되었습니다: {doc_dir}")
    
    # QA 시스템 초기화 및 실행
    qa_system = PDFQuestionAnswering(doc_dir)
    qa_system.process_pdf()
    qa_system.setup_qa_system()
    qa_system.analyze_and_save()

if __name__ == "__main__":
    main() 