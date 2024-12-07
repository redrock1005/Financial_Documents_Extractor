import os
from dotenv import load_dotenv
from tiktoken import get_encoding
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Optional, Dict, Any, Set
import logging
import json
from sheets_manager import GoogleSheetsManager
import pandas as pd
from typing import List, Dict, Tuple
from agentic_chunker import AgenticChunker
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import openai

class PDFQuestionAnswering:
    def __init__(self, doc_dir: str, index_path: str = None):
        load_dotenv()
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY가 설정되지 않았습니다.")
            
        self.doc_dir = doc_dir
        self.index_path = index_path
        
        if index_path:
            self.metadata_path = os.path.join(
                os.path.dirname(index_path), 
                "processed_files.json"
            )
        else:
            self.metadata_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "processed_files.json"
            )
            
        self.processed_files = self.load_processed_files() if index_path else set()
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        self.chunker = AgenticChunker(openai_api_key=self.api_key)
        
        # 프롬프트 수정
        self.prompt = """당신은 은행에서 15년 이상 근무한 가계대출 전산 전문가입니다. 
주어진 대출 상품 설명서를 분석하여 금융상품을 구성하는 속성들을 의미에 따라 체계적으로 분류해주세요.

다음 규칙을 반드시 준수해주세요:
1. 모든 속성은 명사형으로 작성
2. 최대 5단계까지의 위계로 구분 가능 (구분/대분류/중분류/소분류/상세)
3. 분류체계에 포함시키기 어려운 속성은 '구분' 열에만 내용을 작성하고 나머지 열은 비워두기

결과는 반드시 다음과 같은 JSON 형식으로 반환해주세요:
{
    "classified": [
        {
            "구분": "",
            "대분류": "",
            "중분류": "",
            "소분류": "",
            "상세": ""
        }
    ]
}

위 예시를 참고하여, 벡터 저장소의 내용을 체계적으로 분석해주세요."""
        
    def process_pdf(self) -> None:
        """PDF 문서 처리 및 벡터 저장소 생성/업데이트"""
        try:
            new_pdf_files = self.get_new_pdf_files()
            
            if not new_pdf_files:
                print("\n처리할 새로운 PDF 파일이 없습니다.")
                if self.index_path and os.path.exists(os.path.join(os.path.dirname(self.index_path), "index.faiss")):
                    print("기존 인덱스를 로드합니다.")
                    self.load_index()
                else:
                    print("인덱스를 새로 생성합니다.")
                    new_pdf_files = self.get_pdf_files()
                
            if new_pdf_files:
                print("\n=== PDF 문서 처리 시작 ===")
                for pdf_path in new_pdf_files:
                    print(f"\n처리 중: {os.path.basename(pdf_path)}")
                    loader = PyPDFLoader(pdf_path)
                    pages = loader.load_and_split()
                    
                    # 각 페이지를 의미 있는 단위로 분할
                    for page in pages:
                        # 페이지를 문장/단락 단위로 분할
                        text_blocks = self._split_into_blocks(page.page_content)
                        
                        # AgenticChunker를 사용하여 청킹
                        self.chunker.add_propositions(text_blocks)
                    
                    self.processed_files.add(os.path.basename(pdf_path))
                    
                # 청킹된 결과를 벡터 저장소로 변환
                chunks = self.chunker.get_chunks(get_type='list_of_strings')
                docs = self._convert_chunks_to_documents(chunks)
                
                # 벡터 저장소 생성
                self.vector_store = FAISS.from_documents(
                    documents=docs,
                    embedding=self.embeddings
                )
                
                if self.index_path:
                    os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
                    self.vector_store.save_local(folder_path=os.path.dirname(self.index_path))
                    self.save_processed_files()
                    
                print("\n=== 청킹 결과 ===")
                self.chunker.pretty_print_chunks()
                print("\n✅ 벡터 저장소가 성공적으로 생성/업데이트되었습니다.")
                
        except Exception as e:
            logging.error(f"PDF 처리 중 오류 발생: {str(e)}")
            raise
            
    def _split_into_blocks(self, text: str) -> List[str]:
        """텍스트를 의미 있는 블록으로 분할"""
        # 기본적인 분할 (향후 개선 가능)
        blocks = []
        current_block = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:  # 빈 줄을 만나면
                if current_block:  # 현재 블록이 있으면 저장
                    blocks.append(' '.join(current_block))
                    current_block = []
            else:
                current_block.append(line)
                
        # 마지막 블록 처리
        if current_block:
            blocks.append(' '.join(current_block))
            
        return blocks
        
    def _convert_chunks_to_documents(self, chunks: List[str]) -> List[Document]:
        """청크를 Document 객체로 변환"""
        docs = []
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        'chunk_index': i,
                        'summary': self.chunker.chunks[list(self.chunker.chunks.keys())[i]]['summary'],
                        'title': self.chunker.chunks[list(self.chunker.chunks.keys())[i]]['title']
                    }
                )
            )
        return docs

    def setup_tokenizer(self) -> None:
        """토크나이저 설정"""
        tokenizer = get_encoding("cl100k_base")
        self.tiktoken_len = lambda text: len(tokenizer.encode(text))
        
    def setup_embeddings(self) -> None:
        """임베딩 모델 설정"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jhgan/ko-sbert-nli",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def get_pdf_files(self) -> List[str]:
        """PDF 파일 목록 반환"""
        target_file = "토스뱅크_가계대출_상품설명서.pdf"
        pdf_path = os.path.join(self.doc_dir, target_file)
        
        if os.path.exists(pdf_path):
            return [pdf_path]
        else:
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {target_file}")
    
    def load_processed_files(self) -> Set[str]:
        """처리된 PDF 파일 목록 로드"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        return set()

    def save_processed_files(self) -> None:
        """처리된 PDF 파일 목록 저장"""
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(list(self.processed_files), f, ensure_ascii=False, indent=2)

    def get_new_pdf_files(self) -> list:
        """새로 추가된 PDF 파일 경로 반환"""
        all_pdfs = set(os.path.basename(f) for f in self.get_pdf_files())
        
        # 인덱스 파일이 없으면 모든 PDF를 새로 처리
        if not os.path.exists(self.index_path):
            print("인덱스 파일이 없습니다. 모든 PDF를 다시 처리합니다.")
            return [os.path.join(self.doc_dir, pdf) for pdf in all_pdfs]
        
        # 인스가 있는 경우 새로운 파일만 처리
        new_pdfs = all_pdfs - self.processed_files
        return [os.path.join(self.doc_dir, pdf) for pdf in new_pdfs]

    def load_index(self) -> None:
        """저장된 FAISS 인덱스 로드"""
        try:
            index_file = os.path.join(os.path.dirname(self.index_path), "index.faiss")
            if not os.path.exists(index_file):
                raise ValueError(f"인덱스 파일을 찾을 수 없습니다: {index_file}")

            self.vector_store = FAISS.load_local(
                os.path.dirname(self.index_path),  # 디렉토리 경로만 전달
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("✅ 기존 인덱스를 성공적으로 로드했습니다.")
        except Exception as e:
            logging.error(f"인덱스 로드 중 오류 발생: {str(e)}")
            raise
    
    def setup_qa_system(self, 
                       temperature: float = 0,
                       search_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """QA 시스템 설정"""
        if search_kwargs is None:
            search_kwargs = {
                'k': 10,  # 3 -> 10으로 증가
                'score_threshold': 0.5  # 유사도 임계값 추가
            }
            
        # ChatGPT 모델 설정
        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model="gpt-3.5-turbo-16k",  # 더 큰 컨텍스트 윈도우를 가진 모델 사용
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            temperature=temperature
        )
        
        # QA 체인 설정
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs=search_kwargs
            ),
            return_source_documents=True
        )
        
    def ask(self, query: str) -> Dict[str, Any]:
        """질문에 대한 답변 생성"""
        if not hasattr(self, 'qa'):
            raise RuntimeError("QA 시스템이 설정되지 않았습니다. setup_qa_system()을 먼저 실행하세요.")
        return self.qa.invoke(query)

    def extract_loan_factors(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """대출 상품의 주요 요소들을 추출하여 DataFrame으로 반환"""
        try:
            # 벡터 저장소의 모든 청크를 하나의 문자열로 결합
            all_chunks = []
            results = self.vector_store.similarity_search("", k=100)  # 모든 청크 검색
            for doc in results:
                all_chunks.append(doc.page_content)
            context = "\n".join(all_chunks)

            print("\n=== 컨텍스트 내용 ===")
            print(context[:500] + "...")  # 컨텍스트 일부 출력

            # GPT에 분석 요청 (새로운 API 방식)
            client = openai.OpenAI(api_key=self.api_key)
            
            print("\n=== GPT 요청 중 ===")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=[
                    {"role": "system", "content": self.prompt},
                    {"role": "user", "content": f"다음 내용을 분석해주세요:\n\n{context}"}
                ],
                temperature=0
            )
            
            result = response.choices[0].message.content
            print("\n=== GPT 응답 ===")
            print(result)

            # JSON 문자열 정제 시도
            result = result.strip()
            if result.startswith("```json"):
                result = result.split("```json")[1]
            if result.endswith("```"):
                result = result.split("```")[0]
            result = result.strip()
            
            print("\n=== 정제된 JSON 문자열 ===")
            print(result)

            # JSON 파싱
            try:
                parsed_result = json.loads(result)
                print("\n=== 파싱된 JSON ===")
                print(json.dumps(parsed_result, indent=2, ensure_ascii=False))
            except json.JSONDecodeError as e:
                print(f"\nJSON 파싱 오류: {e}")
                print("원본 응답:", result)
                raise ValueError("GPT 응답을 JSON으로 파싱할 수 없습니다")

            # 필수 키 확인
            if "classified" not in parsed_result:
                print("\n=== 누락된 키 오류 ===")
                print("응답에서 'classified' 키를 찾을 수 없습니다")
                print("파싱된 결과의 키들:", list(parsed_result.keys()))
                raise ValueError("응답에 필수 키가 누락되었습니다")

            # DataFrame 생성
            classified_data = parsed_result["classified"]
            if not classified_data:  # 빈 리스트 체크
                raise ValueError("분류된 데이터가 비어있습니다")

            print("\n=== DataFrame 생성 ===")
            classified_df = pd.DataFrame(classified_data)
            print("생성된 DataFrame 열:", classified_df.columns.tolist())

            # 필요한 열이 없으면 빈 문자열로 채우기
            required_columns = ["구분", "대분류", "중분류", "소분류", "상세"]
            for col in required_columns:
                if col not in classified_df.columns:
                    print(f"누락된 열 추가: {col}")
                    classified_df[col] = ""

            # 열 순서 정렬
            classified_df = classified_df[required_columns]
            
            print("\n=== 최종 DataFrame 형태 ===")
            print(classified_df.head())
            
            return classified_df, pd.DataFrame()

        except Exception as e:
            logging.error(f"결과 처리 중 오류 발생: {str(e)}")
            print(f"\n상세 오류 정보: {str(e)}")
            raise

    def save_to_sheets(self, classified_df: pd.DataFrame, unclassified_df: pd.DataFrame):
        """결과를 구글 시트에 저장"""
        try:
            # .env 파일 다시 로드 (안전을 위해)
            load_dotenv(dotenv_path='.env')  # 경로 수정
            
            credentials_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'credentials',
                'google_sheets_credentials.json'
            )
            sheets_manager = GoogleSheetsManager(credentials_path)
            
            # 디버깅을 위한 출력 추가
            sheet_id = os.getenv('GOOGLE_SHEETS_ID')
            print(f"읽어온 GOOGLE_SHEETS_ID: {sheet_id}")  # 디버깅용
            
            if not sheet_id:
                raise ValueError("GOOGLE_SHEETS_ID가 .env 파일에 설정되지 않았습니다.")
            
            # 분류된 항목 저장
            if not classified_df.empty:
                values = [classified_df.columns.tolist()] + classified_df.values.tolist()
                sheets_manager.update_values(sheet_id, '시트1!A1', values)
                
            # 미분류 항목 저장
            if not unclassified_df.empty:
                values = [unclassified_df.columns.tolist()] + unclassified_df.values.tolist()
                sheets_manager.update_values(sheet_id, '시트1!H1', values)
                
            return sheet_id
            
        except Exception as e:
            logging.error(f"구글 시트 저장 중 오류 발생: {str(e)}")
            raise

    def get_index_stats(self) -> Dict[str, Any]:
        """인덱스의 통계 정보 반환"""
        try:
            if not hasattr(self, 'vector_store'):
                raise ValueError("벡터 저장소가 초기화되지 않았습니다.")
            
            # FAISS 인덱스에서 문서 내용 가져오기
            docs = self.vector_store.docstore._dict.values()
            
            # 청크 길이 계산
            chunk_lengths = [len(doc.page_content) for doc in docs]
            
            return {
                'total_chunks': len(chunk_lengths),
                'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths) if chunk_lengths else 0,
                'min_chunk_length': min(chunk_lengths) if chunk_lengths else 0,
                'max_chunk_length': max(chunk_lengths) if chunk_lengths else 0
            }
            
        except Exception as e:
            logging.error(f"인덱스 정보 조회 중 오류 발생: {str(e)}")
            return {
                'total_chunks': 0,
                'avg_chunk_length': 0,
                'min_chunk_length': 0,
                'max_chunk_length': 0
            }

def main():
    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    doc_dir = os.path.join(current_dir, "doc")
    index_dir = os.path.join(current_dir, "faiss_index")
    
    # doc 디렉토리 존재 확인
    if not os.path.exists(doc_dir):
        raise FileNotFoundError(f"doc 디렉토리를 찾을 수 없습니다: {doc_dir}")
    
    """
    # 기존 인덱스 파일 삭제 부분 주석 처리
    if os.path.exists(index_dir):
        print("\n=== 기존 인덱스 삭제 중 ===")
        import shutil
        shutil.rmtree(index_dir)
        print("✅ 기존 인덱스가 삭제되었습니다.")
    """
    
    # QA 시스템 초기화
    qa_system = PDFQuestionAnswering(doc_dir, index_dir)
    
    # PDF 처리 및 인덱스 생성 (또는 기존 인덱스 로드)
    qa_system.process_pdf()
    
    print("\n=== PDF 분석 및 구조화 시작 ===")
    qa_system.setup_qa_system()
    classified_df, unclassified_df = qa_system.extract_loan_factors()
    
    print("\n=== 구조화된 데이터 구글 시트 저장 중 ===")
    sheet_id = qa_system.save_to_sheets(classified_df, unclassified_df)
    print(f"\n✅ 분석 완료! 구글 시트 ID: {sheet_id}")
    print(f"https://docs.google.com/spreadsheets/d/{sheet_id}")

if __name__ == "__main__":
    main()