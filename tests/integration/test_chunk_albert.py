import pytest
import os
import numpy as np
from src.processors.chunk import ChunkProcessor
from src.embedders.kb_albert import KBAlbertEmbedder
from src.processors.pdf import PDFProcessor  # PDF 처리를 위해 추가

class TestChunkAlbertIntegration:
    @pytest.fixture(scope="class")
    def processor(self):
        return ChunkProcessor()

    @pytest.fixture(scope="class")
    def embedder(self):
        return KBAlbertEmbedder()

    @pytest.fixture(scope="class")
    def pdf_processor(self):
        return PDFProcessor()

    def test_real_product_document(self, processor, embedder, pdf_processor):
        """실제 금융 상품 설명서 테스트"""
        # PDF 파일 경로
        pdf_path = "/Users/jungseokoh/Desktop/Cursor_prac_2024/product_factory/doc/토스뱅크_가계대출_상품설명서.pdf"
        assert os.path.exists(pdf_path), "PDF 파일이 존재하지 않습니다"

        # 1. PDF 텍스트 추출
        print("\n1. PDF 텍스트 추출 중...")
        pdf_text = pdf_processor.process(pdf_path)
        assert pdf_text, "PDF에서 텍스트를 추출할 수 없습니다"
        print(f"추출된 텍스트 길이: {len(pdf_text)}")

        # 2. 텍스트 청킹
        print("\n2. 텍스트 청킹 중...")
        chunks_result = processor.process(pdf_text)
        chunks = chunks_result["chunks"]
        assert len(chunks) > 0, "청크를 생성할 수 없습니다"
        print(f"생성된 청크 수: {len(chunks)}")

        # 3. 청크 분석
        print("\n3. 청크 분석 중...")
        for i, chunk in enumerate(chunks):
            print(f"\n청크 {i+1}/{len(chunks)}:")
            print(f"텍스트 길이: {len(chunk['text'])}")
            print(f"토큰 수: {chunk['tokens']}")
            print(f"텍스트 미리보기: {chunk['text'][:100]}...")
            
            # 토큰 수 제한 확인
            assert chunk["tokens"] <= processor.max_tokens, f"청크 {i+1}의 토큰 수가 제한을 초과합니다"

        # 4. 임베딩 생성
        print("\n4. 임베딩 생성 중...")
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        # 배치 처리로 임베딩 생성
        embeddings = embedder.batch_embed(chunk_texts)
        assert embeddings is not None, "임베딩 생성에 실패했습니다"
        print(f"임베딩 shape: {embeddings.shape}")

        # 5. 임베딩 품질 확인
        print("\n5. 임베딩 품질 확인 중...")
        assert embeddings.shape[0] == len(chunks), "청크 수와 임베딩 수가 일치하지 않습니다"
        assert embeddings.shape[1] == 768, "임베딩 차원이 올바르지 않습니다"
        
        # 임베딩 값 범위 확인
        assert not np.any(np.isnan(embeddings)), "임베딩에 NaN 값이 있습니다"
        assert not np.any(np.isinf(embeddings)), "임베딩에 Inf 값이 있습니다"

        print("\n테스트 완료!")
        return {
            "num_chunks": len(chunks),
            "embedding_shape": embeddings.shape,
            "avg_chunk_length": sum(len(chunk["text"]) for chunk in chunks) / len(chunks)
        }