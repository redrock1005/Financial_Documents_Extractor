import pytest
import os
import numpy as np
from src.processors.pdf import PDFProcessor
from src.processors.chunk import ChunkProcessor
from src.embedders.kb_albert import KBAlbertEmbedder
from src.utils.tokenizer import TokenizerUtils

class TestComponentIntegration:
    @pytest.fixture(scope="class")
    def pdf_processor(self):
        return PDFProcessor()
    
    @pytest.fixture(scope="class")
    def chunk_processor(self):
        return ChunkProcessor()
    
    @pytest.fixture(scope="class")
    def embedder(self):
        return KBAlbertEmbedder()
    
    @pytest.fixture(scope="class")
    def tokenizer(self):
        return TokenizerUtils()

    def test_full_processing_flow(self, pdf_processor, chunk_processor, embedder, tokenizer):
        """PDF -> 청크 -> 임베딩 전체 흐름 테스트"""
        # 1. PDF 파일 로드 및 텍스트 추출
        print("\n1. PDF 처리 중...")
        pdf_path = "/Users/jungseokoh/Desktop/Cursor_prac_2024/product_factory/doc/토스뱅크_가계대출_상품설명서.pdf"
        assert os.path.exists(pdf_path), "PDF 파일이 존재하지 않습니다"
        
        text = pdf_processor.process(pdf_path)
        assert text, "PDF 텍스트 추출 실패"
        print(f"추출된 텍스트 길이: {len(text)}")

        # 2. 텍스트 청킹
        print("\n2. 텍스트 청킹 중...")
        chunks_result = chunk_processor.process(text)
        chunks = chunks_result["chunks"]
        assert chunks, "청크 생성 실패"
        print(f"생성된 청크 수: {len(chunks)}")
        
        # 청크 상세 정보 출력
        print("\n=== 청크 상세 정보 ===")
        for i, chunk in enumerate(chunks):
            print(f"\n[청크 {i+1}/{len(chunks)}]")
            print(f"길이: {len(chunk['text'])} 문자")
            print(f"토큰 수: {chunk['tokens']}")
            print(f"시작 위치: {chunk['start_pos']}")
            print(f"텍스트 미리보기: {chunk['text'][:100]}...")

        # 3. 토큰화 검증
        print("\n3. 토큰화 검증 중...")
        for i, chunk in enumerate(chunks):
            tokens = tokenizer.tokenize(chunk["text"])
            assert tokens["num_tokens"] <= tokenizer.max_length, f"청크 {i}의 토큰 수가 제한을 초과합니다"
            print(f"청크 {i} 토큰 수: {tokens['num_tokens']}")

        # 4. 임베딩 생성
        print("\n4. 임베딩 생성 중...")
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.batch_embed(chunk_texts)
        assert embeddings is not None, "임베딩 생성 실패"
        print(f"임베딩 shape: {embeddings.shape}")
        
        # 임베딩 상세 정보 출력
        print("\n=== 임베딩 상세 정보 ===")
        for i, embedding in enumerate(embeddings):
            print(f"\n[임베딩 {i+1}/{len(embeddings)}]")
            print(f"차원: {embedding.shape}")
            print(f"최소값: {embedding.min():.4f}")
            print(f"최대값: {embedding.max():.4f}")
            print(f"평균값: {embedding.mean():.4f}")
            print(f"표준편차: {embedding.std():.4f}")

        # 5. 결과 검증
        print("\n5. 결과 검증 중...")
        assert embeddings.shape[0] == len(chunks), "청크 수와 임베딩 수가 일치하지 않습니다"
        assert embeddings.shape[1] == 768, "임베딩 차원이 올바르지 않습니다"
        assert not np.any(np.isnan(embeddings)), "임베딩에 NaN 값이 있습니다"

        # 6. 임베딩 품질 검사
        print("\n6. 임베딩 품질 검사 중...")
        similarity_matrix = np.zeros((len(chunks), len(chunks)))
        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
                if similarity > 0.9:  # 매우 유사한 청크 출력
                    print(f"\n높은 유사도 발견 (청크 {i+1}와 {j+1}): {similarity:.4f}")
                    print(f"청크 {i+1} 미리보기: {chunks[i]['text'][:100]}...")
                    print(f"청크 {j+1} 미리보기: {chunks[j]['text'][:100]}...")

        print("\n전체 통합 테스트 완료!")
        return {
            "num_chunks": len(chunks),
            "embedding_shape": embeddings.shape,
            "avg_chunk_length": sum(len(chunk["text"]) for chunk in chunks) / len(chunks)
        } 