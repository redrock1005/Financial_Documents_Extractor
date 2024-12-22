import pytest
from src.processors.chunk import ChunkProcessor
from src.embedders.kb_albert import KBAlbertEmbedder

@pytest.fixture
def processor():
    return ChunkProcessor(min_tokens=100, max_tokens=512)

def test_processor_initialization(processor):
    """청크 프로세서 초기화 테스트"""
    assert processor is not None
    assert processor.min_tokens == 100
    assert processor.max_tokens == 512
    assert processor.tokenizer is not None
    assert isinstance(processor.section_patterns, list)
    assert len(processor.section_patterns) > 0

def test_token_length_calculation(processor):
    """토큰 수 계산 테스트"""
    text = "KB 주택담보대출 금리 3.5%"
    token_length = processor._get_token_length(text)
    assert token_length > 0
    assert isinstance(token_length, int)

def test_chunk_size_limits(processor):
    """청크 크기 제한 테스트"""
    # 작은 텍스트도 의미가 있다면 청크로 포함
    small_text = "작은 텍스트"
    small_result = processor.process(small_text)
    assert len(small_result["chunks"]) > 0  # 청크가 있어야 함
    
    # 최대 크기보다 큰 텍스트
    large_text = "주택담보대출 신청 자격\n" * 100
    large_result = processor.process(large_text)
    assert all(
        chunk.get("tokens", 0) <= processor.max_tokens
        for chunk in large_result["chunks"]
    )

def test_section_pattern_recognition(processor):
    """섹션 패턴 인식 테스트"""
    text = """1. 대출 신청 자격
    • 주택소유자 및 무주택자
    • 신용점수 600점 이상
    
    2. 대출 한도
    • 담보가치의 최대 70%
    • 소득대비 원리금상환비율 고려
    
    3. 금리 조건
    ① 기본금리 3.5%
    ② 우대금리 최대 1.0%
    """
    
    result = processor.process(text)
    assert len(result["chunks"]) > 0
    # 각 청크가 토큰 제한을 준수하는지 확인
    assert all(
        chunk.get("tokens", 0) <= processor.max_tokens 
        for chunk in result["chunks"]
    )
    # 섹션이 적절히 인식되었는지 확인
    assert any("대출 신청 자격" in chunk["text"] for chunk in result["chunks"])
    assert any("금리 조건" in chunk["text"] for chunk in result["chunks"])

def test_metadata_structure(processor):
    """메타데이터 구조 테스트"""
    text = """1. 대출 상품 개요
    • 대출대상: 주택소유자
    • 대출한도: 최대 5억원
    • 대출기간: 1년~30년
    """
    
    result = processor.process(text)
    assert "metadata" in result
    assert "num_chunks" in result["metadata"]
    assert "avg_chunk_size" in result["metadata"]
    # 청크에 토큰 수 정보가 포함되어 있는지 확인
    if result["chunks"]:
        assert "tokens" in result["chunks"][0]
        assert isinstance(result["chunks"][0]["tokens"], int)

def test_chunk_with_albert():
    """ChunkProcessor와 KB-ALBERT 통합 테스트"""
    processor = ChunkProcessor()
    albert = KBAlbertEmbedder()
    
    test_text = """1. 대출 상품 개요
    • 대출한도: 최대 5억원
    • 대출기간: 1년~30년
    """
    
    chunks = processor.process(test_text)
    for chunk in chunks["chunks"]:
        embeddings = albert.embed(chunk["text"])
        assert embeddings is not None
        assert len(embeddings) > 0
