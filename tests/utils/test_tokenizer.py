import pytest
from src.utils.tokenizer import TokenizerUtils

@pytest.fixture
def tokenizer():
    return TokenizerUtils()

def test_basic_tokenization(tokenizer):
    """기본 토큰화 테스트"""
    text = "금리 3.5% 우대금리 적용"
    result = tokenizer.tokenize(text)
    
    assert result["tokens"]
    assert result["token_ids"]
    assert result["num_tokens"] > 0
    assert isinstance(result["num_tokens"], int)

def test_batch_tokenization(tokenizer):
    """배치 토큰화 테스트"""
    texts = [
        "대출한도 5억원",
        "금리 3.5%",
        "우대금리 적용"
    ]
    
    results = tokenizer.batch_tokenize(texts)
    assert len(results) == len(texts)
    for result in results:
        assert result["num_tokens"] > 0

def test_token_length(tokenizer):
    """토큰 길이 계산 테스트"""
    text = "대출한도 5억원"
    length = tokenizer.get_token_length(text)
    assert length > 0
    assert length < 512  # max_length 이내

def test_split_by_token_length(tokenizer):
    """토큰 길이 기준 분할 테스트"""
    # 긴 텍스트 생성
    long_text = "대출상품 설명" * 200
    chunks = tokenizer.split_by_token_length(long_text, max_tokens=100)
    
    # 각 청크가 max_tokens 이하인지 확인
    for chunk in chunks:
        assert tokenizer.get_token_length(chunk) <= 100

def test_empty_input(tokenizer):
    """빈 입력 처리 테스트"""
    result = tokenizer.tokenize("")
    assert result["num_tokens"] == 0
    assert not result["tokens"]
    assert not result["token_ids"]

    chunks = tokenizer.split_by_token_length("")
    assert not chunks