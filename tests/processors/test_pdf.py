import pytest
import os
from src.processors.pdf import PDFProcessor

def test_pdf_processor():
    processor = PDFProcessor()
    pdf_path = "/Users/jungseokoh/Desktop/Cursor_prac_2024/product_factory/doc/토스뱅크_가계대출_상품설명서.pdf"
    
    # PDF 파일 존재 확인
    assert os.path.exists(pdf_path), "PDF 파일이 존재하지 않습니다"
    
    # 텍스트 추출
    text = processor.process(pdf_path)
    assert text, "텍스트 추출 실패"
    assert len(text) > 0, "추출된 텍스트가 비어있습니다"
    
    print(f"\n추출된 텍스트 길이: {len(text)}")
    print(f"텍스트 미리보기:\n{text[:500]}...")
