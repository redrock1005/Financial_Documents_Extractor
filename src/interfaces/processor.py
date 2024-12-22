from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseProcessor(ABC):
    """문서 처리를 위한 기본 인터페이스"""
    
    @abstractmethod
    def process(self, data: Any) -> Dict[str, Any]:
        """문서 처리 메소드

        Args:
            data: 처리할 데이터

        Returns:
            처리된 결과를 담은 딕셔너리
        """
        pass 