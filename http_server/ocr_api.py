from dataclasses import dataclass
from typing import Union, List, Optional


@dataclass
class OcrRow:
    """ocr单行信息"""
    bbox: List[Union[int, float]]  # left, top, right, bottom
    text: str
    score: float


@dataclass
class OcrResult:
    """ocr检测结果"""
    err_msg: Optional[str]
    rows: Optional[List[OcrRow]]

    def __str__(self) -> str:
        if self.err_msg:
            return self.err_msg
        return str(self.rows)


class OcrAPI:

    def get_name(self) -> str:
        """返回该 ocr的名字, 要求唯一"""
        raise ValueError('you should implement it!')

    def process(self, cv2_img) -> OcrResult:
        """输入cv2_img, ocr处理后得到 OcrResult"""
        raise ValueError('you should implement it!')
