from typing import List

from http_server.ocr_api import OcrAPI, OcrRow, OcrResult
from http_server.ppocr_api_v2 import PpocrApi

ocr_api: OcrAPI = PpocrApi()


def _is_text_same_line(row1: OcrRow, row2: OcrRow):
    """判断2个文本是否在同一行"""
    l1_top, l1_btm = row1.bbox[1], row1.bbox[3]
    l2_top, l2_btm = row2.bbox[1], row2.bbox[3]

    # case1: l1 contain l2 in height
    if l1_top <= l2_top and l1_btm >= l2_btm:
        return True

    # case2: l2 contain l1 in height
    if l2_top <= l1_top and l2_btm >= l1_btm:
        return True

    # 共同的部分 > 各自的 1/2
    common_h = min(l1_btm, l2_btm) - max(l1_top, l2_top)
    l1_height = l1_btm - l1_top
    l2_height = l2_btm - l2_top
    if (common_h / l1_height > 0.5) and (common_h / l2_height > 0.5):
        return True
    return False


def _combine_ocr_rows(result: OcrResult):
    """ ocr识别得到多行文本, 将这些文本进行组装 """
    min_score = 1
    combine_text = ''
    last_row = None
    for row in result.rows:
        min_score = min(min_score, row.score)
        if last_row is None:
            combine_text += row.text
        else:
            if _is_text_same_line(last_row, row):
                combine_text += ' ' + row.text  # 因为 ocr 分开了, 至少加个空格
            else:
                combine_text += '\n' + row.text
        last_row = row
    return combine_text, min_score


def service_img_ocr(cv2_img) -> dict:
    result: OcrResult = ocr_api.process(cv2_img)
    if (result.err_msg is not None) or (len(result.rows) == 0):  # 解析失败 or  未解析到文字
        return {'text': '', 'score': 0}
    #
    text, score = _combine_ocr_rows(result)
    return {'text': text, 'score': score}


def service_det_and_rec(cv2_img) -> List[OcrRow]:
    result: OcrResult = ocr_api.process(cv2_img)
    if (result.err_msg is not None) or (len(result.rows) == 0):
        return []
    return result.rows
