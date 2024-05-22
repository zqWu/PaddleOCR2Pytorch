import traceback
from typing import List

import cv2

from http_server.debug.predict_system_copy import TextSystemCopy
from http_server.logger import get_logger
from http_server.ocr_api import OcrAPI, OcrResult, OcrRow
from http_server.ocr_utils import get_cv2_image_w_h, add_white_border
from project_root import relative_path_in_root

logger = get_logger(__name__)


class PpocrApi(OcrAPI):
    _border_size = 10  # 添加border的尺寸, first=10, still not good, then to 20. by 朱雨辰

    def __init__(self):
        super().__init__()
        self.text_system = TextSystemCopy()

    def get_name(self) -> str:
        return self.__class__.__name__

    def process(self, cv2_img) -> OcrResult:
        border_cv2 = self._preprocess(cv2_img)
        x: OcrResult = self._detection_and_recognition(border_cv2)
        self._post_process(cv2_img, x)
        return x

    def _post_process(self, cv2_img, x: OcrResult):
        """使用 ocr进行记录"""
        # 修正坐标, 因为 增加了 border
        w, h = get_cv2_image_w_h(cv2_img)
        for a_row in x.rows:
            bbox: List = a_row.bbox
            left = max(0, bbox[0] - self._border_size)
            top = max(0, bbox[1] - self._border_size)
            right = min(bbox[2] - self._border_size, w)
            bottom = min(bbox[3] - self._border_size, h)
            a_row.bbox = [left, top, right, bottom]
            a_row.score = float(a_row.score)

    def _preprocess(self, cv2_img):
        """预处理:
        加边框
        """
        cv2_img = add_white_border(cv2_img, border_size=self._border_size)
        return cv2_img

    def _detection_and_recognition(self, cv2_img) -> OcrResult:
        """ppocr 文本检测 & 识别"""
        err_msg = None
        rows: List = []
        try:
            filter_boxes, filter_rec_res = self.text_system(cv2_img)
            for i in range(len(filter_boxes)):
                p1, p2, p3, p4 = filter_boxes[i]
                x_s = [p1[0], p2[0], p3[0], p4[0]]
                y_s = [p1[1], p2[1], p3[1], p4[1]]
                bbox = [min(x_s), min(y_s), max(x_s), max(y_s)]
                #
                text, score = filter_rec_res[i]
                a_row = OcrRow(bbox=bbox, text=text, score=score)
                rows.append(a_row)
        except Exception as ex:
            traceback.print_exc()
            err_msg = f'ppocr detection_and_recognition error: {ex}'
        #
        return OcrResult(err_msg=err_msg, rows=rows)


if __name__ == '__main__':
    img_file = relative_path_in_root('doc/imgs_2/3_2.png')
    cv2_img = cv2.imread(img_file)
    ocr_api: OcrAPI = PpocrApi()
    ocr_result: OcrResult = ocr_api.process(cv2_img)
    print(ocr_result)
