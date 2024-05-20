import traceback
from typing import List

import cv2

from http_server.logger import get_logger
from http_server.ocr_api import OcrAPI, OcrResult, OcrRow
from http_server.ocr_det_npu import default_text_detector
from http_server.ocr_rec_npu import default_text_recognizer
from http_server.ocr_utils import get_cv2_image_w_h, add_white_border, crop_cv2_image

logger = get_logger(__name__)


class PpocrApi(OcrAPI):
    _border_size = 10  # 添加border的尺寸, first=10, still not good, then to 20. by 朱雨辰

    def __init__(self):
        super().__init__()
        self.text_recognizer = default_text_recognizer()
        self.text_detector = default_text_detector()

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
        if cv2_img is None:
            raise ValueError('_detection_and_recognition, cv2_img is None')
        err_msg = None
        rows: List = []
        try:
            dt_boxes, elapse = self.text_detector(cv2_img)
            bbox_list = []
            for dots in dt_boxes:
                # [[  6.  50.] [926.  50.] [926.  73.] [  6.  73.]]
                x_s = [e[0] for e in dots]
                y_s = [e[1] for e in dots]
                left, top, right, btm = int(min(x_s)), int(min(y_s)), int(max(x_s)), int(max(y_s))
                bbox = [left, top, right, btm]
                bbox_list.append(bbox)

            bbox_list.sort(key=lambda bbox: bbox[1])  # 按照top重新排序

            for bbox in bbox_list:
                sub_img = crop_cv2_image(cv2_img, bbox)
                rec_res, predict_time = self.text_recognizer([sub_img])

                text, score = "", 0
                if len(rec_res) == 1:
                    text, score = rec_res[0]
                else:
                    raise ValueError(f'ocr rec 未知情况')
                logger.info(f'rec_res={rec_res}')  # rec_res=[('韩国小馆', 0.996515)]
                a_row = OcrRow(bbox=bbox, text=text, score=score)
                rows.append(a_row)
        except Exception as ex:
            traceback.print_exc()
            err_msg = f'ppocr detection_and_recognition error: {ex}'
        #
        return OcrResult(err_msg=err_msg, rows=rows)


if __name__ == '__main__':
    cv2_img = cv2.imread('../doc/imgs_2/3_2.png')
    ocr_api: OcrAPI = PpocrApi()
    ocr_result: OcrResult = ocr_api.process(cv2_img)
    print(ocr_result)
