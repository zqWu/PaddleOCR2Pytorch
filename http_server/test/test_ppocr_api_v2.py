import cv2

from http_server.ocr_api import OcrResult
from http_server.ppocr_api_v2 import PpocrApi
from project_root import relative_path_in_root


def test_one(image_name):
    ppocr_api = PpocrApi()
    cv2_img = cv2.imread(image_name)
    result: OcrResult = ppocr_api.process(cv2_img)
    print(result)


if __name__ == '__main__':
    test_one(relative_path_in_root('http_server/test/test_case/eval_0.png'))
    # run_all_test_case(ppocr_api)
