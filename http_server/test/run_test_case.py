import json
import os
import traceback
from dataclasses import dataclass, asdict
from typing import List, Optional, Union

import cv2

from http_server.ocr_api import OcrAPI, OcrResult
from http_server.ocr_utils import current_datetime
from project_root import relative_path_in_root

spec_eval_jsonl_file = relative_path_in_root('test/test_case/eval.jsonl')
spec_test_result_file = relative_path_in_root('test/output/test_result.jsonl')


@dataclass
class TestInput:
    """目前没有包含"""
    id: str
    img: str
    bbox: List[Union[int, float]]  # bbox, 目前不做评测
    txt: List[str]

    def get_img_cv2(self):
        img_file = relative_path_in_root(f'test/test_case/{self.img}')
        return cv2.imread(img_file)


@dataclass
class TestOutput:
    id: str
    diff_content: List[List[str]]  # 对比文本


def dataclass_to_str(obj_dataclass):
    """ dataclass 对象 => str """
    return json.dumps(asdict(obj_dataclass), ensure_ascii=False)


def run_all_test_case(ocr_api: OcrAPI, save_file: bool = True) -> List[TestOutput]:
    """运行 ocr test case"""
    diff_list = []
    with open(spec_eval_jsonl_file, 'r') as file:
        line_num = 1
        for line in file:
            try:
                _dict = json.loads(line)
                d_input = TestInput(**_dict)
                d_output = _ocr_and_compare(d_input, ocr_api)
                if d_output:
                    diff_list.append(d_output)
            except Exception as ex:
                traceback.print_exc()
                # 这里出错就直接抛异常, 通常是 bug 或 eval.jsonl出错, 解决后重跑即可
                raise ValueError(f'parse jsonl出错, line_num={line_num}, content={line}, ex={ex}')

    if save_file:
        if not os.path.exists("output"):
            os.makedirs("output")
        with open(spec_test_result_file, 'w', encoding='utf8') as f:
            # date time
            # ocr
            f.write(f'time: {current_datetime()}')
            f.write(f'\nocr: {ocr_api.get_name()}')
            if len(diff_list) > 0:
                for ele in diff_list:
                    f.writelines('\n' + dataclass_to_str(ele))
    #
    return diff_list


def _ocr_and_compare(test_data: TestInput, ocr_api: OcrAPI) -> Optional[TestOutput]:
    """进行ocr并对比结果, 如果不同则返回不同的内容. 全部相同则不返回"""
    cv2_image = test_data.get_img_cv2()
    ocr_result: OcrResult = ocr_api.process(cv2_image)
    # 进行判断
    diff_content = []  # 记录不同的文本
    if ((ocr_result.err_msg is None)
            and (ocr_result.rows is not None)
            and (len(test_data.txt) == len(ocr_result.rows))):
        for (gt, ocr_row) in zip(test_data.txt, ocr_result.rows):
            if gt != ocr_row.text:
                diff_content.append([gt, ocr_row.text])
    if len(diff_content) > 0:
        return TestOutput(test_data.id, diff_content)
    else:
        print(f'pass: {ocr_api.get_name()}, id={test_data.id}')
