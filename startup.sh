#!/usr/bin/env bash

# in project root dir
cd $(dirname "$BASH_SOURCE")

eval "$(conda shell.bash hook)"
conda activate doc_parser_ocr

export ASCEND_RT_VISIBLE_DEVICES=2,3

uvicorn http_server.rest_api:app --host 0.0.0.0 --port 32008 --workers 1 --reload
#uvicorn http_server.rest_api:app --host 0.0.0.0 --port 22008 --workers 1 > log 2>&1

#PYTHONPATH=. python http_server/rest_api.py