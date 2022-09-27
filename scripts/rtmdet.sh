#!/bin/sh

python demo/demo_trt_onnx.py --cfg-path ./configs/rtmdet/t.yaml
python demo/demo_trt_onnx.py --cfg-path ./configs/rtmdet/s.yaml
python demo/demo_trt_onnx.py --cfg-path ./configs/rtmdet/m.yaml
python demo/demo_trt_onnx.py --cfg-path ./configs/rtmdet/l.yaml
python demo/demo_trt_onnx.py --cfg-path ./configs/rtmdet/x.yaml
