#!/bin/sh

python demo/demo_trt_onnx.py --cfg-path ./configs/yolov6/n.yaml
python demo/demo_trt_onnx.py --cfg-path ./configs/yolov6/t.yaml
python demo/demo_trt_onnx.py --cfg-path ./configs/yolov6/s.yaml
python demo/demo_trt_onnx.py --cfg-path ./configs/yolov6/m.yaml
python demo/demo_trt_onnx.py --cfg-path ./configs/yolov6/l.yaml
python demo/demo_trt_onnx.py --cfg-path ./configs/yolov6/l_relu.yaml
