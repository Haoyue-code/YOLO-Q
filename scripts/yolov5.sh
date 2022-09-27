#!/bin/sh

python demo/demo_trt_onnx.py --cfg-path ./configs/yolov5/n.yaml
python demo/demo_trt_onnx.py --cfg-path ./configs/yolov5/s.yaml
python demo/demo_trt_onnx.py --cfg-path ./configs/yolov5/m.yaml
python demo/demo_trt_onnx.py --cfg-path ./configs/yolov5/l.yaml
python demo/demo_trt_onnx.py --cfg-path ./configs/yolov5/x.yaml
