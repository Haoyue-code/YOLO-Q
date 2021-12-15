from yolo.trt import build_from_configs
from yolo.api.trt_inference import TRTPredictor
from yolo.api.visualization import Visualizer
from yolo.utils.metrics import MeterBuffer
from yolo.utils.gpu_metrics import gpu_mem_usage, gpu_use
import cv2
import pycuda.driver as cuda
from loguru import logger
import time

global_settings = {
    './configs/config_trt.yaml': {
        'batch': 1,
        'model': 'n',
        'size': (384, 640)
    },
    './configs/config_trt15n.yaml': {
        'batch': 15,
        'model': 'n',
        'size': (384, 640)
    },
    './configs/config_trt15n640.yaml': {
        'batch': 15,
        'model': 'n',
        'size': (640, 640)
    },
    './configs/config_trt15s.yaml': {
        'batch': 15,
        'model': 's',
        'size': (384, 640)
    },
}

if __name__ == "__main__":

    device = "0"
    cuda.init()
    ctx = cuda.Device(int(device)).make_context()
    stream = cuda.Stream()
    # stream = None

    pre_multi = False  # 多线程速度较慢
    infer_multi = False  # 多线程速度较慢
    post_multi = False  # 多线程速度较慢

    cfg_path = './configs/config_trt.yaml'
    test_frames = 500
    setting = global_settings[cfg_path]

    test_batch = setting['batch']
    test_model = setting['model']
    test_size = setting['size']

    # logger.add("trt15.log", format="{message}")
    # logger.add("trt1.log", format="{message}")
    # logger.add("trt15.log")

    model = build_from_configs(cfg_path=cfg_path,
                               ctx=ctx,
                               stream=stream)
    predictor = TRTPredictor(
        img_hw=test_size,
        models=model,
        stream=stream,
        pre_multi=pre_multi,
        infer_multi=infer_multi,
        post_multi=post_multi,
    )

    if predictor.multi_model:
        vis = [Visualizer(names=model.names) for model in predictor.models]
    else:
        vis = [Visualizer(names=predictor.models.names)]
        # vis.draw_imgs(img, outputs[i])
    # vis = Visualizer(names=model[1].names)

    meter = MeterBuffer(window_size=500)

    cap = cv2.VideoCapture('/e/1.avi')
    frame_num = 1
    while cap.isOpened():
        ts = time.time()
        frame_num += 1
        # if frame_num % 2 == 0:
        #     continue
        if frame_num == test_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        outputs = predictor.inference([frame for _ in range(test_batch)])
        # for i, v in enumerate(vis):
        #     v.draw_imgs(frame, outputs[i])
        # cv2.imshow('p', frame)
        # if cv2.waitKey(1) == ord('q'):
        #     break
        # te = time.time()
        # print(f"frame {frame_num} time: {te - ts}")
        memory = gpu_mem_usage()
        utilize = gpu_use()
        logger.info(f"{predictor.times}, {memory}, {utilize}")
        meter.update(memory=memory, utilize=utilize, **predictor.times)

    logger.info("-------------------------------------------------------")
    logger.info(
        f"Tensort, {test_batch}x5, yolov5{test_model}, {test_size}, {test_frames}frames average time."
    )
    logger.info(f"pre_multi: {pre_multi}")
    logger.info(f"infer_multi: {infer_multi}")
    logger.info(f"post_multi: {post_multi}")
    logger.info(f"Average preprocess: {meter['preprocess'].avg}s")
    logger.info(f"Average inference: {meter['inference'].avg}s")
    logger.info(f"Average postprocess: {meter['postprocess'].avg}s")
    logger.info(f"Average memory: {meter['memory'].avg}MB")
    logger.info(f"Average utilize: {meter['utilize'].avg}%")
    logger.info(f"Max utilize: {meter['utilize'].max}%")

    ctx.pop()

# multi stream one thread
# Average preprocess: 0.027459805749027604s
# Average inference: 0.029808317801081032s
# Average postprocess: 0.023598009323978042s
# Average memory: 2369.6875MB
# Average utilize: 36.43574297188755%
# Max utilize: 40%

# multi stream multi thread
# Average preprocess: 0.0272097893986836s
# Average inference: 0.027776270506372415s
# Average postprocess: 0.023574815696501827s
# Average memory: 2356.715612449799MB
# Average utilize: 35.06024096385542%
# Max utilize: 38%

# one stream one thread
# Average preprocess: 0.027554046198067415s
# Average inference: 0.02978426720722612s
# Average postprocess: 0.02378575820999452s
# Average memory: 2369.6875MB
# Average utilize: 36.238955823293175%
# Max utilize: 40%

# ------------1x5-----------
# multi stream one thread
# Average preprocess: 0.001693828996405544s
# Average inference: 0.0054869872020430355s
# Average postprocess: 0.0017274603786238704s
# Average memory: 1847.6875MB
# Average utilize: 45.69477911646587%
# Max utilize: 55%

# multi stream multi thread
# Average preprocess: 0.0018568163415993073s
# Average inference: 0.0054639515627818895s
# Average postprocess: 0.0024826655904930757s
# Average memory: 1847.6875MB
# Average utilize: 36.74497991967871%
# Max utilize: 45%


# -----------5000 frames-----------
# -----------two models------------
# single thread total time: 85.99742603302002
# multi thread total time: 66.30611062049866

# -----------three models------------
# single thread total time: 117.53678607940674
# multi thread total time: 78.62954616546631

# -----------three models, two pic------------
# single thread total time: 136.21081161499023
# multi thread total time: 107.52954616546631

# -----------1000 frames-----------
# -----------four model, two pic-----------
# single thread total time: 43.65745544433594
# multi thread total time: 32.65745544433594
