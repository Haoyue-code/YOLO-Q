import matplotlib.pyplot as plt

yolov5 = dict(
    N=dict(speed=0.9, mAP=28.0),
    S=dict(speed=1.2, mAP=37.4),
    M=dict(speed=2.1, mAP=45.4),
    L=dict(speed=3.8, mAP=49.0),
    X=dict(speed=5.9, mAP=50.7),
)

yolom = dict(
    # M=dict(speed=2.3, mAP=46.5),
    base=dict(speed=2.5, mAP=48.2),
    p1a=dict(speed=2.6, mAP=48.6),
)

yolov6 = dict(
    N=dict(speed=0.8, mAP=35.9),
    T=dict(speed=1.2, mAP=40.3),
    S=dict(speed=1.5, mAP=43.5),
    M=dict(speed=2.9, mAP=49.5),
    L=dict(speed=4.8, mAP=52.5),
)

rtmdet = dict(
    T=dict(speed=1.4, mAP=40.9),
    S=dict(speed=1.7, mAP=44.5),
    M=dict(speed=3.1, mAP=49.1),
    L=dict(speed=5.5, mAP=51.3),
    X=dict(speed=8.5, mAP=52.6),
)

plt.plot(
    [y5["speed"] for t, y5 in yolov5.items()],
    [y5["mAP"] for t, y5 in yolov5.items()],
    "*-",
    linewidth=2,
    markersize=8,
    label="yolov5",
    color="#1f77b4",
)
plt.plot(
    [y6["speed"] for t, y6 in yolov6.items()],
    [y6["mAP"] for t, y6 in yolov6.items()],
    "D-",
    linewidth=2,
    markersize=8,
    label="yolov6",
    color="#ff7f0e",
)
plt.plot(
    [rt["speed"] for t, rt in rtmdet.items()],
    [rt["mAP"] for t, rt in rtmdet.items()],
    "o-",
    linewidth=2,
    markersize=8,
    label="rtmdet",
    color="#2ca02c",
)

plt.plot(
    [y5["speed"] for t, y5 in yolom.items()],
    [y5["mAP"] for t, y5 in yolom.items()],
    "p-",
    linewidth=2,
    markersize=8,
    label="yolom",
    color="#d62728",
)

for k, v in yolov5.items():
    plt.text(
        v["speed"],
        v["mAP"],
        k,
        fontsize=15,
        # style="italic",
        weight="light",
        # verticalalignment="center",
        horizontalalignment="right",
        # rotation=45,
        color="#1f77b4",
    )

for k, v in yolov6.items():
    plt.text(
        v["speed"],
        v["mAP"],
        k,
        fontsize=15,
        # style="italic",
        weight="light",
        # verticalalignment="center",
        horizontalalignment="right",
        # rotation=45,
        color="#ff7f0e",
    )

for k, v in rtmdet.items():
    plt.text(
        v["speed"],
        v["mAP"],
        k,
        fontsize=15,
        # style="italic",
        weight="light",
        # verticalalignment="center",
        horizontalalignment="right",
        # rotation=45,
        color="#2ca02c",
    )

for k, v in yolom.items():
    plt.text(
        v["speed"],
        v["mAP"],
        k,
        fontsize=15,
        # style="italic",
        weight="light",
        # verticalalignment="center",
        horizontalalignment="right",
        # rotation=45,
        color="#d62728",
    )

plt.grid(alpha=0.5)
plt.xlabel("TensorRT-FP16 Latency(ms) on RTX3060")
plt.ylabel("COCO AP val(%)")
plt.legend(loc="lower right")

plt.show()
# plt.savefig("result_new.png")
