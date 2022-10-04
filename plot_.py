import matplotlib.pyplot as plt

yolov5 = dict(
    N=dict(time=3.3, params=1.9),
    S=dict(time=3.3, params=7.2),
    M=dict(time=5.5, params=21.2),
    # L=dict(time=3.8, params=49.0),
    # X=dict(time=5.9, params=50.7),
)

yolov6 = dict(
    N=dict(time=4, params=4.3),
    T=dict(time=5, params=15.0),
    S=dict(time=5.5, params=17.2),
    # M=dict(time=2.9, params=49.5),
    # L=dict(time=4.8, params=52.5),
)

plt.plot(
    [y5["time"] for t, y5 in yolov5.items()],
    [y5["params"] for t, y5 in yolov5.items()],
    "*-",
    linewidth=2,
    markersize=8,
    label="yolov5",
    color="#1f77b4",
)
plt.plot(
    [y6["time"] for t, y6 in yolov6.items()],
    [y6["params"] for t, y6 in yolov6.items()],
    "D-",
    linewidth=2,
    markersize=8,
    label="yolov6",
    color="#ff7f0e",
)
for k, v in yolov5.items():
    plt.text(
        v["time"],
        v["params"],
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
        v["time"],
        v["params"],
        k,
        fontsize=15,
        # style="italic",
        weight="light",
        # verticalalignment="center",
        horizontalalignment="right",
        # rotation=45,
        color="#ff7f0e",
    )

plt.grid(alpha=0.5)
plt.xlabel("Params(M)")
plt.ylabel("Training speed(mins/epoch)")
plt.legend(loc="lower right")

plt.show()
# plt.savefig("result_new.png")
