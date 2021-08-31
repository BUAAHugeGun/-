import matplotlib.pyplot as plt
import math
from matplotlib.font_manager import FontProperties

if __name__ == "__main__":
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'color': 'black',
            'size': 14
            }
    fid = [
        [37.4933, 39.9258, 40.0716, 40.06388, 40.2336, 40.6938, 44.3687],
        [39.4855, 42.2285, 42.0408, 41.9602, 42.2707, 42.4953, 44.9346]
    ]
    psnr = [
        [10.8957, 10.9171, 10.9200, 10.9262, 10.9232, 10.9442, 11.0080],
        [10.8697, 10.8926, 10.8812, 10.8927, 10.9078, 10.9358, 10.9895]
    ]
    msssim = [
        [0.6001, 0.600196, 0.600163, 0.600271, 0.600317, 0.6004, 0.6009],
        [0.6013, 0.601382, 0.601470, 0.601410, 0.601507, 0.601607, 0.602088]
    ]
    # for l in msssim:
    #     for i in range(len(l)):
    #         l[i]=-10*math.log10(1-l[i])
    tv = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]

    # plt.axhline(y=37.3588, color='red')
    # plt.axhline(y=10.8474, color='red')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 这两行需要手动设置
    plt.figure(figsize=(10, 4.5))
    plt.axhline(y=37.4588, color='red')
    plt.grid()
    plt.plot(tv, fid[0], color="dodgerblue", marker="o", label="实验A")
    plt.plot(tv, fid[1], color="orange", marker="v", label="实验B")
    plt.ylabel("FID", fontdict=font)
    plt.xlabel("\u03bb_TVLoss", fontdict=font)
    plt.legend(loc="best")

    plt.show()
