from IPython.display import display, Audio,YouTubeVideo
import pyroomacoustics as pra
import numpy as np
import matplotlib.pyplot as plt

fs = 44100        # 샘플링 레이트 (1초를 44,100개로 쪼갬)
duration = 1.0    # 1초 동안 재생
f = 440           # 주파수 440Hz


m = pra.make_materials(
    ceiling=(0.1, 0.05),  # (흡음, 산란)
    floor=(0.2, 0.1),
    east=(0.1, 0.05),
    west=(0.1, 0.05),
    north=(0.1, 0.05),
    south=(0.1, 0.05)
)

room=pra.room.ShoeBox(
    p= [9, 7.5, 3.5], # 2次元[幅、奥行き], ３次元[幅、奥行き、高さ]
    fs=48000, # サンプリング周波数
    t0=0.0, # シミュレーションの開始時間
    absorption=None,  # 平均壁面吸音率（materialsを設定するときは不要）
    max_order=1, # 鏡像法における反射回数の上限 *1
    sigma2_awgn=None, # ガウシアンノイズを乗せる
    sources=None, # 音源データ(リスト)。マイク設定と一緒に使う
    mics=None,  # マイク座標のリスト（リスト）。音源設定と一緒に使う。
    materials=m, # 建材の設定。詳しくはこの後に書きます。
    temperature=None, # 室温(摂氏)。音速に影響。特に書かないと343m/s
    humidity=None, # 湿度(%)。音速に影響。
    air_absorption=False, # 空気吸収。*2
    ray_tracing=False, # レイトレーシングと鏡像法のハイブリッド利用
    use_rand_ism=False, # よくわからん。音像位置がランダムになる？
    max_rand_disp=0.0 # 上のと一緒に使う
)
mic_locs=[]
micarray_ichi=np.linspace(4,5,10,endpoint=False)
print('마이크 어레이의 y위치\n',micarray_ichi,type(micarray_ichi))
mic_locs.append([6.3,i,1.2]for i in micarray_ichi)
mic_locs=np.array(mic_locs).T
print('마이크 어레이의 최종 위치\n',mic_locs,type(mic_locs))
