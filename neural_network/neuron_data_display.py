import math
import matplotlib.pyplot as plt


# シグモイド関数
def sigmoid(a):
    return 1.0 / (1.0 + math.exp(-a))


# ニューロン
class Neuron:
    input_sum = 0.0
    output = 0.0

    def setInput(self, inp):
        self.input_sum += inp

    def getOutput(self):
        self.output = sigmoid(self.input_sum)
        return self.output


# ニューラルネットワーク
class NeuralNetwork:
    # 入力の重み
    w = [1.5, -2.51, 0.5]
    neuron = Neuron()

    # 実行
    def commit(self, input_data):
        self.neuron.setInput(input_data[0] * self.w[0])
        self.neuron.setInput(input_data[1] * self.w[1])
        self.neuron.setInput(input_data[1] * self.w[1])
        return self.neuron.getOutput()


# 基準点(データの範囲を0.0-1.0の範囲に収めるため)
refer_point_0 = 34.5
refer_point_1 = 137.5

# ファイルの読み込み
trial_data = []
trial_data_file = open("trial_data.txt", "r")  # 読み込みのみ
for line in trial_data_file:
    line = line.rstrip().split(",")  # rstripeで行末の文字を取り除く。「,」で区切ってリストにする。
    trial_data.append(
        [float(line[0]) - refer_point_0,
         float(line[1]) - refer_point_1])  # 井戸から、基準点を引いたもの、軽度から基準点を引いたものを格納する。（全ての値が0~1なっている）
trial_data_file.close()

# ニューラルネットワークのインスタンス
nurral_network = NeuralNetwork()

# 実行
position = [[], []]
for data in trial_data:
    position[0].append(data[0] + refer_point_1)  # 緯度
    position[1].append(data[1] + refer_point_0)  # 経度

# プロット
plt.scatter(position[0], position[1], c="red", label="position", marker="+")  # scatterは散布図

plt.legend()  # 散布図の描画にはlegendの表記が必要
plt.show()
