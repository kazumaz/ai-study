import math
import matplotlib.pyplot as plt

# これは動かない。

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

    def reset(self):
        self.input_sum = 0
        self.output = 0


# ニューラルネットワーク
class NeuralNetwork:
    # 入力の重み
    # 入力層と中間層の間の重み（１つめの要素は入力そうの最初のニューロンと中間層のかくニューロのとの間の重み）
    # 二番目の要素は入力層の二番目のニューロンと中間層の各ニューロンとの間の重み
    # 最後の要素は入力層のバイアスと中間層の間の各要素の重み
    w_im = [[0.496, 0.512], [-0.501, 0.998], [0.498, -0.502]]
    # 最初の要素は中間層の最初のニューロンと出力層のニューロンの間の重み
    # 中間層の二番目のニューロンと出力そうのニューロンの間の重み
    # 中間層のバイアスと、出力層のニューロンの間の重みになる
    w_mo = [0.121, -0.4996, 0.200]  # 中間層と出力おすの間の重み

    # 各層の宣言
    input_layer = [0.0, 0.0, 1.0]  # 2つのニューロンと、バイアスがが１つ　入力層には、入力値がそのまま入るからニューロンクラスではなく、数値がそのまま入る。
    middle_layer = [Neuron(), Neuron(), 1.0]  # 中間層を表す。
    output_layer = Neuron()

    # 実行
    def commit(self, input_data):
        self.neuron.reset()

        bias = 1.0
        self.neuron.setInput(input_data[0] * self.w[0])
        self.neuron.setInput(input_data[1] * self.w[1])
        self.neuron.setInput(bias * self.w[2])
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
position_tokyo = [[], []]
position_kanagawa = [[], []]
for data in trial_data:
    if nurral_network.commit(data) < 0.5:
        position_tokyo[0].append(data[1] + refer_point_1)
        position_tokyo[1].append(data[0] + refer_point_0)
    else:
        position_kanagawa[0].append(data[1] + refer_point_1)
        position_kanagawa[1].append(data[0] + refer_point_0)

# プロット
plt.scatter(position_tokyo[0], position_tokyo[1], c="red", label="Tokyo", marker="+")
plt.scatter(position_kanagawa[0], position_kanagawa[1], c="blue", label="Kanagawa", marker="+")

plt.legend()  # 散布図の描画にはlegendの表記が必要
plt.show()
