import math


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
    neuron = Neuron()

    # 実行
    def commit(self, input_data):
        for data in input_data:
            self.neuron.setInput(data)
        return self.neuron.getOutput()


# ニューラルネットワークのインスタンス
nurral_network = NeuralNetwork()

# 実行
trial_data = [1.0, 2.0, 3.0]
print(nurral_network.commit(trial_data))
