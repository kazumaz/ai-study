from sklearn import datasets
from sklearn import svm
from sklearn import metrics  # 文字認識の正解率の表示に用いる
import matplotlib.pyplot as plt

# 数字データの読み込み
digits = datasets.load_digits()

# データの形式を確認
print(digits.data)
print(digits.data.shape)

################
# 実行結果
#
# 各画像のデータは8*8　ピクセル　画像の数は1797個。各要素の数字はそこのピクセルのいろを表している。
#
# [[ 0.  0.  5. ...  0.  0.  0.]  この行で１つの画像を表現している。
# [ 0.  0.  0. ... 10.  0.  0.]
# [ 0.  0.  0. ... 16.  9.  0.]
# ...
# [ 0.  0.  1. ...  6.  0.  0.]
# [ 0.  0.  2. ... 12.  0.  0.]
# [ 0.  0. 10. ... 12.  1.  0.]]
# (1797, 64)
#
###################

# データの数
n = len(digits.data)

# 画像と正解値の表示
# images = digits.images  # 画像データをimagesに入れる
# labels = digits.target  # 正解値(ターゲット)# をlabelに入れる
# for i in range(10):
#     plt.subplot(2, 5, i + 1)  # subplotで、複数の表示を行う 2行、5列で i+1で表示の位置を決める
#     plt.imshow(images[i], cmap=plt.cm.gray_r,
#                interpolation="nearest")  # imshowで画像の表示　gray表示で白黒 interpolationでピクセル間の保管がnearlestになる
#     plt.axis("off")  # 軸の表示は行わない
#     plt.title("Training: " + str(labels[i]))
# plt.show()

################
# 実行結果
#
# 下記のような画像が表示される
# 1   2   3   4  5
# 6   7   8   9  10
###################

# サポートベクターマシーン
# 複雑な訓練データに対応するためにリニアSVCではなく、SVCを用いる。
# svmは引数で複数の設定をできる、。gammaは１つの訓練データが与える影響の大きさを表す。Cは誤認識の許容度
clf = svm.SVC(gamma=0.001, C=100.0)
# サポートベクターマシーンによる訓練 (6割のデータを使用。残りの４割は検証よう。）
# fitで訓練を行う。
# 先頭から初めて[:n * 6 / 10]のインデックス（6割）までデータを取得する。　targetは正解値
clf.fit(digits.data[:n * 6 // 10], digits.target[:n * 6 // 10])

# 最後の10個のデータをチェックする
# 正解（マイナスを指定すると末尾からの範囲）
# 正解値のprintを行う
print(digits.target[-10:])

# 予想を行う（数字を読み取る）
# 訓練された分類機を元に予測を行う すなわち数字の読み取りを行う
# 画像データの末尾から10個を取り出し、predictに渡す
print(clf.predict(digits.data[-10]))
