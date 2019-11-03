from sklearn import datasets
from sklearn import svm  # SV¥ = サポートベクターマシーン

# Iris(アヤメ)の測定データの読み込み
iris = datasets.load_iris()

# データの形式を確認
print(iris.data)
print(iris.data.shape)

# データ量
n = len(iris.data)
print(n)

# 線形サポートベクラーマシーン
clf = svm.LinearSVC()
# サポートベクターマシーンによる訓練
clf.fit(iris.data, iris.target)  # iris.dataはアイリスのデータ　iris.targetは正解値(品種に対応)

# 品種を判定する
# predictメソッドで判定　品種が0か1か2の値で判定
print(clf.predict([[5.1, 3.5, 1.4, 0.1], [6.5, 2.5, 4.4, 1.4], [5.9, 3.0, 5.2, 1.5]]))
