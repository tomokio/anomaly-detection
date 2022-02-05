# おもちゃの稼働音を使った異常音検知
おもちゃの稼働音を収録した公開データセット、ToyADMOSデータセット[1]を用いた異常音検知プログラム

>[1] Yuma Koizumi, Shoichiro Saito, Noboru Harada, Hisashi Uematsu and Keisuke Imoto, "ToyADMOS: A Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection," in Proc of Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA), 2019.
> Paper URL: https://arxiv.org/abs/1908.03299

## 主なファイル、フォルダ
- 信号解析に用いたJupyter Notebook
  - https://github.com/tomokio/anomaly-detection/blob/main/result/signal_analysis.pdf
  - https://github.com/tomokio/anomaly-detection/blob/main/src/signal_analysis.ipynb
- 機械学習プログラム
  - https://github.com/tomokio/anomaly-detection/tree/main/src
- その他
  - 解析ログ　　　：https://github.com/tomokio/anomaly-detection/tree/main/result
  - 最終評価結果　：https://github.com/tomokio/anomaly-detection/blob/main/result/result_final_evaluation.txt
  - 設定ファイル　：https://github.com/tomokio/anomaly-detection/blob/main/src/config.yml
  - 学習済みモデル：https://github.com/tomokio/anomaly-detection/tree/main/model

## フォルダ構成
- src：機械学習プログラムを格納しています

- result：機械学習による分類結果、信号解析の結果を格納しています

- model：学習済みの機械学習モデルを格納しています

- dataset：データセットを配置するためのフォルダです

## 機械学習プログラム
- train.py：機械学習モデルの学習を行うためのプログラムです

- test.py：機械学習モデルのテストを行うためのプログラムです

- dataset.py：データセットの分割や特徴量抽出を行うためのプログラムです

- config.yml：ハイパーパラメータや特徴量抽出時の設定を記述した設定ファイルです

## 実行方法
1. 下記URLから`ToyConveyor`の名前がついた7zファイルをすべてダウンロードし解凍する
> https://zenodo.org/record/3351307#.Yf5Il_tUuXx
2. プロジェクトのルートに`data`フォルダを追加し、解凍後出てきたフォルダを移動する
> /anomaly-detection/data/ToyConveyor
3. srcフォルダ内にある`make_dataset_for_car_and_conveyor.py`を実行する
4. srcフォルダ内にある`train.py`を実行し、機械学習モデルの学習を行う
5. srcフォルダ内にある`test.py`を実行し、機械学習モデルのテストを行う