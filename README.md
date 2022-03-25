# 研究用リポジトリ

# 環境構築

`docker-compose up -d`

## jupyter lab 開発環境

`vscode` で remote containers を使う。

# テストおよびライブラリノートの.py化の実行

## ブラウザ＆dockerを使用する場合

1. コンテナが動作している状態で`localhost:8080`をブラウザで開く
2. jupyter labでterminalを起動する
3. `/root/src/lib` へ移動
4. `make` を実行

## vscode を使用する場合

1. remote containers を用いて vscode による開発環境を立ち上げる
2. vscode のターミナルにて下記のコマンドを実行 
    1. `cd /workspace/lib`
    2. `make`
