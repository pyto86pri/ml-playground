# 機械学習環境テンプレート
## 概要
機械学習系のタスクをする環境のテンプレート
### 前提
- [poetry]()
### セットアップ
```
$ make install
```
## モデル構築
### 学習
### 検証
### チューニング
## 推論エンドポイント構築
## Lab起動
Jupyterサーバ立ち上げ
```
$ make lab
```
# メモ
## Tensorboardエラー
```
raise VersionConflict(dist, req).with_context(dependent_req)
pkg_resources.VersionConflict: (setuptools 40.6.2 (/Users/pyto86/my/github.com/ml-playground/.venv/lib/python3.7/site-packages), Requirement.parse('setuptools>=41.0.0'))
```
poetry環境のsetuptoolsのバージョンが低いことが原因
```
$ poetry run pip install --upgrade setuptools
```