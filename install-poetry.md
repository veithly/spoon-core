## 环境配置
#### 1. 用户目录下安装 poetry，不要在当前工程下的 venv 环境中安装
```
pip install poetry --user
```

#### 2.检验是否安装成功
```
poetry --version

output: Poetry (version x.x.x)
```

#### 3.使用
```
1. 安装 pyproject.toml 中的所有依赖
poetry install 　　　　　　　　　　#安装pyproject.toml文件中的全部依赖
poetry install --no-dev 　　　　 #只安装[tool.poetry.dependencies]下的（一般部署时使用）

2. 安装包
poetry add gym　　　　　　　　　   #安装最新稳定版本的gym
poetry add gym --dev　　　　　　  #指定为开发依赖（写到pyproject.toml中的[tool.poetry.dev-dependencies]下）
poetry add gym=0.26.2 　　　　　　#指定具体的版本
```