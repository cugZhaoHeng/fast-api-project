# fast-api-project

一个以 Python 为主的个人实验仓库，内容覆盖了接口开发、Protobuf 通信、LangGraph/LLM 工作流、Selenium 自动化、以及 VAE/扩散模型等方向。  
从代码结构来看，这个仓库更像“按主题拆分的练习平台”，而不是一个只有单一入口的完整产品。

## 项目特点

- 同时包含 `FastAPI`、`Flask`、`Protobuf`、`LangGraph`、`PyTorch`、`Selenium` 等多类实验代码
- 部分实验带有明显的油气/地质建模背景，例如：
  - `src/pipeline` 中的天然气管网代理模型预测接口
  - `utils/tnavigator.py` 中的 `tNavigator`/`.inc` 数据处理
  - `src/vae` 中的 3D facies VAE 训练与生成
- 仓库内保留了较多样例数据、生成结果和中间文件，便于直接复现实验
- 不同目录通常有各自独立的运行入口，使用时建议按模块进入对应目录运行

## 目录概览

```text
.
├─ data/                    # 公共数据目录，包含 npy 样本、MNIST、ChromeDriver 等
├─ log/                     # 日志输出目录
├─ notebooks/               # Jupyter 笔记本实验
├─ src/
│  ├─ pipeline/             # 天然气管网代理模型预测接口、CSV/JSON/Protobuf 数据处理
│  ├─ protobuf_demo/        # FastAPI + Protobuf 通信示例
│  ├─ protobuf_test/        # 简化版 Protobuf 请求/响应示例
│  ├─ flask/                # Flask SSE / JSON 接口实验
│  ├─ flask_test/           # Flask 路由与接口练习
│  ├─ langgraph/            # LangGraph、LangChain、OpenAI 兼容模型调用实验
│  ├─ vae/                  # VAE / LDM / 3D facies 生成实验
│  ├─ diffuser/             # 基于 MNIST 的 VAE + latent diffusion 实验
│  ├─ fracture/             # 裂缝图像 VAE / diffusion 相关实验
│  ├─ selenium-test/        # Selenium 抓取与网页自动化实验
│  ├─ tnavigator_test/      # 地学网格 / INC 文件相关练习
│  └─ ...                   # 其他基础语法、NumPy、Pandas、时间处理等练习
└─ utils/                   # 通用日志、日期、JSON、tNavigator 工具
```

## 重点模块

### 1. `src/pipeline`

这是仓库里最接近“业务原型”的模块，主要围绕天然气管网代理模型预测做接口封装和数据转换。

代码中可以看到的能力包括：

- 读取本地 `flow` / `sensor` JSON 样例数据并提供接口
- 接收 `CSV` 上传后返回本地预测结果压缩包
- 接收 `JSON` 请求并流式返回 `Protobuf`
- 将预测 CSV 聚合为 `inputKey` / `inputValue` 结构
- 提供 FastAPI 路由与跨域配置

主要文件：

- `src/pipeline/entrance.py`：FastAPI 启动入口
- `src/pipeline/main_api.py`：核心接口定义
- `src/pipeline/process_csv_to_json.py`：将多个预测 CSV 合并为 JSON
- `src/pipeline/client_receive_protobuf.py`：Protobuf 流式客户端示例

### 2. `src/protobuf_demo` 与 `src/protobuf_test`

这两组代码用于练习 `FastAPI + Protobuf` 请求解析与返回，适合做二进制接口联调验证。

- `src/protobuf_demo/server.py`：接收 `ModelFullParam`，转字典后处理，再返回 Protobuf
- `src/protobuf_test/request_main.py`：接收简单 `Person` Protobuf 并返回 JSON

### 3. `src/langgraph`

这一部分主要是 LLM 工作流实验，包含：

- 使用 `LangGraph` 组织多节点 agent 流程
- 使用 `langchain_openai.ChatOpenAI` 连接 OpenAI 兼容接口
- 工具调用、消息记录、回调日志等封装

其中 `src/langgraph/main.py` 展示了一个“天气助手 + 美食助手”的串联式多 Agent 工作流。

### 4. `src/vae` / `src/diffuser` / `src/fracture`

这是生成模型实验区，方向比较集中：

- `src/vae/train_vae_facies.py`
  - 使用 `data/npy_files/` 下的三维 facies 数据训练 3D VAE
  - 数据被编码为 mud / sand / fluid 三类 one-hot
- `src/diffuser/main.py`
  - 基于 `MNIST` 训练 VAE，再在潜空间里做 diffusion
- `src/fracture/models.py`
  - 面向裂缝图像的 VAE 与扩散模型结构定义

### 5. `utils/tnavigator.py`

该工具脚本体现出仓库的地学建模背景，主要用于：

- 读取 `.inc` 文件
- 解析 `a*b` 形式的网格数据表达
- 将数据 reshape 为 `16 x 64 x 64`
- 批量转换为 `.npy` 文件，供后续模型训练使用

## 环境准备

仓库里目前没有统一的 `pyproject.toml` 或完整锁定环境文件，因此更适合按“基础依赖 + 可选模块依赖”方式安装。

建议使用 Python 3.10+。

### 1. 创建虚拟环境

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. 安装基础依赖

```powershell
pip install -r requirements.txt
```

### 3. 安装深度学习依赖

```powershell
pip install -r requirements-torch.txt
```

### 4. 安装代码里实际使用但未写入 `requirements.txt` 的依赖

```powershell
pip install fastapi uvicorn flask flask-cors protobuf pydantic python-multipart concurrent-log-handler
pip install langgraph langchain-core langchain-openai openai
```

说明：

- `src/pipeline`、`src/protobuf_demo`、`src/protobuf_test` 依赖 `FastAPI` / `uvicorn` / `protobuf`
- `src/flask`、`src/flask_test` 依赖 `Flask` / `flask-cors`
- `utils/logger.py` 依赖 `concurrent-log-handler`
- `src/langgraph` 依赖 `langgraph`、`langchain-core`、`langchain-openai`

## 运行方式

这个仓库没有统一入口。很多脚本使用了相对路径，所以最好进入对应子目录后再运行。

### 1. 启动天然气管网预测 FastAPI

```powershell
cd src\pipeline
python entrance.py
```

默认端口：

- `http://127.0.0.1:6112`
- 文档页通常可访问 `http://127.0.0.1:6112/docs`

### 2. 启动 Protobuf Demo 服务

```powershell
cd src\protobuf_demo
python server.py
```

默认端口：

- `http://127.0.0.1:8082`

### 3. 启动简化版 Protobuf 测试服务

```powershell
cd src\protobuf_test
python request_main.py
```

默认端口：

- `http://127.0.0.1:8000`

### 4. 启动 Flask SSE 示例

```powershell
cd src\flask
python main.py
```

默认端口：

- `http://127.0.0.1:6010`

### 5. 运行 LangGraph 示例

```powershell
cd src\langgraph
python main.py
```

注意：

- 该目录中的部分脚本写死了 OpenAI 兼容接口地址或测试模型名，运行前请按自己的环境修改

### 6. 训练 3D facies VAE

```powershell
cd src\vae
python train_vae_facies.py
```

依赖数据：

- `data/npy_files/`

### 7. 运行 MNIST latent diffusion 实验

```powershell
cd src\diffuser
python main.py
```

依赖数据：

- `data/MNIST/`

### 8. 运行 Selenium 抓取示例

```powershell
cd src\selenium-test
python douban.py
```

仓库已包含一个 ChromeDriver：

- `data/chromedriver-win64/chromedriver.exe`

如 Chrome 版本不匹配，请自行替换，常用参考：

- 当前版本查询：https://googlechromelabs.github.io/chrome-for-testing/
- 历史版本下载：https://chromedriver.storage.googleapis.com/index.html

## 数据与输出说明

- `data/npy_files/`：大量 `.npy` 三维模型样本，可供 `src/vae` 训练使用
- `data/MNIST/`：MNIST 数据集，供 `src/diffuser` 使用
- `src/pipeline/data/`：接口调试用 JSON、Proto、ZIP 数据
- `log/`：日志输出目录，由 `utils/logger.py` 自动创建与滚动写入

## 使用建议

- 把这个仓库当作“实验集合”来用，不要期待一个统一启动命令
- 优先从 `src/pipeline`、`src/protobuf_demo`、`src/langgraph`、`src/vae` 这几个目录开始阅读
- 运行脚本前先确认当前工作目录，因为不少脚本直接使用相对路径读取数据
- 部分脚本写死了本地路径、远程地址、端口或模型名，运行前建议先检查文件顶部配置
- 根目录 `main.py` 目前更像临时调试/数据片段记录，不建议作为仓库入口

## Git 提交规范

仓库原 README 中保留了 Conventional Commits 的约定，这个习惯仍然推荐继续使用。

常见类型：

- `feat`：新功能
- `fix`：修复问题
- `docs`：文档更新
- `style`：格式调整，不影响逻辑
- `refactor`：重构
- `perf`：性能优化
- `test`：测试相关
- `chore`：杂项维护
- `ci`：CI 配置调整
- `build`：构建或依赖相关修改

推荐格式：

```text
<type>(<scope>): <subject>
```

示例：

```text
feat(api): add pagination to user list endpoint
```
