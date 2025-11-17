# 🚀 DeepSeek OCR - 基于 DeepSeek-OCR 的 AI 文本识别系统

> 完整的 OCR 系统，使用 **DeepSeek-OCR（2025 年 10 月发布）** 模型，提供现代化 Web 界面与生产级的 REST API（适用于开发/测试环境）。

[![License: MIT](https://img.shields.io/badge/License-MIT%20(Dev%20Only)-yellow.svg)](LICENSE)
 [![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)](https://www.docker.com/)
 [![Python](https://img.shields.io/badge/Python-3.10+-green?logo=python)](https://www.python.org/)
 [![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-teal?logo=fastapi)](https://fastapi.tiangolo.com/)
 [![CUDA](https://img.shields.io/badge/CUDA-11.8+-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)

⚠️ **注意：本项目仅用于开发与测试，不适合生产环境。**

------

## ✨ 功能特点

- 🤖 **最新 DeepSeek-OCR AI 模型**
- 🌐 **现代化 Web 界面**（拖拽上传、实时进度）
- 📊 **模型下载进度条**
- 🎮 **Demo 演示模式（无需下载模型）**
- 🔌 **完整 REST API（基于 FastAPI）**
- 🐳 **支持 Docker Compose 一键部署**
- ⚡ **兼容 NVIDIA GPU / CUDA 加速**
- 📝 **多种 OCR 模式：Free、Markdown、Grounding、Parse Figure、Detailed**
- 🔓 **完全开源（MIT-Dev 测试许可）**

------

## 📦 系统要求

- **Docker 20.10+ / Docker Compose 2.0+**
- **NVIDIA GPU（CUDA 11.8+）**
- 推荐 **8GB+ VRAM**
- **10GB** 磁盘空间（模型缓存）
- 支持 **Windows / Linux / macOS（Docker Desktop）**

------

## 🚀 快速开始

### 1. 克隆仓库

```
git clone https://github.com/YOUR_USERNAME/deepseek-ocr.git
cd deepseek-ocr
```

### 2. 创建 `.env`

```
cp .env.example .env
```

### 3. 启动程序

```
docker-compose up -d
```

### 4. 访问服务

| 功能     | 地址                         |
| -------- | ---------------------------- |
| Web 界面 | http://localhost:3000        |
| API 文档 | http://localhost:8000/docs   |
| 健康检查 | http://localhost:8000/health |

### 5. 首次使用

通过 Web 界面点击“下载模型”即可自动下载 DeepSeek-OCR。
 如果不想等待，可使用 **Demo 模式** 体验界面。

------

## 📡 API 使用示例

```
curl -X POST "http://localhost:8000/api/ocr" \
  -F "file=@document.jpg" \
  -F "mode=markdown"
```

返回示例：

```
{
  "text": "# 文档标题\n\n识别内容...",
  "mode": "markdown",
  "processing_time": 2.5,
  "image_size": [1024, 768],
  "tokens": 2257
}
```

------

## 📜 支持的识别模式

| 模式         | 描述               | 用途               |
| ------------ | ------------------ | ------------------ |
| free_ocr     | 快速 OCR，无结构   | 普通文本           |
| markdown     | 输出 Markdown 格式 | 文档结构化         |
| grounding    | OCR + 坐标信息     | OCR 分析、表单识别 |
| detailed     | 图像详细描述       | 图像理解           |
| parse_figure | 提取图表内容       | 学术/数据图表      |

------

## 🧱 项目结构

```
deepseek-ocr/
├── backend/          # FastAPI 后端
├── frontend/         # 网页前端（Nginx）
├── uploads/          # 上传文件
├── outputs/          # OCR 输出
├── docs/             # 文档
└── docker-compose.yml
```

------

## 🔧 配置（Environment Variables）

```
environment:
  - CUDA_VISIBLE_DEVICES=0
  - MODEL_NAME=deepseek-ai/DeepSeek-OCR
  - HF_HOME=/root/.cache/huggingface
```

如需预下载模型：

```
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="deepseek-ai/DeepSeek-OCR",
    cache_dir="/path/to/local/cache",
    resume_download=True
)
```

------

## 🐳 Docker 常用命令

```
docker-compose up -d      # 后台启动
docker-compose logs -f    # 查看日志
docker-compose restart    # 重启服务
docker-compose down       # 停止并删除容器
docker-compose build --no-cache  # 强制重建镜像
```

------

## 🐛 常见问题排查

### 1. GPU 未被 Docker 识别

```
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 2. 模型无法下载

- 检查网络
- 检查磁盘空间
- 查看 API 容器日志

### 3. 显存不足（OOM）

修改 `backend/config.py`：

```
BASE_SIZE = 640
```

------

## 📊 性能（A100 40GB）

| 模式      | 耗时 | 质量 | 结构信息      |
| --------- | ---- | ---- | ------------- |
| Free OCR  | ~24s | ⭐⭐⭐  | 基础          |
| Markdown  | ~39s | ⭐⭐⭐  | 完整 Markdown |
| Grounding | ~58s | ⭐⭐   | 坐标信息      |
| Detailed  | ~9s  | -    | 图像描述      |

------

## 🔒 安全说明

⚠️ **本项目无任何安全加固，不适用于生产环境**

- 无认证系统
- 无权限控制
- 对外开放 API 存在风险
- 仅供开发测试用途

请阅读：`SECURITY.md`

------

## 📝 许可证

MIT License（仅允许开发与测试用途）
 使用于生产环境需自担风险。

------

## 🤝 贡献方式

欢迎 PR！

步骤：

1. Fork 仓库
2. 创建新分支
3. 提交修改
4. 发起 Pull Request

请遵循 `CONTRIBUTING.md` 与 `CODE_OF_CONDUCT.md`。

------

## ⭐ 支持本项目

如果它对你有帮助：

- 欢迎在 GitHub ⭐Star
- 分享给他人
- 提交改进建议
