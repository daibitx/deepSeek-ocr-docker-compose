from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoModel, AutoTokenizer
import torch
import os
from pathlib import Path
import time
from datetime import datetime
from typing import Optional, Literal
import shutil
from PIL import Image
import logging
import asyncio
import json
from huggingface_hub import snapshot_download

from config import settings, PROMPTS

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="DeepSeek OCR API",
    description="使用 DeepSeek-OCR 的光学字符识别 API",
    version="1.0.0"
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型全局变量
model = None
tokenizer = None
model_loaded = False
model_loading = False
model_error = None
download_progress = {"status": "idle", "progress": 0, "message": ""}


def load_model():
    """加载 DeepSeek-OCR 模型"""
    global model, tokenizer, model_loaded, model_loading, model_error, download_progress
    
    if model_loaded:
        return
    
    if model_loading:
        return
    
    model_loading = True
    model_error = None
    download_progress["status"] = "downloading"
    download_progress["progress"] = 0
    download_progress["message"] = "正在初始化模型下载..."

    try:
        logger.info(f"正在加载模型 {settings.MODEL_NAME}...")

        download_progress["progress"] = 10
        download_progress["message"] = "正在下载 tokenizer..."

        tokenizer = AutoTokenizer.from_pretrained(
            settings.MODEL_NAME,
            trust_remote_code=True
        )

        download_progress["progress"] = 30
        download_progress["message"] = "Tokenizer 下载完成，正在下载模型..."

        # 优先尝试 flash_attention_2
        try:
            logger.info("尝试使用 flash_attention_2 加载模型...")
            download_progress["message"] = "使用 flash_attention_2 加载模型..."
            model = AutoModel.from_pretrained(
                settings.MODEL_NAME,
                _attn_implementation='flash_attention_2',
                trust_remote_code=True,
                use_safetensors=True
            )
            logger.info("✓ 使用 flash_attention_2 加载成功")
        except Exception as e:
            logger.warning(f"flash_attention_2 不可用: {e}")
            logger.info("尝试使用 eager attention 加载模型...")
            download_progress["message"] = "使用 eager attention 加载模型..."
            model = AutoModel.from_pretrained(
                settings.MODEL_NAME,
                _attn_implementation='eager',
                trust_remote_code=True,
                use_safetensors=True
            )
            logger.info("✓ 使用 eager attention 加载成功")

        download_progress["progress"] = 80
        download_progress["message"] = "模型下载完成，正在初始化..."

        # 移动到 GPU
        if settings.DEVICE == "cuda" and torch.cuda.is_available():
            logger.info("正在将模型移动至 GPU...")
            download_progress["message"] = "模型移动至 GPU..."
            model = model.eval().cuda().to(torch.bfloat16)
            logger.info(f"✓ 模型加载到 GPU：{torch.cuda.get_device_name(0)}")
        else:
            model = model.eval()
            logger.info("✓ 模型加载到 CPU")

        download_progress["progress"] = 100
        download_progress["status"] = "completed"
        download_progress["message"] = "✓ 模型加载完成，已准备就绪"

        model_loaded = True
        model_loading = False
        logger.info("✓ 模型加载完成")

    except Exception as e:
        model_loading = False
        model_error = str(e)
        download_progress["status"] = "error"
        download_progress["progress"] = 0
        download_progress["message"] = f"加载失败: {str(e)}"
        logger.error(f"加载模型时发生错误: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    logger.info("启动 DeepSeek OCR API...")

    # 创建目录
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    logger.info("✓ API 已就绪（模型将在首次调用时加载）")


@app.get("/")
async def root():
    """根接口"""
    return {
        "message": "DeepSeek OCR API",
        "version": "1.0.0",
        "model": settings.MODEL_NAME,
        "model_loaded": model_loaded,
        "device": settings.DEVICE,
        "endpoints": {
            "health": "/health",
            "ocr": "/api/ocr",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "正常运行",
        "model_loaded": model_loaded,
        "model_loading": model_loading,
        "model_error": model_error,
        "download_progress": download_progress,
        "device": settings.DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/download-model")
async def download_model(background_tasks: BackgroundTasks):
    """后台下载模型"""
    global model_loading, download_progress

    if model_loaded:
        return {"status": "already_loaded", "message": "模型已加载"}

    if model_loading:
        return {"status": "downloading", "message": "下载中", "progress": download_progress}

    background_tasks.add_task(load_model)

    return {"status": "started", "message": "模型下载已开始"}


@app.get("/api/download-progress")
async def get_download_progress():
    """获取模型下载进度"""
    return {
        "model_loaded": model_loaded,
        "model_loading": model_loading,
        "progress": download_progress
    }


@app.post("/api/ocr")
async def process_ocr(
    file: UploadFile = File(...),
    mode: Literal["free_ocr", "markdown", "grounding", "parse_figure", "detailed"] = Form("markdown"),
    custom_prompt: Optional[str] = Form(None)
):
    """
    处理 OCR 图像并提取文本

    参数:
        file: 上传的图片文件（JPG, PNG, PDF, WEBP）
        mode: OCR 模式
        custom_prompt: 自定义提示词（覆盖 mode）

    返回:
        JSON：提取的文本内容和相关信息
    """

    # 检查模型
    if not model_loaded:
        try:
            load_model()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"模型不可用：{str(e)}"
            )

    # 校验文件类型
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型。允许：{settings.ALLOWED_EXTENSIONS}"
        )

    # 生成唯一文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = f"{timestamp}_{file.filename}"
    upload_path = os.path.join(settings.UPLOAD_DIR, unique_id)

    try:
        # 保存上传文件
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 校验文件大小
        file_size = os.path.getsize(upload_path)
        if file_size > settings.MAX_FILE_SIZE:
            os.remove(upload_path)
            raise HTTPException(
                status_code=400,
                detail=f"文件过大，最大支持 {settings.MAX_FILE_SIZE / 1024 / 1024}MB"
            )

        # 校验图片格式
        try:
            img = Image.open(upload_path)
            img_size = img.size
            img.close()
        except Exception as e:
            os.remove(upload_path)
            raise HTTPException(
                status_code=400,
                detail=f"文件不是有效的图片：{str(e)}"
            )

        # 选择 prompt
        prompt = custom_prompt if custom_prompt else PROMPTS.get(mode, PROMPTS["markdown"])

        # 创建输出目录
        output_dir = os.path.join(settings.OUTPUT_DIR, timestamp)
        os.makedirs(output_dir, exist_ok=True)

        # 执行 OCR
        logger.info(f"正在处理：{unique_id}（模式：{mode}）")
        start_time = time.time()

        if model is None or tokenizer is None:
            raise HTTPException(
                status_code=503,
                detail="模型未正确初始化"
            )

        result = model.infer(
            tokenizer,
            prompt=prompt,
            image_file=upload_path,
            output_path=output_dir,
            base_size=settings.BASE_SIZE,
            image_size=settings.IMAGE_SIZE,
            crop_mode=settings.CROP_MODE,
            save_results=True,
            test_compress=True
        )

        processing_time = time.time() - start_time
        logger.info(f"✓ 处理完成，耗时 {processing_time:.2f} 秒")

        # 读取结果文件
        result_file = os.path.join(output_dir, "result.mmd")
        text_content = ""

        if os.path.exists(result_file):
            with open(result_file, 'r', encoding='utf-8') as f:
                text_content = f.read()

        # 返回结果
        return JSONResponse(content={
            "success": True,
            "text": text_content or result,
            "mode": mode,
            "prompt": prompt,
            "processing_time": round(processing_time, 2),
            "image_size": img_size,
            "file_size": file_size,
            "timestamp": timestamp,
            "output_dir": output_dir,
            "metadata": {
                "filename": file.filename,
                "unique_id": unique_id,
                "device": settings.DEVICE
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR 处理失败: {str(e)}")
        if os.path.exists(upload_path):
            os.remove(upload_path)
        raise HTTPException(
            status_code=500,
            detail=f"处理图片时发生错误：{str(e)}"
        )


@app.get("/api/modes")
async def get_modes():
    """返回可用 OCR 模式"""
    return {
        "modes": {
            "free_ocr": {
                "description": "快速 OCR，不保留结构",
                "speed": "⚡⚡⚡ 非常快",
                "use_case": "提取一般文本"
            },
            "markdown": {
                "description": "转换为结构化 Markdown",
                "speed": "⚡⚡ 中速",
                "use_case": "文档结构化 OCR"
            },
            "grounding": {
                "description": "返回带坐标的文本（含 Bounding Box）",
                "speed": "⚡ 较慢",
                "use_case": "需要文本位置的 OCR"
            },
            "parse_figure": {
                "description": "解析图像中的图表、表格、流程图",
                "speed": "⚡⚡ 中速",
                "use_case": "图表识别"
            },
            "detailed": {
                "description": "输出非常详细的图像描述",
                "speed": "⚡⚡⚡ 很快",
                "use_case": "视觉分析"
            }
        }
    }


@app.delete("/api/cleanup")
async def cleanup_old_files(days: int = 7):
    """清理过期文件"""
    try:
        current_time = time.time()
        days_in_seconds = days * 24 * 60 * 60

        cleaned = {"uploads": 0, "outputs": 0}

        # 清理上传文件
        for file in Path(settings.UPLOAD_DIR).iterdir():
            if current_time - file.stat().st_mtime > days_in_seconds:
                file.unlink()
                cleaned["uploads"] += 1

        # 清理输出文件
        for folder in Path(settings.OUTPUT_DIR).iterdir():
            if folder.is_dir() and current_time - folder.stat().st_mtime > days_in_seconds:
                shutil.rmtree(folder)
                cleaned["outputs"] += 1

        return {
            "success": True,
            "cleaned": cleaned,
            "days": days
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
