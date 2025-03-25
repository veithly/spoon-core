#!/usr/bin/env python
# spoon_ai/monitoring/standalone.py

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import sys

# 将父目录添加到sys.path，确保可以导入spoon_ai包
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("monitoring-service")

# 创建FastAPI应用
app = FastAPI(
    title="Crypto Monitoring Service",
    description="A service for monitoring cryptocurrency metrics and sending alerts",
    version="0.1.0",
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 导入并注册API路由
from .api.routes import router as monitoring_router
app.include_router(monitoring_router)

# 添加健康检查端点
@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok", "service": "monitoring"}

# 启动任务调度器
from .core.tasks import MonitoringTaskManager
task_manager = MonitoringTaskManager()

@app.on_event("startup")
async def startup_event():
    """服务启动时的事件处理"""
    logger.info("Starting monitoring service...")
    # 调度器已经在MonitoringTaskManager初始化时启动

@app.on_event("shutdown")
async def shutdown_event():
    """服务关闭时的事件处理"""
    logger.info("Shutting down monitoring service...")
    # 停止调度器
    task_manager.scheduler.stop()

# 如果直接运行此文件，则启动服务
if __name__ == "__main__":
    # 获取配置参数，可以从环境变量读取
    host = os.getenv("MONITORING_HOST", "0.0.0.0")
    port = int(os.getenv("MONITORING_PORT", "8080"))
    
    logger.info(f"Starting monitoring service on {host}:{port}")
    uvicorn.run("spoon_ai.monitoring.main:app", host=host, port=port, reload=True)