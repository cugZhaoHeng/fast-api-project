import os
import json
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="File Operation Demo")

# 在当前目录下创建一个测试文件夹
WORK_DIR = Path(__file__).parent / "demo_files"
WORK_DIR.mkdir(exist_ok=True)

# 记录操作历史的文件
HISTORY_FILE = WORK_DIR / "history.json"


@app.get("/")
def root():
    """根路径，返回可用操作列表"""
    return {
        "available_operations": [
            "GET /create/{filename}",
            "GET /write/{filename}?content=xxx",
            "GET /read/{filename}",
            "GET /delete/{filename}",
            "GET /rename/{old}/{new}",
            "GET /exists/{filename}",
            "GET /list",
            "GET /create_with_registry"
        ]
    }


@app.get("/create/{filename}")
def create_file(filename: str):
    """创建空文件"""
    filepath = WORK_DIR / filename
    
    # 确保目录存在
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # 创建文件
    filepath.touch()
    
    return {
        "operation": "create",
        "filename": str(filepath),
        "status": "created"
    }


@app.get("/write/{filename}")
def write_file(filename: str, content: str = "Hello World!"):
    """写入内容到文件"""
    filepath = WORK_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return {
        "operation": "write",
        "filename": str(filepath),
        "content_length": len(content),
        "status": "written"
    }


@app.get("/read/{filename}")
def read_file(filename: str):
    """读取文件内容"""
    filepath = WORK_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return {
        "operation": "read",
        "filename": str(filepath),
        "content": content,
        "status": "read"
    }


@app.get("/delete/{filename}")
def delete_file(filename: str):
    """删除文件"""
    filepath = WORK_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    os.remove(filepath)
    
    return {
        "operation": "delete",
        "filename": str(filepath),
        "status": "deleted"
    }


@app.get("/rename/{old}/{new}")
def rename_file(old: str, new: str):
    """重命名文件"""
    old_path = WORK_DIR / old
    new_path = WORK_DIR / new
    
    if not old_path.exists():
        raise HTTPException(status_code=404, detail="Source file not found")
    
    if new_path.exists():
        raise HTTPException(status_code=409, detail="Target file already exists")
    
    old_path.rename(new_path)
    
    return {
        "operation": "rename",
        "from": str(old_path),
        "to": str(new_path),
        "status": "renamed"
    }


@app.get("/exists/{filename}")
def exists_file(filename: str):
    """检查文件是否存在"""
    filepath = WORK_DIR / filename
    exists = filepath.exists()
    
    return {
        "operation": "exists",
        "filename": str(filepath),
        "exists": exists
    }


@app.get("/list")
def list_files():
    """列出工作目录下的所有文件"""
    files = []
    for item in WORK_DIR.iterdir():
        files.append({
            "name": item.name,
            "is_file": item.is_file(),
            "size": item.stat().st_size if item.is_file() else None
        })
    
    return {
        "operation": "list",
        "directory": str(WORK_DIR),
        "files": files
    }


@app.post("/batch")
def batch_operations():
    """批量操作：创建多个文件，写入内容，然后删除一个"""
    results = []
    
    # 创建3个文件
    for i in range(1, 4):
        filename = f"batch_{i}.txt"
        filepath = WORK_DIR / filename
        
        filepath.touch()
        results.append({"op": "create", "file": filename})
        
        # 写入内容
        with open(filepath, 'w') as f:
            f.write(f"This is batch file {i}")
        results.append({"op": "write", "file": filename})
    
    # 读取第一个文件
    with open(WORK_DIR / "batch_1.txt", 'r') as f:
        content = f.read()
    results.append({"op": "read", "file": "batch_1.txt", "content": content})
    
    # 删除第三个文件
    os.remove(WORK_DIR / "batch_3.txt")
    results.append({"op": "delete", "file": "batch_3.txt"})
    
    return {"batch_results": results}


@app.get("/slow/{filename}")
def slow_operation(filename: str, delay: float = 2.0):
    """带延迟的操作，方便观察 Process Monitor 中的时间线"""
    filepath = WORK_DIR / filename
    
    # 模拟慢操作
    time.sleep(delay)
    filepath.touch()
    time.sleep(delay // 2)
    
    with open(filepath, 'w') as f:
        f.write("Written after delay")
    
    return {
        "operation": "slow_create_and_write",
        "filename": str(filepath),
        "delay_seconds": delay,
        "status": "completed"
    }


# 可选：模拟注册表操作
try:
    import winreg
    HAS_WINREG = True
except ImportError:
    HAS_WINREG = False


@app.get("/create_with_registry")
def create_with_registry(filename: str = "from_registry.txt"):
    """创建文件并写入注册表记录"""
    filepath = WORK_DIR / filename
    filepath.touch()
    
    results = {"file_created": str(filepath)}
    
    # 尝试写注册表（需要管理员权限可能失败，只是演示）
    if HAS_WINREG:
        try:
            key = winreg.CreateKey(winreg.HKEY_CURRENT_USER, 
                                   r"Software\ProcMonDemo")
            winreg.SetValueEx(key, "last_file", 0, winreg.REG_SZ, str(filepath))
            winreg.CloseKey(key)
            results["registry_written"] = r"HKCU\Software\ProcMonDemo\last_file"
        except Exception as e:
            results["registry_error"] = str(e)
    
    return results


if __name__ == "__main__":
    import uvicorn
    print(f"工作目录: {WORK_DIR.absolute()}")
    print(f"启动服务: http://localhost:8000")
    print(f"API文档: http://localhost:8000/docs")
    uvicorn.run(app, host="127.0.0.1", port=8000)