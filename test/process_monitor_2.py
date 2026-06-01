import os
import tempfile
import shutil
import uuid
import json
from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse

app = FastAPI(title="Temp File Demo - 展示中间文件的创建与删除")

# 工作目录
WORK_DIR = Path(__file__).parent / "output_files"
WORK_DIR.mkdir(exist_ok=True)


@app.get("/generate_report")
def generate_report(
    format: str = Query("json", description="输出格式: json, csv, 或 text"),
    rows: int = Query(10, description="生成的数据行数")
):
    """
    模拟生成报告的过程：
    1. 创建临时目录
    2. 生成多个中间文件（原始数据、处理中数据、备份）
    3. 处理数据
    4. 生成最终文件
    5. 清理所有中间文件
    """
    
    # 生成唯一的任务ID
    task_id = str(uuid.uuid4())[:8]
    final_filename = WORK_DIR / f"report_{task_id}.{format}"
    
    # === 第1步：创建临时目录 ===
    temp_dir = Path(tempfile.mkdtemp(prefix=f"report_{task_id}_"))
    print(f"[Step 1] 创建临时目录: {temp_dir}")
    
    # === 第2步：创建中间文件1 - 原始数据 ===
    raw_data_file = temp_dir / "raw_data.json"
    raw_data = []
    for i in range(rows):
        raw_data.append({
            "id": i + 1,
            "value": i * 100,
            "name": f"Item_{i+1}"
        })
    
    with open(raw_data_file, 'w', encoding='utf-8') as f:
        json.dump(raw_data, f, indent=2)
    print(f"[Step 2] 创建原始数据文件: {raw_data_file}")
    
    # === 第3步：创建中间文件2 - 处理中数据（添加计算字段）===
    processing_file = temp_dir / "processing_data.json"
    processed_data = []
    for item in raw_data:
        processed_item = item.copy()
        processed_item["value_squared"] = item["value"] ** 2
        processed_item["value_halved"] = item["value"] / 2
        processed_data.append(processed_item)
    
    with open(processing_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2)
    print(f"[Step 3] 创建处理中数据文件: {processing_file}")
    
    # === 第4步：创建中间文件3 - 备份文件（防止处理失败）===
    backup_file = temp_dir / "processing_data.backup"
    shutil.copy2(processing_file, backup_file)
    print(f"[Step 4] 创建备份文件: {backup_file}")
    
    # === 第5步：再次读取处理文件进行二次处理 ===
    with open(processing_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 添加更多计算
    for item in data:
        item["processed_at"] = "2026-01-01 12:00:00"
    
    # 写入更新后的数据（覆盖）
    with open(processing_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"[Step 5] 更新处理文件: {processing_file}")
    
    # === 第6步：创建最终文件（根据格式生成）===
    if format == "json":
        # 直接复制处理后的数据
        shutil.copy2(processing_file, final_filename)
        print(f"[Step 6] 生成最终 JSON 文件: {final_filename}")
        
    elif format == "csv":
        # 生成 CSV 文件
        import csv
        with open(final_filename, 'w', newline='', encoding='utf-8') as f:
            if processed_data:
                writer = csv.DictWriter(f, fieldnames=processed_data[0].keys())
                writer.writeheader()
                writer.writerows(processed_data)
        print(f"[Step 6] 生成最终 CSV 文件: {final_filename}")
        
    elif format == "text":
        # 生成文本报告
        with open(final_filename, 'w', encoding='utf-8') as f:
            f.write("=" * 50 + "\n")
            f.write(f"报告生成时间: 2026-01-01 12:00:00\n")
            f.write(f"数据行数: {rows}\n")
            f.write("=" * 50 + "\n\n")
            for item in processed_data:
                f.write(f"ID: {item['id']}, Name: {item['name']}, Value: {item['value']}\n")
        print(f"[Step 6] 生成最终 TEXT 文件: {final_filename}")
    
    # === 第7步：删除所有中间文件和临时目录 ===
    # 删除处理文件
    if processing_file.exists():
        os.remove(processing_file)
        print(f"[Step 7] 删除中间文件: {processing_file}")
    
    # 删除备份文件
    if backup_file.exists():
        os.remove(backup_file)
        print(f"[Step 7] 删除备份文件: {backup_file}")
    
    # 删除原始数据文件
    if raw_data_file.exists():
        os.remove(raw_data_file)
        print(f"[Step 7] 删除原始数据文件: {raw_data_file}")
    
    # 删除临时目录
    try:
        os.rmdir(temp_dir)  # 目录应该是空的
        print(f"[Step 7] 删除临时目录: {temp_dir}")
    except OSError:
        print(f"[Step 7] 临时目录非空，强制删除: {temp_dir}")
        shutil.rmtree(temp_dir)
    
    return {
        "status": "success",
        "task_id": task_id,
        "final_file": str(final_filename),
        "temporary_files_created": [
            str(raw_data_file),
            str(processing_file),
            str(backup_file)
        ],
        "temporary_files_deleted": True,
        "note": "所有中间文件已被删除，只有最终文件保留"
    }


@app.get("/generate_with_checkpoint")
def generate_with_checkpoint(
    checkpoint: bool = Query(True, description="是否保留检查点文件")
):
    """
    模拟带检查点的生成过程，可以看到中间文件的存在
    checkpoint=False 时正常清理，checkpoint=True 时保留中间文件用于观察
    """
    
    task_id = str(uuid.uuid4())[:8]
    temp_dir = Path(tempfile.mkdtemp(prefix=f"checkpoint_{task_id}_"))
    final_file = WORK_DIR / f"final_{task_id}.txt"
    
    # 创建阶段文件
    stage1 = temp_dir / "stage1_raw.txt"
    stage2 = temp_dir / "stage2_processed.txt"
    stage3 = temp_dir / "stage3_filtered.txt"
    
    # Stage 1: 原始数据
    with open(stage1, 'w') as f:
        for i in range(100):
            f.write(f"Line {i}: Original data\n")
    
    # Stage 2: 处理
    with open(stage1, 'r') as fin, open(stage2, 'w') as fout:
        for line in fin:
            fout.write(line.upper())
    
    # Stage 3: 过滤
    with open(stage2, 'r') as fin, open(stage3, 'w') as fout:
        for i, line in enumerate(fin):
            if i % 10 == 0:  # 只保留每10行
                fout.write(line)
    
    # 最终输出
    shutil.copy2(stage3, final_file)
    
    files_info = {
        "stage1": str(stage1),
        "stage2": str(stage2),
        "stage3": str(stage3),
        "final": str(final_file)
    }
    
    if checkpoint:
        # 保留检查点文件用于观察
        checkpoint_dir = WORK_DIR / f"checkpoint_{task_id}"
        checkpoint_dir.mkdir(exist_ok=True)
        shutil.move(str(stage1), checkpoint_dir / "stage1_raw.txt")
        shutil.move(str(stage2), checkpoint_dir / "stage2_processed.txt")
        shutil.move(str(stage3), checkpoint_dir / "stage3_filtered.txt")
        shutil.rmtree(temp_dir)
        
        return {
            "status": "checkpoint_saved",
            "checkpoint_directory": str(checkpoint_dir),
            "files": files_info,
            "note": "中间文件已保留在 checkpoint_directory 中，方便观察"
        }
    else:
        # 正常情况：删除所有中间文件
        os.remove(stage1)
        os.remove(stage2)
        os.remove(stage3)
        os.rmdir(temp_dir)
        
        return {
            "status": "success",
            "final_file": str(final_file),
            "files_created_and_deleted": [
                str(stage1), str(stage2), str(stage3)
            ],
            "note": "所有中间文件已删除"
        }


@app.get("/list_outputs")
def list_outputs():
    """列出已生成的所有最终文件"""
    files = []
    for item in WORK_DIR.iterdir():
        if item.is_file():
            files.append({
                "name": item.name,
                "size": item.stat().st_size,
                "modified": item.stat().st_mtime
            })
    return {"output_files": files}


@app.get("/clean_outputs")
def clean_outputs():
    """清理所有生成的文件"""
    count = 0
    for item in WORK_DIR.iterdir():
        if item.is_file():
            os.remove(item)
            count += 1
    return {"cleaned": count, "files_removed": count}


if __name__ == "__main__":
    import uvicorn
    print(f"输出目录: {WORK_DIR.absolute()}")
    print(f"启动服务: http://localhost:8000")
    print(f"\n可用接口:")
    print(f"  - GET /generate_report?format=json&rows=5   → 生成报告（自动清理中间文件）")
    print(f"  - GET /generate_with_checkpoint?checkpoint=true  → 生成文件并保留中间文件")
    print(f"  - GET /list_outputs  → 查看所有输出文件")
    print(f"  - GET /clean_outputs → 清理输出目录")
    uvicorn.run(app, host="127.0.0.1", port=8000)