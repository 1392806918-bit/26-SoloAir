#!/usr/bin/env python3
import subprocess
import os
import sys
import re

QWEN_DIR = "/home/elf/model/demo_Linux_aarch64"
OUTPUT_FILE = "/home/elf/model/recorded_text.txt"
REPORT_FILE = "/home/elf/model/report.txt"

SHERPA_CMD = [
    "/home/elf/miniconda3/envs/sherpa_env/bin/sherpa-onnx-alsa",
    "--provider=rknn",
    "--encoder=./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/encoder.rknn",
    "--decoder=./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/decoder.rknn",
    "--joiner=./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/joiner.rknn",
    "--tokens=./sherpa-onnx-rk3588-streaming-zipformer-small-bilingual-zh-en-2023-02-16/tokens.txt",
    "plughw:4,0"
]

def run_sherpa_and_save():
    print("🎤 开始语音识别... 请说话，按 Ctrl+C 结束")
    print("="*50)
    
    process = subprocess.Popen(
        SHERPA_CMD,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd="/home/elf/model"
    )
    
    current_text = ""
    pattern = re.compile(r'^\s*(\d+)[:：]\s*(.+)$')
    
    try:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            line = line.strip()
            match = pattern.match(line)
            if match:
                text = match.group(2)
                text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)
                if text:
                    current_text = text
    except KeyboardInterrupt:
        process.terminate()
        print("\n" + "="*50)
        print("⏹️ 录音结束")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(current_text)
    
    print(f"\n📁 识别结果已保存: {OUTPUT_FILE}")
    print(f"识别内容: {current_text}")
    return current_text

def get_image_description():
    print("\n📷 正在分析图片...")
    
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = './lib'
    env['RKLLM_LOG_LEVEL'] = '1'
    
    demo_path = f"{QWEN_DIR}/demo"
    cmd = [
        demo_path,
        "test1.png",
        "../qwen2_vl_2b_vision_rk3588.rknn",
        "../qwen2-vl-2b-instruct_W8A8_rk3588.rkllm",
        "2048", "4096", "3",
        "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"
    ]
    
    try:
        import pexpect
        child = pexpect.spawn(cmd[0], cmd[1:], cwd=QWEN_DIR, env=env, encoding='utf-8', timeout=120)
        child.expect(['user:', 'user: '], timeout=60)
        child.sendline("1")
        child.expect(['user:', 'user: '], timeout=120)
        output = child.before
        child.sendline('exit')
        child.terminate()
        
        # 提取描述
        lines = output.split('\n')
        for line in lines:
            if 'robot:' in line:
                parts = line.split('robot:', 1)
                desc = parts[1].strip()
                if desc and len(desc) > 10:
                    print(f"📷 图片描述: {desc[:200]}")
                    return desc
        return None
    except Exception as e:
        print(f"❌ 图片分析失败: {e}")
        return None

def generate_inspiration(image_desc, user_topic):
    print("\n💡 正在生成视频灵感...")
    
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = './lib'
    env['RKLLM_LOG_LEVEL'] = '1'
    
    demo_path = f"{QWEN_DIR}/demo"
    cmd = [
        demo_path,
        "test1.png",
        "../qwen2_vl_2b_vision_rk3588.rknn",
        "../qwen2-vl-2b-instruct_W8A8_rk3588.rkllm",
        "2048", "4096", "3",
        "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"
    ]
    
    prompt = f"""基于以下图片内容生成视频拍摄灵感：图片内容：{image_desc}/n用户主题：{user_topic}请输出：1. 视频标题：1个;2. 拍摄灵感：3-5条，每条一句话;3. 推荐标签：8-12个，用#开头"""
    
    try:
        import pexpect
        child = pexpect.spawn(cmd[0], cmd[1:], cwd=QWEN_DIR, env=env, encoding='utf-8', timeout=180)
        child.expect(['user:', 'user: '], timeout=60)
        print(f"发送的 prompt: {prompt}")  # 添加这行调试
        child.sendline(prompt)
        child.expect(['user:', 'user: '], timeout=180)
        output = child.before
        child.sendline('exit')
        child.terminate()
        
        # 提取结果
        result = ""
        lines = output.split('\n')
        in_result = False
        for line in lines:
            if 'robot:' in line:
                parts = line.split('robot:', 1)
                result = parts[1].strip()
                in_result = True
            elif in_result and line.strip() and 'user:' not in line:
                result += "\n" + line.strip()
            elif 'user:' in line:
                in_result = False
        
        if not result:
            for line in lines:
                line = line.strip()
                if line and not any(skip in line for skip in ['I rkllm:', 'main:', 'Peak Memory']):
                    if 'user:' not in line:
                        result = line
                        break
        
        with open(REPORT_FILE, 'w', encoding='utf-8') as f:
            f.write(f"=== 图片描述 ===\n{image_desc}\n\n")
            f.write(f"=== 用户主题 ===\n{user_topic}\n\n")
            f.write(f"=== 生成的灵感/标签 ===\n{result}")
        
        print(f"\n📄 灵感已保存到: {REPORT_FILE}")
        print("\n" + "="*50)
        print("生成的灵感/标签：")
        print("="*50)
        print(result)
        return result
    except Exception as e:
        print(f"❌ 灵感生成失败: {e}")
        return None

def main():
    print("="*50)
    print("智能语音助手 - 图片分析 + 视频灵感生成")
    print("="*50)
    
    user_topic = run_sherpa_and_save()
    
    if not user_topic:
        print("⚠️ 没有识别到内容，将仅基于图片生成灵感")
    
    image_desc = get_image_description()
    
    if not image_desc:
        print("❌ 无法获取图片描述")
        return
    
    generate_inspiration(image_desc, user_topic if user_topic else "生活日常")

if __name__ == "__main__":
    main()