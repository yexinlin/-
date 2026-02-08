# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import time
import pyautogui
import easyocr
import numpy as np
import re
from functools import lru_cache
from deep_translator import GoogleTranslator
from rich.console import Console
from PIL import Image, ImageOps

# ========== 固定坐标参数 ==========
QUESTION_REGION = (345, 584, 591, 195)
OPTIONS_REGION = (327, 837, 637, 672)

pyautogui.PAUSE = 0.0001 
console = Console()

# 初始化时关闭段落检测以提速
READER = easyocr.Reader(['en', 'ch_sim'], gpu=False) 
TRANSLATOR = GoogleTranslator(source='en', target='zh-CN')

CORRECT_MAP = {'1f': 'if', 'lf': 'if', '1t': 'it', 'lt': 'it', 'll': 'if', 'at': 'at'}

@lru_cache(maxsize=1024)
def get_translation(word):
    if word in CORRECT_MAP: word = CORRECT_MAP[word]
    try:
        res = TRANSLATOR.translate(word).strip()
        res = re.sub(r'\(.*?\)|（.*?）|[^\u4e00-\u9fa5]', '', res)
        return res[:4]
    except: return ""

def get_img_hash(img):
    """4x4 哈希提速"""
    return list(img.resize((4, 4), Image.Resampling.NEAREST).convert('L').getdata())

# 增加全局计时器
stuck_timer = time.time()

def solve_extreme_speed(last_h):
    global stuck_timer
    start_time = time.perf_counter()
    
    # 1. 快速截图与画面对比
    q_snap_raw = pyautogui.screenshot(region=QUESTION_REGION)
    curr_h = get_img_hash(q_snap_raw)
    
    # 检查是否卡死超过 5 秒
    is_stuck = (time.time() - stuck_timer) > 5.0
    
    if not is_stuck and last_h and sum(abs(a - b) for a, b in zip(last_h, curr_h))/16 < 3:
        return False, last_h

    if is_stuck:
        console.print("[bold red]⏳ 检测到超时，强制刷新重试...[/bold red]")
        stuck_timer = time.time() # 重置计时器

    # 2. 题目识别：针对短词开启局部轻量放大
    q_gray = ImageOps.grayscale(q_snap_raw)
    # 局部放大 2 倍对 CPU 负担很小，但对识别 'if' 至关重要
    w, h = q_gray.size
    q_big = q_gray.resize((w*2, h*2), Image.Resampling.BILINEAR)
    
    q_res = READER.readtext(np.array(q_big), detail=0, paragraph=False, min_size=2)
    
    word_match = re.search(r'[a-zA-Z]+', "".join(q_res))
    if not word_match: return False, curr_h
    
    word = word_match.group(0).lower()
    target_cn = get_translation(word)
    if not target_cn: return False, curr_h

    # 3. 选项识别
    opt_snap = pyautogui.screenshot(region=OPTIONS_REGION)
    opt_array = np.array(ImageOps.grayscale(opt_snap))
    opt_res = READER.readtext(opt_array, detail=1, decoder='greedy')

    bins = ["" for _ in range(4)]
    h_step = OPTIONS_REGION[3] / 4
    for (bbox, text, prob) in opt_res:
        if not any(u'\u4e00' <= c <= u'\u9fff' for c in text): continue
        mid_y = (bbox[0][1] + bbox[2][1]) / 2
        idx = int(mid_y // h_step)
        if 0 <= idx < 4: bins[idx] += text

    # 4. 判定与点击
    best_idx, max_score = -1, -1
    for i, opt_text in enumerate(bins):
        if not opt_text: continue
        score = (100 if target_cn in opt_text else 0) + len(set(target_cn) & set(opt_text)) * 20
        if score > max_score:
            max_score, best_idx = score, i

    if best_idx != -1 and max_score >= 15:
        click_x = OPTIONS_REGION[0] + OPTIONS_REGION[2] / 2
        click_y = OPTIONS_REGION[1] + (best_idx + 0.5) * h_step
        pyautogui.click(click_x, click_y)
        
        # 成功点击，重置超时计时器
        stuck_timer = time.time()
        
        duration = time.perf_counter() - start_time
        console.print(f"⚡ [bold yellow]{word:10}[/bold yellow] | [green]{target_cn:6}[/green] | [bold cyan]{duration:.2f}s[/bold cyan]")
        return True, curr_h
    
    return False, curr_h

def main():
    console.print("[bold red]🚀 百词斩自动答题...[/bold red]")
    last_h = None
    while True:
        try:
            success, new_h = solve_extreme_speed(last_h)
            last_h = new_h
            # 答题成功后等待动画，失败则极速重试
            time.sleep(0.4 if success else 0.01)
        except KeyboardInterrupt: break
        except: continue

if __name__ == "__main__":
    main()
