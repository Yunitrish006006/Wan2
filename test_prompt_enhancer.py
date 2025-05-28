# -*- coding: utf-8 -*-
# Test script for the prompt enhancer

from prompt_enhancer import PromptEnhancer

def test_prompt_enhancer():
    enhancer = PromptEnhancer()
    
    test_prompts = [
        "紅色手機在桌上旋轉",
        "模特展示新款連衣裙",
        "室內明亮客廳場景",
        "空白提示詞"
    ]
    
    print("=== Prompt Enhancer Test ===")
    for prompt in test_prompts:
        enhanced = enhancer.enhance_prompt(prompt, detail_level=2)
        print(f"原始提示詞: {prompt}")
        print(f"增強提示詞: {enhanced}")
        print("-" * 50)

if __name__ == "__main__":
    test_prompt_enhancer()
