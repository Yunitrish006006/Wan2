# -*- coding: utf-8 -*-
# Demo for the prompt enhancer

import gradio as gr
from prompt_enhancer import PromptEnhancer

# Initialize the prompt enhancer
enhancer = PromptEnhancer()

def enhance_prompt_function(prompt, detail_level):
    """Enhance the prompt using the PromptEnhancer"""
    if not prompt:
        return "請先輸入提示詞", "", ""
    
    result = enhancer.enhance_prompt_with_examples(prompt, detail_level)
    enhanced = result["enhanced_prompt"]
    examples = result["examples"]
    
    # Prepare example output
    examples_text = "相似提示詞範例：\n" + "\n".join(f"- {ex}" for ex in examples)
    
    # Calculate added keywords
    added = enhanced.replace(prompt, "").strip()
    if added.startswith(","):
        added = added[1:].strip()
    
    return enhanced, added, examples_text

# Create a simple Gradio interface
with gr.Blocks(title="提示詞增強示範") as demo:
    gr.Markdown("## 提示詞增強工具")
    gr.Markdown("此工具可以將簡單的提示詞增強為更加詳細與精準的提示詞，提高生成影片的品質。")
    
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="原始提示詞",
                placeholder="請輸入簡單的提示詞...",
                lines=2
            )
            detail_level = gr.Slider(
                label="詳細程度",
                minimum=1,
                maximum=3,
                step=1,
                value=2
            )
            enhance_button = gr.Button("增強提示詞")
        
        with gr.Column():
            enhanced_output = gr.Textbox(
                label="增強後的提示詞",
                lines=3
            )
            added_keywords = gr.Textbox(
                label="新增關鍵字",
                lines=2
            )
            examples_output = gr.Textbox(
                label="範例參考",
                lines=4
            )
    
    enhance_button.click(
        enhance_prompt_function,
        inputs=[prompt_input, detail_level],
        outputs=[enhanced_output, added_keywords, examples_output]
    )
    
    gr.Markdown("### 使用說明")
    gr.Markdown("""
    1. 在「原始提示詞」欄位中輸入簡單的描述
    2. 調整「詳細程度」的滑桿（1=基本增強，3=最大增強）
    3. 點擊「增強提示詞」按鈕
    4. 查看增強後的提示詞、新增的關鍵字以及參考範例
    
    範例輸入：
    - 紅色手機在桌上旋轉
    - 模特展示新款連衣裙
    - 室內明亮客廳場景
    """)

if __name__ == "__main__":
    demo.launch()
