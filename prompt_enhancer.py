# -*- coding: utf-8 -*-
# Copyright (c) 2025
# Prompt enhancer for VACE

import re
import torch
import os

class PromptEnhancer:
    """
    A class to enhance user prompts by making them more detailed and precise.
    This enhances the quality of the generated videos by the Wan2 VACE model.
    """
    
    def __init__(self, base_keywords=None):
        """
        Initialize the PromptEnhancer.
        
        Args:
            base_keywords (list, optional): List of keywords to be used for enhancement.
        """
        self.base_keywords = base_keywords or [
            "高品質", "細節豐富", "清晰", "高解析度", "專業錄影",
            "自然光線", "完美構圖", "流暢動作", "高影格", "生動色彩"
        ]
        
        # Common categories for enhancement
        self.categories = {
            "商品": ["商品展示", "產品特寫", "商品細節", "360度展示", "使用示範", "專業照明", "質感呈現", "立體感", "完美比例", "精緻質地"],
            "人物": ["自然表情", "專業妝容", "舒適姿勢", "生動表情", "自然互動", "高質感服裝", "專業造型", "完美燈光", "膚色自然", "細節刻畫"],
            "場景": ["整潔背景", "適當照明", "環境協調", "專業佈置", "氛圍營造", "豐富紋理", "和諧色調", "精心設計", "空間層次", "細節專注"],
            "風格": ["時尚", "現代", "簡約", "奢華", "溫暖", "活力", "專業", "精緻", "高端", "典雅", "前衛", "復古", "自然"]
        }
        
        # Video specific enhancements
        self.video_quality = ["順暢轉場", "連貫敘事", "穩定視角", "專業運鏡", "流暢節奏", "精準聚焦"]
        
        # Different light conditions
        self.lighting = ["自然採光", "柔和打光", "專業燈光", "高對比度", "暖色調光源", "冷色調光源", "側光打亮", "頂光照明"]
        
    def detect_category(self, prompt):
        """Detect which category the prompt belongs to."""
        prompt_lower = prompt.lower()
        
        # Simple detection based on keywords
        if any(k in prompt_lower for k in ["商品", "產品", "展示", "銷售", "貨品"]):
            return "商品"
        elif any(k in prompt_lower for k in ["人物", "模特", "主持人", "人像"]):
            return "人物"
        elif any(k in prompt_lower for k in ["場景", "背景", "環境", "室內", "戶外"]):
            return "場景"
        
        # Default to general enhancement
        return None
      def enhance_prompt(self, prompt, detail_level=2):
        """
        Enhance the user prompt by adding more detailed descriptors.
        
        Args:
            prompt (str): The original user prompt
            detail_level (int): Level of detail to add (1-3)
                1 = minimal enhancement
                2 = moderate enhancement
                3 = maximum enhancement
                
        Returns:
            str: The enhanced prompt
        """
        if not prompt or prompt.strip() == "":
            return prompt
            
        # Detect the category
        category = self.detect_category(prompt)
        
        # Base enhancement
        enhancements = []
        
        # Add category-specific enhancements
        if category and category in self.categories:
            # Select a number of category keywords based on detail level
            num_keywords = min(detail_level * 2, len(self.categories[category]))
            category_keywords = self.categories[category][:num_keywords]
            enhancements.extend(category_keywords)
        
        # Add general quality enhancements
        num_base = min(detail_level * 2, len(self.base_keywords))
        enhancements.extend(self.base_keywords[:num_base])
        
        # Add video-specific quality keywords
        num_video = min(detail_level, len(self.video_quality))
        enhancements.extend(self.video_quality[:num_video])
        
        # Add lighting suggestions based on detail level
        if detail_level >= 2:
            # For higher detail levels, add lighting suggestions
            num_light = min(detail_level - 1, len(self.lighting))
            enhancements.extend(self.lighting[:num_light])
            
        # Combine with original prompt
        # Check if the prompt already contains these keywords
        final_enhancements = []
        for keyword in enhancements:
            if keyword.lower() not in prompt.lower():
                final_enhancements.append(keyword)
        
        # Shuffle the enhancements slightly to add variety
        import random
        if len(final_enhancements) > 3:
            random.shuffle(final_enhancements)
        
        enhanced_prompt = prompt
        if final_enhancements:
            enhanced_prompt = f"{prompt}, {', '.join(final_enhancements)}"
            
        return enhanced_prompt
    
    def enhance_prompt_with_examples(self, prompt, detail_level=2):
        """
        Enhanced version that also shows examples of similar prompts
        
        Args:
            prompt (str): The original user prompt
            detail_level (int): Level of detail to add (1-3)
                
        Returns:
            dict: Dictionary containing the enhanced prompt and examples
        """
        enhanced = self.enhance_prompt(prompt, detail_level)
        
        # Common examples by category
        examples = {
            "商品": [
                "紅色手機360度展示，高品質，細節豐富，專業照明，質感呈現，完美比例",
                "藍色連衣裙商品特寫，高解析度，清晰，生動色彩，專業打光，立體感"
            ],
            "人物": [
                "女模特展示新款連衣裙，自然表情，專業妝容，高質感服裝，膚色自然，柔和打光",
                "男模特展示西裝，舒適姿勢，專業造型，完美燈光，細節刻畫，流暢動作"
            ],
            "場景": [
                "明亮客廳場景，整潔背景，適當照明，環境協調，豐富紋理，和諧色調，自然採光",
                "現代辦公室環境，專業佈置，氛圍營造，空間層次，細節專注，專業燈光"
            ]
        }
        
        category = self.detect_category(prompt)
        example_prompts = []
        
        if category and category in examples:
            example_prompts = examples[category]
        else:
            # General examples
            example_prompts = [
                "產品展示視頻，高品質，細節豐富，清晰，專業照明，完美構圖",
                "模特走秀，流暢動作，高解析度，專業錄影，自然光線"
            ]
            
        return {
            "enhanced_prompt": enhanced,
            "examples": example_prompts[:2]  # Return up to 2 examples
        }
