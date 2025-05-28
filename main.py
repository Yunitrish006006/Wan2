# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
# python main.py --ckpt_dir ./Wan2.1-VACE-1.3B
import argparse
import datetime
import os
import sys
import time

import imageio
import numpy as np
import torch

import gradio as gr
from prompt_enhancer import PromptEnhancer

sys.path.insert(
    0, os.path.sep.join(os.path.realpath(__file__).split(os.path.sep)[:-2]))
import wan
from wan import WanVace, WanVaceMP
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS


class FixedSizeQueue:

    def __init__(self, max_size):
        self.max_size = max_size
        self.queue = []

    def add(self, item):
        self.queue.insert(0, item)
        if len(self.queue) > self.max_size:
            self.queue.pop()

    def get(self):
        return self.queue

    def __repr__(self):
        return str(self.queue)


class VACEInference:

    def __init__(self,
                 cfg,
                 skip_load=False,
                 gallery_share=True,
                 gallery_share_limit=5):
        self.cfg = cfg
        self.save_dir = cfg.save_dir
        self.gallery_share = gallery_share
        self.gallery_share_data = FixedSizeQueue(max_size=gallery_share_limit)
        self.prompt_enhancer = PromptEnhancer()
        if not skip_load:
            if not args.mp:
                self.pipe = WanVace(
                    config=WAN_CONFIGS[cfg.model_name],
                    checkpoint_dir=cfg.ckpt_dir,
                    device_id=0,
                    rank=0,
                    t5_fsdp=False,
                    dit_fsdp=False,
                    use_usp=False,
                )
            else:
                self.pipe = WanVaceMP(
                    config=WAN_CONFIGS[cfg.model_name],
                    checkpoint_dir=cfg.ckpt_dir,
                    use_usp=True,
                    ulysses_size=cfg.ulysses_size,
                    ring_size=cfg.ring_size)

    def create_ui(self, *args, **kwargs):
        gr.Markdown("""
                    <div style="text-align: center; font-size: 24px; font-weight: bold; margin-bottom: 15px;">
                        <a href="https://ali-vilab.github.io/VACE-Page/" style="text-decoration: none; color: inherit;">直播帶貨 Demo</a>
                    </div>
                    """)
        with gr.Row(variant='panel', equal_height=True):
            with gr.Column(scale=1, min_width=0):
                self.src_video = gr.Video(
                    label="src_video",
                    sources=['upload'],
                    value=None,
                    interactive=True)
            with gr.Column(scale=1, min_width=0):
                self.src_mask = gr.Video(
                    label="src_mask",
                    sources=['upload'],
                    value=None,
                    interactive=True)
        #
        with gr.Row(variant='panel', equal_height=True):
            with gr.Column(scale=1, min_width=0):
                with gr.Row(equal_height=True):
                    self.src_ref_image_1 = gr.Image(
                        label='src_ref_image_1',
                        height=200,
                        interactive=True,
                        type='filepath',
                        image_mode='RGB',
                        sources=['upload'],
                        elem_id="src_ref_image_1",
                        format='png')
                    self.src_ref_image_2 = gr.Image(
                        label='src_ref_image_2',
                        height=200,
                        interactive=True,
                        type='filepath',
                        image_mode='RGB',
                        sources=['upload'],
                        elem_id="src_ref_image_2",
                        format='png')
                    self.src_ref_image_3 = gr.Image(
                        label='src_ref_image_3',
                        height=200,
                        interactive=True,
                        type='filepath',
                        image_mode='RGB',
                        sources=['upload'],
                        elem_id="src_ref_image_3",
                        format='png')
        with gr.Row(variant='panel', equal_height=True):
            with gr.Column(scale=1):
                self.prompt = gr.Textbox(
                    show_label=False,
                    placeholder="positive_prompt_input",
                    elem_id='positive_prompt',
                    container=True,
                    autofocus=True,
                    elem_classes='type_row',
                    visible=True,
                    lines=2)
                with gr.Row():
                    self.enhance_prompt_button = gr.Button(
                        value='增強提示詞',
                        elem_classes='type_row',
                        elem_id='enhance_prompt_button',
                        visible=True)
                    self.detail_level = gr.Slider(
                        label='詳細程度',
                        minimum=1,
                        maximum=3,
                        step=1,
                        value=2,
                        interactive=True)
                self.negative_prompt = gr.Textbox(
                    show_label=False,
                    value=self.pipe.config.sample_neg_prompt,
                    placeholder="negative_prompt_input",
                    elem_id='negative_prompt',
                    container=True,
                    autofocus=False,
                    elem_classes='type_row',
                    visible=True,
                    interactive=True,
                    lines=1)
        #
        with gr.Row(variant='panel', equal_height=True):
            with gr.Column(scale=1, min_width=0):
                gr.Markdown("""
                ### 參數說明:
                - **shift_scale** (0-100): 控制生成影片時的變化程度，值越大變化越明顯
                - **sample_steps** (1-100): 生成影片的採樣步驟數，步驟越多品質越好但耗時更長
                - **context_scale** (0-2): 控制參考圖像的影響程度，值越大越強調參考圖像特徵
                - **guide_scale** (1-10): 控制文字提示詞的引導強度，值越大越貼近提示詞描述
                """)
                with gr.Row(equal_height=True):
                    self.shift_scale = gr.Slider(
                        label='shift_scale',
                        minimum=0.0,
                        maximum=100.0,
                        step=1.0,
                        value=16.0,
                        interactive=True)
                    self.sample_steps = gr.Slider(
                        label='sample_steps',
                        minimum=1,
                        maximum=100,
                        step=1,
                        value=10,
                        interactive=True)
                    self.context_scale = gr.Slider(
                        label='context_scale',
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                        interactive=True)
                    self.guide_scale = gr.Slider(
                        label='guide_scale',
                        minimum=1,
                        maximum=10,
                        step=0.5,
                        value=5.0,
                        interactive=True)
                    self.infer_seed = gr.Slider(
                        minimum=-1, maximum=10000000, value=2025, label="Seed")
        #
        with gr.Accordion(label="Usable without source video", open=False):
            with gr.Row(equal_height=True):
                self.output_height = gr.Textbox(
                    label='resolutions_height',
                    value=480,
                    # value=720,
                self.generate_button = gr.Button(
                    value='Run',
                    elem_classes='type_row',
                    elem_id='generate_button',
                    visible=True)
            with gr.Column(scale=1):
                self.refresh_button = gr.Button(value='\U0001f504')  # 🔄
        #
        self.output_gallery = gr.Gallery(
            label="output_gallery",
            value=[],
            interactive=False,
            allow_preview=True,
            preview=True)
        
        self.output_info = gr.Textbox(
            label="影片資訊",
            interactive=False,
            lines=4,
            max_lines=10,
            visible=True)

    def enhance_prompt(self, prompt, detail_level):
        """
        Enhance the user prompt using the PromptEnhancer.
        
        Args:
            prompt (str): The original user prompt
            detail_level (int): Level of detail to add (1-3)
            
        Returns:
            str: The enhanced prompt
        """
        enhanced = self.prompt_enhancer.enhance_prompt(prompt, detail_level=detail_level)
        return enhanced

    def generate(self, output_gallery, src_video, src_mask, src_ref_image_1,
                 src_ref_image_2, src_ref_image_3, prompt, negative_prompt,
                 shift_scale, sample_steps, context_scale, guide_scale,
                 infer_seed, output_height, output_width, frame_rate,
                 num_frames):
        start_time = time.time()
        output_height, output_width, frame_rate, num_frames = int(
            output_height), int(output_width), int(frame_rate), int(num_frames)
        src_ref_images = [
            x for x in [src_ref_image_1, src_ref_image_2, src_ref_image_3]
            if x is not None
        ]
        src_video, src_mask, src_ref_images = self.pipe.prepare_source(
            [src_video], [src_mask], [src_ref_images],
            num_frames=num_frames,
            image_size=SIZE_CONFIGS[f"{output_width}*{output_height}"],
            device=self.pipe.device)
        video = self.pipe.generate(
            prompt,
            src_video,
            src_mask,
            src_ref_images,
            size=(output_width, output_height),
            context_scale=context_scale,
            shift=shift_scale,
            sampling_steps=sample_steps,
            guide_scale=guide_scale,
            n_prompt=negative_prompt,
            seed=infer_seed,
            offload_model=True)

        name = '{0:%Y%m%d-%H%M%S}'.format(datetime.datetime.now())
        video_path = os.path.join(self.save_dir, f'cur_gallery_{name}.mp4')
        video_frames = (
            torch.clamp(video / 2 + 0.5, min=0.0, max=1.0).permute(1, 2, 3, 0) *
            255).cpu().numpy().astype(np.uint8)

        try:
            writer = imageio.get_writer(
                video_path,
                fps=frame_rate,
                codec='libx264',
                quality=8,
                macro_block_size=1)
            for frame in video_frames:
                writer.append_data(frame)
            writer.close()
            print(video_path)
        except Exception as e:
            raise gr.Error(f"Video save error: {e}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        info_text = f"""影片生成時間：{duration:.2f} 秒
        解析度：{output_width}x{output_height}
        幀率：{frame_rate}
        幀數：{num_frames}
        shift_scale：{shift_scale}
        sample_steps：{sample_steps}
        context_scale：{context_scale}
        guide_scale：{guide_scale}
        seed：{infer_seed}
        """

        if self.gallery_share:
            self.gallery_share_data.add(video_path)
            return self.gallery_share_data.get(), info_text
        else:
            return [video_path], info_text

    def set_callbacks(self, **kwargs):
        self.gen_inputs = [
            self.output_gallery, self.src_video, self.src_mask,
            self.src_ref_image_1, self.src_ref_image_2, self.src_ref_image_3,
            self.prompt, self.negative_prompt, self.shift_scale,
            self.sample_steps, self.context_scale, self.guide_scale,
            self.infer_seed, self.output_height, self.output_width,
            self.frame_rate, self.num_frames
        ]
        self.gen_outputs = [self.output_gallery, self.output_info]
        self.generate_button.click(
            self.generate,
            inputs=self.gen_inputs,
            outputs=self.gen_outputs,
            queue=True)
        self.enhance_prompt_button.click(
            self.enhance_prompt,
            inputs=[self.prompt, self.detail_level],
            outputs=[self.prompt],
            queue=True)
        self.refresh_button.click(
            lambda x: self.gallery_share_data.get()
            if self.gallery_share else x,
            inputs=[self.output_gallery],
            outputs=[self.output_gallery])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Argparser for VACE-WAN Demo:\n')
    parser.add_argument(
        '--server_port', dest='server_port', help='', type=int, default=7860)
    parser.add_argument(
        '--server_name', dest='server_name', help='', default='127.0.0.1')
    parser.add_argument('--root_path', dest='root_path', help='', default=None)
    parser.add_argument('--save_dir', dest='save_dir', help='', default='cache')
    parser.add_argument(
        "--mp",
        action="store_true",
        help="Use Multi-GPUs",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="vace-14B",
        choices=list(WAN_CONFIGS.keys()),
        help="The model name to run.")
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.")
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        # default='models/VACE-Wan2.1-1.3B-Preview',
        # default='models/Wan2.1-VACE-14B/',
        default='./Wan2.1-VACE-1.3B',
        help="The path to the checkpoint directory.",
    )
    parser.add_argument(
        "--offload_to_cpu",
        action="store_true",
        help="Offloading unnecessary computations to CPU.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    with gr.Blocks() as demo:
        infer_gr = VACEInference(
            args, skip_load=False, gallery_share=True, gallery_share_limit=5)
        infer_gr.create_ui()
        infer_gr.set_callbacks()
        allowed_paths = [args.save_dir]
        demo.queue(status_update_rate=1).launch(
            server_name=args.server_name,
            server_port=args.server_port,
            root_path=args.root_path,
            allowed_paths=allowed_paths,
            show_error=True,
            debug=True)
