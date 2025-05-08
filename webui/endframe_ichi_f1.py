import os
import sys
sys.path.append(os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './submodules/FramePack'))))

from diffusers_helper.hf_login import login

import os
import random
import time
import subprocess
import traceback  # デバッグログ出力用
# クロスプラットフォーム対応のための条件付きインポート
import yaml
import zipfile

import argparse

# PNGメタデータ処理モジュールのインポート
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from eichi_utils.png_metadata import (
    embed_metadata_to_png, extract_metadata_from_png, extract_metadata_from_numpy_array,
    PROMPT_KEY, SEED_KEY, SECTION_PROMPT_KEY, SECTION_NUMBER_KEY
)

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='127.0.0.1')
parser.add_argument("--port", type=int, default=8001)
parser.add_argument("--inbrowser", action='store_true')
parser.add_argument("--lang", type=str, default='ja', help="Language: ja, zh-tw, en")
args = parser.parse_args()

# Load translations from JSON files
from locales.i18n_extended import (set_lang, translate)
set_lang(args.lang)

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False
import json
import traceback
from datetime import datetime, timedelta

if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))
    print(translate("HF_HOMEを設定: {0}").format(os.environ['HF_HOME']))
else:
    print(translate("既存のHF_HOMEを使用: {0}").format(os.environ['HF_HOME']))
temp_dir = "./temp_for_zip_section_info"

# LoRAサポートの確認
has_lora_support = False
try:
    import lora_utils
    has_lora_support = True
    print(translate("LoRAサポートが有効です"))
except ImportError:
    print(translate("LoRAサポートが無効です（lora_utilsモジュールがインストールされていません）"))

# 設定モジュールをインポート（ローカルモジュール）
import os.path
from eichi_utils.video_mode_settings import (
    VIDEO_MODE_SETTINGS, get_video_modes, get_video_seconds, get_important_keyframes,
    get_copy_targets, get_max_keyframes_count, get_total_sections, generate_keyframe_guide_html,
    handle_mode_length_change, process_keyframe_change, MODE_TYPE_NORMAL
    # F1モードでは不要な機能のインポートを削除
)

# 設定管理モジュールをインポート
from eichi_utils.settings_manager import (
    get_settings_file_path,
    get_output_folder_path,
    initialize_settings,
    load_settings,
    save_settings,
    open_output_folder
)

# プリセット管理モジュールをインポート
from eichi_utils.preset_manager import (
    initialize_presets,
    load_presets,
    get_default_startup_prompt,
    save_preset,
    delete_preset
)

# キーフレーム処理モジュールをインポート
from eichi_utils.keyframe_handler import (
    ui_to_code_index,
    code_to_ui_index,
    unified_keyframe_change_handler,
    unified_input_image_change_handler,
    print_keyframe_debug_info
)

# 拡張キーフレーム処理モジュールをインポート
from eichi_utils.keyframe_handler_extended import extended_mode_length_change_handler
import gradio as gr
# UI関連モジュールのインポート
from eichi_utils.ui_styles import get_app_css
import torch
import einops
import safetensors.torch as sf
import numpy as np
import math

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, device_type, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

from eichi_utils.transformer_manager import TransformerManager
from eichi_utils.text_encoder_manager import TextEncoderManager

# Get available memory and determine if high VRAM mode should be used
free_mem_gb = get_cuda_free_memory_gb(gpu)

# For MPS (Apple Silicon), use a different threshold
if device_type == 'mps':
    # Apple Silicon typically has unified memory, so we use a lower threshold
    high_vram = free_mem_gb > 12  # Lower threshold for Apple Silicon
else:
    # Original threshold for CUDA devices
    high_vram = free_mem_gb > 100

print(translate('Free Memory {0} GB').format(free_mem_gb))
print(translate('High-VRAM Mode: {0}').format(high_vram))

# グローバルなモデル状態管理インスタンスを作成
# F1モードではuse_f1_model=Trueを指定
transformer_manager = TransformerManager(device=gpu, high_vram_mode=high_vram, use_f1_model=True)
text_encoder_manager = TextEncoderManager(device=gpu, high_vram_mode=high_vram)

try:
    tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
    tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
    vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

    # text_encoderとtext_encoder_2の初期化
    if not text_encoder_manager.ensure_text_encoder_state():
        raise Exception(translate("text_encoderとtext_encoder_2の初期化に失敗しました"))
    text_encoder, text_encoder_2 = text_encoder_manager.get_text_encoders()

    # transformerの初期化
    transformer_manager.ensure_download_models()
    transformer = transformer_manager.get_transformer()  # 仮想デバイス上のtransformerを取得

    # 他のモデルの読み込み
    feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
    image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()
except Exception as e:
    print(translate("モデル読み込みエラー: {0}").format(e))
    print(translate("プログラムを終了します..."))
    import sys
    sys.exit(1)

vae.eval()
image_encoder.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)

vae.requires_grad_(False)
image_encoder.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    # Only use DynamicSwapInstaller for CUDA devices, as it may not work properly with MPS
    if device_type == 'cuda':
        DynamicSwapInstaller.install_model(transformer, device=gpu) # クラスを操作するので仮想デバイス上のtransformerでもOK
else:
    image_encoder.to(gpu)
    vae.to(gpu)

stream = AsyncStream()

# 設定管理モジュールをインポート
from eichi_utils.settings_manager import (
    get_settings_file_path,
    get_output_folder_path,
    initialize_settings,
    load_settings,
    save_settings,
    open_output_folder
)

# フォルダ構造を先に定義
webui_folder = os.path.dirname(os.path.abspath(__file__))

# 設定保存用フォルダの設定
settings_folder = os.path.join(webui_folder, 'settings')
os.makedirs(settings_folder, exist_ok=True)

# 設定ファイル初期化
initialize_settings()

# ベースパスを定義
base_path = os.path.dirname(os.path.abspath(__file__))

# 設定から出力フォルダを取得
app_settings = load_settings()
output_folder_name = app_settings.get('output_folder', 'outputs')
print(translate("設定から出力フォルダを読み込み: {0}").format(output_folder_name))

# 出力フォルダのフルパスを生成
outputs_folder = get_output_folder_path(output_folder_name)
os.makedirs(outputs_folder, exist_ok=True)

@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf=16, all_padding_value=1.0, image_strength=1.0, keep_section_videos=False, lora_files=None, lora_files2=None, lora_scales_text="0.8,0.8", output_dir=None, save_section_frames=False, use_all_padding=False, use_lora=False, save_tensor_data=False, tensor_data_input=None, fp8_optimization=False, resolution=640, batch_index=None, save_latent_frames=False, save_last_section_frames=False):

    # フレーム保存フラグのタイプと値を確認（必ずブール値であるべき）
    print(translate("[DEBUG] worker関数に渡されたフラグ - save_latent_frames型: {0}, 値: {1}").format(type(save_latent_frames).__name__, save_latent_frames))
    print(translate("[DEBUG] worker関数に渡されたフラグ - save_last_section_frames型: {0}, 値: {1}").format(type(save_last_section_frames).__name__, save_last_section_frames))
    
    # 万が一文字列が渡された場合の防御コード
    if isinstance(save_latent_frames, str):
        # 文字列の場合は、条件判定して適切なブール値に変換
        if save_latent_frames == translate("全フレーム画像保存"):
            save_latent_frames = True
        else:
            save_latent_frames = False
        print(translate("[WARN] save_latent_framesが文字列でした。ブール値に変換: {0}").format(save_latent_frames))
    
    if isinstance(save_last_section_frames, str):
        # 文字列の場合は、条件判定して適切なブール値に変換
        # 注意: UIでは「最終セクションのみ全フレーム画像保存」という表記を使っている
        if save_last_section_frames == translate("最終セクションのみフレーム画像保存") or save_last_section_frames == translate("最終セクションのみ全フレーム画像保存"):
            save_last_section_frames = True
        else:
            save_last_section_frames = False
        print(translate("[WARN] save_last_section_framesが文字列でした。ブール値に変換: {0}").format(save_last_section_frames))
    
    # 最終的に必ずブール型に変換しておく
    save_latent_frames = bool(save_latent_frames)
    save_last_section_frames = bool(save_last_section_frames)
    
    print(translate("[DEBUG] 最終変換後のフラグ - save_latent_frames: {0}, save_last_section_frames: {1}").format(save_latent_frames, save_last_section_frames))

    # 入力画像または表示されている最後のキーフレーム画像のいずれかが存在するか確認
    print(translate("[DEBUG] worker内 input_imageの型: {0}").format(type(input_image)))
    if isinstance(input_image, str):
        print(translate("[DEBUG] input_imageはファイルパスです: {0}").format(input_image))
        has_any_image = (input_image is not None)
    else:
        print(translate("[DEBUG] input_imageはファイルパス以外です").format())
        has_any_image = (input_image is not None)
    last_visible_section_image = None
    last_visible_section_num = -1

    if not has_any_image and section_settings is not None:
        # 現在の動画長設定から表示されるセクション数を計算
        total_display_sections = None
        try:
            # 動画長を秒数で取得
            seconds = get_video_seconds(total_second_length)

            # フレームサイズ設定からlatent_window_sizeを計算
            current_latent_window_size = 4.5 if frame_size_setting == "0.5秒 (17フレーム)" else 9
            frame_count = current_latent_window_size * 4 - 3

            # セクション数を計算
            total_frames = int(seconds * 30)
            total_display_sections = int(max(round(total_frames / frame_count), 1))
            print(translate("[DEBUG] worker内の現在の設定によるセクション数: {0}").format(total_display_sections))
        except Exception as e:
            print(translate("[ERROR] worker内のセクション数計算エラー: {0}").format(e))

        # 有効なセクション番号を収集
        valid_sections = []
        for section in section_settings:
            if section and len(section) > 1 and section[0] is not None and section[1] is not None:
                try:
                    section_num = int(section[0])
                    # 表示セクション数が計算されていれば、それ以下のセクションのみ追加
                    if total_display_sections is None or section_num < total_display_sections:
                        valid_sections.append((section_num, section[1]))
                except (ValueError, TypeError):
                    continue

        # 有効なセクションがあれば、最大の番号（最後のセクション）を探す
        if valid_sections:
            # 番号でソート
            valid_sections.sort(key=lambda x: x[0])
            # 最後のセクションを取得
            last_visible_section_num, last_visible_section_image = valid_sections[-1]
            print(translate("[DEBUG] worker内の最後のキーフレーム確認: セクション{0} (画像あり)").format(last_visible_section_num))

    has_any_image = has_any_image or (last_visible_section_image is not None)
    if not has_any_image:
        raise ValueError("入力画像または表示されている最後のキーフレーム画像のいずれかが必要です")

    # 入力画像がない場合はキーフレーム画像を使用
    if input_image is None and last_visible_section_image is not None:
        print(translate("[INFO] 入力画像が指定されていないため、セクション{0}のキーフレーム画像を使用します").format(last_visible_section_num))
        input_image = last_visible_section_image

    # 出力フォルダの設定
    global outputs_folder
    global output_folder_name
    if output_dir and output_dir.strip():
        # 出力フォルダパスを取得
        outputs_folder = get_output_folder_path(output_dir)
        print(translate("出力フォルダを設定: {0}").format(outputs_folder))

        # フォルダ名が現在の設定と異なる場合は設定ファイルを更新
        if output_dir != output_folder_name:
            settings = load_settings()
            settings['output_folder'] = output_dir
            if save_settings(settings):
                output_folder_name = output_dir
                print(translate("出力フォルダ設定を保存しました: {0}").format(output_dir))
    else:
        # デフォルト設定を使用
        outputs_folder = get_output_folder_path(output_folder_name)
        print(translate("デフォルト出力フォルダを使用: {0}").format(outputs_folder))

    # フォルダが存在しない場合は作成
    os.makedirs(outputs_folder, exist_ok=True)

    # 処理時間計測の開始
    process_start_time = time.time()

    # グローバル変数で状態管理しているモデル変数を宣言する
    global transformer, text_encoder, text_encoder_2

    # text_encoderとtext_encoder_2を確実にロード
    if not text_encoder_manager.ensure_text_encoder_state():
        raise Exception(translate("text_encoderとtext_encoder_2の初期化に失敗しました"))
    text_encoder, text_encoder_2 = text_encoder_manager.get_text_encoders()

    # 既存の計算方法を保持しつつ、設定からセクション数も取得する
    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    # 現在のモードを取得（UIから渡された情報から）
    # セクション数を全セクション数として保存
    total_sections = total_latent_sections

    # 現在のバッチ番号が指定されていれば使用する
    batch_suffix = f"_batch{batch_index+1}" if batch_index is not None else ""
    job_id = generate_timestamp() + batch_suffix

    # F1モードでは順生成を行うため、latent_paddingsのロジックは使用しない
    # 全セクション数を設定
    total_sections = total_latent_sections
    
    # 正確なセクション数の再計算と確認（トラブルシューティング用）
    if total_second_length > 0:
        sections_by_frames = int(max(round((total_second_length * 30) / (latent_window_size * 4 - 3)), 1))
        print(translate("[DEBUG] セクション数再計算チェック: 秒数={0}, フレーム計算={1}, 設定値={2}").format(
            total_second_length, sections_by_frames, total_sections))
        if sections_by_frames != total_sections:
            print(translate("[WARN] セクション数に不一致があります！計算値を優先します"))
            total_sections = sections_by_frames

    print(translate("\u25a0 セクション生成詳細 (F1モード):"))
    print(translate("  - 合計セクション数: {0} (最終確定値)").format(total_sections))
    frame_count = latent_window_size * 4 - 3
    print(translate("  - 各セクションのフレーム数: 約{0}フレーム (latent_window_size: {1})").format(frame_count, latent_window_size))

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # F1モードのプロンプト処理
        section_map = None
        section_numbers_sorted = []

        # Clean GPU
        if not high_vram:
            # モデルをCPUにアンロード
            unload_complete_models(
                image_encoder, vae
            )

        # Text encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("Text encoding ...")))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)


        # これ以降の処理は text_encoder, text_encoder_2 は不要なので、メモリ解放してしまって構わない
        if not high_vram:
            text_encoder, text_encoder_2 = None, None
            text_encoder_manager.dispose_text_encoders()

        # テンソルデータのアップロードがあれば読み込み
        uploaded_tensor = None
        if tensor_data_input is not None:
            try:
                # リスト型の場合、最初の要素を取得
                if isinstance(tensor_data_input, list):
                    if tensor_data_input and hasattr(tensor_data_input[0], 'name'):
                        tensor_data_input = tensor_data_input[0]
                    else:
                        tensor_data_input = None
                
                if tensor_data_input is not None and hasattr(tensor_data_input, 'name'):
                    tensor_path = tensor_data_input.name
                    print(translate("テンソルデータを読み込み: {0}").format(os.path.basename(tensor_path)))
                    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate('Loading tensor data ...')))))

                    # safetensorsからテンソルを読み込み
                    tensor_dict = sf.load_file(tensor_path)

                    # テンソルに含まれているキーとシェイプを確認
                    print(translate("テンソルデータの内容:"))
                    for key, tensor in tensor_dict.items():
                        print(f"  - {key}: shape={tensor.shape}, dtype={tensor.dtype}")

                    # history_latentsと呼ばれるキーが存在するか確認
                    if "history_latents" in tensor_dict:
                        uploaded_tensor = tensor_dict["history_latents"]
                        print(translate("テンソルデータ読み込み成功: shape={0}, dtype={1}").format(uploaded_tensor.shape, uploaded_tensor.dtype))
                        stream.output_queue.push(('progress', (None, translate('Tensor data loaded successfully!'), make_progress_bar_html(10, translate('Tensor data loaded successfully!')))))
                    else:
                        print(translate("警告: テンソルデータに 'history_latents' キーが見つかりません"))
            except Exception as e:
                print(translate("テンソルデータ読み込みエラー: {0}").format(e))
                traceback.print_exc()

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("Image processing ...")))))

        def preprocess_image(img_path_or_array, resolution=640):
            """Pathまたは画像配列を処理して適切なサイズに変換する"""
            print(translate("[DEBUG] preprocess_image: img_path_or_array型 = {0}").format(type(img_path_or_array)))

            if img_path_or_array is None:
                # 画像がない場合は指定解像度の黒い画像を生成
                img = np.zeros((resolution, resolution, 3), dtype=np.uint8)
                height = width = resolution
                return img, img, height, width

            # TensorからNumPyへ変換する必要があれば行う
            if isinstance(img_path_or_array, torch.Tensor):
                img_path_or_array = img_path_or_array.cpu().numpy()

            # Pathの場合はPILで画像を開く
            if isinstance(img_path_or_array, str) and os.path.exists(img_path_or_array):
                print(translate("[DEBUG] ファイルから画像を読み込み: {0}").format(img_path_or_array))
                img = np.array(Image.open(img_path_or_array).convert('RGB'))
            else:
                # NumPy配列の場合はそのまま使う
                img = img_path_or_array

            H, W, C = img.shape
            # 解像度パラメータを使用してサイズを決定
            height, width = find_nearest_bucket(H, W, resolution=resolution)
            img_np = resize_and_center_crop(img, target_width=width, target_height=height)
            img_pt = torch.from_numpy(img_np).float() / 127.5 - 1
            img_pt = img_pt.permute(2, 0, 1)[None, :, None]
            return img_np, img_pt, height, width

        input_image_np, input_image_pt, height, width = preprocess_image(input_image, resolution=resolution)
        Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'))
        # 入力画像にメタデータを埋め込んで保存
        initial_image_path = os.path.join(outputs_folder, f'{job_id}.png')
        Image.fromarray(input_image_np).save(initial_image_path)

        # メタデータの埋め込み
        # print(translate("\n[DEBUG] 入力画像へのメタデータ埋め込み開始: {0}").format(initial_image_path))
        # print(f"[DEBUG] prompt: {prompt}")
        # print(f"[DEBUG] seed: {seed}")
        metadata = {
            PROMPT_KEY: prompt,
            SEED_KEY: seed
        }
        # print(translate("[DEBUG] 埋め込むメタデータ: {0}").format(metadata))
        embed_metadata_to_png(initial_image_path, metadata)

        # VAE encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("VAE encoding ...")))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)

        # アップロードされたテンソルがあっても、常に入力画像から通常のエンコーディングを行う
        # テンソルデータは後で後付けとして使用するために保持しておく
        if uploaded_tensor is not None:
            print(translate("アップロードされたテンソルデータを検出: 動画生成後に後方に結合します"))
            # 入力画像がNoneの場合、テンソルからデコードして表示画像を生成
            if input_image is None:
                try:
                    # テンソルの最初のフレームから画像をデコードして表示用に使用
                    preview_latent = uploaded_tensor[:, :, 0:1, :, :].clone()
                    if preview_latent.device != torch.device('cpu'):
                        preview_latent = preview_latent.cpu()
                    if preview_latent.dtype != torch.float16:
                        preview_latent = preview_latent.to(dtype=torch.float16)

                    decoded_image = vae_decode(preview_latent, vae)
                    decoded_image = (decoded_image[0, :, 0] * 127.5 + 127.5).permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
                    # デコードした画像を保存
                    Image.fromarray(decoded_image).save(os.path.join(outputs_folder, f'{job_id}_tensor_preview.png'))
                    # デコードした画像を入力画像として設定
                    input_image = decoded_image
                    # 前処理用のデータも生成
                    input_image_np, input_image_pt, height, width = preprocess_image(input_image)
                    print(translate("テンソルからデコードした画像を生成しました: {0}x{1}").format(height, width))
                except Exception as e:
                    print(translate("テンソルからのデコード中にエラーが発生しました: {0}").format(e))
                    # デコードに失敗した場合は通常の処理を続行

            # UI上でテンソルデータの情報を表示
            tensor_info = translate("テンソルデータ ({0}フレーム) を検出しました。動画生成後に後方に結合します。").format(uploaded_tensor.shape[2])
            stream.output_queue.push(('progress', (None, tensor_info, make_progress_bar_html(10, translate('テンソルデータを後方に結合')))))

        # 常に入力画像から通常のエンコーディングを行う
        start_latent = vae_encode(input_image_pt, vae)

        # 簡略化設計: section_latents機能を削除

        # CLIP Vision

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("CLIP Vision encoding ...")))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

        # Dtype

        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, translate("Start sampling ...")))))

        rnd = torch.Generator("cpu").manual_seed(seed)
        # latent_window_sizeが4.5の場合は特別に17フレームとする
        if latent_window_size == 4.5:
            num_frames = 17  # 5 * 4 - 3 = 17
        else:
            num_frames = int(latent_window_size * 4 - 3)

        # 初期フレーム準備
        history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None

        # 開始フレームをhistory_latentsに追加
        history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
        total_generated_latent_frames = 1  # 最初のフレームを含むので1から開始

        # -------- LoRA 設定 START ---------

        # LoRAの環境変数設定（PYTORCH_CUDA_ALLOC_CONF）
        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
            old_env = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            print(translate("CUDA環境変数設定: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (元の値: {0})").format(old_env))

        # 次回のtransformer設定を更新
        current_lora_paths = []
        current_lora_scales = []
        
        if use_lora and has_lora_support:
            # LoRAファイルを収集
            if lora_files is not None:
                if isinstance(lora_files, list):
                    # 複数のLoRAファイル（将来のGradioバージョン用）
                    current_lora_paths.extend([file.name for file in lora_files])
                else:
                    # 単一のLoRAファイル
                    current_lora_paths.append(lora_files.name)
            
            # 2つ目のLoRAファイルがあれば追加
            if lora_files2 is not None:
                if isinstance(lora_files2, list):
                    # 複数のLoRAファイル（将来のGradioバージョン用）
                    current_lora_paths.extend([file.name for file in lora_files2])
                else:
                    # 単一のLoRAファイル
                    current_lora_paths.append(lora_files2.name)
            
            # スケール値をテキストから解析
            if current_lora_paths:  # LoRAパスがある場合のみ解析
                try:
                    scales_text = lora_scales_text.strip()
                    if scales_text:
                        # カンマ区切りのスケール値を解析
                        scales = [float(scale.strip()) for scale in scales_text.split(',')]
                        current_lora_scales = scales
                    else:
                        # スケール値が指定されていない場合は全て0.8を使用
                        current_lora_scales = [0.8] * len(current_lora_paths)
                except Exception as e:
                    print(translate("LoRAスケール解析エラー: {0}").format(e))
                    print(translate("デフォルトスケール 0.8 を使用します"))
                    current_lora_scales = [0.8] * len(current_lora_paths)
                
                # スケール値の数がLoRAパスの数と一致しない場合は調整
                if len(current_lora_scales) < len(current_lora_paths):
                    # 足りない分は0.8で埋める
                    current_lora_scales.extend([0.8] * (len(current_lora_paths) - len(current_lora_scales)))
                elif len(current_lora_scales) > len(current_lora_paths):
                    # 余分は切り捨て
                    current_lora_scales = current_lora_scales[:len(current_lora_paths)]
        
        # LoRA設定を更新（リロードは行わない）
        transformer_manager.set_next_settings(
            lora_paths=current_lora_paths,
            lora_scales=current_lora_scales,
            fp8_enabled=fp8_optimization,
            high_vram_mode=high_vram,
        )

        # -------- LoRA 設定 END ---------

        # セクション処理開始前にtransformerの状態を確認
        print(translate("\nセクション処理開始前のtransformer状態チェック..."))
        try:
            # transformerの状態を確認し、必要に応じてリロード
            if not transformer_manager.ensure_transformer_state():
                raise Exception(translate("transformer状態の確認に失敗しました"))

            # 最新のtransformerインスタンスを取得
            transformer = transformer_manager.get_transformer()
            print(translate("transformer状態チェック完了"))
        except Exception as e:
            print(translate("transformer状態チェックエラー: {0}").format(e))
            traceback.print_exc()
            raise e

        # セクション順次処理
        for i_section in range(total_sections):
            # 先に変数を定義
            is_first_section = i_section == 0

            # 単純なインデックスによる判定
            is_last_section = i_section == total_sections - 1

            # F1モードではオールパディング機能は無効化されているため、常に固定値を使用
            # この値はF1モードでは実際には使用されないが、ログ出力のために計算する
            latent_padding = 1  # 固定値

            latent_padding_size = int(latent_padding * latent_window_size)

            # 定義後にログ出力（F1モードではオールパディングは常に無効）
            padding_info = translate("パディング値: {0} (F1モードでは影響なし)").format(latent_padding)
            print(translate("\n■ セクション{0}の処理開始 ({1})").format(i_section, padding_info))
            print(translate("  - 現在の生成フレーム数: {0}フレーム").format(total_generated_latent_frames * 4 - 3))
            print(translate("  - 生成予定フレーム数: {0}フレーム").format(num_frames))
            print(translate("  - 最初のセクション?: {0}").format(is_first_section))
            print(translate("  - 最後のセクション?: {0}").format(is_last_section))
            # set current_latent here
            # 常に開始フレームを使用
            current_latent = start_latent


            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            # 共通プロンプトを使用
            current_llama_vec, current_clip_l_pooler, current_llama_attention_mask = llama_vec, clip_l_pooler, llama_attention_mask

            print(translate('latent_padding_size = {0}, is_last_section = {1}').format(latent_padding_size, is_last_section))


            # COMMENTED OUT: セクション処理前のメモリ解放（処理速度向上のため）
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            #     torch.cuda.empty_cache()

            # latent_window_sizeが4.5の場合は特別に5を使用
            effective_window_size = 5 if latent_window_size == 4.5 else int(latent_window_size)
            # 必ず整数のリストを使用
            indices = torch.arange(0, sum([1, 16, 2, 1, effective_window_size])).unsqueeze(0)
            clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, effective_window_size], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

            clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
            clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)

            if not high_vram:
                unload_complete_models()
                # GPUメモリ保存値を明示的に浮動小数点に変換
                preserved_memory = float(gpu_memory_preservation) if gpu_memory_preservation is not None else 6.0
                print(translate('Setting transformer memory preservation to: {0} GB').format(preserved_memory))
                move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=preserved_memory)

            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                preview = d['denoised']
                preview = vae_decode_fake(preview)

                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    raise KeyboardInterrupt('User ends the task.')

                current_step = d['i'] + 1
                percentage = int(100.0 * current_step / steps)
                hint = translate('Sampling {0}/{1}').format(current_step, steps)
                # セクション情報を追加（現在のセクション/全セクション）
                section_info = translate('セクション: {0}/{1}').format(i_section+1, total_sections)
                desc = f"{section_info} " + translate('生成フレーム数: {total_generated_latent_frames}, 動画長: {video_length:.2f} 秒 (FPS-30). 動画が生成中です ...').format(section_info=section_info, total_generated_latent_frames=int(max(0, total_generated_latent_frames * 4 - 3)), video_length=max(0, (total_generated_latent_frames * 4 - 3) / 30))
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

            # Image影響度を計算：大きい値ほど始点の影響が強くなるよう変換
            # 1.0/image_strengthを使用し、最小値を0.01に制限
            strength_value = max(0.01, 1.0 / image_strength)
            print(translate('Image影響度: UI値={0:.2f}（{1:.0f}%）→計算値={2:.4f}（値が小さいほど始点の影響が強い）').format(
                image_strength, image_strength * 100, strength_value))

            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                # shift=3.0,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=current_llama_vec,  # セクションごとのプロンプトを使用
                prompt_embeds_mask=current_llama_attention_mask,  # セクションごとのマスクを使用
                prompt_poolers=current_clip_l_pooler,  # セクションごとのプロンプトを使用
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=gpu,
                dtype=torch.bfloat16,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                initial_latent=current_latent,  # 開始潜在空間を設定
                strength=strength_value,        # 計算した影響度を使用
                callback=callback,
            )

            # if is_last_section:
            #     generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            # 後方にフレームを追加
            history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)

            if not high_vram:
                # 減圧時に使用するGPUメモリ値も明示的に浮動小数点に設定
                preserved_memory_offload = 8.0  # こちらは固定値のまま
                print(translate('Offloading transformer with memory preservation: {0} GB').format(preserved_memory_offload))
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=preserved_memory_offload)
                load_model_as_complete(vae, target_device=gpu)

            # 最新フレームは末尾から切り出し
            real_history_latents = history_latents[:, :, -total_generated_latent_frames:, :, :]

            # COMMENTED OUT: VAEデコード前のメモリクリア（処理速度向上のため）
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            #     torch.cuda.empty_cache()
            #     print(translate("VAEデコード前メモリ: {0:.2f}GB").format(torch.cuda.memory_allocated()/1024**3))

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                # latent_window_sizeが4.5の場合は特別に5を使用
                if latent_window_size == 4.5:
                    section_latent_frames = 11 if is_last_section else 10  # 5 * 2 + 1 = 11, 5 * 2 = 10
                    overlapped_frames = 17  # 5 * 4 - 3 = 17
                else:
                    # +1は逆方向生成時の start_latent 分なのでカット
                    section_latent_frames = int(latent_window_size * 2) if is_last_section else int(latent_window_size * 2)
                    overlapped_frames = int(latent_window_size * 4 - 3)

                # F1モードでは最新フレームは末尾にあるため、後方のセクションを取得
                current_pixels = vae_decode(real_history_latents[:, :, -section_latent_frames:], vae).cpu()

                # 引数の順序を修正 - history_pixelsが先、新しいcurrent_pixelsが後
                if history_pixels is None:
                    history_pixels = current_pixels
                else:
                    history_pixels = soft_append_bcthw(history_pixels, current_pixels, overlapped_frames)

            # 各セクションの最終フレームを静止画として保存（セクション番号付き）
            if save_section_frames and history_pixels is not None:
                try:
                    if i_section == 0 or current_pixels is None:
                        # 最初のセクションは history_pixels の最後
                        last_frame = history_pixels[0, :, -1, :, :]
                    else:
                        # 2セクション目以降は current_pixels の最後
                        last_frame = current_pixels[0, :, -1, :, :]
                    last_frame = einops.rearrange(last_frame, 'c h w -> h w c')
                    last_frame = last_frame.cpu().numpy()
                    last_frame = np.clip((last_frame * 127.5 + 127.5), 0, 255).astype(np.uint8)
                    last_frame = resize_and_center_crop(last_frame, target_width=width, target_height=height)

                    # メタデータを埋め込むための情報を収集
                    print(translate("\n[DEBUG] セクション{0}のメタデータ埋め込み準備").format(i_section))
                    section_metadata = {
                        PROMPT_KEY: prompt,  # メインプロンプト
                        SEED_KEY: seed,
                        SECTION_NUMBER_KEY: i_section
                    }
                    print(translate("[DEBUG] 基本メタデータ: {0}").format(section_metadata))

                    # セクション固有のプロンプトがあれば取得
                    if section_map and i_section in section_map:
                        _, section_prompt = section_map[i_section]
                        if section_prompt and section_prompt.strip():
                            section_metadata[SECTION_PROMPT_KEY] = section_prompt
                            print(translate("[DEBUG] セクションプロンプトを追加: {0}").format(section_prompt))

                    # 画像の保存とメタデータの埋め込み
                    if is_first_section:
                        frame_path = os.path.join(outputs_folder, f'{job_id}_{i_section}_end.png')
                        print(translate("[DEBUG] セクション画像パス: {0}").format(frame_path))
                        Image.fromarray(last_frame).save(frame_path)
                        print(translate("[DEBUG] メタデータ埋め込み実行: {0}").format(section_metadata))
                        embed_metadata_to_png(frame_path, section_metadata)
                    else:
                        frame_path = os.path.join(outputs_folder, f'{job_id}_{i_section}.png')
                        print(translate("[DEBUG] セクション画像パス: {0}").format(frame_path))
                        Image.fromarray(last_frame).save(frame_path)
                        print(translate("[DEBUG] メタデータ埋め込み実行: {0}").format(section_metadata))
                        embed_metadata_to_png(frame_path, section_metadata)

                    print(translate("\u2713 セクション{0}のフレーム画像をメタデータ付きで保存しました").format(i_section))
                except Exception as e:
                    print(translate("[WARN] セクション{0}最終フレーム画像保存時にエラー: {1}").format(i_section, e))

            # 全フレーム画像保存機能
            # 「全フレーム画像保存」または「最終セクションのみ全フレーム画像保存かつ最終セクション」が有効な場合
            # 最終セクションかどうかの判定をtotal_sectionsから正確に取得
            is_last_section = i_section == total_sections - 1
            print(translate("\n[DEBUG] 現在のセクション: {0}, 総セクション数: {1}, 最終セクションと判定: {2}").format(i_section, total_sections, is_last_section))
            
            # save_latent_frames と save_last_section_frames の値をcopy
            # ループ内の変数を変更してもグローバルな値は変わらないため
            # 注意：既にここに来る前に万が一の文字列→ブール変換処理が済んでいるはず
            
            # デバッグ情報を追加：実際に使用されるフラグの値を確認
            print(translate("[DEBUG] セクション{0}の処理開始時 - 現在のsave_latent_frames型: {1}, 値: {2}").format(
                i_section, type(save_latent_frames).__name__, save_latent_frames
            ))
            print(translate("[DEBUG] セクション{0}の処理開始時 - 現在のsave_last_section_frames型: {1}, 値: {2}").format(
                i_section, type(save_last_section_frames).__name__, save_last_section_frames
            ))
            
            # 値のコピーではなく、明示的に新しい変数に適切な値を設定
            # BooleanかStringかの型変換ミスを防ぐ
            is_save_all_frames = bool(save_latent_frames)
            is_save_last_frame_only = bool(save_last_section_frames)
            
            # デバッグ情報を追加：変換後の値を確認
            print(translate("[DEBUG] セクション{0}の処理 - 変換後のis_save_all_frames型: {1}, 値: {2}").format(
                i_section, type(is_save_all_frames).__name__, is_save_all_frames
            ))
            print(translate("[DEBUG] セクション{0}の処理 - 変換後のis_save_last_frame_only型: {1}, 値: {2}").format(
                i_section, type(is_save_last_frame_only).__name__, is_save_last_frame_only
            ))
            
            if is_save_all_frames:
                should_save_frames = True
                print(translate("[DEBUG] 全フレーム画像保存が有効: 全セクションでフレーム保存します"))
            elif is_save_last_frame_only and is_last_section:
                should_save_frames = True
                print(translate("[DEBUG] 最終セクションのみ全フレーム画像保存が有効: 現在のセクション{0}が最終セクションと判定されたため保存します").format(i_section))
            else:
                should_save_frames = False
                if is_save_last_frame_only:
                    print(translate("[DEBUG] 最終セクションのみ全フレーム画像保存が有効: 現在のセクション{0}は最終セクションではないためスキップします").format(i_section))
                else:
                    print(translate("[DEBUG] フレーム画像保存は無効です"))
            
            if should_save_frames:
                try:
                    # source_pixelsは、このセクションで使用するピクセルデータ
                    source_pixels = None
                    
                    # i_section=0の場合、current_pixelsが定義される前に参照されるためエラーとなる
                    # history_pixelsを優先して使用するよう処理順序を変更
                    if history_pixels is not None:
                        source_pixels = history_pixels
                        print(translate("\n[INFO] フレーム画像保存: history_pixelsを使用します"))
                    elif 'current_pixels' in locals() and current_pixels is not None:
                        source_pixels = current_pixels
                        print(translate("\n[INFO] フレーム画像保存: current_pixelsを使用します"))
                    else:
                        print(translate("\n[WARN] フレーム画像保存: 有効なピクセルデータがありません"))
                        return
                        
                    # フレーム数（1秒モードでは9フレーム、0.5秒モードでは5フレーム）
                    latent_frame_count = source_pixels.shape[2]
                    
                    # 保存モードに応じたメッセージを表示
                    # グローバル変数ではなく、ローカルのcopyを使用
                    if is_save_all_frames:
                        print(translate("[INFO] 全フレーム画像保存: セクション{0}の{1}フレームを保存します").format(i_section, latent_frame_count))
                    elif is_save_last_frame_only and is_last_section:
                        # 強調して最終セクションであることを表示
                        print(translate("[INFO] 最終セクションのみ全フレーム画像保存: セクション{0}/{1}の{2}フレームを保存します (最終セクション)").format(
                            i_section, total_sections-1, latent_frame_count))
                    else:
                        print(translate("[INFO] フレーム画像保存: セクション{0}の{1}フレームを保存します").format(i_section, latent_frame_count))
                    
                    # セクションごとのフォルダを作成
                    frames_folder = os.path.join(outputs_folder, f'{job_id}_frames_section{i_section}')
                    os.makedirs(frames_folder, exist_ok=True)
                    
                    # 各フレームの保存
                    for frame_idx in range(latent_frame_count):
                        # フレームを取得
                        frame = source_pixels[0, :, frame_idx, :, :]
                        frame = einops.rearrange(frame, 'c h w -> h w c')
                        frame = frame.cpu().numpy()
                        frame = np.clip((frame * 127.5 + 127.5), 0, 255).astype(np.uint8)
                        frame = resize_and_center_crop(frame, target_width=width, target_height=height)
                        
                        # メタデータの準備
                        frame_metadata = {
                            PROMPT_KEY: prompt,  # メインプロンプト
                            SEED_KEY: seed,
                            SECTION_NUMBER_KEY: i_section,
                            "FRAME_NUMBER": frame_idx  # フレーム番号も追加
                        }
                        
                        # 画像の保存とメタデータの埋め込み
                        frame_path = os.path.join(frames_folder, f'frame_{frame_idx:03d}.png')
                        Image.fromarray(frame).save(frame_path)
                        embed_metadata_to_png(frame_path, frame_metadata)
                    
                    # 保存モードに応じたメッセージを表示
                    # グローバル変数ではなく、ローカルのcopyを使用
                    if is_save_all_frames:
                        print(translate("[INFO] 全フレーム画像保存: セクション{0}の{1}個のフレーム画像を保存しました: {2}").format(
                            i_section, latent_frame_count, frames_folder))
                    elif is_save_last_frame_only and is_last_section:
                        print(translate("[INFO] 最終セクションのみ全フレーム画像保存: セクション{0}/{1}の{2}個のフレーム画像を保存しました (最終セクション): {3}").format(
                            i_section, total_sections-1, latent_frame_count, frames_folder))
                    else:
                        print(translate("[INFO] セクション{0}の{1}個のフレーム画像を保存しました: {2}").format(
                            i_section, latent_frame_count, frames_folder))
                except Exception as e:
                    print(translate("[WARN] セクション{0}のフレーム画像保存中にエラー: {1}").format(i_section, e))
                    traceback.print_exc()

            if not high_vram:
                unload_complete_models()

            output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')

            # MP4保存前のデータ検証を追加
            print(translate("[DEBUG] MP4保存前のhistory_pixels形状: {0}").format(history_pixels.shape))
            print(translate("[DEBUG] MP4保存前のhistory_pixelsデータ範囲: min={0}, max={1}").format(history_pixels.min().item(), history_pixels.max().item()))

            # MP4保存前のデバッグ情報を出力
            print(translate("[DEBUG] MP4保存前のhistory_pixels形状: {0}").format(history_pixels.shape))
            print(translate("[DEBUG] MP4保存前のhistory_pixelsデータ範囲: min={0}, max={1}").format(history_pixels.min().item(), history_pixels.max().item()))

            # GPUメモリ状況の確認
            if torch.cuda.is_available():
                print(translate("[DEBUG] MP4保存前GPUメモリ: {0:.2f}GB/{1:.2f}GB").format(torch.cuda.memory_allocated()/1024**3, torch.cuda.get_device_properties(0).total_memory/1024**3))

            # 出力ファイルパスの確認
            print(translate("[DEBUG] MP4保存パス: {0}").format(os.path.abspath(output_filename)))

            # もしhistory_pixelsの値が不適切な範囲にある場合、範囲を修正
            if history_pixels.min() < -1.0 or history_pixels.max() > 1.0:
                print(translate("[DEBUG] history_pixelsの値範囲を[-1.0, 1.0]に修正します"))
                history_pixels = torch.clamp(history_pixels, -1.0, 1.0)

            # MP4を保存
            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            print(translate('Decoded. Current latent shape {0}; pixel shape {1}').format(real_history_latents.shape, history_pixels.shape))

            # COMMENTED OUT: セクション処理後の明示的なメモリ解放（処理速度向上のため）
            # if torch.cuda.is_available():
            #     torch.cuda.synchronize()
            #     torch.cuda.empty_cache()
            #     import gc
            #     gc.collect()
            #     memory_allocated = torch.cuda.memory_allocated()/1024**3
            #     memory_reserved = torch.cuda.memory_reserved()/1024**3
            #     print(translate("セクション後メモリ状態: 割当={0:.2f}GB, 予約={1:.2f}GB").format(memory_allocated, memory_reserved))

            print(translate("■ セクション{0}の処理完了").format(i_section))
            print(translate("  - 現在の累計フレーム数: {0}フレーム").format(int(max(0, total_generated_latent_frames * 4 - 3))))
            print(translate("  - レンダリング時間: {0}秒").format(f"{max(0, (total_generated_latent_frames * 4 - 3) / 30):.2f}"))
            print(translate("  - 出力ファイル: {0}").format(output_filename))

            stream.output_queue.push(('file', output_filename))

            if is_last_section:
                combined_output_filename = None
                # 全セクション処理完了後、テンソルデータを後方に結合
                if uploaded_tensor is not None:
                    try:
                        original_frames = real_history_latents.shape[2]  # 元のフレーム数を記録
                        uploaded_frames = uploaded_tensor.shape[2]  # アップロードされたフレーム数

                        print(translate("テンソルデータを後方に結合します: アップロードされたフレーム数 = {uploaded_frames}").format(uploaded_frames=uploaded_frames))
                        # UI上で進捗状況を更新
                        stream.output_queue.push(('progress', (None, translate("テンソルデータ({uploaded_frames}フレーム)の結合を開始します...").format(uploaded_frames=uploaded_frames), make_progress_bar_html(80, translate('テンソルデータ結合準備')))))

                        # テンソルデータを後方に結合する前に、互換性チェック
                        # デバッグログを追加して詳細を出力
                        print(translate("[DEBUG] テンソルデータの形状: {0}, 生成データの形状: {1}").format(uploaded_tensor.shape, real_history_latents.shape))
                        print(translate("[DEBUG] テンソルデータの型: {0}, 生成データの型: {1}").format(uploaded_tensor.dtype, real_history_latents.dtype))
                        print(translate("[DEBUG] テンソルデータのデバイス: {0}, 生成データのデバイス: {1}").format(uploaded_tensor.device, real_history_latents.device))

                        if uploaded_tensor.shape[3] != real_history_latents.shape[3] or uploaded_tensor.shape[4] != real_history_latents.shape[4]:
                            print(translate("警告: テンソルサイズが異なります: アップロード={0}, 現在の生成={1}").format(uploaded_tensor.shape, real_history_latents.shape))
                            print(translate("テンソルサイズの不一致のため、前方結合をスキップします"))
                            stream.output_queue.push(('progress', (None, translate("テンソルサイズの不一致のため、前方結合をスキップしました"), make_progress_bar_html(85, translate('互換性エラー')))))
                        else:
                            # デバイスとデータ型を合わせる
                            processed_tensor = uploaded_tensor.clone()
                            if processed_tensor.device != real_history_latents.device:
                                processed_tensor = processed_tensor.to(real_history_latents.device)
                            if processed_tensor.dtype != real_history_latents.dtype:
                                processed_tensor = processed_tensor.to(dtype=real_history_latents.dtype)

                            # 元の動画を品質を保ちつつ保存
                            original_output_filename = os.path.join(outputs_folder, f'{job_id}_original.mp4')
                            save_bcthw_as_mp4(history_pixels, original_output_filename, fps=30, crf=mp4_crf)
                            print(translate("元の動画を保存しました: {original_output_filename}").format(original_output_filename=original_output_filename))

                            # 元データのコピーを取得
                            combined_history_latents = real_history_latents.clone()
                            combined_history_pixels = history_pixels.clone() if history_pixels is not None else None

                            # 各チャンクの処理前に明示的にメモリ解放
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                                print(translate("[MEMORY] チャンク処理前のGPUメモリ確保状態: {memory:.2f}GB").format(memory=torch.cuda.memory_allocated()/1024**3))

                            # VAEをGPUに移動
                            if not high_vram and vae.device != torch.device('cuda'):
                                print(translate("[SETUP] VAEをGPUに移動: {0} → cuda").format(vae.device))
                                vae.to('cuda')

                            # 各チャンクを処理
                            # チャンクサイズを設定(各セクションと同等のサイズにする)
                            chunk_size = min(5, uploaded_frames)  # 最大チャンクサイズを5フレームに設定（メモリ使用量を減らすため）

                            # チャンク数を計算
                            num_chunks = (uploaded_frames + chunk_size - 1) // chunk_size

                            # テンソルデータの詳細を出力
                            print(translate("[DEBUG] テンソルデータの詳細分析:"))
                            print(translate("  - 形状: {0}").format(processed_tensor.shape))
                            print(translate("  - 型: {0}").format(processed_tensor.dtype))
                            print(translate("  - デバイス: {0}").format(processed_tensor.device))
                            print(translate("  - 値範囲: 最小={0:.4f}, 最大={1:.4f}, 平均={2:.4f}").format(processed_tensor.min().item(), processed_tensor.max().item(), processed_tensor.mean().item()))
                            print(translate("  - チャンク数: {0}, チャンクサイズ: {1}").format(num_chunks, chunk_size))
                            tensor_size_mb = (processed_tensor.element_size() * processed_tensor.nelement()) / (1024 * 1024)
                            print(translate("  - テンソルデータ全体サイズ: {0:.2f} MB").format(tensor_size_mb))
                            print(translate("  - フレーム数: {0}フレーム（制限無し）").format(uploaded_frames))
                            # 各チャンクを処理
                            for chunk_idx in range(num_chunks):
                                chunk_start = chunk_idx * chunk_size
                                chunk_end = min(chunk_start + chunk_size, uploaded_frames)
                                chunk_frames = chunk_end - chunk_start

                                # 進捗状況を更新
                                chunk_progress = (chunk_idx + 1) / num_chunks * 100
                                progress_message = translate("テンソルデータ結合中: チャンク {0}/{1} (フレーム {2}-{3}/{4})").format(chunk_idx+1, num_chunks, chunk_start+1, chunk_end, uploaded_frames)
                                stream.output_queue.push(('progress', (None, progress_message, make_progress_bar_html(int(80 + chunk_progress * 0.1), translate('テンソルデータ処理中')))))

                                # 現在のチャンクを取得
                                current_chunk = processed_tensor[:, :, chunk_start:chunk_end, :, :]
                                print(translate("チャンク{0}/{1}処理中: フレーム {2}-{3}/{4}").format(chunk_idx+1, num_chunks, chunk_start+1, chunk_end, uploaded_frames))

                                # メモリ状態を出力
                                if torch.cuda.is_available():
                                    print(translate("[MEMORY] チャンク{0}処理前のGPUメモリ: {1:.2f}GB/{2:.2f}GB").format(chunk_idx+1, torch.cuda.memory_allocated()/1024**3, torch.cuda.get_device_properties(0).total_memory/1024**3))
                                    # メモリキャッシュをクリア
                                    torch.cuda.empty_cache()

                                try:
                                    # 各チャンク処理前にGPUメモリを解放
                                    if torch.cuda.is_available():
                                        torch.cuda.synchronize()
                                        torch.cuda.empty_cache()
                                        import gc
                                        gc.collect()
                                    # チャンクをデコード
                                    # VAEデコードは時間がかかるため、進行中であることを表示
                                    print(translate("チャンク{0}のVAEデコード開始...").format(chunk_idx+1))
                                    stream.output_queue.push(('progress', (None, translate("チャンク{0}/{1}のVAEデコード中...").format(chunk_idx+1, num_chunks), make_progress_bar_html(int(80 + chunk_progress * 0.1), translate('デコード処理')))))

                                    # VAEデコード前にテンソル情報を詳しく出力
                                    print(translate("[DEBUG] チャンク{0}のデコード前情報:").format(chunk_idx+1))
                                    print(translate("  - 形状: {0}").format(current_chunk.shape))
                                    print(translate("  - 型: {0}").format(current_chunk.dtype))
                                    print(translate("  - デバイス: {0}").format(current_chunk.device))
                                    print(translate("  - 値範囲: 最小={0:.4f}, 最大={1:.4f}, 平均={2:.4f}").format(current_chunk.min().item(), current_chunk.max().item(), current_chunk.mean().item()))

                                    # 明示的にデバイスを合わせる
                                    if current_chunk.device != vae.device:
                                        print(translate("  - デバイスをVAEと同じに変更: {0} → {1}").format(current_chunk.device, vae.device))
                                        current_chunk = current_chunk.to(vae.device)

                                    # 型を明示的に合わせる
                                    if current_chunk.dtype != torch.float16:
                                        print(translate("  - データ型をfloat16に変更: {0} → torch.float16").format(current_chunk.dtype))
                                        current_chunk = current_chunk.to(dtype=torch.float16)

                                    # VAEデコード処理
                                    chunk_pixels = vae_decode(current_chunk, vae).cpu()
                                    print(translate("チャンク{0}のVAEデコード完了 (フレーム数: {1})").format(chunk_idx+1, chunk_frames))

                                    # デコード後のピクセルデータ情報を出力
                                    print(translate("[DEBUG] チャンク{0}のデコード結果:").format(chunk_idx+1))
                                    print(translate("  - 形状: {0}").format(chunk_pixels.shape))
                                    print(translate("  - 型: {0}").format(chunk_pixels.dtype))
                                    print(translate("  - デバイス: {0}").format(chunk_pixels.device))
                                    print(translate("  - 値範囲: 最小={0:.4f}, 最大={1:.4f}, 平均={2:.4f}").format(chunk_pixels.min().item(), chunk_pixels.max().item(), chunk_pixels.mean().item()))

                                    # メモリ使用量を出力
                                    if torch.cuda.is_available():
                                        print(translate("[MEMORY] チャンク{0}デコード後のGPUメモリ: {1:.2f}GB").format(chunk_idx+1, torch.cuda.memory_allocated()/1024**3))

                                    # 結合する
                                    if combined_history_pixels is None:
                                        # 初回のチャンクの場合はそのまま設定
                                        combined_history_pixels = chunk_pixels
                                    else:
                                        # 2回目以降は結合
                                        print(translate("[DEBUG] 結合前の情報:"))
                                        print(translate("  - 既存: {0}, 型: {1}, デバイス: {2}").format(combined_history_pixels.shape, combined_history_pixels.dtype, combined_history_pixels.device))
                                        print(translate("  - 新規: {0}, 型: {1}, デバイス: {2}").format(chunk_pixels.shape, chunk_pixels.dtype, chunk_pixels.device))

                                        # 既存データと新規データで型とデバイスを揃える
                                        if combined_history_pixels.dtype != chunk_pixels.dtype:
                                            print(translate("  - データ型の不一致を修正: {0} → {1}").format(combined_history_pixels.dtype, chunk_pixels.dtype))
                                            combined_history_pixels = combined_history_pixels.to(dtype=chunk_pixels.dtype)

                                        # 両方とも必ずCPUに移動してから結合
                                        if combined_history_pixels.device != torch.device('cpu'):
                                            combined_history_pixels = combined_history_pixels.cpu()
                                        if chunk_pixels.device != torch.device('cpu'):
                                            chunk_pixels = chunk_pixels.cpu()

                                        # 結合処理
                                        combined_history_pixels = torch.cat([combined_history_pixels, chunk_pixels], dim=2)

                                    # 結合後のフレーム数を確認
                                    current_total_frames = combined_history_pixels.shape[2]
                                    print(translate("チャンク{0}の結合完了: 現在の組み込みフレーム数 = {1}").format(chunk_idx+1, current_total_frames))

                                    # 中間結果の保存（チャンクごとに保存すると効率が悪いので、最終チャンクのみ保存）
                                    if chunk_idx == num_chunks - 1 or (chunk_idx > 0 and (chunk_idx + 1) % 5 == 0):
                                        # 5チャンクごと、または最後のチャンクで保存
                                        interim_output_filename = os.path.join(outputs_folder, f'{job_id}_combined_interim_{chunk_idx+1}.mp4')
                                        print(translate("中間結果を保存中: チャンク{0}/{1}").format(chunk_idx+1, num_chunks))
                                        stream.output_queue.push(('progress', (None, translate("中間結果のMP4変換中... (チャンク{0}/{1})").format(chunk_idx+1, num_chunks), make_progress_bar_html(int(85 + chunk_progress * 0.1), translate('MP4保存中')))))

                                        # MP4として保存
                                        save_bcthw_as_mp4(combined_history_pixels, interim_output_filename, fps=30, crf=mp4_crf)
                                        print(translate("中間結果を保存しました: {0}").format(interim_output_filename))

                                        # 結合した動画をUIに反映するため、出力フラグを立てる
                                        stream.output_queue.push(('file', interim_output_filename))
                                except Exception as e:
                                    print(translate("チャンク{0}の処理中にエラーが発生しました: {1}").format(chunk_idx+1, e))
                                    traceback.print_exc()

                                    # エラー情報の詳細な出力
                                    print(translate("[ERROR] 詳細エラー情報:"))
                                    print(translate("  - チャンク情報: {0}/{1}, フレーム {2}-{3}/{4}").format(chunk_idx+1, num_chunks, chunk_start+1, chunk_end, uploaded_frames))
                                    if 'current_chunk' in locals():
                                        print(translate("  - current_chunk: shape={0}, dtype={1}, device={2}").format(current_chunk.shape, current_chunk.dtype, current_chunk.device))
                                    if 'vae' in globals():
                                        print(translate("  - VAE情報: device={0}, dtype={1}").format(vae.device, next(vae.parameters()).dtype))

                                    # GPUメモリ情報
                                    if torch.cuda.is_available():
                                        print(translate("  - GPU使用量: {0:.2f}GB/{1:.2f}GB").format(torch.cuda.memory_allocated()/1024**3, torch.cuda.get_device_properties(0).total_memory/1024**3))

                                    stream.output_queue.push(('progress', (None, translate("エラー: チャンク{0}の処理に失敗しました - {1}").format(chunk_idx+1, str(e)), make_progress_bar_html(90, translate('エラー')))))
                                    break

                            # 処理完了後に明示的にメモリ解放
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                                print(translate("[MEMORY] チャンク処理後のGPUメモリ確保状態: {0:.2f}GB").format(torch.cuda.memory_allocated()/1024**3))

                            # 全チャンクの処理が完了したら、最終的な結合動画を保存
                            if combined_history_pixels is not None:
                                # 結合された最終結果の情報を出力
                                print(translate("[DEBUG] 最終結合結果:"))
                                print(translate("  - 形状: {0}").format(combined_history_pixels.shape))
                                print(translate("  - 型: {0}").format(combined_history_pixels.dtype))
                                print(translate("  - デバイス: {0}").format(combined_history_pixels.device))
                                # 最終結果の保存
                                print(translate("最終結果を保存中: 全{0}チャンク完了").format(num_chunks))
                                stream.output_queue.push(('progress', (None, translate("結合した動画をMP4に変換中..."), make_progress_bar_html(95, translate('最終MP4変換処理')))))

                                # 最終的な結合ファイル名
                                combined_output_filename = os.path.join(outputs_folder, f'{job_id}_combined.mp4')

                                # MP4として保存
                                save_bcthw_as_mp4(combined_history_pixels, combined_output_filename, fps=30, crf=mp4_crf)
                                print(translate("最終結果を保存しました: {0}").format(combined_output_filename))
                                print(translate("結合動画の保存場所: {0}").format(os.path.abspath(combined_output_filename)))

                                # 中間ファイルの削除処理
                                print(translate("中間ファイルの削除を開始します..."))
                                deleted_files = []
                                try:
                                    # 現在のジョブIDに関連する中間ファイルを正規表現でフィルタリング
                                    import re
                                    interim_pattern = re.compile(f'{job_id}_combined_interim_\d+\.mp4')

                                    for filename in os.listdir(outputs_folder):
                                        if interim_pattern.match(filename):
                                            interim_path = os.path.join(outputs_folder, filename)
                                            try:
                                                os.remove(interim_path)
                                                deleted_files.append(filename)
                                                print(translate("  - 中間ファイルを削除しました: {0}").format(filename))
                                            except Exception as e:
                                                print(translate("  - ファイル削除エラー ({0}): {1}").format(filename, e))

                                    if deleted_files:
                                        print(translate("合計 {0} 個の中間ファイルを削除しました").format(len(deleted_files)))
                                        # 削除ファイル名をユーザーに表示
                                        files_str = ', '.join(deleted_files)
                                        stream.output_queue.push(('progress', (None, translate("中間ファイルを削除しました: {0}").format(files_str), make_progress_bar_html(97, translate('クリーンアップ完了')))))
                                    else:
                                        print(translate("削除対象の中間ファイルは見つかりませんでした"))
                                except Exception as e:
                                    print(translate("中間ファイル削除中にエラーが発生しました: {0}").format(e))
                                    traceback.print_exc()

                                # 結合した動画をUIに反映するため、出力フラグを立てる
                                stream.output_queue.push(('file', combined_output_filename))

                                # 結合後の全フレーム数を計算して表示
                                combined_frames = combined_history_pixels.shape[2]
                                combined_size_mb = (combined_history_pixels.element_size() * combined_history_pixels.nelement()) / (1024 * 1024)
                                print(translate("結合完了情報: テンソルデータ({0}フレーム) + 新規動画({1}フレーム) = 合計{2}フレーム").format(uploaded_frames, original_frames, combined_frames))
                                print(translate("結合動画の再生時間: {0:.2f}秒").format(combined_frames / 30))
                                print(translate("データサイズ: {0:.2f} MB（制限無し）").format(combined_size_mb))

                                # UI上で完了メッセージを表示
                                stream.output_queue.push(('progress', (None, translate("テンソルデータ({0}フレーム)と動画({1}フレーム)の結合が完了しました。\n合計フレーム数: {2}フレーム ({3:.2f}秒) - サイズ制限なし").format(uploaded_frames, original_frames, combined_frames, combined_frames / 30), make_progress_bar_html(100, translate('結合完了')))))
                            else:
                                print(translate("テンソルデータの結合に失敗しました。"))
                                stream.output_queue.push(('progress', (None, translate("テンソルデータの結合に失敗しました。"), make_progress_bar_html(100, translate('エラー')))))


                            # real_history_latentsとhistory_pixelsを結合済みのものに更新
                            real_history_latents = combined_history_latents
                            history_pixels = combined_history_pixels

                            # 結合した動画をUIに反映するため、出力フラグを立てる
                            stream.output_queue.push(('file', combined_output_filename))

                            # 出力ファイル名を更新
                            output_filename = combined_output_filename

                            # 結合後の全フレーム数を計算して表示
                            combined_frames = combined_history_pixels.shape[2]
                            combined_size_mb = (combined_history_pixels.element_size() * combined_history_pixels.nelement()) / (1024 * 1024)
                            print(translate("結合完了情報: テンソルデータ({0}フレーム) + 新規動画({1}フレーム) = 合計{2}フレーム").format(uploaded_frames, original_frames, combined_frames))
                            print(translate("結合動画の再生時間: {0:.2f}秒").format(combined_frames / 30))
                            print(translate("データサイズ: {0:.2f} MB（制限無し）").format(combined_size_mb))

                            # UI上で完了メッセージを表示
                            stream.output_queue.push(('progress', (None, translate("テンソルデータ({0}フレーム)と動画({1}フレーム)の結合が完了しました。\n合計フレーム数: {2}フレーム ({3:.2f}秒)").format(uploaded_frames, original_frames, combined_frames, combined_frames / 30), make_progress_bar_html(100, translate('結合完了')))))
                    except Exception as e:
                        print(translate("テンソルデータ結合中にエラーが発生しました: {0}").format(e))
                        traceback.print_exc()
                        stream.output_queue.push(('progress', (None, translate("エラー: テンソルデータ結合に失敗しました - {0}").format(str(e)), make_progress_bar_html(100, translate('エラー')))))

                # 処理終了時に通知
                if HAS_WINSOUND:
                    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)
                else:
                    print(translate("\n✓ 処理が完了しました！"))  # Linuxでの代替通知

                # メモリ解放を明示的に実行
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                    print(translate("[MEMORY] 処理完了後のメモリクリア: {memory:.2f}GB/{total_memory:.2f}GB").format(memory=torch.cuda.memory_allocated()/1024**3, total_memory=torch.cuda.get_device_properties(0).total_memory/1024**3))

                # テンソルデータの保存処理
                print(translate("[DEBUG] worker: テンソルデータ保存判定時の値: {0} (型: {1})").format(save_tensor_data, type(save_tensor_data).__name__))
                if save_tensor_data:
                    try:
                        # 結果のテンソルを保存するファイルパス
                        tensor_file_path = os.path.join(outputs_folder, f'{job_id}.safetensors')

                        # 保存するデータを準備
                        print(translate("=== テンソルデータ保存処理開始 ==="))
                        print(translate("保存対象フレーム数: {frames}").format(frames=real_history_latents.shape[2]))

                        # サイズ制限を完全に撤廃し、全フレームを保存
                        tensor_to_save = real_history_latents.clone().cpu()

                        # テンソルデータの保存サイズの概算
                        tensor_size_mb = (tensor_to_save.element_size() * tensor_to_save.nelement()) / (1024 * 1024)

                        print(translate("テンソルデータを保存中... shape: {shape}, フレーム数: {frames}, サイズ: {size:.2f} MB").format(shape=tensor_to_save.shape, frames=tensor_to_save.shape[2], size=tensor_size_mb))
                        stream.output_queue.push(('progress', (None, translate('テンソルデータを保存中... ({frames}フレーム)').format(frames=tensor_to_save.shape[2]), make_progress_bar_html(95, translate('テンソルデータの保存')))))

                        # メタデータの準備（フレーム数も含める）
                        metadata = torch.tensor([height, width, tensor_to_save.shape[2]], dtype=torch.int32)

                        # safetensors形式で保存
                        tensor_dict = {
                            "history_latents": tensor_to_save,
                            "metadata": metadata
                        }
                        sf.save_file(tensor_dict, tensor_file_path)

                        print(translate("テンソルデータを保存しました: {path}").format(path=tensor_file_path))
                        print(translate("保存済みテンソルデータ情報: {frames}フレーム, {size:.2f} MB").format(frames=tensor_to_save.shape[2], size=tensor_size_mb))
                        print(translate("=== テンソルデータ保存処理完了 ==="))
                        stream.output_queue.push(('progress', (None, translate("テンソルデータが保存されました: {path} ({frames}フレーム, {size:.2f} MB)").format(path=os.path.basename(tensor_file_path), frames=tensor_to_save.shape[2], size=tensor_size_mb), make_progress_bar_html(100, translate('処理完了')))))

                        # アップロードされたテンソルデータがあれば、それも結合したものを保存する
                        if tensor_data_input is not None and uploaded_tensor is not None:
                            try:
                                # アップロードされたテンソルデータのファイル名を取得
                                uploaded_tensor_filename = os.path.basename(tensor_data_input.name)
                                tensor_combined_path = os.path.join(outputs_folder, f'{job_id}_combined_tensors.safetensors')

                                print(translate("=== テンソルデータ結合処理開始 ==="))
                                print(translate("生成テンソルと入力テンソルを結合して保存します"))
                                print(translate("生成テンソル: {frames}フレーム").format(frames=tensor_to_save.shape[2]))
                                print(translate("入力テンソル: {frames}フレーム").format(frames=uploaded_tensor.shape[2]))

                                # データ型とデバイスを統一
                                if uploaded_tensor.dtype != tensor_to_save.dtype:
                                    uploaded_tensor = uploaded_tensor.to(dtype=tensor_to_save.dtype)
                                if uploaded_tensor.device != tensor_to_save.device:
                                    uploaded_tensor = uploaded_tensor.to(device=tensor_to_save.device)

                                # サイズチェック
                                if uploaded_tensor.shape[3] != tensor_to_save.shape[3] or uploaded_tensor.shape[4] != tensor_to_save.shape[4]:
                                    print(translate("警告: テンソルサイズが一致しないため結合できません: {uploaded_shape} vs {tensor_shape}").format(uploaded_shape=uploaded_tensor.shape, tensor_shape=tensor_to_save.shape))
                                else:
                                    # 結合（生成テンソルの後にアップロードされたテンソルを追加）
                                    combined_tensor = torch.cat([tensor_to_save, uploaded_tensor], dim=2)
                                    combined_frames = combined_tensor.shape[2]
                                    combined_size_mb = (combined_tensor.element_size() * combined_tensor.nelement()) / (1024 * 1024)

                                    # メタデータ更新
                                    combined_metadata = torch.tensor([height, width, combined_frames], dtype=torch.int32)

                                    # 結合したテンソルを保存
                                    combined_tensor_dict = {
                                        "history_latents": combined_tensor,
                                        "metadata": combined_metadata
                                    }
                                    sf.save_file(combined_tensor_dict, tensor_combined_path)

                                    print(translate("結合テンソルを保存しました: {path}").format(path=tensor_combined_path))
                                    print(translate("結合テンソル情報: 合計{0}フレーム ({1}+{2}), {3:.2f} MB").format(frames, tensor_to_save.shape[2], uploaded_tensor.shape[2], size))
                                    print(translate("=== テンソルデータ結合処理完了 ==="))
                                    stream.output_queue.push(('progress', (None, translate("テンソルデータ結合が保存されました: 合計{frames}フレーム").format(frames=combined_frames), make_progress_bar_html(100, translate('結合テンソル保存完了')))))
                            except Exception as e:
                                print(translate("テンソルデータ結合保存エラー: {0}").format(e))
                                traceback.print_exc()
                    except Exception as e:
                        print(translate("テンソルデータ保存エラー: {0}").format(e))
                        traceback.print_exc()
                        stream.output_queue.push(('progress', (None, translate("テンソルデータの保存中にエラーが発生しました。"), make_progress_bar_html(100, translate('処理完了')))))

                # 全体の処理時間を計算
                process_end_time = time.time()
                total_process_time = process_end_time - process_start_time
                hours, remainder = divmod(total_process_time, 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = ""
                if hours > 0:
                    time_str = translate("{0}時間 {1}分 {2}秒").format(int(hours), int(minutes), f"{seconds:.1f}")
                elif minutes > 0:
                    time_str = translate("{0}分 {1}秒").format(int(minutes), f"{seconds:.1f}")
                else:
                    time_str = translate("{0:.1f}秒").format(seconds)
                print(translate("\n全体の処理時間: {0}").format(time_str))

                # 完了メッセージの設定（結合有無によって変更）
                if combined_output_filename is not None:
                    # テンソル結合が成功した場合のメッセージ
                    combined_filename_only = os.path.basename(combined_output_filename)
                    completion_message = translate("すべてのセクション({sections}/{total_sections})が完了しました。テンソルデータとの後方結合も完了しました。結合ファイル名: {filename}\n全体の処理時間: {time}").format(sections=sections, total_sections=total_sections, filename=combined_filename_only, time=time_str)
                    # 最終的な出力ファイルを結合したものに変更
                    output_filename = combined_output_filename
                else:
                    # 通常の完了メッセージ
                    completion_message = translate("すべてのセクション({sections}/{total_sections})が完了しました。全体の処理時間: {time}").format(sections=total_sections, total_sections=total_sections, time=time_str)

                stream.output_queue.push(('progress', (None, completion_message, make_progress_bar_html(100, translate('処理完了')))))

                # 中間ファイルの削除処理
                if not keep_section_videos:
                    # 最終動画のフルパス
                    final_video_path = output_filename
                    final_video_name = os.path.basename(final_video_path)
                    # job_id部分を取得（タイムスタンプ部分）
                    job_id_part = job_id

                    # ディレクトリ内のすべてのファイルを取得
                    files = os.listdir(outputs_folder)
                    deleted_count = 0

                    for file in files:
                        # 同じjob_idを持つMP4ファイルかチェック
                        # 結合ファイル('combined'を含む)は消さないように保護
                        if file.startswith(job_id_part) and file.endswith('.mp4') \
                           and file != final_video_name \
                           and 'combined' not in file:  # combinedファイルは保護
                            file_path = os.path.join(outputs_folder, file)
                            try:
                                os.remove(file_path)
                                deleted_count += 1
                                print(translate("[削除] 中間ファイル: {0}").format(file))
                            except Exception as e:
                                print(translate("[エラー] ファイル削除時のエラー {0}: {1}").format(file, e))

                    if deleted_count > 0:
                        print(translate("[済] {0}個の中間ファイルを削除しました。最終ファイルは保存されています: {1}").format(deleted_count, final_video_name))
                        final_message = translate("中間ファイルを削除しました。最終動画と結合動画は保存されています。")
                        stream.output_queue.push(('progress', (None, final_message, make_progress_bar_html(100, translate('処理完了')))))

                break
    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream.output_queue.push(('end', None))
    return


# 画像のバリデーション関数
def validate_images(input_image, section_settings, length_radio=None, frame_size_radio=None):
    """入力画像または画面に表示されている最後のキーフレーム画像のいずれかが有効かを確認する"""
    # 入力画像をチェック
    if input_image is not None:
        return True, ""

    # 現在の設定から表示すべきセクション数を計算
    total_display_sections = None
    if length_radio is not None and frame_size_radio is not None:
        try:
            # 動画長を秒数で取得
            seconds = get_video_seconds(length_radio.value)

            # フレームサイズ設定からlatent_window_sizeを計算
            latent_window_size = 4.5 if frame_size_radio.value == translate("0.5秒 (17フレーム)") else 9
            frame_count = latent_window_size * 4 - 3

            # セクション数を計算
            total_frames = int(seconds * 30)
            total_display_sections = int(max(round(total_frames / frame_count), 1))
            print(translate("[DEBUG] 現在の設定によるセクション数: {0}").format(total_display_sections))
        except Exception as e:
            print(translate("[ERROR] セクション数計算エラー: {0}").format(e))

    # 入力画像がない場合、表示されているセクションの中で最後のキーフレーム画像をチェック
    last_visible_section_image = None
    last_visible_section_num = -1

    if section_settings is not None and not isinstance(section_settings, bool):
        # 有効なセクション番号を収集
        valid_sections = []
        try:
            for section in section_settings:
                if section and len(section) > 1 and section[0] is not None:
                    try:
                        section_num = int(section[0])
                        # 表示セクション数が計算されていれば、それ以下のセクションのみ追加
                        if total_display_sections is None or section_num < total_display_sections:
                            valid_sections.append((section_num, section[1]))
                    except (ValueError, TypeError):
                        continue
        except (TypeError, ValueError):
            # section_settingsがイテラブルでない場合（ブール値など）、空のリストとして扱う
            valid_sections = []
            print(f"[DEBUG] section_settings is not iterable: {type(section_settings)}")


        # 有効なセクションがあれば、最大の番号（最後のセクション）を探す
        if valid_sections:
            # 番号でソート
            valid_sections.sort(key=lambda x: x[0])
            # 最後のセクションを取得
            last_visible_section_num, last_visible_section_image = valid_sections[-1]

            print(translate("[DEBUG] 最後のキーフレーム確認: セクション{0} (画像あり: {1})").format(last_visible_section_num, last_visible_section_image is not None))

    # 最後のキーフレーム画像があればOK
    if last_visible_section_image is not None:
        return True, ""

    # どちらの画像もない場合はエラー
    error_html = f"""
    <div style="padding: 15px; border-radius: 10px; background-color: #ffebee; border: 1px solid #f44336; margin: 10px 0;">
        <h3 style="color: #d32f2f; margin: 0 0 10px 0;">{translate('❗️ 画像が選択されていません')}</h3>
        <p>{translate('生成を開始する前に「Image」欄または表示されている最後のキーフレーム画像に画像をアップロードしてください。これはあまねく叡智の始発点となる重要な画像です。')}</p>
    </div>
    """
    error_bar = make_progress_bar_html(100, translate('画像がありません'))
    return False, error_html + error_bar

def process(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, use_random_seed, mp4_crf=16, all_padding_value=1.0, image_strength=1.0, frame_size_setting="1秒 (33フレーム)", keep_section_videos=False, lora_files=None, lora_files2=None, lora_scales_text="0.8,0.8", output_dir=None, save_section_frames=False, use_all_padding=False, use_lora=False, save_tensor_data=False, section_settings=None, tensor_data_input=None, fp8_optimization=False, resolution=640, batch_count=1, save_latent_frames=False, save_last_section_frames=False):
    # メイン生成処理
    global stream
    global batch_stopped

    # バッチ処理開始時に停止フラグをリセット
    batch_stopped = False


    # フレームサイズ設定に応じてlatent_window_sizeを先に調整
    if frame_size_setting == "0.5秒 (17フレーム)":
        # 0.5秒の場合はlatent_window_size=4.5に設定（実際には4.5*4-3=17フレーム≒0.5秒@30fps）
        latent_window_size = 4.5
        print(translate('フレームサイズを0.5秒モードに設定: latent_window_size = {0}').format(latent_window_size))
    else:
        # デフォルトの1秒モードではlatent_window_size=9を使用（9*4-3=33フレーム≒1秒@30fps）
        latent_window_size = 9
        print(translate('フレームサイズを1秒モードに設定: latent_window_size = {0}').format(latent_window_size))

    # バッチ処理回数を確認し、詳細を出力
    batch_count = max(1, min(int(batch_count), 100))  # 1〜100の間に制限
    print(translate("\u25c6 バッチ処理回数: {0}回").format(batch_count))

    # 解像度を安全な値に丸めてログ表示
    from diffusers_helper.bucket_tools import SAFE_RESOLUTIONS

    # 解像度値を表示
    print(translate("\u25c6 UIから受け取った解像度値: {0}（型: {1}）").format(resolution, type(resolution).__name__))

    # 安全な値に丸める
    if resolution not in SAFE_RESOLUTIONS:
        closest_resolution = min(SAFE_RESOLUTIONS, key=lambda x: abs(x - resolution))
        print(translate('安全な解像度値ではないため、{0}から{1}に自動調整しました').format(resolution, closest_resolution))
        resolution = closest_resolution

    # 解像度設定を出力
    print(translate('解像度を設定: {0}').format(resolution))

    # 動画生成の設定情報をログに出力
    # 4.5の場合は5として計算するための特別処理
    if latent_window_size == 4.5:
        frame_count = 17  # 5 * 4 - 3 = 17
    else:
        frame_count = int(latent_window_size * 4 - 3)
    total_latent_sections = int(max(round((total_second_length * 30) / frame_count), 1))

    # F1モードでは常に通常のみ
    mode_name = translate("通常モード")

    print(translate("\n==== 動画生成開始 ====="))
    print(translate("\u25c6 生成モード: {0}").format(mode_name))
    print(translate("\u25c6 動画長: {0}秒").format(total_second_length))
    print(translate("\u25c6 フレームサイズ: {0}").format(frame_size_setting))
    print(translate("\u25c6 生成セクション数: {0}回").format(total_latent_sections))
    print(translate("\u25c6 サンプリングステップ数: {0}").format(steps))
    print(translate("\u25c6 TeaCache使用: {0}").format(use_teacache))
    # TeaCache使用の直後にSEED値の情報を表示
    print(translate("\u25c6 使用SEED値: {0}").format(seed))
    print(translate("\u25c6 LoRA使用: {0}").format(use_lora))

    # FP8最適化設定のログ出力
    print(translate("\u25c6 FP8最適化: {0}").format(fp8_optimization))

    # オールパディング設定のログ出力（F1モードでは常に無効）
    print(translate("\u25c6 オールパディング: F1モードでは無効化されています"))

    # LoRA情報のログ出力
    if use_lora and has_lora_support:
        all_lora_files = []
        
        # 1つ目のLoRAファイルを処理
        if lora_files is not None:
            if isinstance(lora_files, list):
                all_lora_files.extend(lora_files)
            else:
                all_lora_files.append(lora_files)
                
        # 2つ目のLoRAファイルを処理
        if lora_files2 is not None:
            if isinstance(lora_files2, list):
                all_lora_files.extend(lora_files2)
            else:
                all_lora_files.append(lora_files2)
        
        # スケール値を解析
        try:
            scales = [float(s.strip()) for s in lora_scales_text.split(',')]
        except:
            # 解析エラーの場合はデフォルト値を使用
            scales = [0.8] * len(all_lora_files)
            
        # スケール値の数を調整
        if len(scales) < len(all_lora_files):
            scales.extend([0.8] * (len(all_lora_files) - len(scales)))
        elif len(scales) > len(all_lora_files):
            scales = scales[:len(all_lora_files)]
            
        # LoRAファイル情報を出力
        if len(all_lora_files) == 1:
            # 単一ファイル
            print(translate("\u25c6 LoRAファイル: {0}").format(os.path.basename(all_lora_files[0].name)))
            print(translate("\u25c6 LoRA適用強度: {0}").format(scales[0]))
        elif len(all_lora_files) > 1:
            # 複数ファイル
            print(translate("\u25c6 LoRAファイル (複数):"))
            for i, file in enumerate(all_lora_files):
                print(f"   - {os.path.basename(file.name)} (スケール: {scales[i]})")
        else:
            # LoRAファイルなし
            print(translate("\u25c6 LoRA: 使用しない"))

    # セクションごとのキーフレーム画像の使用状況をログに出力
    valid_sections = []
    if section_settings is not None:
        for i, sec_data in enumerate(section_settings):
            if sec_data and sec_data[1] is not None:  # 画像が設定されている場合
                valid_sections.append(sec_data[0])

    if valid_sections:
        print(translate("\u25c6 使用するキーフレーム画像: セクション{0}").format(', '.join(map(str, valid_sections))))
    else:
        print(translate("◆ キーフレーム画像: デフォルト設定のみ使用"))

    print("=============================\n")

    # バッチ処理の全体停止用フラグ
    batch_stopped = False

    # 元のシード値を保存（バッチ処理用）
    original_seed = seed

    if use_random_seed:
        seed = random.randint(0, 2**32 - 1)
        # UIのseed欄もランダム値で更新
        yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True), gr.update(value=seed)
        # ランダムシードの場合は最初の値を更新
        original_seed = seed
    else:
        yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True), gr.update()

    stream = AsyncStream()

    # stream作成後、バッチ処理前もう一度フラグを確認
    if batch_stopped:
        print(translate("\nバッチ処理が中断されました（バッチ開始前）"))
        yield (
            None,
            gr.update(visible=False),
            translate("バッチ処理が中断されました"),
            '',
            gr.update(interactive=True),
            gr.update(interactive=False, value=translate("End Generation")),
            gr.update()
        )
        return

    # バッチ処理ループの開始
    for batch_index in range(batch_count):
        # 停止フラグが設定されている場合は全バッチ処理を中止
        if batch_stopped:
            print(translate("\nバッチ処理がユーザーによって中止されました"))
            yield (
                None,
                gr.update(visible=False),
                translate("バッチ処理が中止されました。"),
                '',
                gr.update(interactive=True),
                gr.update(interactive=False, value=translate("End Generation")),
                gr.update()
            )
            break

        # 現在のバッチ番号を表示
        if batch_count > 1:
            batch_info = translate("バッチ処理: {0}/{1}").format(batch_index + 1, batch_count)
            print(f"\n{batch_info}")
            # UIにもバッチ情報を表示
            yield None, gr.update(visible=False), batch_info, "", gr.update(interactive=False), gr.update(interactive=True), gr.update()

        # バッチインデックスに応じてSEED値を設定
        current_seed = original_seed + batch_index
        if batch_count > 1:
            print(translate("現在のSEED値: {0}").format(current_seed))
        # 現在のバッチ用のシードを設定
        seed = current_seed

        # もう一度停止フラグを確認 - worker処理実行前
        if batch_stopped:
            print(translate("バッチ処理が中断されました。worker関数の実行をキャンセルします。"))
            # 中断メッセージをUIに表示
            yield (None,
                   gr.update(visible=False),
                   translate("バッチ処理が中断されました（{0}/{1}）").format(batch_index, batch_count),
                   '',
                   gr.update(interactive=True),
                   gr.update(interactive=False, value=translate("End Generation")),
                   gr.update())
            break

        # GPUメモリの設定値をデバッグ出力し、正しい型に変換
        gpu_memory_value = float(gpu_memory_preservation) if gpu_memory_preservation is not None else 6.0
        print(translate('Using GPU memory preservation setting: {0} GB').format(gpu_memory_value))

        # 出力フォルダが空の場合はデフォルト値を使用
        if not output_dir or not output_dir.strip():
            output_dir = "outputs"
        print(translate('Output directory: {0}').format(output_dir))

        # 先に入力データの状態をログ出力（デバッグ用）
        if input_image is not None:
            if isinstance(input_image, str):
                print(translate("[DEBUG] input_image path: {0}, type: {1}").format(input_image, type(input_image)))
            else:
                print(translate("[DEBUG] input_image shape: {0}, type: {1}").format(input_image.shape, type(input_image)))
        if section_settings is not None:
            print(translate("[DEBUG] section_settings count: {0}").format(len(section_settings)))
            valid_images = sum(1 for s in section_settings if s and s[1] is not None)
            print(translate("[DEBUG] Valid section images: {0}").format(valid_images))

        # バッチ処理の各回で実行
        # worker関数の引数順とips配列の順序をデバッグログから確認し、正確に追跡
        print(translate("[DEBUG] async_runに渡すsave_tensor_data: {0}").format(save_tensor_data))
        print(translate("[DEBUG] フレーム保存モード (オリジナル): {0}").format(frame_save_mode))
        
        # Gradioオブジェクトから実際の値を取得
        if hasattr(frame_save_mode, 'value'):
            frame_save_mode_actual = frame_save_mode.value
        else:
            frame_save_mode_actual = frame_save_mode
            
        print(translate("[DEBUG] フレーム保存モード (実際の値): {0}").format(frame_save_mode_actual))
        print(translate("[DEBUG] save_latent_frames: {0}").format(save_latent_frames))
        print(translate("[DEBUG] save_last_section_frames: {0}").format(save_last_section_frames))
        async_run(
            worker,
            input_image,
            prompt,
            n_prompt,
            seed,
            total_second_length,
            latent_window_size,
            steps,
            cfg,
            gs,
            rs,
            gpu_memory_value,
            use_teacache,
            mp4_crf,
            all_padding_value,
            image_strength,
            keep_section_videos,
            lora_files,
            lora_files2,
            lora_scales_text,
            output_dir,
            save_section_frames,
            use_all_padding,
            use_lora,
            save_tensor_data,
            tensor_data_input,
            fp8_optimization,
            resolution,
            batch_index,
            save_latent_frames,
            save_last_section_frames
        )

        # 現在のバッチの出力ファイル名
        batch_output_filename = None

        # 現在のバッチの処理結果を取得
        while True:
            flag, data = stream.output_queue.next()

            if flag == 'file':
                batch_output_filename = data
                # より明確な更新方法を使用し、preview_imageを明示的にクリア
                yield batch_output_filename, gr.update(value=None, visible=False), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), gr.update()

            if flag == 'progress':
                preview, desc, html = data
                # バッチ処理中は現在のバッチ情報を追加
                if batch_count > 1:
                    batch_info = translate("バッチ処理: {0}/{1} - ").format(batch_index + 1, batch_count)
                    desc = batch_info + desc
                # preview_imageを明示的に設定
                yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True), gr.update()

            if flag == 'end':
                # このバッチの処理が終了
                if batch_index == batch_count - 1 or batch_stopped:
                    # 最終バッチの場合は処理完了を通知
                    completion_message = ""
                    if batch_stopped:
                        completion_message = translate("バッチ処理が中止されました（{0}/{1}）").format(batch_index + 1, batch_count)
                    else:
                        completion_message = translate("バッチ処理が完了しました（{0}/{1}）").format(batch_count, batch_count)
                    yield (
                        batch_output_filename,
                        gr.update(value=None, visible=False),
                        completion_message,
                        '',
                        gr.update(interactive=True),
                        gr.update(interactive=False, value=translate("End Generation")),
                        gr.update()
                    )
                else:
                    # 次のバッチに進むメッセージを表示
                    next_batch_message = translate("バッチ処理: {0}/{1} 完了、次のバッチに進みます...").format(batch_index + 1, batch_count)
                    yield (
                        batch_output_filename,
                        gr.update(value=None, visible=False),
                        next_batch_message,
                        '',
                        gr.update(interactive=False),
                        gr.update(interactive=True),
                        gr.update()
                    )
                break

        # 最終的な出力ファイル名を更新
        output_filename = batch_output_filename

        # バッチ処理が停止されている場合はループを抜ける
        if batch_stopped:
            print(translate("バッチ処理ループを中断します"))
            break


def end_process():
    global stream
    global batch_stopped

    # 現在のバッチと次のバッチ処理を全て停止するフラグを設定
    batch_stopped = True
    print(translate("\n停止ボタンが押されました。バッチ処理を停止します..."))
    # 現在実行中のバッチを停止
    stream.input_queue.push('end')

    # ボタンの名前を一時的に変更することでユーザーに停止処理が進行中であることを表示
    return gr.update(value=translate("停止処理中..."))





# 既存のQuick Prompts（初期化時にプリセットに変換されるので、互換性のために残す）
quick_prompts = [
    'A character doing some simple body movements.',
    'A character uses expressive hand gestures and body language.',
    'A character walks leisurely with relaxed movements.',
    'A character performs dynamic movements with energy and flowing motion.',
    'A character moves in unexpected ways, with surprising transitions poses.',
]
quick_prompts = [[x] for x in quick_prompts]


css = get_app_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.HTML('<h1>FramePack<span class="title-suffix">-<s>eichi</s> F1</span></h1>')

    # デバッグ情報の表示
    # print_keyframe_debug_info()

    # 一番上の行に「生成モード、セクションフレームサイズ、オールパディング、動画長」を配置
    with gr.Row():
        with gr.Column(scale=1):
            # 生成モードのラジオボタン（F1モードでは通常のみ）
            mode_radio = gr.Radio(choices=[MODE_TYPE_NORMAL], value=MODE_TYPE_NORMAL, label=translate("生成モード"), info=translate("F1モードでは通常のみ利用可能"))
        with gr.Column(scale=1):
            # フレームサイズ切替用のUIコントロール（名前を「セクションフレームサイズ」に変更）
            frame_size_radio = gr.Radio(
                choices=[translate("1秒 (33フレーム)"), translate("0.5秒 (17フレーム)")],
                value=translate("1秒 (33フレーム)"),
                label=translate("セクションフレームサイズ"),
                info=translate("1秒 = 高品質・通常速度 / 0.5秒 = よりなめらかな動き（実験的機能）")
            )
        with gr.Column(scale=1):
            # オールパディング設定 (F1モードでは無効化)
            use_all_padding = gr.Checkbox(
                label=translate("オールパディング"),
                value=False,
                info=translate("F1モードでは使用できません。無印モードでのみ有効です。"),
                elem_id="all_padding_checkbox",
                interactive=False  # F1モードでは非活性化
            )
            all_padding_value = gr.Slider(
                label=translate("パディング値"),
                minimum=0.2,
                maximum=3,
                value=1,
                step=0.1,
                info=translate("F1モードでは使用できません"),
                visible=False,
                interactive=False  # F1モードでは非活性化
            )

            # オールパディングのチェックボックス状態に応じてスライダーの表示/非表示を切り替える
            def toggle_all_padding_visibility(use_all_padding):
                return gr.update(visible=use_all_padding)

            use_all_padding.change(
                fn=toggle_all_padding_visibility,
                inputs=[use_all_padding],
                outputs=[all_padding_value]
            )
        with gr.Column(scale=1):
            # 設定から動的に選択肢を生成
            length_radio = gr.Radio(choices=get_video_modes(), value=translate("1秒"), label=translate("動画長"), info=translate("動画の長さを設定。F1モードでは右下の「動画の総長（秒）」で20秒より長い動画長を設定可能です"))

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources=['upload', 'clipboard'], type="filepath", label="Image", height=320)

            # テンソルデータ設定をグループ化して灰色のタイトルバーに変更
            with gr.Group():
                gr.Markdown(f"### " + translate("テンソルデータ設定"))

                # テンソルデータ使用有無のチェックボックス
                use_tensor_data = gr.Checkbox(label=translate("テンソルデータを使用する"), value=False, info=translate("チェックをオンにするとテンソルデータをアップロードできます"))

                # テンソルデータ設定コンポーネント（初期状態では非表示）
                with gr.Group(visible=False) as tensor_data_group:
                    tensor_data_input = gr.File(
                        label=translate("テンソルデータアップロード (.safetensors) - 生成動画の後方(末尾)に結合されます"),
                        file_types=[".safetensors"]
                    )

                    gr.Markdown(translate("※ テンソルデータをアップロードすると通常の動画生成後に、その動画の後方（末尾）に結合されます。\n結合した動画は「元のファイル名_combined.mp4」として保存されます。\n※ テンソルデータの保存機能を有効にすると、生成とアップロードのテンソルを結合したデータも保存されます。\n※ テンソルデータの結合は別ツール `python eichi_utils/tensor_combiner.py --ui` でもできます。"))

                # チェックボックスの状態によってテンソルデータ設定の表示/非表示を切り替える関数
                def toggle_tensor_data_settings(use_tensor):
                    return gr.update(visible=use_tensor)

                # チェックボックスの変更イベントに関数を紐づけ
                use_tensor_data.change(
                    fn=toggle_tensor_data_settings,
                    inputs=[use_tensor_data],
                    outputs=[tensor_data_group]
                )

            # テンソルデータ設定の下に解像度スライダーとバッチ処理回数を追加
            with gr.Group():
                with gr.Row():
                    with gr.Column(scale=2):
                        resolution = gr.Dropdown(
                            label=translate("解像度"),
                            choices=[512, 640, 768, 960, 1080],
                            value=640,
                            info=translate("出力動画の基準解像度。640推奨。960/1080は高負荷・高メモリ消費")
                        )
                    with gr.Column(scale=1):
                        batch_count = gr.Slider(
                            label=translate("バッチ処理回数"),
                            minimum=1,
                            maximum=100,
                            value=1,
                            step=1,
                            info=translate("同じ設定で連続生成する回数。SEEDは各回で+1されます")
                        )

            # 開始・終了ボタン
            with gr.Row():
                start_button = gr.Button(value=translate("Start Generation"))
                end_button = gr.Button(value=translate("End Generation"), interactive=False)

            # セクション入力用のリストを初期化
            section_number_inputs = []
            section_image_inputs = []
            section_prompt_inputs = []  # プロンプト入力欄用のリスト
            section_row_groups = []  # 各セクションのUI行を管理するリスト

            # 設定から最大キーフレーム数を取得
            max_keyframes = get_max_keyframes_count()

            # 現在の動画モードで必要なセクション数を取得する関数
            def get_current_sections_count():
                mode_value = length_radio.value
                if mode_value in VIDEO_MODE_SETTINGS:
                    # sections値をそのまま使用 - 注：これは0から始めた場合の最大値となる
                    return VIDEO_MODE_SETTINGS[mode_value]["sections"]
                return max_keyframes  # デフォルト値

            # 現在の必要セクション数を取得
            initial_sections_count = get_current_sections_count()
            # 簡略化セクション表示
            # セクションタイトルの関数は削除し、固定メッセージのみ表示

            # 埋め込みプロンプトおよびシードを複写するチェックボックスの定義
            # グローバル変数として定義し、後で他の場所から参照できるようにする
            global copy_metadata
            copy_metadata = gr.Checkbox(
                label=translate("埋め込みプロンプトおよびシードを複写する"),
                value=False,
                info=translate("チェックをオンにすると、画像のメタデータからプロンプトとシードを自動的に取得します")
            )

            # F1モードではセクション設定は完全に削除
            # 隠しコンポーネント（互換性のため）
            section_image_inputs = []
            section_number_inputs = []
            section_prompt_inputs = []
            section_row_groups = []


            # メタデータ抽出関数を定義（後で登録する）
            def update_from_image_metadata(image_path, copy_enabled=False):
                """Imageアップロード時にメタデータを抽出してUIに反映する
                F1モードではキーフレームコピー機能を削除済みのため、単純化
                """
                print(translate("\n[DEBUG] F1モード：update_from_image_metadata関数が実行されました"))
                print(translate("[DEBUG] メタデータ複写機能: {0}").format(copy_enabled))

                # 複写機能が無効の場合は何もしない
                if not copy_enabled:
                    print(translate("[DEBUG] メタデータ複写機能が無効化されているため、処理をスキップします"))
                    return [gr.update()] * 2

                if image_path is None:
                    print(translate("[DEBUG] image_pathはNoneです"))
                    return [gr.update()] * 2

                print(translate("[DEBUG] 画像パス: {0}").format(image_path))

                try:
                    # ファイルパスから直接メタデータを抽出
                    print(translate("[DEBUG] extract_metadata_from_pngをファイルパスから直接呼び出します"))
                    metadata = extract_metadata_from_png(image_path)

                    if not metadata:
                        print(translate("[DEBUG] メタデータが抽出されませんでした"))
                        print(translate("アップロードされた画像にメタデータが含まれていません"))
                        return [gr.update()] * 2

                    print(translate("[DEBUG] メタデータサイズ: {0}, 内容: {1}").format(len(metadata), metadata))
                    print(translate("画像からメタデータを抽出しました: {0}").format(metadata))

                    # プロンプトとSEEDをUIに反映
                    prompt_update = gr.update()
                    seed_update = gr.update()

                    if PROMPT_KEY in metadata and metadata[PROMPT_KEY]:
                        prompt_update = gr.update(value=metadata[PROMPT_KEY])
                        print(translate("[DEBUG] プロンプトを更新: {0}").format(metadata[PROMPT_KEY]))
                        print(translate("プロンプトを画像から取得: {0}").format(metadata[PROMPT_KEY]))

                    if SEED_KEY in metadata and metadata[SEED_KEY]:
                        # SEED値を整数に変換
                        try:
                            seed_value = int(metadata[SEED_KEY])
                            seed_update = gr.update(value=seed_value)
                            print(translate("[DEBUG] SEED値を更新: {0}").format(seed_value))
                            print(translate("SEED値を画像から取得: {0}").format(seed_value))
                        except (ValueError, TypeError):
                            print(translate("[DEBUG] SEED値の変換エラー: {0}").format(metadata[SEED_KEY]))
                            print(translate("SEED値の変換エラー: {0}").format(metadata[SEED_KEY]))

                    print(translate("[DEBUG] 更新結果: prompt_update={0}, seed_update={1}").format(prompt_update, seed_update))
                    return [prompt_update, seed_update]
                except Exception as e:
                    print(translate("[ERROR] メタデータ抽出処理中のエラー: {0}").format(e))
                    traceback.print_exc()
                    print(translate("メタデータ抽出エラー: {0}").format(e))
                    return [gr.update()] * 2

            # 注意: イベント登録は変数定義後に行うため、後で実行する
            # メタデータ抽出処理の登録は、promptとseed変数の定義後に移動します

            # LoRA設定グループを追加
            with gr.Group(visible=has_lora_support) as lora_settings_group:
                gr.Markdown(f"### " + translate("LoRA設定"))

                # LoRA使用有無のチェックボックス
                use_lora = gr.Checkbox(label=translate("LoRAを使用する"), value=False, info=translate("チェックをオンにするとLoRAを使用します（要16GB VRAM以上）"))

                # LoRA設定コンポーネント（初期状態では非表示）
                # メインのLoRAファイル
                lora_files = gr.File(
                    label=translate("LoRAファイル (.safetensors, .pt, .bin)"),
                    file_types=[".safetensors", ".pt", ".bin"],
                    visible=False
                )
                # 追加のLoRAファイル
                lora_files2 = gr.File(
                    label=translate("LoRAファイル2 (.safetensors, .pt, .bin)"),
                    file_types=[".safetensors", ".pt", ".bin"],
                    visible=False
                )
                # スケール値の入力フィールド
                lora_scales_text = gr.Textbox(
                    label=translate("LoRA適用強度 (カンマ区切り)"),
                    value="0.8,0.8",
                    info=translate("各LoRAのスケール値をカンマ区切りで入力 (例: 0.8,0.5)"),
                    visible=False
                )
                fp8_optimization = gr.Checkbox(
                    label=translate("FP8最適化"),
                    value=False,
                    info=translate("メモリ使用量を削減し、速度を改善します（PyTorch 2.1以上が必要）"),
                    visible=False
                )
                lora_blocks_type = gr.Dropdown(
                    label=translate("LoRAブロック選択"),
                    choices=["all", "single_blocks", "double_blocks", "db0-9", "db10-19", "sb0-9", "sb10-19", "important"],
                    value="all",
                    info=translate("選択するブロックタイプ（all=すべて、その他=メモリ節約）"),
                    visible=False
                )

                # チェックボックスの状態によって他のLoRA設定の表示/非表示を切り替える関数
                def toggle_lora_settings(use_lora):
                    return [
                        gr.update(visible=use_lora),  # lora_files
                        gr.update(visible=use_lora),  # lora_files2
                        gr.update(visible=use_lora),  # lora_scales_text
                        gr.update(visible=use_lora),  # fp8_optimization
                    ]

                # チェックボックスの変更イベントに関数を紋づけ
                use_lora.change(fn=toggle_lora_settings,
                           inputs=[use_lora],
                           outputs=[lora_files, lora_files2, lora_scales_text, fp8_optimization])

                # LoRAサポートが無効の場合のメッセージ
                if not has_lora_support:
                    gr.Markdown(translate("LoRAサポートは現在無効です。lora_utilsモジュールが必要です。"))

            prompt = gr.Textbox(label=translate("Prompt"), value=get_default_startup_prompt(), lines=6)

            # プロンプト管理パネルの追加
            with gr.Group(visible=True) as prompt_management:
                gr.Markdown(f"### " + translate("プロンプト管理"))

                # 編集画面を常時表示する
                with gr.Group(visible=True):
                    # 起動時デフォルトの初期表示用に取得
                    default_prompt = ""
                    default_name = ""
                    for preset in load_presets()["presets"]:
                        if preset.get("is_startup_default", False):
                            default_prompt = preset["prompt"]
                            default_name = preset["name"]
                            break

                    with gr.Row():
                        edit_name = gr.Textbox(label=translate("プリセット名"), placeholder=translate("名前を入力..."), value=default_name)

                    edit_prompt = gr.Textbox(label=translate("プロンプト"), lines=5, value=default_prompt)

                    with gr.Row():
                        # 起動時デフォルトをデフォルト選択に設定
                        default_preset = translate("起動時デフォルト")
                        # プリセットデータから全プリセット名を取得
                        presets_data = load_presets()
                        choices = [preset["name"] for preset in presets_data["presets"]]
                        default_presets = [name for name in choices if any(p["name"] == name and p.get("is_default", False) for p in presets_data["presets"])]
                        user_presets = [name for name in choices if name not in default_presets]
                        sorted_choices = [(name, name) for name in sorted(default_presets) + sorted(user_presets)]
                        preset_dropdown = gr.Dropdown(label=translate("プリセット"), choices=sorted_choices, value=default_preset, type="value")

                    with gr.Row():
                        save_btn = gr.Button(value=translate("保存"), variant="primary")
                        apply_preset_btn = gr.Button(value=translate("反映"), variant="primary")
                        clear_btn = gr.Button(value=translate("クリア"))
                        delete_preset_btn = gr.Button(value=translate("削除"))

                # メッセージ表示用
                result_message = gr.Markdown("")

            # プリセットの説明文を削除

            # 互換性のためにQuick Listも残しておくが、非表示にする
            with gr.Row(visible=False):
                example_quick_prompts = gr.Dataset(samples=quick_prompts, label=translate("Quick List"), samples_per_page=1000, components=[prompt])
                example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            # 以下の設定ブロックは右カラムに移動しました

                # セクション設定のリストは既にアコーディオン内で初期化されています
                # section_number_inputs
                # section_image_inputs
                # section_prompt_inputs
                # section_row_groups

                # collect_section_settings関数は未使用のため削除

                # シンプルな互換性のためのダミーステートを作成
                section_settings = gr.State([[None, None, ""] for _ in range(max_keyframes)])
                section_inputs = []

                # update_section_settings関数は未使用のため削除

                # フレームサイズ変更時の処理を追加
                def update_section_calculation(frame_size, mode, length):
                    """フレームサイズ変更時にセクション数を再計算して表示を更新"""
                    # 動画長を取得
                    seconds = get_video_seconds(length)

                    # latent_window_sizeを設定
                    latent_window_size = 4.5 if frame_size == translate("0.5秒 (17フレーム)") else 9
                    frame_count = latent_window_size * 4 - 3

                    # セクション数を計算
                    total_frames = int(seconds * 30)
                    total_sections = int(max(round(total_frames / frame_count), 1))

                    # 計算詳細を表示するHTMLを生成
                    html = f"""<div style='padding: 10px; background-color: #f5f5f5; border-radius: 5px; font-size: 14px;'>
                    {translate('<strong>計算詳細</strong>: フレームサイズ={0}, 総フレーム数={1}, セクションあたり={2}フレーム, 必要セクション数={3}').format(frame_size, total_frames, frame_count, total_sections)}
                    <br>
                    {translate('動画モード {0} とフレームサイズ {1} で必要なセクション数: <strong>{2}</strong>').format(length, frame_size, total_sections)}
                    </div>"""

                    # デバッグ用ログ
                    print(translate("計算結果: モード=通常, フレームサイズ={0}, latent_window_size={1}, 総フレーム数={2}, 必要セクション数={3}").format(frame_size, latent_window_size, total_frames, total_sections))

                    return html

                # 初期化時にも計算を実行
                initial_html = update_section_calculation(frame_size_radio.value, mode_radio.value, length_radio.value)
                section_calc_display = gr.HTML(value=initial_html, label="")

                # フレームサイズ変更イベント - HTML表示の更新とセクションタイトルの更新を行う
                frame_size_radio.change(
                    fn=update_section_calculation,
                    inputs=[frame_size_radio, mode_radio, length_radio],
                    outputs=[section_calc_display]
                )

                # セクション表示機能をシンプル化
                def update_section_visibility(mode, length, frame_size=None):
                    """F1モードではシンプル化された関数"""
                    # 秒数だけ計算して返す
                    seconds = get_video_seconds(length)
                    print(translate("F1モード：シンプル設定（不要な機能を削除済み）"))

                    # 最低限の返値（入力に対応するだけの空更新）
                    return [gr.update()] * 2 + [] + [gr.update(value=seconds)] + []

                # 注意: この関数のイベント登録は、total_second_lengthのUIコンポーネント定義後に行うため、
                # ここでは関数の定義のみ行い、実際のイベント登録はUIコンポーネント定義後に行います。

                # 動画長変更イベントでもセクション数計算を更新
                length_radio.change(
                    fn=update_section_calculation,
                    inputs=[frame_size_radio, mode_radio, length_radio],
                    outputs=[section_calc_display]
                )

                # F1モードではセクションタイトルは不要

                # モード変更時にも計算を更新
                mode_radio.change(
                    fn=update_section_calculation,
                    inputs=[frame_size_radio, mode_radio, length_radio],
                    outputs=[section_calc_display]
                )

                # F1モードではセクションタイトルは不要

                # モード変更時の処理もtotal_second_lengthコンポーネント定義後に行います

                # 動画長変更時のセクション表示更新もtotal_second_lengthコンポーネント定義後に行います

                # F1モードでは終端フレームとループモード関連の機能をすべて削除

                # キーフレーム処理関数とZipファイルアップロード処理関数は未使用のため削除


        with gr.Column():
            result_video = gr.Video(
                label=translate("Finished Frames"),
                autoplay=True,
                show_share_button=False,
                height=512,
                loop=True,
                format="mp4",
                interactive=False,
            )
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)

            # フレームサイズ切替用のUIコントロールは上部に移動したため削除

            # 計算結果を表示するエリア
            section_calc_display = gr.HTML("", label="")

            use_teacache = gr.Checkbox(label=translate('Use TeaCache'), value=True, info=translate('Faster speed, but often makes hands and fingers slightly worse.'))

            # Use Random Seedの初期値
            use_random_seed_default = True
            seed_default = random.randint(0, 2**32 - 1) if use_random_seed_default else 1

            use_random_seed = gr.Checkbox(label=translate("Use Random Seed"), value=use_random_seed_default)

            n_prompt = gr.Textbox(label=translate("Negative Prompt"), value="", visible=False)  # Not used
            seed = gr.Number(label=translate("Seed"), value=seed_default, precision=0)

            # ここで、メタデータ取得処理の登録を移動する
            # ここでは、promptとseedの両方が定義済み
            input_image.change(
                fn=update_from_image_metadata,
                inputs=[input_image, copy_metadata],
                outputs=[prompt, seed]
            )

            # チェックボックスの変更時に再読み込みを行う
            def check_metadata_on_checkbox_change(copy_enabled, image_path):
                if not copy_enabled or image_path is None:
                    return [gr.update()] * 2
                # チェックボックスオン時に、画像があれば再度メタデータを読み込む
                return update_from_image_metadata(image_path, copy_enabled)

            # update_section_metadata_on_checkbox_change関数は未使用のため削除

            copy_metadata.change(
                fn=check_metadata_on_checkbox_change,
                inputs=[copy_metadata, input_image],
                outputs=[prompt, seed]
            )


            def set_random_seed(is_checked):
                if is_checked:
                    return random.randint(0, 2**32 - 1)
                else:
                    return gr.update()
            use_random_seed.change(fn=set_random_seed, inputs=use_random_seed, outputs=seed)

            total_second_length = gr.Slider(label=translate("Total Video Length (Seconds)"), minimum=1, maximum=120, value=1, step=1)
            latent_window_size = gr.Slider(label=translate("Latent Window Size"), minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
            steps = gr.Slider(label=translate("Steps"), minimum=1, maximum=100, value=25, step=1, info=translate('Changing this value is not recommended.'))

            cfg = gr.Slider(label=translate("CFG Scale"), minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
            gs = gr.Slider(label=translate("Distilled CFG Scale"), minimum=1.0, maximum=32.0, value=10.0, step=0.01, info=translate('Changing this value is not recommended.'))
            rs = gr.Slider(label=translate("CFG Re-Scale"), minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

            available_cuda_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3))
            default_gpu_memory_preservation_gb = 6 if available_cuda_memory_gb >= 20 else (8 if available_cuda_memory_gb > 16 else 10)
            gpu_memory_preservation = gr.Slider(label=translate("GPU Memory to Preserve (GB) (smaller = more VRAM usage)"), minimum=6, maximum=128, value=default_gpu_memory_preservation_gb, step=0.1, info=translate("空けておくGPUメモリ量を指定。小さい値=より多くのVRAMを使用可能=高速、大きい値=より少ないVRAMを使用=安全"))

            # MP4圧縮設定スライダーを追加
            mp4_crf = gr.Slider(label=translate("MP4 Compression"), minimum=0, maximum=100, value=16, step=1, info=translate("数値が小さいほど高品質になります。0は無圧縮。黒画面が出る場合は16に設定してください。"))

            # セクションごとの動画保存チェックボックスを追加（デフォルトOFF）
            keep_section_videos = gr.Checkbox(label=translate("完了時にセクションごとの動画を残す - チェックがない場合は最終動画のみ保存されます（デフォルトOFF）"), value=False)

            # テンソルデータ保存チェックボックス违加
            save_tensor_data = gr.Checkbox(
                label=translate("完了時にテンソルデータ(.safetensors)も保存 - このデータを別の動画の後に結合可能"),
                value=False,
                info=translate("チェックすると、生成されたテンソルデータを保存します。アップロードされたテンソルがあれば、結合したテンソルデータも保存されます。")
            )

            # セクションごとの静止画保存チェックボックスを追加（デフォルトOFF）
            save_section_frames = gr.Checkbox(label=translate("Save Section Frames"), value=False, info=translate("各セクションの最終フレームを静止画として保存します（デフォルトOFF）"))
            
            # フレーム画像保存のラジオボタンを追加（デフォルトは「保存しない」）
            gr.Markdown(translate("### フレーム画像保存設定"))
            frame_save_mode = gr.Radio(
                label=translate("フレーム画像保存モード"),
                choices=[
                    translate("保存しない"),
                    translate("全フレーム画像保存"),
                    translate("最終セクションのみ全フレーム画像保存")
                ],
                value=translate("保存しない"),
                info=translate("フレーム画像の保存方法を選択します。過去セクション分も含めて保存します。全セクションか最終セクションのみか選択できます。")
            )

            # UIコンポーネント定義後のイベント登録
            # F1モードではセクション機能を削除済み - シンプル化したイベントハンドラ
            mode_radio.change(
                fn=update_section_visibility,
                inputs=[mode_radio, length_radio, frame_size_radio],
                outputs=[input_image, input_image, total_second_length]
            )

            # フレームサイズ変更時の処理（シンプル化）
            frame_size_radio.change(
                fn=update_section_visibility,
                inputs=[mode_radio, length_radio, frame_size_radio],
                outputs=[input_image, input_image, total_second_length]
            )

            # 動画長変更時の処理（シンプル化）
            length_radio.change(
                fn=update_section_visibility,
                inputs=[mode_radio, length_radio, frame_size_radio],
                outputs=[input_image, input_image, total_second_length]
            )


            # Image影響度調整スライダー
            with gr.Group():
                gr.Markdown("### " + translate("Image影響度調整"))
                image_strength = gr.Slider(
                    label=translate("Image影響度"),
                    minimum=1.00,
                    maximum=1.02,
                    value=1.00,
                    step=0.001,
                    info=translate("開始フレーム(Image)が動画に与える影響の強さを調整します。1.00が通常の動作（100%）です。値を大きくすると始点の影響が強まり、変化が少なくなります。100%-102%の範囲で0.1%刻みの微調整が可能です。")
                )

            # 出力フォルダ設定
            gr.Markdown(translate("※ 出力先は `webui` 配下に限定されます"))
            with gr.Row(equal_height=True):
                with gr.Column(scale=4):
                    # フォルダ名だけを入力欄に設定
                    output_dir = gr.Textbox(
                        label=translate("出力フォルダ名"),
                        value=output_folder_name,  # 設定から読み込んだ値を使用
                        info=translate("動画やキーフレーム画像の保存先フォルダ名"),
                        placeholder="outputs"
                    )
                with gr.Column(scale=1, min_width=100):
                    open_folder_btn = gr.Button(value=translate("📂 保存および出力フォルダを開く"), size="sm")

            # 実際の出力パスを表示
            with gr.Row(visible=False):
                path_display = gr.Textbox(
                    label=translate("出力フォルダの完全パス"),
                    value=os.path.join(base_path, output_folder_name),
                    interactive=False
                )

            # フォルダを開くボタンのイベント
            def handle_open_folder_btn(folder_name):
                """フォルダ名を保存し、そのフォルダを開く"""
                if not folder_name or not folder_name.strip():
                    folder_name = "outputs"

                # フォルダパスを取得
                folder_path = get_output_folder_path(folder_name)

                # 設定を更新して保存
                settings = load_settings()
                old_folder_name = settings.get('output_folder')

                if old_folder_name != folder_name:
                    settings['output_folder'] = folder_name
                    save_result = save_settings(settings)
                    if save_result:
                        # グローバル変数も更新
                        global output_folder_name, outputs_folder
                        output_folder_name = folder_name
                        outputs_folder = folder_path
                    print(translate("出力フォルダ設定を保存しました: {folder_name}").format(folder_name=folder_name))

                # フォルダを開く
                open_output_folder(folder_path)

                # 出力ディレクトリ入力欄とパス表示を更新
                return gr.update(value=folder_name), gr.update(value=folder_path)

            open_folder_btn.click(fn=handle_open_folder_btn, inputs=[output_dir], outputs=[output_dir, path_display])

            # プロンプト管理パネル（右カラムから左カラムに移動済み）

    # 実行前のバリデーション関数
    def validate_and_process(*args):
        """入力画像または最後のキーフレーム画像のいずれかが有効かどうかを確認し、問題がなければ処理を実行する"""
        input_img = args[0]  # 入力の最初が入力画像
        # 注意: 引数のインデックスを正確に確認するためのデバッグ出力
        print(translate("[DEBUG] validate_and_process: 引数の総数: {0}").format(len(args)))
        print(translate("[DEBUG] validate_and_process: 引数20-26の値:"))
        for i in range(20, 27):
            if i < len(args):
                print(translate("[DEBUG] args[{0}] = {1} (型: {2})").format(i, args[i], type(args[i]).__name__))
        
        # UIのセットアップとips配列:
        # [0]input_image, [1]prompt, [2]n_prompt, [3]seed, [4]total_second_length, [5]latent_window_size, 
        # [6]steps, [7]cfg, [8]gs, [9]rs, [10]gpu_memory_preservation, [11]use_teacache, [12]use_random_seed, 
        # [13]mp4_crf, [14]all_padding_value, [15]image_strength, [16]frame_size_radio, [17]keep_section_videos, 
        # [18]lora_files, [19]lora_files2, [20]lora_scales_text, [21]output_dir, [22]save_section_frames, 
        # [23]use_all_padding, [24]use_lora, [25]save_tensor_data, [26]section_settings, [27]tensor_data_input, 
        # [28]fp8_optimization, [29]resolution, [30]batch_count
        
        # 各引数を明示的に取得
        output_dir = args[21] if len(args) > 21 else None
        save_section_frames = args[22] if len(args) > 22 else False
        use_all_padding = args[23] if len(args) > 23 else False
        use_lora = args[24] if len(args) > 24 else False
        save_tensor_data = args[25] if len(args) > 25 else False  # 正しいインデックスに修正
        section_settings = args[26] if len(args) > 26 else None   # 正しいインデックスに修正
        tensor_data_input = args[27] if len(args) > 27 else None  # 正しいインデックスに修正
        fp8_optimization = args[28] if len(args) > 28 else False
        resolution_value = args[29] if len(args) > 29 else 640
        batch_count = args[30] if len(args) > 30 else 1
        # 旧パラメータの代わりにフレーム保存モードを取得
        frame_save_mode = args[31] if len(args) > 31 else translate("保存しない")
        
        # デバッグ：フレーム保存モードの型と値を確認
        print(translate("[DEBUG] frame_save_modeの型: {0}, 値: {1}").format(type(frame_save_mode).__name__, frame_save_mode))
        
        # Gradioのラジオボタンオブジェクトが直接渡されているか、文字列値が渡されているかを確認
        if hasattr(frame_save_mode, 'value'):
            # Gradioオブジェクトの場合は値を取得
            frame_save_mode_value = frame_save_mode.value
            print(translate("[DEBUG] Gradioオブジェクトから値を取得: {0}").format(frame_save_mode_value))
        else:
            # 文字列などの通常の値の場合はそのまま使用
            frame_save_mode_value = frame_save_mode
            print(translate("[DEBUG] 通常の値として使用: {0}").format(frame_save_mode_value))
        
        # モード選択に基づいてフラグを設定
        # 選択肢は3つ: "保存しない", "全フレーム画像保存", "最終セクションのみ全フレーム画像保存"
        save_latent_frames = False  # 最初にFalseに設定
        save_last_section_frames = False  # 最初にFalseに設定
        
        if frame_save_mode_value == translate("全フレーム画像保存"):
            save_latent_frames = True
        elif frame_save_mode_value == translate("最終セクションのみ全フレーム画像保存"):
            save_last_section_frames = True
            
        print(translate("[DEBUG] 設定済みフラグ - save_latent_frames: {0}, save_last_section_frames: {1}").format(save_latent_frames, save_last_section_frames))
        
        print(translate("[DEBUG] 重要な引数の値:"))
        print(translate("[DEBUG] save_tensor_data = {0} (型: {1})").format(save_tensor_data, type(save_tensor_data).__name__))
        print(translate("[DEBUG] use_lora = {0} (型: {1})").format(use_lora, type(use_lora).__name__))
        print(translate("[DEBUG] resolution = {0}").format(resolution_value))

        # バッチ回数を有効な範囲に制限
        batch_count = max(1, min(int(batch_count), 100))

        # F1モードでは簡易化
        section_settings = [[None, None, ""] for _ in range(50)]

        # 現在の動画長設定とフレームサイズ設定を渡す
        is_valid, error_message = validate_images(input_img, section_settings, length_radio, frame_size_radio)

        if not is_valid:
            # 画像が無い場合はエラーメッセージを表示して終了
            yield None, gr.update(visible=False), translate("エラー: 画像が選択されていません"), error_message, gr.update(interactive=True), gr.update(interactive=False), gr.update()
            return

        # 画像がある場合は通常の処理を実行
        # 元のパラメータを使用
        new_args = list(args)
        
        # 引数を正しいインデックスで設定
        if len(new_args) > 25:
            new_args[25] = save_tensor_data  # save_tensor_dataを確実に正しい値に
        if len(new_args) > 26:
            new_args[26] = section_settings  # section_settingsを更新
        
        # その他の引数も必要に応じて設定
        if len(new_args) <= 32:
            # 不足している場合は追加
            if len(new_args) <= 31:
                if len(new_args) <= 30:
                    if len(new_args) <= 29:
                        # resolutionもない場合
                        new_args.append(resolution_value)  # resolutionを追加
                    new_args.append(batch_count)  # batch_countを追加
                new_args.append(save_latent_frames)  # save_latent_framesを追加
            new_args.append(save_last_section_frames)  # save_last_section_framesを追加
        else:
            # 既に存在する場合は更新
            new_args[29] = resolution_value  # resolution
            new_args[30] = batch_count  # batch_count
            new_args[31] = save_latent_frames  # save_latent_frames
            new_args[32] = save_last_section_frames  # save_last_section_frames

        # process関数に渡す前に重要な値を確認
        print(translate("[DEBUG] process関数への引数: save_tensor_data = {0}").format(new_args[25] if len(new_args) > 25 else "未設定"))
        print(translate("[DEBUG] フレーム保存モード = {0}").format(frame_save_mode))
        print(translate("[DEBUG] save_latent_frames = {0}").format(save_latent_frames))
        print(translate("[DEBUG] save_last_section_frames = {0}").format(save_last_section_frames))
        
        # process関数のジェネレータを返す
        yield from process(*new_args)

    # 実行ボタンのイベント
    # ips配列の引数の順序と、validate_and_processで実際に処理する引数の順序を正確に一致させる
    # コメントの最新版と引数の順序が一致していることを確認
    #  [0]input_image, [1]prompt, [2]n_prompt, [3]seed, [4]total_second_length, [5]latent_window_size, 
    #  [6]steps, [7]cfg, [8]gs, [9]rs, [10]gpu_memory_preservation, [11]use_teacache, [12]use_random_seed, 
    #  [13]mp4_crf, [14]all_padding_value, [15]image_strength, [16]frame_size_radio, [17]keep_section_videos, 
    #  [18]lora_files, [19]lora_files2, [20]lora_scales_text, [21]output_dir, [22]save_section_frames, 
    #  [23]use_all_padding, [24]use_lora, [25]save_tensor_data, [26]section_settings, [27]tensor_data_input, 
    #  [28]fp8_optimization, [29]resolution, [30]batch_count
    ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, use_random_seed, mp4_crf, all_padding_value, image_strength, frame_size_radio, keep_section_videos, lora_files, lora_files2, lora_scales_text, output_dir, save_section_frames, use_all_padding, use_lora, save_tensor_data, section_settings, tensor_data_input, fp8_optimization, resolution, batch_count, frame_save_mode]

    start_button.click(fn=validate_and_process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button, seed])
    end_button.click(fn=end_process, outputs=[end_button])

    # F1モードではセクション機能とキーフレームコピー機能を削除済み

    # 注: create_single_keyframe_handler関数はフレームサイズや動画長に基づいた動的セクション数を計算します
    # UIでフレームサイズや動画長を変更すると、動的に計算されたセクション数に従ってコピー処理が行われます

    # プリセット保存ボタンのイベント
    def save_button_click_handler(name, prompt_text):
        """保存ボタンクリック時のハンドラ関数"""

        # 重複チェックと正規化
        if "A character" in prompt_text and prompt_text.count("A character") > 1:
            sentences = prompt_text.split(".")
            if len(sentences) > 0:
                prompt_text = sentences[0].strip() + "."
                # 重複を検出したため正規化

        # プリセット保存
        result_msg = save_preset(name, prompt_text)

        # プリセットデータを取得してドロップダウンを更新
        presets_data = load_presets()
        choices = [preset["name"] for preset in presets_data["presets"]]
        default_presets = [n for n in choices if any(p["name"] == n and p.get("is_default", False) for p in presets_data["presets"])]
        user_presets = [n for n in choices if n not in default_presets]
        sorted_choices = [(n, n) for n in sorted(default_presets) + sorted(user_presets)]

        # メインプロンプトは更新しない（保存のみを行う）
        return result_msg, gr.update(choices=sorted_choices), gr.update()

    # 保存ボタンのクリックイベントを接続
    save_btn.click(
        fn=save_button_click_handler,
        inputs=[edit_name, edit_prompt],
        outputs=[result_message, preset_dropdown, prompt]
    )

    # クリアボタン処理
    def clear_fields():
        return gr.update(value=""), gr.update(value="")

    clear_btn.click(
        fn=clear_fields,
        inputs=[],
        outputs=[edit_name, edit_prompt]
    )

    # プリセット読込処理
    def load_preset_handler(preset_name):
        # プリセット選択時に編集欄のみを更新
        for preset in load_presets()["presets"]:
            if preset["name"] == preset_name:
                return gr.update(value=preset_name), gr.update(value=preset["prompt"])
        return gr.update(), gr.update()

    # プリセット選択時に編集欄に反映
    def load_preset_handler_wrapper(preset_name):
        # プリセット名がタプルの場合も処理する
        if isinstance(preset_name, tuple) and len(preset_name) == 2:
            preset_name = preset_name[1]  # 値部分を取得
        return load_preset_handler(preset_name)

    preset_dropdown.change(
        fn=load_preset_handler_wrapper,
        inputs=[preset_dropdown],
        outputs=[edit_name, edit_prompt]
    )

    # 反映ボタン処理 - 編集画面の内容をメインプロンプトに反映
    def apply_to_prompt(edit_text):
        """編集画面の内容をメインプロンプトに反映する関数"""
        # 編集画面のプロンプトをメインに適用
        return gr.update(value=edit_text)

    # プリセット削除処理
    def delete_preset_handler(preset_name):
        # プリセット名がタプルの場合も処理する
        if isinstance(preset_name, tuple) and len(preset_name) == 2:
            preset_name = preset_name[1]  # 値部分を取得

        result = delete_preset(preset_name)

        # プリセットデータを取得してドロップダウンを更新
        presets_data = load_presets()
        choices = [preset["name"] for preset in presets_data["presets"]]
        default_presets = [name for name in choices if any(p["name"] == name and p.get("is_default", False) for p in presets_data["presets"])]
        user_presets = [name for name in choices if name not in default_presets]
        sorted_names = sorted(default_presets) + sorted(user_presets)
        updated_choices = [(name, name) for name in sorted_names]

        return result, gr.update(choices=updated_choices)

    apply_preset_btn.click(
        fn=apply_to_prompt,
        inputs=[edit_prompt],
        outputs=[prompt]
    )

    delete_preset_btn.click(
        fn=delete_preset_handler,
        inputs=[preset_dropdown],
        outputs=[result_message, preset_dropdown]
    )

# F1モードではキーフレームコピー機能を削除済み

allowed_paths = [os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './outputs')))]

# 起動コード
try:
    block.launch(
        server_name=args.server,
        server_port=args.port,
        share=args.share,
        allowed_paths=allowed_paths,
        inbrowser=args.inbrowser,
    )
except OSError as e:
    if "Cannot find empty port" in str(e):
        print("\n======================================================")
        print(translate("エラー: FramePack-eichiは既に起動しています。"))
        print(translate("同時に複数のインスタンスを実行することはできません。"))
        print(translate("現在実行中のアプリケーションを先に終了してください。"))
        print("======================================================\n")
        input(translate("続行するには何かキーを押してください..."))
    else:
        # その他のOSErrorの場合は元のエラーを表示
        print(translate("\nエラーが発生しました: {e}").format(e=e))
        input(translate("続行するには何かキーを押してください..."))
