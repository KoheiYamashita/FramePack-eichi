"""
FramePack-eichi DynamicSwap対応LoRAモジュール
DynamicSwapメモリ管理システムと互換性のあるLoRA適用機能を提供します。
DynamicSwap処理による効率的なメモリ管理を実装し、緩やかなマッチングも対応。
"""

import os
import torch
import logging
import traceback
from typing import Dict, List, Optional, Union, Tuple, Any

from locales import i18n

# ロギング設定
logger = logging.getLogger("dynamic_swap_lora")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class DynamicSwapLoRAManager:
    """
    DynamicSwapと互換性のあるLoRA管理クラス
    DynamicSwapのシステムを使用してメモリ効率良くLoRAを適用します
    緩やかなマッチングによりキーの不一致にも対応
    """

    def __init__(self):
        # LoRAの状態を保持
        self.lora_state_dict = None
        # 適用スケール
        self.scale = 0.0
        # 適用したレイヤーの記録
        self.applied_layers = set()
        # LoRAの適用有無フラグ
        self.is_active = False
        # 元の状態保存用
        self.original_states = {}
        # キーマッピング用のキャッシュ
        self.key_mapping_cache = {}
        # 適用する最大パラメータ数
        self.max_parameters = 100

    def load_lora(self, lora_path: str, is_diffusers: bool = False) -> None:
        """
        LoRAファイルを読み込む

        Args:
            lora_path: LoRAファイルへのパス
            is_diffusers: Diffusersフォーマットかどうか
        """
        # LoRAの読み込み
        try:
            from .lora_loader import load_lora_weights, detect_lora_format
            from .lora_loader import convert_diffusers_lora_to_hunyuan, check_for_musubi

            lora_state_dict = load_lora_weights(lora_path)

            # フォーマット変換（必要な場合）
            format_type = detect_lora_format(lora_state_dict)
            logger.info(i18n.translate("検出されたLoRAフォーマット: {format_type}").format(format_type=format_type))

            if format_type == 'diffusers' or is_diffusers:
                lora_state_dict = convert_diffusers_lora_to_hunyuan(lora_state_dict)
            elif format_type == 'musubi':
                lora_state_dict = check_for_musubi(lora_state_dict)

            logger.info(i18n.translate("読み込み完了: {len(lora_state_dict)} パラメータ").format(len(lora_state_dict)))

            # 最終的な状態設定
            self.lora_state_dict = lora_state_dict
            # 適用記録をクリア
            self.applied_layers.clear()
            self.is_active = True
            # キーマッピングキャッシュもクリア
            self.key_mapping_cache = {}

        except Exception as e:
            logger.error(i18n.translate("LoRA読み込みエラー: {e}").format(e=e))
            logger.error(traceback.format_exc())
            self.lora_state_dict = None
            self.is_active = False
            raise

    def set_scale(self, scale: float) -> None:
        """
        LoRA適用スケールを設定

        Args:
            scale: 適用スケール (0.0-1.0)
        """
        self.scale = max(0.0, min(1.0, scale))
        logger.info(i18n.translate("LoRA適用スケールを設定: {scale}").format(scale=self.scale))

    def install_hooks(self, model: torch.nn.Module, test_apply: bool = True) -> None:
        """
        DynamicSwapをインストールして動的にLoRAを適用

        Args:
            model: DynamicSwapが設定されたモデル
            test_apply: 試験的に適用してパラメータ数を診断するかどうか
        """
        if not self.is_active or self.lora_state_dict is None:
            logger.info(i18n.translate("LoRAが読み込まれていないか非アクティブです。フック設定をスキップします。"))
            return

        # モデルの元の状態を保存
        self.save_original_states(model)
        logger.info(i18n.translate("元の状態を保存: {len(self.original_states)} パラメータ").format(len(self.original_states)))

        # 読み込み前フックを定義（LoRA適用）
        def pre_load_hook(module, hook_name):
            if not self.is_active:
                return

            logger.debug(i18n.translate("DynamicSwap呼び出し: {hook_name}").format(hook_name=hook_name))
            self.apply_to_layer(hook_name, module)

        # アンロード後フックを定義（元の状態に戻す）
        def post_unload_hook(module, hook_name):
            if not self.is_active:
                return

            if hook_name in self.applied_layers:
                logger.debug(i18n.translate("DynamicSwapアンロード後呼び出し: {hook_name}").format(hook_name=hook_name))
                self.remove_from_layer(hook_name, module)

        # DynamicSwapの内部状態を取得
        for name, module in model.named_modules():
            if hasattr(module, '_hook_register_handle_pre_load'):
                # 既存のフックに加えて、LoRAフックを登録
                old_pre_load = module._hook_register_handle_pre_load

                def combined_pre_load(mod, name=name):  # 名前を固定するためにデフォルト引数を使用
                    old_pre_load(mod)
                    pre_load_hook(mod, name)

                module._hook_register_handle_pre_load = combined_pre_load

            if hasattr(module, '_hook_register_handle_post_unload'):
                # 既存のフックに加えて、LoRAフックを登録
                old_post_unload = module._hook_register_handle_post_unload

                def combined_post_unload(mod, name=name):  # 名前を固定するためにデフォルト引数を使用
                    post_unload_hook(mod, name)
                    old_post_unload(mod)

                module._hook_register_handle_post_unload = combined_post_unload

        # _lora_appliedフラグを設定（診断用）
        model._lora_applied = True
        model._lora_source = "dynamic_swap_hooks"

        logger.info(i18n.translate("DynamicSwapを設定しました"))

        # 試験的に適用してパラメータ数を診断
        if test_apply:
            total_applicable_params = self.diagnose_applicable_parameters(model)
            logger.info(i18n.translate("適用可能なパラメータ数: {total_applicable_params}").format(total_applicable_params=total_applicable_params))

    def diagnose_applicable_parameters(self, model: torch.nn.Module) -> int:
        """
        モデルにLoRAが適用可能なパラメータ数を診断
        最大数に制限された実際の診断結果を表示

        Args:
            model: 診断対象のモデル

        Returns:
            int: 適用可能なパラメータ数
        """
        try:
            # 診断用の一時的なカウンター
            applicable_params = 0
            applicable_modules = 0
            applicable_layers = set()
            total_potential_params = 0

            # 実際に適用されるパラメータ数をカウント
            applicable_params = len(self.original_states.keys())

            # 表示用に全体的な非適用パラメータ数もカウント
            for name, module in model.named_modules():
                if len(list(module.parameters(recurse=False))) > 0:
                    # パラメータを持つモジュールのみ処理
                    for param_name, param in module.named_parameters(recurse=False):
                        full_param_name = f"{name}.{param_name}"

                        # LoRAキーの検索（緩やかなマッチングを含む）
                        lora_down_key, lora_up_key = self.find_matching_lora_keys(full_param_name)

                        if lora_down_key and lora_up_key:
                            total_potential_params += 1
                            if name in self.original_states and param_name in self.original_states[name]:
                                applicable_layers.add(name)

            # モジュール数をカウント
            applicable_modules = len(self.original_states)

            logger.info(i18n.translate("診断結果: {applicable_modules}モジュール中の{applicable_params}パラメータにLoRA適用").format(applicable_modules=applicable_modules, applicable_params=applicable_params))
            logger.info(i18n.translate("全体の適用可能パラメータ数: {total_potential_params} (制限数: {self.max_parameters})").format(total_potential_params=total_potential_params, self=self))

            if applicable_layers:
                logger.info(i18n.translate("適用層の例: {applicable_layers}...").format(applicable_layers=list(applicable_layers)[:3]))

            return applicable_params

        except Exception as e:
            logger.error(i18n.translate("診断中にエラーが発生: {e}").format(e=e))
            logger.error(traceback.format_exc())
            return 0

    def save_original_states(self, model: torch.nn.Module) -> None:
        """
        モデルの元の状態を保存（緩やかなマッチングを使用）
        最大パラメータ数に制限

        Args:
            model: モデル
        """
        self.original_states = {}
        saved_params = 0
        all_matches = []

        # 最初に全ての有望なマッチを探す
        with torch.no_grad():
            for name, module in model.named_modules():
                if len(list(module.parameters(recurse=False))) > 0:
                    # パラメータを持つモジュールのみ処理
                    for param_name, param in module.named_parameters(recurse=False):
                        full_param_name = f"{name}.{param_name}"

                        # LoRAキーの検索（緩やかなマッチングを含む）
                        lora_down_key, lora_up_key = self.find_matching_lora_keys(full_param_name)

                        if lora_down_key and lora_up_key:
                            # 優先度スコアを計算
                            priority_score = 0

                            # 重要なキーワードを含むパラメータを優先
                            high_priority_keywords = ["attention", "attn", "transformer", "blocks.0", "blocks.1"]
                            for keyword in high_priority_keywords:
                                if keyword in full_param_name:
                                    priority_score += 10

                            # レイヤー0と1は特に重要（最初の層）
                            if any(f".{i}." in full_param_name for i in range(2)):
                                priority_score += 5

                            # q, k, v関連は重要
                            if any(key in full_param_name for key in ["q_proj", "k_proj", "v_proj", "qkv"]):
                                priority_score += 3

                            all_matches.append((name, param_name, param, priority_score, full_param_name))

        # 優先度でソート
        all_matches.sort(key=lambda x: x[3], reverse=True)

        # 最大パラメータ数を拡大（1.5倍に増加 - メモリ改善）
        self.max_parameters = max(self.max_parameters, min(1500, len(all_matches)))  # 最大1500パラメータまで自動拡張

        # 最大パラメータ数に制限
        selected_matches = all_matches[:self.max_parameters]

        # 選択されたパラメータのみ保存
        for name, param_name, param, score, full_param_name in selected_matches:
            # モジュールのストレージ初期化
            if name not in self.original_states:
                self.original_states[name] = {}

            # CPUメモリに保存してGPUメモリを節約
            self.original_states[name][param_name] = param.data.detach().cpu().clone()
            saved_params += 1
            # デバッグレベルに変更し、ログの量を減らす
            logger.debug(f"パラメータ保存: {full_param_name} (スコア: {score})")

            # 100件ごとにメモリ状態をログ出力
            if saved_params % 100 == 0 and torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated()/1024**3
                logger.info(f"保存進捗: {saved_params}/{len(selected_matches)} - メモリ使用: {memory_allocated:.2f}GB")

        logger.info(f"元の状態を保存: {saved_params}パラメータ（最大{self.max_parameters}個まで処理）")

        # 明示的なメモリクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    def find_matching_lora_keys(self, param_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        パラメータ名に対応するLoRAキーを検索（より激しい緩和マッチングを含む）

        Args:
            param_name: パラメータ名

        Returns:
            Tuple[Optional[str], Optional[str]]: LoRA downキーとLoRA upキー
        """
        # キャッシュをチェック
        if param_name in self.key_mapping_cache:
            return self.key_mapping_cache[param_name]

        # LoRAキーリスト
        lora_keys = list(self.lora_state_dict.keys())
        down_keys = [k for k in lora_keys if "lora_down" in k or "lora_A" in k]
        up_keys = [k for k in lora_keys if "lora_up" in k or "lora_B" in k]

        # すべてのLoRAキーをログに出力（初回のみ）
        if not hasattr(self, "_logged_keys"):
            self._logged_keys = True
            logger.info(i18n.translate("LoRAキーの例（最在5個）: {down_keys[:5]}").format(down_keys=down_keys))

        # 完全一致を最初に試す（最も信頼性が高い）
        lora_down_key = f"{param_name}.lora_down"
        lora_up_key = f"{param_name}.lora_up"

        if lora_down_key in lora_keys and lora_up_key in lora_keys:
            self.key_mapping_cache[param_name] = (lora_down_key, lora_up_key)
            return lora_down_key, lora_up_key

        # 複数の緩和マッチングアプローチを試す

        # 1. パスマッピング変換テーブル
        path_mappings = {
            "diffusion_model": "",  # 接頭辞を除去
            "transformer": "",      # 一般的な接頭辞
            "blocks": "transformer_blocks",  # 一般的なブロック名
            "attn": "attention",    # 注意機構の名称の違い
            "transformer_blocks": "blocks",  # 逆方向のマッピング
            "attention": "attn",    # 逆方向のマッピング
            "_": ".",               # 区切り文字の変換
            ".": "_"                # 逆方向の区切り文字変換
        }

        # シンプルなキー抽出（最後の部分だけに注目）
        param_parts = param_name.split(".")
        simple_name = param_parts[-1] if len(param_parts) > 0 else param_name

        # 2. パスとパラメータ名の変換を試す
        for key_prefix in ["", "diffusion_model.", "transformer.", "model."]:
            # パスマッピングを適用した変換
            converted_param = param_name
            for old, new in path_mappings.items():
                converted_param = converted_param.replace(old, new)

            # 各種キー形式を試す
            test_keys = [
                f"{key_prefix}{converted_param}.lora_down",
                f"{key_prefix}{converted_param}.lora_up",
                f"{key_prefix}{param_name}.lora_down",
                f"{key_prefix}{param_name}.lora_up",
                f"{key_prefix}{simple_name}.lora_down",
                f"{key_prefix}{simple_name}.lora_up"
            ]

            # downキーを探す
            found_down_key = None
            for test_key in test_keys:
                if test_key in down_keys:
                    found_down_key = test_key
                    break

            # 対応するupキーを探す
            if found_down_key:
                # downキーからupキーを生成
                base_key = found_down_key.replace(".lora_down", "")
                found_up_key = f"{base_key}.lora_up"

                if found_up_key in up_keys:
                    self.key_mapping_cache[param_name] = (found_down_key, found_up_key)
                    return found_down_key, found_up_key

        # 3. パターンマッチングで重要な要素を抽出
        important_keywords = ["block", "transformer", "attn", "mlp", "layer", "norm", "conv", "linear", "proj", "q", "k", "v"]
        important_parts = []
        digit_parts = []

        # 重要なキーワードと数字を抽出
        for part in param_name.split("."):
            for keyword in important_keywords:
                if keyword in part.lower():
                    important_parts.append(part)
                    break

            # 数字（レイヤー番号など）も重要
            if any(c.isdigit() for c in part):
                # 数字部分だけを抽出
                digits = "".join([c for c in part if c.isdigit()])
                digit_parts.append(digits)

        # 重要な要素がない場合はパラメータ名自体を使用
        if not important_parts:
            important_parts = [simple_name]

        # パターンマッチングで適合度を計算
        best_match = None
        best_score = 0

        for down_key in down_keys:
            score = 0
            # 重要キーワードのマッチをスコアに加算
            for part in important_parts:
                if part in down_key:
                    score += 2  # 重要キーワードのマッチを優先

            # 数字のマッチも重要
            for digit in digit_parts:
                if digit in down_key:
                    score += 3  # 数字のマッチはさらに重要

            # パラメータタイプの一致もチェック
            if simple_name in down_key:
                score += 4  # パラメータ名の一致は最も重要

            if score > best_score:
                # 対応するup_keyを確認
                base_key = down_key.replace(".lora_down", "")
                up_key = f"{base_key}.lora_up"

                if up_key in up_keys:
                    best_score = score
                    best_match = (down_key, up_key)

        # 最後の手段: 最スコアの高いマッチを返す
        if best_match and best_score >= 2:  # 最低スコアのしきい値
            self.key_mapping_cache[param_name] = best_match
            # スコアが高いマッチをログに出力
            if best_score >= 4:  # 良いマッチを見つけた場合はログに出力
                logger.info(i18n.translate("良いマッチを発見: {param_name} -> {best_match[0]} (スコア: {best_score})").format(param_name=param_name, best_match=best_match, best_score=best_score))
            return best_match

        # 最後の手段: 単純なリストベースのマッピング
        # 最初の20個のキーに対して順番に割り当て
        if not hasattr(self, "_list_mapping"):
            self._list_mapping = {}
            # モデルのパラメータを取得
            model_params = set()

            # 先頭の20組のキーペアを置き換え用に設定
            for i in range(min(20, len(down_keys))):
                if i < len(down_keys) and i < len(up_keys):
                    self._list_mapping[f"param_{i}"] = (down_keys[i], up_keys[i])

        # モデルパラメータキーをインデックスに変換
        param_index = hash(param_name) % min(20, len(down_keys))
        param_key = f"param_{param_index}"

        if param_key in self._list_mapping:
            self.key_mapping_cache[param_name] = self._list_mapping[param_key]
            return self._list_mapping[param_key]

        # マッチするものがない
        self.key_mapping_cache[param_name] = (None, None)
        return None, None

    def apply_to_layer(self, layer_name: str, layer: torch.nn.Module) -> torch.nn.Module:
        """
        単一レイヤーにLoRAを適用

        Args:
            layer_name: レイヤー名
            layer: レイヤーモジュール

        Returns:
            torch.nn.Module: LoRAが適用されたレイヤー
        """
        if not self.is_active or self.lora_state_dict is None:
            return layer

        if layer_name in self.applied_layers:
            logger.debug(i18n.translate("レイヤーは既にLoRA適用済み: {layer_name}").format(layer_name=layer_name))
            return layer

        try:
            # メモリ使用状況を記録（改良）
            if torch.cuda.is_available() and layer_name.endswith("0"):  # 主要レイヤーのみログ出力
                memory_allocated = torch.cuda.memory_allocated()/1024**3
                logger.info(f"レイヤー適用前メモリ: {layer_name}, 使用中={memory_allocated:.2f}GB")

            applied_count = 0
            with torch.no_grad():
                # 適用予定パラメータを参照
                orig_names = set()
                if layer_name in self.original_states:
                    orig_names = set(self.original_states[layer_name].keys())

                # レイヤーのパラメータを取得
                for param_name, param in layer.named_parameters(recurse=False):
                    # 保存されているパラメータのみ処理（高速化）
                    if param_name not in orig_names:
                        continue

                    full_param_name = f"{layer_name}.{param_name}"

                    # LoRAキーの検索（緩やかなマッチングを含む）
                    lora_down_key, lora_up_key = self.find_matching_lora_keys(full_param_name)

                    if lora_down_key and lora_up_key:
                        try:
                            # LoRAの重みを取得
                            lora_down = self.lora_state_dict[lora_down_key].to(param.device)
                            lora_up = self.lora_state_dict[lora_up_key].to(param.device)

                            # LoRAの演算
                            delta = torch.matmul(lora_up, lora_down) * self.scale

                            # 不要になった中間テンソルを明示的に削除
                            del lora_down, lora_up

                            # 形状を確認して調整
                            if delta.shape != param.shape:
                                logger.warning(f"形状不一致: {delta.shape} != {param.shape}, スキップ: {full_param_name}")
                                continue

                            # 元のパラメータに適用
                            param.data += delta
                            applied_count += 1
                            logger.debug(f"LoRA適用: {full_param_name}")

                            # deltaも不要になったので削除
                            del delta

                        except Exception as param_error:
                            logger.warning(f"パラメータ適用エラー: {full_param_name}, {param_error}")

            # 適用したパラメータがあれば記録
            if applied_count > 0:
                self.applied_layers.add(layer_name)
                logger.info(f"レイヤー「{layer_name}」に{applied_count}個のパラメータを適用")

            # メモリ使用状況の記録（適用後）
            if torch.cuda.is_available() and (applied_count > 0 or layer_name.endswith("0")):
                # 昕やかなメモリ使用状況をデバッグ用に記録
                torch.cuda.synchronize()  # 特に重要なレイヤーの場合は同期する
                memory_allocated = torch.cuda.memory_allocated()/1024**3
                logger.debug(f"レイヤー適用後メモリ: {layer_name}, 使用中={memory_allocated:.2f}GB")

            return layer

        except Exception as e:
            logger.error(i18n.translate("レイヤーへのLoRA適用エラー: {layer_name}, {e}").format(layer_name=layer_name, e=e))
            logger.error(traceback.format_exc())
            return layer

    def remove_from_layer(self, layer_name: str, layer: torch.nn.Module) -> torch.nn.Module:
        """
        レイヤーからLoRAの効果を削除（元の状態に戻す）

        Args:
            layer_name: レイヤー名
            layer: レイヤーモジュール

        Returns:
            torch.nn.Module: 元の状態に戻されたレイヤー
        """
        if layer_name not in self.applied_layers:
            return layer

        try:
            with torch.no_grad():
                # レイヤーのパラメータを元に戻す
                if layer_name in self.original_states:
                    for param_name, original_value in self.original_states[layer_name].items():
                        if hasattr(layer, param_name):
                            param = getattr(layer, param_name)
                            if isinstance(param, torch.nn.Parameter):
                                # 保存された元の状態がCPUにある場合は、デバイスを合わせる
                                if original_value.device != param.device:
                                    original_value = original_value.to(param.device)

                                # 現在のパラメータを元の状態に戻す
                                param.data.copy_(original_value)
                                logger.debug(i18n.translate("LoRA効果を削除: {layer_name}.{param_name}").format(layer_name=layer_name, param_name=param_name))

            # 適用記録を更新
            self.applied_layers.remove(layer_name)
            return layer

        except Exception as e:
            logger.error(i18n.translate("レイヤーからのLoRA削除エラー: {layer_name}, {e}").format(layer_name=layer_name, e=e))
            logger.error(traceback.format_exc())
            return layer

    def reset(self) -> None:
        """
        LoRA状態をリセット
        """
        self.lora_state_dict = None
        self.scale = 0.0
        self.applied_layers.clear()
        self.original_states = {}
        self.key_mapping_cache = {}
        self.is_active = False
        logger.info(i18n.translate("LoRA状態をリセットしました"))
