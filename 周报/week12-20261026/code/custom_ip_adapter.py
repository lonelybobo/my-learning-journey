from transformers import AutoModel, AutoImageProcessor
from diffusers.loaders import IPAdapterMixin

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from huggingface_hub.utils import validate_hf_hub_args
from safetensors import safe_open

from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT
from diffusers.utils import (
    _get_model_file,
    is_accelerate_available,
    is_torch_version,
    is_transformers_available,
    logging,
)

logger = logging.get_logger(__name__)


class CustomIPAdapterMixin():
    """
    自定义 IP-Adapter 加载 Mixin。
    与官方 IPAdapterMixin 的核心区别：图像编码器使用 DINOv2-giant 而非 CLIP Vision。
    需要被混入到 DiffusionPipeline 子类中使用。
    """

    @validate_hf_hub_args
    def costum_load_ip_adapter(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, List[str], Dict[str, torch.Tensor]],
        subfolder: Union[str, List[str]],
        weight_name: Union[str, List[str]],
        image_encoder_folder: Optional[str] = "image_encoder",
        **kwargs,
    ):
        """
        加载一个或多个 IP-Adapter 权重到 pipeline 的 UNet 上。

        参数:
            pretrained_model_name_or_path_or_dict:
                模型路径（HF Hub 或本地）或已加载的 state dict。
                支持传入列表以加载多个 IP-Adapter。
            subfolder:
                权重文件所在的子目录，支持列表。
            weight_name:
                权重文件名（如 ip-adapter.bin / .safetensors），支持列表。
            image_encoder_folder:
                图像编码器子目录名；为 None 时不加载图像编码器。
            **kwargs:
                透传给底层的缓存/下载/加载相关参数。
        """

        # ---- 1. 参数规范化：统一转成列表，便于支持多 IP-Adapter ----

        # weight_name 不是列表则包装成单元素列表
        if not isinstance(weight_name, list):
            weight_name = [weight_name]

        # 模型路径/字典不是列表则包装成单元素列表
        if not isinstance(pretrained_model_name_or_path_or_dict, list):
            pretrained_model_name_or_path_or_dict = [pretrained_model_name_or_path_or_dict]
        # 若只给了一个路径但配了多个权重名，则广播复制路径
        if len(pretrained_model_name_or_path_or_dict) == 1:
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict * len(weight_name)

        # subfolder 同理，统一成列表并按需广播
        if not isinstance(subfolder, list):
            subfolder = [subfolder]
        if len(subfolder) == 1:
            subfolder = subfolder * len(weight_name)

        # 三者长度必须一致，否则无法一一对应
        if len(weight_name) != len(pretrained_model_name_or_path_or_dict):
            raise ValueError("`weight_name` and `pretrained_model_name_or_path_or_dict` must have the same length.")

        if len(weight_name) != len(subfolder):
            raise ValueError("`weight_name` and `subfolder` must have the same length.")

        # ---- 2. 解析与下载/加载相关的 kwargs ----

        cache_dir = kwargs.pop("cache_dir", None)                # 缓存目录
        force_download = kwargs.pop("force_download", False)     # 强制重新下载
        proxies = kwargs.pop("proxies", None)                    # 代理
        local_files_only = kwargs.pop("local_files_only", None)  # 仅使用本地文件
        token = kwargs.pop("token", None)                        # HF token
        revision = kwargs.pop("revision", None)                  # 版本/分支
        # 是否低内存加载（默认值来自 diffusers 内部常量）
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)

        # 低内存加载依赖 accelerate，未安装时降级并警告
        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            logger.warning(
                "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                " install accelerate\n```\n."
            )

        # 低内存加载要求 torch >= 1.9.0
        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        # 用于 HF Hub 下载时的 user-agent 标识
        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        # ---- 3. 逐个加载 IP-Adapter 权重 ----

        state_dicts = []  # 收集所有 IP-Adapter 的 state dict
        for pretrained_model_name_or_path_or_dict, weight_name, subfolder in zip(
            pretrained_model_name_or_path_or_dict, weight_name, subfolder
        ):
            # 若传入的是路径字符串（而非现成字典），则先定位/下载权重文件
            if not isinstance(pretrained_model_name_or_path_or_dict, dict):
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                # safetensors 格式：按前缀拆分为 image_proj 与 ip_adapter 两部分
                if weight_name.endswith(".safetensors"):
                    state_dict = {"image_proj": {}, "ip_adapter": {}}
                    with safe_open(model_file, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            if key.startswith("image_proj."):
                                # 去掉前缀，存入 image_proj 子字典
                                state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                            elif key.startswith("ip_adapter."):
                                # 去掉前缀，存入 ip_adapter 子字典
                                state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
                else:
                    # 其他格式（如 .bin），直接整体加载到 CPU
                    state_dict = torch.load(model_file, map_location="cpu")
            else:
                # 已经是字典，直接使用
                state_dict = pretrained_model_name_or_path_or_dict

            # 校验 state_dict 结构必须恰好是这两部分
            keys = list(state_dict.keys())
            if keys != ["image_proj", "ip_adapter"]:
                raise ValueError("Required keys are (`image_proj` and `ip_adapter`) missing from the state dict.")

            state_dicts.append(state_dict)

            # ---- 4. 加载图像编码器（定制点：使用 DINOv2-giant） ----
            # 仅在 pipeline 已有 image_encoder 属性但值为 None 时加载
            if hasattr(self, "image_encoder") and getattr(self, "image_encoder", None) is None:
                if image_encoder_folder is not None:
                    if not isinstance(pretrained_model_name_or_path_or_dict, dict):
                        logger.info(f"loading image_encoder from {pretrained_model_name_or_path_or_dict}")
                        # 计算 image_encoder 子目录（注意：此变量后续未使用，属冗余代码）
                        if image_encoder_folder.count("/") == 0:
                            image_encoder_subfolder = Path(subfolder, image_encoder_folder).as_posix()
                        else:
                            image_encoder_subfolder = Path(image_encoder_folder).as_posix()

                        # 【关键定制点】写死加载 DINOv2-giant，并转为 fp16 放到目标设备
                        image_encoder = AutoModel.from_pretrained("facebook/dinov2-giant").to(self.device, dtype=torch.float16)
                        self.register_modules(image_encoder=image_encoder)
                    else:
                        # 传入的是 state dict，无法推断编码器来源，报错
                        raise ValueError(
                            "`image_encoder` cannot be loaded because `pretrained_model_name_or_path_or_dict` is a state dict."
                        )
                else:
                    # 显式不加载图像编码器，给出提示
                    logger.warning(
                        "image_encoder is not loaded since `image_encoder_folder=None` passed. You will not be able to use `ip_adapter_image` when calling the pipeline with IP-Adapter."
                        "Use `ip_adapter_image_embeds` to pass pre-generated image embedding instead."
                    )

            # ---- 5. 加载特征提取器（图像预处理，同样对应 DINOv2-giant） ----
            # 仅在 pipeline 有 feature_extractor 属性但值为 None 时创建
            if hasattr(self, "feature_extractor") and getattr(self, "feature_extractor", None) is None:
                feature_extractor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant")
                self.register_modules(feature_extractor=feature_extractor)

        # ---- 6. 将 IP-Adapter 权重注入 UNet ----
        # 兼容 pipeline 用 unet_name 指向 UNet 的情况
        unet = getattr(self, self.unet_name) if not hasattr(self, "unet") else self.unet
        # 调用 UNet 内部方法真正挂载 IP-Adapter 注意力层
        unet._load_ip_adapter_weights(state_dicts, low_cpu_mem_usage=low_cpu_mem_usage)