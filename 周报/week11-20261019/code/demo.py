from typing import Optional, Union, Tuple, List, Callable, Dict
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import numpy as np
from PIL import Image, ImageOps
import cv2
from tqdm import tqdm
import argparse
import os

scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

LOW_RESOURCE = False
GUIDANCE_SCALE = 1.0
MAX_NUM_WORDS = 77
DTYPE = torch.float16 if os.environ.get("CROSS_FP16") == "1" else torch.float32
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

ldm_stable = StableDiffusionPipeline.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5', scheduler=scheduler).to(device)

try:
    ldm_stable.disable_xformers_memory_efficient_attention()
except AttributeError:
    print("Attribute disable_xformers_memory_efficient_attention() is missing")

tokenizer = ldm_stable.tokenizer

def load_image(image_path, left=0, right=0, top=0, bottom=0, resize=False):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size                   
    if resize:
        image = np.array(image.resize((512, 512)))
    else:
        width_padding = -h % 8          
        height_padding = -w % 8

        padded_image = ImageOps.expand(image, (
            0, 0, width_padding, height_padding),
            fill=None,
        )
        image = np.array(padded_image)
    return image, h, w

class ODESolve:
    def __init__(self, model, NUM_DDIM_STEPS=50):
        self.model = model
        self.num_ddim_steps = NUM_DDIM_STEPS
        self.tokenizer = self.model.tokenizer
        self.model.scheduler.set_timesteps(self.num_ddim_steps)   
        self.prompt = None       
        self.context = None

    def prev_step(self, model_output, timestep, sample):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        # x̂_0 = (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        # x_{t-Δ} = √ᾱ_{t-Δ}·x̂_0 + √(1-ᾱ_{t-Δ})·ε
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output, timestep, sample):
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        # 此时: timestep = t-Δ（输入所在时刻），next_timestep = t（输出所在时刻）
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        # x̂_0 = (x_{t-Δ} - √(1-ᾱ_{t-Δ})·ε) / √ᾱ_{t-Δ}
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        # x_t = √ᾱ_t·x̂_0 + √(1-ᾱ_t)·ε
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample

    def get_noise_pred_single(self, latents, t, context):
        # 只用一组文本嵌入做一次 UNet 噪声预测（无 CFG 双分支）
        noise_pred = self.model.unet(latents, t, context)["sample"]
        return noise_pred

    def get_noise_pred(self, latents, t, is_forward=True, context=None):
        if context is None:
            context = self.context
        guidance_scale = GUIDANCE_SCALE
        uncond_embeddings, cond_embeddings = context.chunk(2)
        noise_pred_uncond = self.model.unet(latents, t, uncond_embeddings)["sample"]
        noise_prediction_text = self.model.unet(latents, t, cond_embeddings)["sample"]
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        if is_forward:
            latents = self.next_step(noise_pred, t, latents)
        else:
            latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def latent2image(self, latents, return_type='np'):
        latents = 1 / 0.18215 * latents.detach()
        image = self.model.vae.decode(latents)['sample']
        if return_type == 'np':
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()[0]     # BCHW → HWC，取 batch 0
            image = (image * 255).astype(np.uint8)
        return image

    @torch.no_grad()
    def image2latent(self, image):
        with torch.no_grad():
            if type(image) is Image:
                image = np.array(image)
            if type(image) is torch.Tensor and image.dim() == 4:
                latents = image
            else:
                image = torch.from_numpy(image).float() / 127.5 - 1
                image = image.permute(2, 0, 1).unsqueeze(0).to(device)
                latents = self.model.vae.encode(image)['latent_dist'].mean
                latents = latents * 0.18215
        return latents

    @torch.no_grad()
    def init_prompt(self, prompt: str):
        uncond_input = self.model.tokenizer(
            [""], padding="max_length", max_length=self.model.tokenizer.model_max_length,
            return_tensors="pt"
        )
        uncond_embeddings = self.model.text_encoder(uncond_input.input_ids.to(self.model.device))[0]
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        self.context = torch.cat([uncond_embeddings, text_embeddings])
        self.prompt = prompt

    @torch.no_grad()
    def get_text_embeddings(self, prompt: str):
        text_input = self.model.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.model.device))[0]
        return text_embeddings

    @torch.no_grad()
    def ddim_loop(self, latent, is_forward=True):
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in tqdm(range(self.num_ddim_steps)):
            if is_forward:
                t = self.model.scheduler.timesteps[len(self.model.scheduler.timesteps) - i - 1]
            else:
                t = self.model.scheduler.timesteps[i]
            latent = self.get_noise_pred(latent, t, is_forward, self.context)
            all_latent.append(latent)

        return all_latent

    @property
    def scheduler(self):
        return self.model.scheduler

    def save_inter(self, latent, img_name):
        image = self.latent2image(latent)
        cv2.imwrite(img_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def invert(self, prompt, start_latent, is_forward):
        self.init_prompt(prompt)
        latents = self.ddim_loop(start_latent, is_forward=is_forward)
        return latents[-1]

    def invert_i2n2i(self, prompt1, prompt2, image_start_latent, flip=False):

        self.init_prompt(prompt1)
        latent_i2n = self.ddim_loop(image_start_latent, is_forward=True)
        xT = latent_i2n[-1]

        if flip:
            xT = torch.flip(xT, dims=[2])

        self.init_prompt(prompt2)
        latent_n2i = self.ddim_loop(xT, is_forward=False)

        return self.latent2image(image_start_latent), image_start_latent, self.latent2image(latent_n2i[-1]), latent_n2i[-1]


def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    ode = ODESolve(ldm_stable, args.num_steps)

    image_path = args.image_path
    prompt_1 = args.private_key
    prompt_2 = args.public_key

    rev_prompt_1 = prompt_1
    rev_prompt_2 = prompt_2
    need_flip = False

    offsets = (0,0,0,0)
    image_gt, h, w = load_image(image_path, *offsets, resize=True)
    
    image_gt_latent = ode.image2latent(image_gt)
    cv2.imwrite("{:s}/gt.png".format(args.save_path), cv2.cvtColor(image_gt, cv2.COLOR_RGB2BGR))

    # 前向过程 生成stego-cover
    latent_noise = ode.invert(prompt_1, image_gt_latent, is_forward=True)
    stego_latent = ode.invert(prompt_2, latent_noise, is_forward=False)

    # 保存stego
    stego_path = "{:s}/stego.png".format(args.save_path)
    ode.save_inter(stego_latent, stego_path)

    # stego --VAE encode--> 加噪到 noise_step --从该时刻去噪--> cover
    cover_path = "{:s}/cover.png".format(args.save_path)

    # 再生成时使用的文本条件：
    #   empty  : 无条件（与 DiffWMAttacker 原论文默认一致，最容易抹掉隐写信息）
    #   public : 用公钥提示词，通常能更好地保住 stego 的内容
    captions = {}
    if args.cover_prompt_mode == "public":
        captions[os.path.splitext(os.path.basename(stego_path))[0]] = prompt_2

    attacker = DiffWMAttacker(
        RegenPipe(ldm_stable),
        batch_size=1,
        noise_step=args.cover_noise_step,
        captions=captions,
    )
    attacker.attack([stego_path], [cover_path])

    # 可选：对生成的 cover 再走一遍 reveal，检查 secret 是否已被抹掉
    cover_img = load_image(cover_path, *offsets, resize=True)[0]
    cover_latent = ode.image2latent(cover_img)
    cover_reverse_latent = ode.invert(
        prompt_1,
        ode.invert(prompt_2, cover_latent, is_forward=True),
        is_forward=False,
    )
    ode.save_inter(cover_reverse_latent, "{:s}/test.png".format(args.save_path))

class RegenPipe:
    """
    DiffWMAttacker 的调用约定是：
        pipe(prompts, head_start_latents=..., head_start_step=...,
             guidance_scale=7.5, generator=...)
    即从「已经加噪到某个时刻的 latent」出发，只跑剩下的去噪步。
    标准 StableDiffusionPipeline.__call__ 不支持从中间时刻起步，因此这里
    直接用 UNet + scheduler 手写这段去噪循环。
    """

    def __init__(self, pipe):
        self.pipe = pipe
        self.device = pipe.device
        self.unet = pipe.unet
        self.vae = pipe.vae
        self.scheduler = pipe.scheduler
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder

    def _encode(self, texts):
        toks = self.tokenizer(
            texts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return self.text_encoder(toks)[0].to(self.unet.dtype)

    @torch.no_grad()
    def __call__(self, prompts, head_start_latents=None, head_start_step=None,
                 guidance_scale=7.5, generator=None, num_inference_steps=50,
                 output_type="pil", **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]

        do_cfg = guidance_scale > 1.0

        # 1) 文本条件（CFG 时拼成 [uncond; cond]）
        text_emb = self._encode(prompts)
        if do_cfg:
            uncond_emb = self._encode([""] * len(prompts))
            text_emb = torch.cat([uncond_emb, text_emb], dim=0)

        # 2) 时间表：必须与 DiffWMAttacker 中 head_start_step 的 50 步假设一致
        try:
            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        except TypeError:
            self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps

        # 3) 起始 latent（DiffWMAttacker 已经完成 VAE 编码 + add_noise）
        if head_start_latents is None:
            raise ValueError("RegenPipe 只支持 head_start_latents 模式")
        latents = head_start_latents.to(device=self.device, dtype=self.unet.dtype)

        # 4) 从 head_start_step 开始去噪
        start = 0 if head_start_step is None else int(head_start_step)
        for i, t in enumerate(timesteps):
            if i < start:
                continue
            latent_model_input = torch.cat([latents] * 2, dim=0) if do_cfg else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_emb
            ).sample
            if do_cfg:
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # 5) VAE 解码
        latents = latents / self.vae.config.scaling_factor
        out = self.vae.decode(latents.to(self.vae.dtype))
        image = out.sample if hasattr(out, "sample") else out["sample"]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if output_type == "pil":
            images = [Image.fromarray((img * 255).round().astype("uint8")) for img in image]
        else:
            images = image
        return (images, None)


class WMAttacker:
    def attack(self, imgs_path, out_path):
        raise NotImplementedError

class DiffWMAttacker(WMAttacker):
    def __init__(self, pipe, batch_size=20, noise_step=60, captions={}):
        self.pipe = pipe
        self.BATCH_SIZE = batch_size
        self.device = pipe.device
        self.noise_step = noise_step
        self.captions = captions
        # 原始实现把 dtype 写死成 float16；这里跟随 pipeline 实际精度，
        # 否则 fp32 的 ldm_stable 会在 VAE encode 处报 dtype 不匹配。
        self.dtype = pipe.unet.dtype if hasattr(pipe, "unet") else torch.float32
        print(f'Diffuse attack initialized with noise step {self.noise_step} and use prompt {len(self.captions)}')

    def attack(self, image_paths, out_paths, return_latents=False, return_dist=False):
        with torch.no_grad():
            generator = torch.Generator(self.device).manual_seed(1024)
            latents_buf = []
            prompts_buf = []
            outs_buf = []
            timestep = torch.tensor([self.noise_step], dtype=torch.long, device=self.device)
            ret_latents = []

            def batched_attack(latents_buf, prompts_buf, outs_buf):
                latents = torch.cat(latents_buf, dim=0)
                images = self.pipe(prompts_buf,
                                   head_start_latents=latents,
                                   head_start_step=50 - max(self.noise_step // 20, 1),
                                   guidance_scale=7.5,
                                   generator=generator, )
                images = images[0]
                for img, out in zip(images, outs_buf):
                    img.save(out)

            if len(self.captions) != 0:
                prompts = []
                for img_path in image_paths:
                    img_name = os.path.basename(img_path)
                    if img_name[:-4] in self.captions:
                        prompts.append(self.captions[img_name[:-4]])
                    else:
                        prompts.append("")
            else:
                prompts = [""] * len(image_paths)

            for (img_path, out_path), prompt in tqdm(zip(zip(image_paths, out_paths), prompts)):
                img = Image.open(img_path).convert("RGB")
                img = np.asarray(img) / 255
                img = (img - 0.5) * 2
                img = torch.tensor(img, dtype=self.dtype, device=self.device).permute(2, 0, 1).unsqueeze(0)
                latents = self.pipe.vae.encode(img).latent_dist
                latents = latents.sample(generator) * self.pipe.vae.config.scaling_factor
                noise = torch.randn([1, 4, img.shape[-2] // 8, img.shape[-1] // 8], device=self.device)
                if return_dist:
                    return self.pipe.scheduler.add_noise(latents, noise, timestep, return_dist=True)
                latents = self.pipe.scheduler.add_noise(latents, noise, timestep).to(self.dtype)
                latents_buf.append(latents)
                outs_buf.append(out_path)
                prompts_buf.append(prompt)
                if len(latents_buf) == self.BATCH_SIZE:
                    batched_attack(latents_buf, prompts_buf, outs_buf)
                    latents_buf = []
                    prompts_buf = []
                    outs_buf = []
                if return_latents:
                    ret_latents.append(latents.cpu())

            if len(latents_buf) != 0:
                batched_attack(latents_buf, prompts_buf, outs_buf)
            if return_latents:
                return ret_latents

def SC_producer():
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    ode = ODESolve(ldm_stable, args.num_steps)
    image_path = args.image_path
    prompt_1 = args.private_key
    prompt_2 = args.public_key
    offsets = (0,0,0,0)
    image_gt, h, w = load_image(image_path, *offsets, resize=True)

    image_gt_latent = ode.image2latent(image_gt)
    cv2.imwrite("{:s}/gt.png".format(args.save_path), cv2.cvtColor(image_gt, cv2.COLOR_RGB2BGR))
    # 前向过程 生成stego-cover
    latent_noise = ode.invert(prompt_1, image_gt_latent, is_forward=True)
    stego_latent = ode.invert(prompt_2, latent_noise, is_forward=False)
    # 保存stego
    ode.save_inter(stego_latent,"{:s}/stego.png".format(args.save_path))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='./asserts/1.png', help='test image path')
    parser.add_argument('--private_key', type=str, default='Effiel tower', help='text prompt of the private key')
    parser.add_argument('--public_key', type=str, default='a tree', help='text prompt of the public key')
    parser.add_argument('--save_path', type=str, default='./output', help='text prompt of the public key')
    parser.add_argument('--num_steps', type=int, default=50, help='sampling step of DDIM')
    parser.add_argument('--cover_noise_step', type=int, default=60,
                        help='DiffWMAttacker 的加噪步数：越大抹得越狠、cover 越不像 stego')
    parser.add_argument('--cover_prompt_mode', type=str, default='empty', choices=['empty', 'public'],
                        help='再生成时用的文本条件：empty=无条件（原论文默认），public=用公钥提示词')
    args = parser.parse_args()

    main(args=args)
