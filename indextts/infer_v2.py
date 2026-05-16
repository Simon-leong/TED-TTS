import os
from subprocess import CalledProcessError

os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import json
import re
import time
import librosa
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from omegaconf import OmegaConf

from indextts.gpt.model_v2 import UnifiedVoice
from indextts.utils.maskgct_utils import build_semantic_model, build_semantic_codec
from indextts.utils.checkpoint import load_checkpoint
from indextts.utils.front import TextNormalizer, TextTokenizer
from indextts.utils.logger import get_logger, create_progress, ColorfulLogger

from indextts.s2mel.modules.commons import load_checkpoint2, MyModel
from indextts.s2mel.modules.bigvgan import bigvgan
from indextts.s2mel.modules.campplus.DTDNN import CAMPPlus
from indextts.s2mel.modules.audio import mel_spectrogram

from transformers import AutoTokenizer
from modelscope import AutoModelForCausalLM
from huggingface_hub import hf_hub_download
import safetensors
from transformers import SeamlessM4TFeatureExtractor
import random
import torch.nn.functional as F

class IndexTTS2:
    def __init__(
            self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", is_fp16=False, device=None,
            use_cuda_kernel=None,
    ):
        """
        Args:
            cfg_path (str): path to the config file.
            model_dir (str): path to the model directory.
            is_fp16 (bool): whether to use fp16.
            device (str): device to use (e.g., 'cuda:0', 'cpu'). If None, it will be set automatically based on the availability of CUDA or MPS.
            use_cuda_kernel (None | bool): whether to use BigVGan custom fused activation CUDA kernel, only for CUDA device.
        """
        if device is not None:
            self.device = device
            self.is_fp16 = False if device == "cpu" else is_fp16
            self.use_cuda_kernel = use_cuda_kernel is not None and use_cuda_kernel and device.startswith("cuda")
        elif torch.cuda.is_available():
            self.device = "cuda:0"
            self.is_fp16 = is_fp16
            self.use_cuda_kernel = use_cuda_kernel is None or use_cuda_kernel
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
            self.is_fp16 = False  # Use float16 on MPS is overhead than float32
            self.use_cuda_kernel = False
        else:
            self.device = "cpu"
            self.is_fp16 = False
            self.use_cuda_kernel = False
        
        # Initialize logger
        self.logger = get_logger("IndexTTS2")
        
        if self.device == "cpu":
            self.logger.warning("Be patient, it may take a while to run in CPU mode.")
        self.logger.device_info(self.device, self.is_fp16, self.use_cuda_kernel)

        self.cfg = OmegaConf.load(cfg_path)
        self.model_dir = model_dir
        self.dtype = torch.float16 if self.is_fp16 else None
        self.stop_mel_token = self.cfg.gpt.stop_mel_token

        self.qwen_emo = QwenEmotion(os.path.join(self.model_dir, self.cfg.qwen_emo_path))

        self.gpt = UnifiedVoice(**self.cfg.gpt)
        self.gpt_path = os.path.join(self.model_dir, self.cfg.gpt_checkpoint)
        load_checkpoint(self.gpt, self.gpt_path)
        self.gpt = self.gpt.to(self.device)
        if self.is_fp16:
            self.gpt.eval().half()
        else:
            self.gpt.eval()
        self.logger.model_loaded("GPT Model", self.gpt_path)

        use_deepspeed = False
        try:
            # import deepspeed
            pass
        except (ImportError, OSError, CalledProcessError) as e:
            use_deepspeed = False
            self.logger.warning(f"DeepSpeed load failed; falling back to standard inference: {e}")

        self.gpt.post_init_gpt2_config(use_deepspeed=use_deepspeed, kv_cache=True, half=self.is_fp16)

        if self.use_cuda_kernel:
            # preload the CUDA kernel for BigVGAN
            try:
                from indextts.BigVGAN.alias_free_activation.cuda import load

                anti_alias_activation_cuda = load.load()
                self.logger.success(f"Preload custom CUDA kernel for BigVGAN: {anti_alias_activation_cuda}")
            except:
                self.logger.warning("Failed to load custom CUDA kernel for BigVGAN. Falling back to torch.")
                self.use_cuda_kernel = False

        self.extract_features = SeamlessM4TFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.semantic_model, self.semantic_mean, self.semantic_std = build_semantic_model(
            os.path.join(self.model_dir, self.cfg.w2v_stat))
        self.semantic_model = self.semantic_model.to(self.device)
        self.semantic_model.eval()
        self.semantic_mean = self.semantic_mean.to(self.device)
        self.semantic_std = self.semantic_std.to(self.device)

        semantic_codec = build_semantic_codec(self.cfg.semantic_codec)
        semantic_code_ckpt = hf_hub_download("amphion/MaskGCT", filename="semantic_codec/model.safetensors")
        safetensors.torch.load_model(semantic_codec, semantic_code_ckpt)
        self.semantic_codec = semantic_codec.to(self.device)
        self.semantic_codec.eval()
        self.logger.model_loaded("Semantic Codec", semantic_code_ckpt)

        s2mel_path = os.path.join(self.model_dir, self.cfg.s2mel_checkpoint)
        s2mel = MyModel(self.cfg.s2mel, use_gpt_latent=True)
        s2mel, _, _, _ = load_checkpoint2(
            s2mel,
            None,
            s2mel_path,
            load_only_params=True,
            ignore_modules=[],
            is_distributed=False,
        )
        self.s2mel = s2mel.to(self.device)
        self.s2mel.models['cfm'].estimator.setup_caches(max_batch_size=1, max_seq_length=8192)
        self.s2mel.eval()
        self.logger.model_loaded("S2Mel Model", s2mel_path)

        # load campplus_model
        campplus_ckpt_path = hf_hub_download(
            "funasr/campplus", filename="campplus_cn_common.bin"
        )
        campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
        self.campplus_model = campplus_model.to(self.device)
        self.campplus_model.eval()
        self.logger.model_loaded("CAMPPlus Model", campplus_ckpt_path)

        bigvgan_name = self.cfg.vocoder.name
        self.bigvgan = bigvgan.BigVGAN.from_pretrained(bigvgan_name, use_cuda_kernel=self.use_cuda_kernel)
        self.bigvgan = self.bigvgan.to(self.device)
        self.bigvgan.remove_weight_norm()
        self.bigvgan.eval()
        self.logger.model_loaded("BigVGAN Vocoder", bigvgan_name)

        self.bpe_path = os.path.join(self.model_dir, self.cfg.dataset["bpe_model"])
        self.normalizer = TextNormalizer()
        self.normalizer.load()
        self.logger.success("TextNormalizer loaded")
        self.tokenizer = TextTokenizer(self.bpe_path, self.normalizer)
        self.logger.model_loaded("BPE Tokenizer", self.bpe_path)

        emo_matrix = torch.load(os.path.join(self.model_dir, self.cfg.emo_matrix))
        self.emo_matrix = emo_matrix.to(self.device)
        self.emo_num = list(self.cfg.emo_num)

        spk_matrix = torch.load(os.path.join(self.model_dir, self.cfg.spk_matrix))
        self.spk_matrix = spk_matrix.to(self.device)

        self.emo_matrix = torch.split(self.emo_matrix, self.emo_num)
        self.spk_matrix = torch.split(self.spk_matrix, self.emo_num)

        mel_fn_args = {
            "n_fft": self.cfg.s2mel['preprocess_params']['spect_params']['n_fft'],
            "win_size": self.cfg.s2mel['preprocess_params']['spect_params']['win_length'],
            "hop_size": self.cfg.s2mel['preprocess_params']['spect_params']['hop_length'],
            "num_mels": self.cfg.s2mel['preprocess_params']['spect_params']['n_mels'],
            "sampling_rate": self.cfg.s2mel["preprocess_params"]["sr"],
            "fmin": self.cfg.s2mel['preprocess_params']['spect_params'].get('fmin', 0),
            "fmax": None if self.cfg.s2mel['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
            "center": False
        }
        self.mel_fn = lambda x: mel_spectrogram(x, **mel_fn_args)

        # Reference-audio cache.
        self.cache_spk_cond = None
        self.cache_s2mel_style = None
        self.cache_s2mel_prompt = None
        self.cache_spk_audio_prompt = None
        self.cache_emo_cond = None
        self.cache_emo_audio_prompt = None
        self.cache_mel = None

        # Optional external progress handle.
        self.gr_progress = None
        self.colorful_progress = None
        self.model_version = self.cfg.version if hasattr(self.cfg, "version") else None
        
        self.logger.success("All models initialized successfully!")

    @torch.no_grad()
    def get_emb(self, input_features, attention_mask):
        vq_emb = self.semantic_model(
            input_features=input_features,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        feat = vq_emb.hidden_states[17]  # (B, T, C)
        feat = (feat - self.semantic_mean) / self.semantic_std
        return feat

    def remove_long_silence(self, codes: torch.Tensor, silent_token=52, max_consecutive=30):
        """
        Shrink special tokens (silent_token and stop_mel_token) in codes
        codes: [B, T]
        """
        code_lens = []
        codes_list = []
        device = codes.device
        dtype = codes.dtype
        isfix = False
        for i in range(0, codes.shape[0]):
            code = codes[i]
            if not torch.any(code == self.stop_mel_token).item():
                len_ = code.size(0)
            else:
                stop_mel_idx = (code == self.stop_mel_token).nonzero(as_tuple=False)
                len_ = stop_mel_idx[0].item() if len(stop_mel_idx) > 0 else code.size(0)

            count = torch.sum(code == silent_token).item()
            if count > max_consecutive:
                # code = code.cpu().tolist()
                ncode_idx = []
                n = 0
                for k in range(len_):
                    assert code[
                               k] != self.stop_mel_token, f"stop_mel_token {self.stop_mel_token} should be shrinked here"
                    if code[k] != silent_token:
                        ncode_idx.append(k)
                        n = 0
                    elif code[k] == silent_token and n < 10:
                        ncode_idx.append(k)
                        n += 1
                    # if (k == 0 and code[k] == 52) or (code[k] == 52 and code[k-1] == 52):
                    #    n += 1
                # new code
                len_ = len(ncode_idx)
                codes_list.append(code[ncode_idx])
                isfix = True
            else:
                # shrink to len_
                codes_list.append(code[:len_])
            code_lens.append(len_)
        if isfix:
            if len(codes_list) > 1:
                codes = pad_sequence(codes_list, batch_first=True, padding_value=self.stop_mel_token)
            else:
                codes = codes_list[0].unsqueeze(0)
        else:
            # unchanged
            pass
        # clip codes to max length
        max_len = max(code_lens)
        if max_len < codes.shape[1]:
            codes = codes[:, :max_len]
        code_lens = torch.tensor(code_lens, dtype=torch.long, device=device)
        return codes, code_lens

    def insert_interval_silence(self, wavs, sampling_rate=22050, interval_silence=200):
        """
        Insert silences between sentences.
        wavs: List[torch.tensor]
        """

        if not wavs or interval_silence <= 0:
            return wavs

        # get channel_size
        channel_size = wavs[0].size(0)
        # get silence tensor
        sil_dur = int(sampling_rate * interval_silence / 1000.0)
        sil_tensor = torch.zeros(channel_size, sil_dur)

        wavs_list = []
        for i, wav in enumerate(wavs):
            wavs_list.append(wav)
            if i < len(wavs) - 1:
                wavs_list.append(sil_tensor)

        return wavs_list

    def _load_and_cut_audio(self, audio_path, max_audio_length_seconds, verbose=False, sr=None):
        if sr is None:
            audio, sr = librosa.load(audio_path)
        else:
            audio, _ = librosa.load(audio_path, sr=sr)
        if verbose:
            self.logger.debug(f"audio shape: {audio.shape}, sample rate: {sr}")
        if len(audio.shape) == 2:
            audio = torch.tensor(audio).mean(0).unsqueeze(0)
        else:
            audio = torch.tensor(audio).unsqueeze(0)
        max_len_samples = int(max_audio_length_seconds * sr)
        if audio.shape[1] > max_len_samples:
            audio = audio[:, :max_len_samples]
            if verbose:
                self.logger.warning(f"audio too long, cut to first {max_audio_length_seconds} seconds")
        return audio, sr

    def _set_gr_progress(self, value, desc):
        if self.gr_progress is not None:
            self.gr_progress(value, desc=desc)
        if self.colorful_progress is not None:
            self.colorful_progress.update(completed=value * 100, description=desc)

    # Standard inference entry point.
    def infer(self, spk_audio_prompt, text, output_path,
              emo_audio_prompt=None, emo_alpha=1.0,
              emo_vector=None, style_prompt=None,
              use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
              verbose=False, max_text_tokens_per_sentence=120, return_stats=False, save_attention_maps=False, **generation_kwargs):
        self.logger.stage("Starting Inference")
        self._set_gr_progress(0, "start inference...")
        if verbose:
            self.logger.print_dict("Inference Parameters", {
                "text": text,
                "spk_audio_prompt": spk_audio_prompt,
                "emo_audio_prompt": emo_audio_prompt,
                "emo_alpha": emo_alpha,
                "emo_vector": emo_vector,
                "use_emo_text": use_emo_text,
                "emo_text": emo_text
            })
        start_time = time.perf_counter()
        emo_vectors = emo_vector
        if use_emo_text:
            emo_vectors = []
            emo_audio_prompt = None
            emo_alpha = 1.0
            # assert emo_audio_prompt is None
            # assert emo_alpha == 1.0
            if emo_text is None:
                emo_text = text
            emo_texts = emo_text.split("|")
            for emo_text in emo_texts:
                emo_dict = self.qwen_emo.inference(emo_text)
                # convert ordered dict to list of vectors; the order is VERY important!
                emo_vector = list(emo_dict.values())
                emo_vectors.append(emo_vector)
        # emo_vectors = [[1., 0, 0, 0, 0, 0, 0, 0], emo_vector, [1., 0, 0, 0, 0, 0, 0, 0]]
        # emo_vectors = [emo_vector]
        if emo_vectors is not None:
            emo_audio_prompt = None
            emo_alpha = 1.0
            EMOTION_DIMENSIONS = ["happy", "angry", "sad", "afraid", "disgusted", "melancholic", "surprised", "calm"]
            # Collect all emotion vectors for display
            emo_vec_maps = []
            for emo in emo_vectors:
                emo_vec_map = {dim: val for dim, val in zip(EMOTION_DIMENSIONS, emo)}
                emo_vec_maps.append(emo_vec_map)
            # Display all emotion vectors side by side
            self.logger.print_emotion_vector(emo_vec_maps)
            # assert emo_audio_prompt is None
            # assert emo_alpha == 1.0

        if emo_audio_prompt is None:
            emo_audio_prompt = spk_audio_prompt
            emo_alpha = 1.0
            # assert emo_alpha == 1.0

        # Only recompute features when the reference audio actually changes; speeds up repeated calls.
        if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt:
            spk_cond_emb_list = []
            self._set_gr_progress(0.05, "processing reference audio...")
            self.logger.info("Processing reference audio...")
            if isinstance(spk_audio_prompt, str):
                audio, sr = self._load_and_cut_audio(spk_audio_prompt, 15, verbose)
                audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
                audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

                inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
                input_features = inputs["input_features"]
                attention_mask = inputs["attention_mask"]
                input_features = input_features.to(self.device)
                attention_mask = attention_mask.to(self.device)
                spk_cond_emb = self.get_emb(input_features, attention_mask)

                _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
                ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
                ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
                feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(ref_mel.device),
                                                        num_mel_bins=80,
                                                        dither=0,
                                                        sample_frequency=16000)
                feat = feat - feat.mean(dim=0, keepdim=True)  # alternate filter-bank energy features [922, 80]
                style = self.campplus_model(feat.unsqueeze(0))  # reference-audio global style [1, 192]

                prompt_condition = self.s2mel.models['length_regulator'](S_ref,
                                                                        ylens=ref_target_lengths,
                                                                        n_quantizers=3,
                                                                        f0=None)[0]
            elif isinstance(spk_audio_prompt, list):
                ref_mel_list = []
                styles_list = []
                prompt_conditions_list = []
                for spk_path in spk_audio_prompt:
                    audio, sr = self._load_and_cut_audio(spk_path, 15, verbose)
                    audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
                    audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

                    inputs = self.extract_features(audio_16k, sampling_rate=16000, return_tensors="pt")
                    input_features = inputs["input_features"]
                    attention_mask = inputs["attention_mask"]
                    input_features = input_features.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    spk_cond_emb = self.get_emb(input_features, attention_mask)
                    spk_cond_emb_list.append(spk_cond_emb)

                    _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
                    ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
                    ref_mel_list.append(ref_mel)
                    ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
                    feat = torchaudio.compliance.kaldi.fbank(audio_16k.to(ref_mel.device),
                                                            num_mel_bins=80,
                                                            dither=0,
                                                            sample_frequency=16000)
                    feat = feat - feat.mean(dim=0, keepdim=True)  # alternate filter-bank energy features [922, 80]
                    style = self.campplus_model(feat.unsqueeze(0))  # reference-audio global style [1, 192]
                    styles_list.append(style)

                    prompt_condition_tmp = self.s2mel.models['length_regulator'](S_ref,
                                                                                ylens=ref_target_lengths,
                                                                                n_quantizers=3,
                                                                                f0=None)[0]
                    prompt_conditions_list.append(prompt_condition_tmp)
                    
                spk_cond_emb = spk_cond_emb_list
                ref_mel = torch.cat(ref_mel_list, dim=2)
                prompt_condition = torch.cat(prompt_conditions_list, dim=1)
                
                # ref_mel = ref_mel_list[0]
                # prompt_condition = prompt_conditions_list[0]
                if style_prompt is not None:
                    style_audio, sr = self._load_and_cut_audio(style_prompt, 15, verbose)
                    style_audio_16k = torchaudio.transforms.Resample(sr, 16000)(style_audio)
                    
                    feat = torchaudio.compliance.kaldi.fbank(style_audio_16k.to(ref_mel.device),
                                                            num_mel_bins=80,
                                                            dither=0,
                                                            sample_frequency=16000)
                    feat = feat - feat.mean(dim=0, keepdim=True)
                    style = self.campplus_model(feat.unsqueeze(0))  # reference-audio global style [1, 192]
                else:
                    style = torch.stack(styles_list, dim=0).mean(dim=0)
                    # style = styles_list[0]
                
            else:
                raise ValueError("`spk_audio_prompt` must be str or list.")
                
            self.cache_spk_cond = spk_cond_emb
            self.cache_s2mel_style = style
            self.cache_s2mel_prompt = prompt_condition
            self.cache_spk_audio_prompt = spk_audio_prompt
            self.cache_mel = ref_mel
        else:
            style = self.cache_s2mel_style
            prompt_condition = self.cache_s2mel_prompt
            spk_cond_emb = self.cache_spk_cond
            ref_mel = self.cache_mel
        
        # Emotion-matrix module.
        emovec_mats = []
        emo_weight_sums = None
        if emo_vectors is not None and len(emo_vectors) > 0:
            emo_vectors_tensor = torch.as_tensor(emo_vectors, device=self.device, dtype=style.dtype)
            emo_weight_sums = emo_vectors_tensor.sum(dim=1, keepdim=True)

            if use_random:
                for emo_vector in emo_vectors_tensor:
                    random_index = [random.randint(0, x - 1) for x in self.emo_num]
                    emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
                    emo_matrix = torch.cat(emo_matrix, 0)
                    emovec_mat = torch.sum(emo_vector.unsqueeze(1) * emo_matrix, 0, keepdim=True)
                    emovec_mats.append(emovec_mat)
            else:
                random_index = [find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix]
                emo_matrix = [tmp[index].unsqueeze(0) for index, tmp in zip(random_index, self.emo_matrix)]
                emo_matrix = torch.cat(emo_matrix, 0)
                emovec_mats_tensor = emo_vectors_tensor @ emo_matrix
                emovec_mats = list(emovec_mats_tensor.split(1, dim=0))

        if self.cache_emo_cond is None or self.cache_emo_audio_prompt != emo_audio_prompt:
            self._set_gr_progress(0.08, "processing emotion audio...")
            self.logger.info("Processing emotion reference...")
            if isinstance(emo_audio_prompt, str):
                emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt, 15, verbose, sr=16000)
                emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
                emo_input_features = emo_inputs["input_features"]
                emo_attention_mask = emo_inputs["attention_mask"]
                emo_input_features = emo_input_features.to(self.device)
                emo_attention_mask = emo_attention_mask.to(self.device)
                emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)

                self.cache_emo_cond = emo_cond_emb
                self.cache_emo_audio_prompt = emo_audio_prompt
            elif isinstance(emo_audio_prompt, list):
                emo_cond_embs = []
                for emo_audio_path in emo_audio_prompt:
                    emo_audio, _ = self._load_and_cut_audio(emo_audio_path, 15, verbose, sr=16000)
                    emo_inputs = self.extract_features(emo_audio, sampling_rate=16000, return_tensors="pt")
                    emo_input_features = emo_inputs["input_features"]
                    emo_attention_mask = emo_inputs["attention_mask"]
                    emo_input_features = emo_input_features.to(self.device)
                    emo_attention_mask = emo_attention_mask.to(self.device)
                    emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)
                    emo_cond_embs.append(emo_cond_emb)
                
                self.cache_emo_cond = emo_cond_embs
                self.cache_emo_audio_prompt = emo_audio_prompt
            else:
                raise ValueError("emo_audio_prompt must be str or list of str")
        else:
            emo_cond_emb = self.cache_emo_cond

        self._set_gr_progress(0.1, "text processing...")        
        self.logger.info("Processing text...")        # '|' splits the input into segments; each segment carries its own emotion.
        text_list = text.split("|")
        text_tokens_list = []
        for txt in text_list:
            text_tokens_list.append(self.tokenizer.tokenize(txt))
        # sentences = self.tokenizer.split_sentences(text_tokens_list, max_text_tokens_per_sentence)
        
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        autoregressive_batch_size = 1
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)
        target_duration_tokens = generation_kwargs.pop("target_duration_tokens", None)
        duration_mode = generation_kwargs.pop("duration_mode", "none")
        sampling_rate = 22050

        if output_path is not None and "output_path" not in generation_kwargs:
            generation_kwargs["output_path"] = output_path
        
        generation_kwargs["save_attention_maps"] = save_attention_maps

        wavs = []
        gpt_gen_time = 0
        gpt_forward_time = 0
        s2mel_time = 0
        bigvgan_time = 0
        progress = 0
        has_warned = False
        # for sent in sentences:


        text_tokens_list = [self.tokenizer.convert_tokens_to_ids(sent) for sent in text_tokens_list]
        text_tokens_list = [torch.tensor(text_tokens, dtype=torch.int32, device=self.device).unsqueeze(0) for text_tokens in text_tokens_list]
        
        self._set_gr_progress(0.15, "generating speech codes...")        
        self.logger.info("Generating speech codes with GPT...")

        m_start_time = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast(text_tokens_list[0].device.type, enabled=self.dtype is not None, dtype=self.dtype):
                emovecs = []
                spk_cond_lengths_list = []
                emo_cond_lengths_list = []

                if isinstance(spk_audio_prompt, list):
                    for spk_cond_emb_tmp, emo_cond_emb_tmp in zip(spk_cond_emb_list, emo_cond_embs):
                        spk_cond_lengths = torch.tensor([spk_cond_emb_tmp.shape[-1]], device=text_tokens_list[0].device)
                        emo_cond_lengths = torch.tensor([emo_cond_emb_tmp.shape[-1]], device=text_tokens_list[0].device)
                        emovec = self.gpt.merge_emovec(
                            spk_cond_emb_tmp,
                            emo_cond_emb_tmp,
                            spk_cond_lengths,
                            emo_cond_lengths,
                            alpha=emo_alpha
                        )
                        emovecs.append(emovec)
                        spk_cond_lengths_list.append(spk_cond_lengths)
                        emo_cond_lengths_list.append(emo_cond_lengths)
                else:
                    emovec = self.gpt.merge_emovec(
                        spk_cond_emb,
                        emo_cond_emb,
                        torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens_list[0].device),
                        torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens_list[0].device),
                        alpha=emo_alpha
                    )
                    
                if emo_vectors is not None and len(emo_vectors) > 0:
                    if len(emovec_mats) > 0:
                        base_emovec = emovecs[-1] if len(emovecs) > 0 else emovec
                        emovec_mats_tensor = torch.cat(emovec_mats, dim=0)
                        if emo_weight_sums is None:
                            emo_weight_sums = torch.ones(
                                (emovec_mats_tensor.size(0), 1),
                                device=emovec_mats_tensor.device,
                                dtype=emovec_mats_tensor.dtype,
                            )
                        mixed_emovecs = emovec_mats_tensor + (1 - emo_weight_sums.to(base_emovec.dtype)) * base_emovec.expand(emovec_mats_tensor.size(0), -1)
                        emovecs.extend(mixed_emovecs.split(1, dim=0))
                elif len(emovecs) == 0:
                    emovecs.append(emovec)
                
                if len(spk_cond_lengths_list) > 0 and len(emo_cond_lengths_list) > 0:
                    input_cond_lengths = spk_cond_lengths_list
                    input_emo_cond_lengths = emo_cond_lengths_list
                else:
                    input_cond_lengths = torch.tensor([spk_cond_emb.shape[-1]], device=text_tokens_list[0].device)
                    input_emo_cond_lengths = torch.tensor([emo_cond_emb.shape[-1]], device=text_tokens_list[0].device)
                
                # print(f"Merge emovec time: {time.perf_counter() - p_start_time:.2f}s")

                codes, speech_conditioning_latent, attention_mask, seg_lens, token_generation_time = self.gpt.inference_speech(
                    spk_cond_emb,
                    text_tokens_list,
                    emo_cond_emb,
                    cond_lengths=input_cond_lengths,
                    emo_cond_lengths=input_emo_cond_lengths,
                    emo_vecs=emovecs,
                    do_sample=True,
                    top_p=top_p,
                    top_k=top_k,
                    temperature=temperature,
                    num_return_sequences=autoregressive_batch_size,
                    length_penalty=length_penalty,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    max_generate_length=max_mel_tokens,
                    target_duration_tokens=target_duration_tokens,
                    duration_mode=duration_mode,
                    **generation_kwargs
                )

            gpt_gen_time += time.perf_counter() - m_start_time
            self._set_gr_progress(0.4, "speech codes generated")            
            self.logger.success(f"Speech codes generated in {gpt_gen_time:.2f}s")
            if not has_warned and (codes[:, -1] != self.stop_mel_token).any():
                warnings.warn(
                    f"WARN: generation stopped due to exceeding `max_mel_tokens` ({max_mel_tokens}). "
                    f"Input text tokens: {text_tokens_list[0].shape[1]}. "
                    f"Consider reducing `max_text_tokens_per_sentence`({max_text_tokens_per_sentence}) or increasing `max_mel_tokens`.",
                    category=RuntimeWarning
                )
                has_warned = True

            code_lens = torch.tensor([codes.shape[-1]], device=codes.device, dtype=codes.dtype)
            #                 if verbose:
            #                     print(codes, type(codes))
            #                     print(f"codes shape: {codes.shape}, codes type: {codes.dtype}")
            #                     print(f"code len: {code_lens}")

            code_lens = []
            for code in codes:
                if self.stop_mel_token not in code:
                    code_len = len(code)
                else:
                    len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0] + 1
                    code_len = len_ - 1
                code_lens.append(code_len)
            codes = codes[:, :code_len]
            code_lens = torch.LongTensor(code_lens)
            code_lens = code_lens.to(self.device)
            if verbose:
                self.logger.debug(f"Codes: {codes}, type: {type(codes)}")
                self.logger.debug(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                self.logger.debug(f"code len: {code_lens}")

            self._set_gr_progress(0.45, "computing speech latent...")
            self.logger.info("Computing speech latent with GPT forward...")
            m_start_time = time.perf_counter()
            
            if isinstance(spk_cond_emb, list):
                bs = spk_cond_emb[0]
            else:
                bs = spk_cond_emb
            use_speed = torch.zeros(bs.size(0)).to(bs.device).long()
            with torch.amp.autocast(text_tokens_list[0].device.type, enabled=self.dtype is not None, dtype=self.dtype):
                latent = self.gpt(
                    speech_conditioning_latent,
                    torch.cat(text_tokens_list, dim=1),
                    torch.tensor([text_tokens_list[0].shape[-1]], device=text_tokens_list[0].device),
                    codes,
                    torch.tensor([codes.shape[-1]], device=text_tokens_list[0].device),
                    emo_cond_emb,
                    cond_mel_lengths=input_cond_lengths,
                    emo_cond_mel_lengths=input_emo_cond_lengths,
                    emo_vecs=emovecs,
                    use_speed=use_speed,
                    attention_mask=attention_mask,
                )
                gpt_forward_time += time.perf_counter() - m_start_time
                self._set_gr_progress(0.55, "generating mel-spectrogram...")
                self.logger.info("Generating mel-spectrogram with S2Mel...")

            dtype = None
            with torch.amp.autocast(text_tokens_list[0].device.type, enabled=dtype is not None, dtype=dtype):
                m_start_time = time.perf_counter()
                diffusion_steps = 25
                inference_cfg_rate = 0.7
                latent = self.s2mel.models['gpt_layer'](latent)
                S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
                S_infer = S_infer.transpose(1, 2)
                S_infer = S_infer + latent
                target_lengths = (code_lens * 1.72).long()

                cond = self.s2mel.models['length_regulator'](S_infer,
                                                                ylens=target_lengths,
                                                                n_quantizers=3,
                                                                f0=None)[0]
                
                cat_condition = torch.cat([prompt_condition, cond], dim=1)
                vc_target = self.s2mel.models['cfm'].inference(cat_condition,
                                                                torch.LongTensor([cat_condition.size(1)]).to(
                                                                    cond.device),
                                                                ref_mel, style, None, diffusion_steps,
                                                                inference_cfg_rate=inference_cfg_rate)
                vc_target = vc_target[:, :, ref_mel.size(-1):]
                s2mel_time += time.perf_counter() - m_start_time
                self._set_gr_progress(0.75, "synthesizing waveform...")
                self.logger.info("Synthesizing waveform with BigVGAN...")

                m_start_time = time.perf_counter()
                wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
                if verbose:
                    self.logger.debug(f"BigVGAN output shape: {wav.shape}")
                bigvgan_time += time.perf_counter() - m_start_time
                wav = wav.squeeze(1)

            wav = torch.clamp(32767 * wav, -32767.0, 32767.0)
            if verbose:
                self.logger.debug(f"wav shape: {wav.shape}, min: {wav.min():.4f}, max: {wav.max():.4f}")
            # wavs.append(wav[:, :-512])
            wavs.append(wav.cpu())  # to cpu before saving


        end_time = time.perf_counter()
        self._set_gr_progress(0.9, "save audio...")
        wavs = self.insert_interval_silence(wavs, sampling_rate=sampling_rate, interval_silence=interval_silence)
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        total_inference_time = end_time - start_time
        rtf = total_inference_time / wav_length if wav_length > 0 else float("inf")
        
        # Print time statistics with beautiful table
        time_stats = {
            "GPT Generation": gpt_gen_time,
            "GPT Forward": gpt_forward_time,
            "S2Mel": s2mel_time,
            "BigVGAN": bigvgan_time,
        }
        
        inference_stats = {
            "total_inference_time": total_inference_time,
            "wav_length": wav_length,
            "rtf": rtf,
            "token_generation_time": token_generation_time,
            "gpt_gen_time": gpt_gen_time,
            "gpt_forward_time": gpt_forward_time,
            "s2mel_time": s2mel_time,
            "bigvgan_time": bigvgan_time,
        }
        self.logger.print_time_stats(time_stats, total_inference_time, wav_length)

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            # Write the audio to the requested path.
            if os.path.isfile(output_path):
                os.remove(output_path)
                self.logger.debug(f"remove old wav file: {output_path}")
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            self.logger.success(f"wav file saved to: {output_path}")
            if return_stats:
                return output_path, seg_lens, wav_length, inference_stats
            return output_path, seg_lens, wav_length
        else:
            # Return raw samples in the format expected by Gradio.
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            if return_stats:
                return (sampling_rate, wav_data), inference_stats
            return (sampling_rate, wav_data)


def find_most_similar_cosine(query_vector, matrix):
    query_vector = query_vector.float()
    matrix = matrix.float()

    similarities = F.cosine_similarity(query_vector, matrix, dim=1)
    most_similar_index = torch.argmax(similarities)
    return most_similar_index

class QwenEmotion:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.logger = get_logger("QwenEmotion")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            torch_dtype="float16",  # "auto"
            device_map="auto"
        )
        self.prompt = "文本情感分类"
        self.cn_key_to_en = {
            "高兴": "happy",
            "愤怒": "angry",
            "悲伤": "sad",
            "恐惧": "afraid",
            "反感": "disgusted",
            # TODO: the "低落" (melancholic) emotion will always be mapped to
            # "悲伤" (sad) by QwenEmotion's text analysis. it doesn't know the
            # difference between those emotions even if user writes exact words.
            # SEE: `self.melancholic_words` for current workaround.
            "低落": "melancholic",
            "惊讶": "surprised",
            "自然": "calm",
        }
        self.desired_vector_order = ["高兴", "愤怒", "悲伤", "恐惧", "反感", "低落", "惊讶", "自然"]
        self.melancholic_words = {
            # emotion text phrases that will force QwenEmotion's "悲伤" (sad) detection
            # to become "低落" (melancholic) instead, to fix limitations mentioned above.
            "低落",
            "melancholy",
            "melancholic",
            "depression",
            "depressed",
            "gloomy",
        }
        self.max_score = 1.2
        self.min_score = 0.0

    def clamp_score(self, value):
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                return 0.0
        if not isinstance(value, (int, float)):
            return 0.0
        return max(self.min_score, min(self.max_score, value))

    def convert(self, content):
        # generate emotion vector dictionary:
        # - insert values in desired order (Python 3.7+ `dict` remembers insertion order)
        # - convert Chinese keys to English
        # - clamp all values to the allowed min/max range
        # - use 0.0 for any values that were missing in `content`
        emotion_dict = {
            self.cn_key_to_en[cn_key]: self.clamp_score(content.get(cn_key, 0.0))
            for cn_key in self.desired_vector_order
        }

        # default to a calm/neutral voice if all emotion vectors were empty
        if all(val <= 0.0 for val in emotion_dict.values()):
            self.logger.info("no emotions detected; using default calm/neutral voice")
            emotion_dict["calm"] = 1.0

        return emotion_dict

    def inference(self, text_input):
        start = time.time()
        messages = [
            {"role": "system", "content": f"{self.prompt}"},
            {"role": "user", "content": f"{text_input}"}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=2048,
            pad_token_id=self.tokenizer.eos_token_id
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True)

        # decode the JSON emotion detections as a dictionary
        try:
            content = json.loads(content)
        except json.decoder.JSONDecodeError:
            # invalid JSON; fallback to manual string parsing
            # print(">> parsing QwenEmotion response", content)
            content = {
                m.group(1): float(m.group(2))
                for m in re.finditer(r'([^\s":.,]+?)"?\s*:\s*([\d.]+)', content)
            }
            # print(">> dict result", content)

        # workaround for QwenEmotion's inability to distinguish "悲伤" (sad) vs "低落" (melancholic).
        # if we detect any of the IndexTTS "melancholic" words, we swap those vectors
        # to encode the "sad" emotion as "melancholic" (instead of sadness).
        text_input_lower = text_input.lower()
        if any(word in text_input_lower for word in self.melancholic_words):
            # print(">> before vec swap", content)
            content["悲伤"], content["低落"] = content.get("低落", 0.0), content.get("悲伤", 0.0)
            # print(">>  after vec swap", content)

        return self.convert(content)


if __name__ == "__main__":
    prompt_wav = "XX/voice_01.wav"
    text = '欢迎大家来体验TED-TTS，并给予我们意见与反馈，谢谢大家。'

    tts = IndexTTS2(cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_cuda_kernel=False)
    tts.infer(spk_audio_prompt=prompt_wav, text=text, output_path="gen.wav", verbose=True)
