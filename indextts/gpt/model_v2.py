import functools
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import LogitsProcessor
from transformers import GPT2Config, LogitsProcessorList
from indextts.gpt.transformers_gpt2 import GPT2PreTrainedModel, GPT2Model

# from transformers import GPT2Config, GPT2PreTrainedModel, LogitsProcessorList
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.utils.model_parallel_utils import (assert_device_map,
                                                     get_device_map)

from indextts.gpt.conformer_encoder import ConformerEncoder
from indextts.gpt.perceiver import PerceiverResampler
from indextts.utils.arch_util import AttentionBlock
from indextts.utils.typical_sampling import TypicalLogitsWarper
from indextts.utils.logger import get_logger

from .attn_map import AttentionMapProcessor
from .hmm import StreamingHMMAligner
from .transformers_generation_utils import construct_attn_mask

# Initialize logger
_logger = get_logger("GPT")

def null_position_embeddings(range, dim):
    return torch.zeros((range.shape[0], range.shape[1], dim), device=range.device)

class RemainingBudgetEOSProcessor(LogitsProcessor):
    """Per-segment EOS bias processor (final-segment driven).

    1. Tracks which segment each beam is currently in.
    2. Computes per-segment progress against that segment's own target.
    3. Final-segment policy:
         - non-final segments: always maximally suppress EOS (max_negative_bias)
         - final segment:      bias EOS dynamically according to progress
    4. Errors accumulated in earlier segments do not bleed into later segments.
    """

    def __init__(
        self,
        target_tokens_per_segment: list,  # per-segment target length, e.g. [100, 100, 150, ...]
        stop_token_id: int,
        verbose: bool = True,
        # Duration control hyperparameters
        min_ratio: float = 0.5,
        neutral_ratio: tuple = (0.8, 1.2),
        max_ratio: float = 2.0,
        max_negative_bias: float = -5.0,
        max_positive_bias: float = 10.0,
    ):
        self.target_tokens_per_segment = target_tokens_per_segment
        self.num_segments = len(target_tokens_per_segment)
        self.stop_token_id = stop_token_id
        self.verbose = verbose

        self.min_ratio = min_ratio
        self.neutral_start_ratio = neutral_ratio[0]
        self.neutral_end_ratio = neutral_ratio[1]
        self.max_ratio = max_ratio
        self.max_negative_bias = max_negative_bias
        self.max_positive_bias = max_positive_bias

        # Validate
        assert 0.0 < min_ratio < neutral_ratio[0] < neutral_ratio[1] < max_ratio

        self.initial_offset = None  # number of prefix tokens (condition + text)

        self.current_segment_idx = None  # will be [0, 0, ..., 0] for each beam

        # actual_generated_per_segment[beam_id][seg_idx] = tokens actually generated
        self.actual_generated_per_segment = None

        # Token count generated within the current segment.
        self.generated_since_last_segment = None

        # Segment-boundary positions supplied externally.
        self.segment_positions = None
        
        if self.verbose:
            _logger.info("[RemainingBudgetEOS] Initialized")
            _logger.debug(f"  Num segments: {self.num_segments}")
            _logger.debug(f"  Target per segment: {self.target_tokens_per_segment}")
            _logger.debug(f"  Total target: {sum(self.target_tokens_per_segment)}")
            _logger.debug(f"  Control ratios: min={min_ratio}, neutral={neutral_ratio}, max={max_ratio}")
            _logger.debug(f"  Bias range: [{max_negative_bias}, {max_positive_bias}]")
    
    def _initialize_state(self, batch_size: int):
        self.current_segment_idx = [0] * batch_size
        self.generated_since_last_segment = [0] * batch_size
        self.actual_generated_per_segment = [
            [0] * self.num_segments for _ in range(batch_size)
        ]
    
    def update_segment_positions(self, segment_positions: dict):
        """Externally supply the segment-switch positions.

        Args:
            segment_positions: {beam_id: [pos1, pos2, ...]} semantic-token
                positions at which each beam switches segments.
        """
        self.segment_positions = segment_positions

    def _detect_segment_switch(self, current_length: int, beam_idx: int) -> bool:
        """Return True if the given beam has crossed its next segment boundary.

        Args:
            current_length: tokens generated so far for this beam.
            beam_idx:       beam index.
        """
        if self.segment_positions is None:
            return False

        if beam_idx not in self.segment_positions:
            return False

        positions = self.segment_positions[beam_idx]
        current_seg = self.current_segment_idx[beam_idx]

        # Check whether the next segment switch point has been reached.
        if current_seg < len(positions):
            switch_pos = positions[current_seg]
            if current_length >= switch_pos:
                return True

        return False

    def _handle_segment_switch(self, beam_idx: int):
        """Handle a segment-switch event for one beam."""
        old_seg = self.current_segment_idx[beam_idx]

        # Record how many tokens this beam produced for the segment just left.
        self.actual_generated_per_segment[beam_idx][old_seg] = \
            self.generated_since_last_segment[beam_idx]

        # Advance to the next segment.
        self.current_segment_idx[beam_idx] += 1
        self.generated_since_last_segment[beam_idx] = 0
        
        if self.verbose:
            actual_len = self.actual_generated_per_segment[beam_idx][old_seg]
            target_len = self.target_tokens_per_segment[old_seg]
            ratio = actual_len / target_len if target_len > 0 else 0
            _logger.debug(
                f"[RemainingBudgetEOS] Beam {beam_idx}: "
                f"Segment {old_seg} -> {self.current_segment_idx[beam_idx]} | "
                f"Actual: {actual_len} / Target: {target_len} ({ratio:.2f}x)"
            )
    
    def _get_current_segment_target(self, beam_idx: int) -> int:
        """Return the token target for the segment currently active on this beam."""
        current_seg = self.current_segment_idx[beam_idx]

        # Past the last segment? Reuse the final segment's target.
        if current_seg >= self.num_segments:
            return self.target_tokens_per_segment[-1]

        return self.target_tokens_per_segment[current_seg]

    def _compute_bias(self, progress: float, is_last_segment: bool) -> float:
        """Compute the EOS bias from current progress.

        Uses a piecewise-linear schedule:
          - non-final segment: always max_negative_bias (strong EOS suppression).
          - final segment:     bias depends on progress
              - progress < min_ratio                  -> strong suppression
              - min_ratio  .. neutral_start           -> ramp suppression down to 0
              - neutral_start .. neutral_end          -> zero bias (let the model decide)
              - neutral_end   .. max_ratio            -> ramp EOS encouragement up
              - progress > max_ratio                  -> strong encouragement

        Args:
            progress:        per-segment ratio (generated / target).
            is_last_segment: whether the active segment is the final one.

        Returns:
            EOS logit bias.
        """
        if not is_last_segment:
            return self.max_negative_bias

        if progress < self.min_ratio:
            return self.max_negative_bias

        elif progress < self.neutral_start_ratio:
            # Linear ramp from max_negative_bias to 0.
            t = (progress - self.min_ratio) / (self.neutral_start_ratio - self.min_ratio)
            return self.max_negative_bias * (1.0 - t)
        
        elif progress < self.neutral_end_ratio:
            return 0.0
        
        elif progress < self.max_ratio:
            t = (progress - self.neutral_end_ratio) / (self.max_ratio - self.neutral_end_ratio)
            return self.max_positive_bias * t
        
        else:
            return self.max_positive_bias
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Apply the EOS bias to the logits.

        Args:
            input_ids: [batch_size, seq_len].
            scores:    [batch_size, vocab_size].

        Returns:
            Modified scores.
        """
        batch_size = input_ids.shape[0]
        
        if self.initial_offset is None:
            self.initial_offset = input_ids.shape[1]
            self._initialize_state(batch_size)
            
            if self.verbose:
                _logger.debug(f"[RemainingBudgetEOS] Detected offset: {self.initial_offset} tokens")
        
        total_length = input_ids.shape[1]
        
        for beam_idx in range(batch_size):
            current_length = total_length - self.initial_offset
            
            if self._detect_segment_switch(current_length, beam_idx):
                self._handle_segment_switch(beam_idx)
            
            self.generated_since_last_segment[beam_idx] = (
                current_length 
                - sum(self.actual_generated_per_segment[beam_idx])
            )
            
            current_seg = self.current_segment_idx[beam_idx]
            
            # Per-segment progress is computed against this segment's own target.
            current_seg_target = self._get_current_segment_target(beam_idx)
            generated_in_current = self.generated_since_last_segment[beam_idx]

            # progress = tokens generated in current segment / target of the current segment.
            progress = generated_in_current / max(current_seg_target, 1e-6)
            
            # Clamp to reasonable range
            progress = max(0.0, min(progress, 2.0))
            
            is_last_segment = (current_seg >= self.num_segments - 1)
            
            bias = self._compute_bias(progress, is_last_segment)
            
            scores[beam_idx, self.stop_token_id] += bias
            
            if self.verbose and current_length % 50 == 0:
                seg_status = "LAST" if is_last_segment else f"{current_seg}"
                _logger.debug(
                    f"[RemainingBudgetEOS] Beam {beam_idx} | "
                    f"Seg {seg_status}/{self.num_segments} | "
                    f"Generated: {generated_in_current} / Target: {current_seg_target} | "
                    f"Progress: {progress:.2f} | "
                    f"Bias: {bias:+.2f}"
                )
        
        return scores

class ResBlock(nn.Module):
    """
    Basic residual convolutional block that uses GroupNorm.
    """

    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan // 8, chan),
            nn.ReLU(),
            nn.Conv1d(chan, chan, kernel_size=3, padding=1),
            nn.GroupNorm(chan // 8, chan)
        )

    def forward(self, x):
        return F.relu(self.net(x) + x)


class GPT2InferenceModel(GPT2PreTrainedModel):
    def __init__(self, config, gpt, text_pos_emb, embeddings, norm, linear, kv_cache=False):
        super().__init__(config)
        # Note: the argument named `text_pos_emb` here actually represents the mel position embedding
        self.transformer = gpt
        self.text_pos_embedding = text_pos_emb
        self.embeddings = embeddings
        self.final_norm = norm
        self.lm_head = nn.Sequential(norm, linear)
        self.kv_cache = kv_cache

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.cached_mel_emb = None
        self.last_attention_map = None
        
        # attn cache
        self.past_attn_cache = None
        self.past_attn_pos = None
        # MAS
        self.mas_mu = None
        # HMM
        self.hmm = None
        self.segment_positions = None  # recorded segment-switch positions
        self.duration_processor = None  # reference to RemainingBudgetEOSProcessor
        self.duration_controller = None  # reference to IntraSegmentDurationController
        self.trunc_index = None  # number of prefix tokens

    def update_duration_embedding(
        self,
        new_duration_emb: torch.Tensor,
        duration_emb_position: int = -1,
    ):
        """Dynamically replace the duration embedding inside cached_mel_emb.

        Args:
            new_duration_emb: [batch, D] or [D] replacement duration embedding.
            duration_emb_position: position in the conditioning sequence
                (-1 means the last slot).
        """
        self.cached_mel_emb[:, duration_emb_position, :] = new_duration_emb.squeeze(1)

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(max(1, torch.cuda.device_count())))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def store_mel_emb(self, mel_emb):
        self.cached_mel_emb = mel_emb

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        attention_phase = kwargs.get("attention_phase", None)
        phases_attn_mask_ids = kwargs.get("tokenwise_attention_mask", None)
        input_full_attention_mask = kwargs.get("input_full_attention_mask", False)
        input_attention_masks = kwargs.get("input_attention_masks", None)
        dynamic_cond_mask_idx = kwargs.get("dynamic_cond_mask_idx", None)
        trunc_index = input_attention_masks.shape[0] if input_attention_masks is not None else 0
        
        is_decoding = past_key_values is not None
        
        seq_len = input_ids.shape[1]
        device = input_ids.device
        mask_dtype = next(self.parameters()).dtype
        
        # Vectorized handling of attention_phase and attention_masks (avoids per-beam loops).
        if attention_phase is not None:
            attention_masks = kwargs.get("attention_masks")
            attention_masks = torch.stack(attention_masks, dim=0).to(device)  # [B, 1, seq_len]
            
            if isinstance(attention_phase, list):
                attention_phase = torch.tensor(attention_phase, device=device)  # [B, ]
            
            selected_attention_mask = attention_masks[attention_phase]  # [B, 1, seq_len]
            # Drop the singleton dim to get [B, seq_len].
            selected_attention_mask = selected_attention_mask.squeeze(1)  # [B, seq_len]
            
            attention_mask = F.pad(selected_attention_mask, (0, seq_len - selected_attention_mask.shape[1]), value=1)
            kwargs["attention_mask"] = attention_mask
            # Pad the attention mask on the right with 1s up to len(input_ids).
            
            if input_full_attention_mask==False:
                # kwargs["_4d_attention_mask"] = construct_attn_mask(
                #     attention_masks, phases_attn_mask_ids, trunc_index, input_attention_masks if input_full_attention_mask==False else None, dynamic_cond_mask_idx, attention_phase, device, mask_dtype
                # )
                kwargs["_4d_attention_mask"] = self.construct_attn_mask_opt(
                    phase_1dim_attn_masks=attention_masks,
                    phases_attn_mask_ids=phases_attn_mask_ids,
                    trunc_index=trunc_index,
                    input_attention_masks=input_attention_masks if input_full_attention_mask==False else None,
                    dynamic_cond_mask_idx=dynamic_cond_mask_idx,
                    current_phases=attention_phase,
                    device=device,
                    mask_dtype=mask_dtype,
                    is_decoding=is_decoding,
                )
                kwargs["attention_mask"] = torch.ones(input_ids.shape, device=device, dtype=mask_dtype)
            

        token_type_ids = kwargs.get("token_type_ids", None)  # usually None
        if not self.kv_cache:
            past_key_values = None
            
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
            
            if "_4d_attention_mask" in kwargs:
                _4d_attention_mask = kwargs["_4d_attention_mask"]
                kwargs["_4d_attention_mask"] = _4d_attention_mask[:, :, -1:, :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
            
        output_path = kwargs.get("output_path", None)

        # ============ Intra-Segment Duration Control ============
        if self.duration_controller is not None and is_decoding:
            ctrl = self.duration_controller
            text_pos = 0
            if self.hmm is not None and hasattr(self.hmm, "last_center"):
                hmm_center = self.hmm.last_center
                if hmm_center is not None and len(hmm_center) > 0:
                    text_pos = int(hmm_center[0].item())

            attention_phase_val = kwargs.get("attention_phase", None)
            if attention_phase_val is not None:
                current_seg = attention_phase_val[0] if isinstance(attention_phase_val, (list, torch.Tensor)) else attention_phase_val
                if isinstance(current_seg, torch.Tensor):
                    current_seg = current_seg.item()
                if current_seg != ctrl.current_segment_idx:
                    ctrl.switch_to_segment(current_seg)

            try:
                prev_t = ctrl._cached_t
                _, _, new_dur_emb = ctrl.step(text_pos)
                # Only touch the cache when the embedding actually changes (avoids redundant VRAM writes).
                if ctrl._cached_t != prev_t:
                    self.update_duration_embedding(new_dur_emb, duration_emb_position=-1)
            except Exception as e:
                if ctrl.total_steps_generated % 50 == 0:
                    _logger.warning(f"[IntraSegDurCtrl] Update failed: {e}")
        # ============ End Intra-Segment Duration Control ============

        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "_4d_attention_mask": kwargs.get("_4d_attention_mask", None),
            "attention_phase": attention_phase,
            "output_path": output_path,
        }

    def construct_attn_mask_opt(
        self,
        phase_1dim_attn_masks,  
        phases_attn_mask_ids, 
        trunc_index, 
        input_attention_masks, 
        dynamic_cond_mask_idx, 
        current_phases, 
        device, 
        mask_dtype,
        is_decoding=False  # indicates whether we are in the decoding step
    ):
        """Construct the 4D attention mask (optimized variant).

        Functionally identical to construct_attn_mask, but during the decoding
        step only the final row is computed, which avoids rebuilding the full
        [output_len, output_len] matrix every step.

        Original logic recap:
          1. Build attention_mask [1, 1, output_len, output_len] where each
             row is the visibility vector of the corresponding token.
          2. Build an upper-triangular causal_mask (above the diagonal is
             min_dtype, the rest is 0).
          3. padding_mask = (causal_mask + attention_mask) == 0,
             i.e. positions allowed by causal AND zero in attention must be
             additionally masked out.
          4. causal_mask.masked_fill(padding_mask, min_dtype).

        For the last row (the only row required during decoding):
          causal_mask's last row is all zeros (the last token may attend to
          every prior token), so padding_mask[-1] = (attention_mask[-1] == 0).
          The final mask[-1] just fills min_dtype where attention_mask is 0,
          and 0 elsewhere.

        This matches the prefill logic (the last row's causal portion is
        already all zeros).

        Args:
            phase_1dim_attn_masks (`Tensor`): attention masks for each phase,
                shape [num_phases, 1, mask_len].
            phases_attn_mask_ids (`List[List[int]]`): mask_id assigned to each
                generated token (i.e. tokenwise_attention_mask).
            trunc_index (`int`): number of prefix tokens (prompt length).
            input_attention_masks (`Tensor` or `None`): attention masks for the
                prefix tokens, shape [trunc_index, mask_len].
            dynamic_cond_mask_idx (`List[int]` or `None`): positions whose mask
                should be swapped at runtime for the current phase mask.
            current_phases (`Tensor`): attention phase per sample in the batch.
            device (`torch.device`): tensor device.
            mask_dtype (`torch.dtype`): mask dtype.
            is_decoding (`bool`): when True, only compute the last row.

        Returns:
            `Tensor`: 4D attention mask.
                - Prefill: [batch_size, 1, output_len, output_len]
                - Decoding: [batch_size, 1, 1, output_len]
        """
        min_dtype = torch.finfo(mask_dtype).min
        batch_size = len(phases_attn_mask_ids)

        if dynamic_cond_mask_idx is not None and isinstance(dynamic_cond_mask_idx, torch.Tensor):
            dynamic_cond_mask_idx = dynamic_cond_mask_idx.tolist()

        if is_decoding:
            # ==========================================
            # Decoding step: only compute the last row of the mask.
            # ==========================================
            # The last row of causal_mask is all zeros (the new token can see
            # every prior token), so the final mask is min_dtype where
            # attention_mask == 0 and 0 elsewhere.
            # This matches what the prefill path would yield at [-1:, :].
            output_lens = torch.as_tensor(
                [len(mask_ids) + trunc_index for mask_ids in phases_attn_mask_ids],
                device=device,
                dtype=torch.long,
            )
            if not torch.equal(output_lens, output_lens[:1].expand_as(output_lens)):
                raise ValueError(
                    "Decoding expects equal output_len across batch for stacking 4D mask."
                )
            output_len = int(output_lens[0].item())
            last_row_indices = output_lens - 1

            last_mask_ids = torch.as_tensor(
                [mask_ids[-1] for mask_ids in phases_attn_mask_ids],
                device=device,
                dtype=torch.long,
            )

            if isinstance(current_phases, torch.Tensor):
                current_phases_t = current_phases.to(device=device, dtype=torch.long)
            else:
                current_phases_t = torch.as_tensor(
                    current_phases, device=device, dtype=torch.long
                )

            if dynamic_cond_mask_idx is not None:
                dyn_idx = torch.as_tensor(dynamic_cond_mask_idx, device=device, dtype=torch.long)
                use_dynamic = torch.isin(last_row_indices, dyn_idx)
                mask_ids = torch.where(use_dynamic, current_phases_t, last_mask_ids)
            else:
                mask_ids = last_mask_ids

            mask_bank = phase_1dim_attn_masks
            if mask_bank.dim() == 3 and mask_bank.shape[1] == 1:
                mask_bank = mask_bank.squeeze(1)  # [num_phases, mask_len]

            row_mask = mask_bank.index_select(0, mask_ids)
            if row_mask.shape[1] < output_len:
                row_mask = F.pad(row_mask, (0, output_len - row_mask.shape[1]), value=1)
            else:
                row_mask = row_mask[:, :output_len]

            final_rows = torch.zeros((batch_size, output_len), dtype=mask_dtype, device=device)
            final_rows.masked_fill_(row_mask == 0, min_dtype)

            # [batch_size, output_len] -> [batch_size, 1, 1, output_len]
            return final_rows.unsqueeze(1).unsqueeze(1)

        else:
            # ==========================================
            # Prefill step: build the full matrix (matches original logic).
            # ==========================================
            causal_masks = []
            for i in range(batch_size):
                output_len = len(phases_attn_mask_ids[i]) + trunc_index

                # 1. Build each row of the attention_mask matrix.
                if input_attention_masks is None:
                    attention_mask_list = [
                        torch.ones((1, output_len), dtype=torch.long, device=device)
                    ] * trunc_index
                else:
                    # In the original code, the inner list-comprehension `i`
                    # shadows the outer loop variable, so it effectively reads
                    # input_attention_masks[0..trunc_index-1]. Shape is
                    # [trunc_index, mask_len]; pad each row.
                    attention_mask_list = [
                        F.pad(
                            input_attention_masks[k],
                            (0, output_len - input_attention_masks[k].shape[0]),
                            value=1,
                        ).unsqueeze(0)
                        for k in range(trunc_index)
                    ]

                # 2. Append rows for each generated token.
                for mask_id in phases_attn_mask_ids[i]:
                    mask_vec = phase_1dim_attn_masks[mask_id]
                    if mask_vec.dim() == 1:
                        mask_vec = mask_vec.unsqueeze(0)
                    attention_mask_list.append(
                        F.pad(mask_vec, (0, output_len - mask_vec.shape[1]), value=1)
                    )

                # 3. Stack into a [1, 1, output_len, output_len] tensor.
                attention_mask = torch.cat(attention_mask_list, dim=0).unsqueeze(0).unsqueeze(0)

                # 4. Row substitution for dynamic_cond_mask_idx.
                if dynamic_cond_mask_idx is not None:
                    for j in dynamic_cond_mask_idx:
                        phase_mask = phase_1dim_attn_masks[current_phases[i]]
                        if phase_mask.dim() == 1:
                            phase_mask = phase_mask.unsqueeze(0)
                        attention_mask[0, 0, j, :] = F.pad(
                            phase_mask,
                            (0, output_len - phase_mask.shape[-1]),
                            value=1,
                        )

                # 5. Build the causal_mask and combine it.
                causal_mask = torch.full(
                    (output_len, output_len), fill_value=min_dtype, dtype=mask_dtype, device=device
                )
                causal_mask = torch.triu(causal_mask, diagonal=1)
                causal_mask = causal_mask[None, None, :, :]
                # Original logic: padding_mask = (causal_mask + attention_mask) == 0
                padding_mask = (causal_mask + attention_mask) == 0
                causal_mask = causal_mask.masked_fill(padding_mask, min_dtype)
                causal_masks.append(causal_mask)

            return torch.cat(causal_masks, dim=0)
        
    def detect_emotion_twist(self, 
                        next_indices, 
                        next_tokens, 
                        emotion_final_phase, 
                        eos_token_id,
                        output_attentions,
                        **model_kwargs):
        """Detect whether an emotion twist (segment switch) occurred this step.

        Args:
            next_indices (torch.Tensor): per-beam indices for the next step.
            next_tokens (torch.Tensor): per-beam token ids for the next step.
            emotion_final_phase (int): index of the final emotion phase.
            eos_token_id (list): list of stop-token ids.
            method (str): twist-detection method.
                - `eos`:      decide via the EOS token position.
                - `hmm`:      decide via HMM alignment.
                - `max_head`: ablation; pick the head with the largest score.
                - `max_topk`: ablation; pick the top-k heads.
                - `mas`:      monotonic alignment search (greedy).
            output_attentions (`tuple(torch.FloatTensor)`): attention weights
                returned by the model.

        Returns:
            is_emotion_twist (`bool`): whether a twist was detected.
        """
        method = model_kwargs.get("method", "")
        if method == "eos":
            indice_array = next_indices.detach().cpu().numpy()[0]
            token_array = next_tokens.detach().cpu().numpy()[0]
            # Positions in token_array that hold an EOS token -> map back to beam indices.
            eos_positions = [i for i, token_id in enumerate(token_array) if token_id in eos_token_id]
            emotion_end_indices = indice_array[eos_positions]
            # Among those, keep beams whose phase has not yet reached emotion_final_phase.
            emotion_twist_indices = [i for i in emotion_end_indices if model_kwargs["attention_phase"][i] < emotion_final_phase]
            is_emotion_twist = len(emotion_twist_indices) > 0
            for i in emotion_twist_indices:
                model_kwargs["attention_phase"][i] += 1
                _logger.debug(f"[eos] Beam {i} enters phase {model_kwargs['attention_phase'][i]}")
            return is_emotion_twist
        elif method == "hmm":
            text_last_token_position = model_kwargs.get('text_last_token_position', None)
            text_bos_idx = text_last_token_position[0]
            text_eos_idx = text_last_token_position[1][-1] + 1

            # Perf note: avoid materializing the full [L, B, H, T, T] all_attn before slicing.
            # Slicing per-layer first and stacking afterwards cuts memory bandwidth significantly.
            all_layer_attn = torch.stack(
                [layer_attn[:, :, -1, text_bos_idx:text_eos_idx] for layer_attn in output_attentions],
                dim=0,
            )  # [L, B, H, Sk]
            current_total_len = output_attentions[0].shape[-1]
            
            L, B, H, Sk = all_layer_attn.shape  # L: layers, H: heads, B: beams, Sk: text_len

            # # ==== Per-layer "best head" pick (legacy/ablation, disabled) ====
            # # Score by std along the text axis.
            # max_head_attn = torch.std(all_layer_attn, dim=-1)  # [L, B, H, ]
            # max_head = max_head_attn.argmax(dim=-1)  # [L, B]
            # # Gather the resulting attention.
            # attn = all_layer_attn.permute(1, 0, 2, 3)  # [B, L, H, Sk ]
            # idx = max_head.permute(1, 0).unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
            # idx = idx.expand(-1, -1, 1, attn.size(-1))  # [B, L, 1, Sk]
            # attn_seperate = attn.gather(2, idx).squeeze(2)  # [B, L, Sk]

            # # Top-1: rank each head by its mean attention.
            # head_importance = torch.mean(attn_seperate, dim=-1)  # [B, H]
            # # Select the head with the highest score.
            # max_head = head_importance.argmax(dim=-1)  # [B, ]
            # # Gather the corresponding attention vector.
            # max_head = max_head.view(-1, 1, 1).expand(B, 1, Sk)  # [B, 1, Sk]
            # attn_wo_head = attn_seperate.gather(1, max_head).squeeze(1)  # [B, Sk]

            # # ==== Optional visualization hook ====

            # # TODO: log the emotion-twist artifacts for offline analysis.
            # self.attn_map_processor.process_emotion_twist_detect(
            #     method=method,
            #     output_path=model_kwargs.get("output_path", None),
            #     text_last_token_position=model_kwargs.get('text_last_token_position', None),
            # )

            # NOTE: visualization above is auxiliary; the core logic follows below.
            
            attention_phase = model_kwargs.get("attention_phase", None)
            text_last_position_array = model_kwargs.get("_text_last_position_array")
            if text_last_position_array is None:
                text_last_position_array = torch.as_tensor(
                    text_last_token_position[1] - text_last_token_position[0],
                    device=all_layer_attn.device,
                )  # [B, ]
                model_kwargs["_text_last_position_array"] = text_last_position_array
            text_last_pos_per_beam = text_last_position_array[attention_phase]  # [B, ]
            
            
            if self.hmm is None:
                self.hmm = StreamingHMMAligner(num_beams=B, num_text_tokens=Sk, device=all_layer_attn.device, enable_std_head_prune=False)
            
            # Pass the processor only when attention-map recording is requested.
            processor = self.attn_map_processor if model_kwargs.get("save_attention_maps", False) else None
            hmm_center, hmm_idx = self.hmm.step(all_layer_attn, attn_map_processor=processor, **model_kwargs)
            # Stash hmm_center so IntraSegmentDurationController can read it.
            self.hmm.last_center = hmm_center
            suspect_twist_pos = (hmm_center >= text_last_pos_per_beam + 0.5)  # bool tensor
            if suspect_twist_pos.any():
                indices = torch.where(suspect_twist_pos)[0]
                for i in indices:
                    i = i.item()
                    
                    # Record the segment-switch position (in semantic tokens, prefix stripped).
                    trunc_index = self.trunc_index  # prefix token count, stored on self
                    current_sem_len = current_total_len - trunc_index  # semantic-token position

                    # Use attention_phase as the segment index.
                    if self.segment_positions is None:
                        self.segment_positions = dict()
                    if i not in self.segment_positions:
                        self.segment_positions[i] = [current_sem_len]
                    else:
                        if len(self.segment_positions[i]) > model_kwargs["attention_phase"][i]:
                            # The switch position for this segment was already recorded; refresh it.
                            self.segment_positions[i][model_kwargs["attention_phase"][i]] = current_sem_len
                        else:
                            self.segment_positions[i].append(current_sem_len)
                    
                    model_kwargs["attention_phase"][i] += 1
                    _logger.debug(f"[hmm] Beam {i} enters phase {model_kwargs['attention_phase'][i]} at sem {current_total_len} ")
                   
                # ===== Notify duration_processor of the new segment boundaries. =====
                if self.duration_processor is not None:
                    self.duration_processor.update_segment_positions(self.segment_positions)
                return True
            return False
        elif method == "max_head":
            all_attn = torch.stack(output_attentions, dim=0)  # [num_layers, beamsize, num_heads, seq_len, seq_len]

            text_last_token_position = model_kwargs.get('text_last_token_position', (34, [34]))
            text_last_token_position = model_kwargs.get('text_last_token_position', None)
            text_bos_idx = text_last_token_position[0]
            text_eos_idx = text_last_token_position[1][-1] + 1
            all_layer_attn = all_attn[:, :, :, -1, text_bos_idx:text_eos_idx] # [num_layers, beamsize, num_heads, seq_len]; last token only, text region only

            L, B, H, Sk = all_layer_attn.shape  # L: layers, H: heads, B: beams, Sk: text_len

            # ==== Pick the dominant head ====
            # 1. Per-(layer, head, beam) score (mean over the text axis).
            # Result shape: [L, B, H].
            attn_std = torch.mean(all_layer_attn, dim=-1)

            # 2. Reshape so beams come first and (layer, head) is flattened.
            # [L, B, H] -> [B, L, H]
            attn_std = attn_std.permute(1, 0, 2)
            # [B, L, H] -> [B, L*H]
            attn_std_flat = attn_std.reshape(B, -1)

            # 3. For each beam, find the index of the L*H head with the highest score.
            best_head_idx = attn_std_flat.argmax(dim=-1)  # [B, ]

            # 4. Gather the corresponding attention vectors.
            # Reshape the raw attention data to [B, L*H, Sk] for gather.
            attn_data = all_layer_attn.permute(1, 0, 2, 3).reshape(B, L*H, Sk)

            # Build the gather index: [B, 1, Sk].
            gather_idx = best_head_idx.view(B, 1, 1).expand(-1, -1, Sk)

            # Extract [B, 1, Sk] -> squeeze -> [B, Sk].
            attn_wo_head = attn_data.gather(1, gather_idx).squeeze(1)
            
            # Select the highest-attention text token for each beam.
            # token_idx = attn_wo_head.argmax(dim=-1)  # [B, ]
            # token_value = attn_wo_head.gather(-1, token_idx.unsqueeze(-1)).squeeze(-1)  # [B, ]
            token_idx, token_value = self._update_attn_monotonic(attn_wo_head)

            # Prepare per-beam comparison values.
            attention_phase = model_kwargs.get("attention_phase", None)
            text_last_position_array = torch.tensor(
                text_last_token_position[1] - text_last_token_position[0],
                device=attn_wo_head.device)  # [B, ]
            text_last_pos_per_beam = text_last_position_array[attention_phase]  # [B, ]
            
            # TODO: emit emotion-twist artifacts for offline analysis.
            # self.attn_map_processor.process_emotion_twist_detect(
            #     attn_wo_head=attn_wo_head,
            #     method=method,
            #     token_idx=token_idx,
            #     attention_phase=attention_phase,
            #     output_path=model_kwargs.get("output_path", None),
            #     text_last_token_position=model_kwargs.get('text_last_token_position', None),
            # )
            
            # 1. Pull the position from the previous step (handle step 0 / empty history).
            if self.past_attn_pos is not None and len(self.past_attn_pos) > 0:
                prev_token_idx = self.past_attn_pos[-1]
            else:
                # No history yet (first step). Seed with a sentinel that can't match the condition.
                prev_token_idx = torch.full_like(token_idx, -100)
            
            self._update_attn_cache(token_value, token_idx)
            
            suspect_twist_pos = (token_idx > text_last_pos_per_beam) & \
                                (prev_token_idx == token_idx - 1)
                                # bool tensor: alignment passed the last text token
            if suspect_twist_pos.any():
                # Run the emotion-twist comparison.
                indices = torch.where(suspect_twist_pos)[0]
                for i in indices:
                    i = i.item()

                    # Record the segment-switch position (in semantic tokens, prefix stripped).
                    current_total_len = all_attn.shape[-1]  # total length, including prefix
                    trunc_index = self.trunc_index  # prefix token count, stored on self
                    current_sem_len = current_total_len - trunc_index  # semantic-token position

                    # Use attention_phase as the segment index.
                    if self.segment_positions is None:
                        self.segment_positions = dict()
                    if i not in self.segment_positions:
                        self.segment_positions[i] = [current_sem_len]
                    else:
                        if len(self.segment_positions[i]) > model_kwargs["attention_phase"][i]:
                            # The switch position for this segment was already recorded; refresh it.
                            self.segment_positions[i][model_kwargs["attention_phase"][i]] = current_sem_len
                        else:
                            self.segment_positions[i].append(current_sem_len)
                    
                    model_kwargs["attention_phase"][i] += 1
                    _logger.debug(f"[Max head] Beam {i} enters phase {model_kwargs['attention_phase'][i]} at sem {all_attn.shape[-1]} ")
                # ===== Notify duration_processor of the new segment boundaries. =====
                if self.duration_processor is not None:
                    self.duration_processor.update_segment_positions(self.segment_positions)
                return True
            return False
        elif method == "max_topk":
            all_attn = torch.stack(output_attentions, dim=0)  # [num_layers, beamsize, num_heads, seq_len, seq_len]

            text_last_token_position = model_kwargs.get('text_last_token_position', None)
            text_bos_idx = text_last_token_position[0]
            text_eos_idx = text_last_token_position[1][-1] + 1
            all_layer_attn = all_attn[:, :, :, -1, text_bos_idx:text_eos_idx] # [num_layers, beamsize, num_heads, seq_len]; last token only, text region only

            L, B, H, Sk = all_layer_attn.shape  # L: layers, H: heads, B: beams, Sk: text_len

            # ==== Pick the top-k L*H heads. ====
            all_layer_attn = all_layer_attn.permute(0,2,1,3) # [L, H, B, Sk]
            all_layer_attn = all_layer_attn.reshape(-1, B, Sk) # [L * H, B, Sk]
            scores = all_layer_attn.mean(dim=-1)
            topk_scores, topk_indices = torch.topk(scores, k=3, dim=0)
            topk_indices_exp = topk_indices.unsqueeze(-1).expand(-1, -1, Sk) # (topk, B, Sk)
            topk_attn_per_beam = torch.gather(all_layer_attn, 0, topk_indices_exp) #  (topk, B, Sk)
            attn_seperate = topk_attn_per_beam.permute(1, 0, 2)  # [B, topk, Sk]
            
            head_importance = torch.mean(attn_seperate, dim=-1)  # [B, H]
            # Pick the top-k heads.
            topk = 3
            topk_heads = torch.topk(head_importance, k=topk, dim=-1).indices  # [B, topk]
            topk_values = torch.topk(head_importance, k=topk, dim=-1).values  # [B, topk]
            # Min-max normalize, then turn into a weighted mixture.
            weight = (topk_values - topk_values.min(dim=-1, keepdim=True).values) / (topk_values.max(dim=-1, keepdim=True).values - topk_values.min(dim=-1, keepdim=True).values + 1e-8)
            weight = weight / weight.sum(dim=-1, keepdim=True)  # [B, topk]
            # Gather the corresponding attention vectors.
            topk_heads = topk_heads.view(-1, topk, 1).expand(B, topk, Sk)  # [B, topk, Sk]
            topk_attn = attn_seperate.gather(1, topk_heads)  # [B, topk, Sk]
            
            attn_wo_head = (topk_attn * weight.unsqueeze(-1)).sum(dim=1)  # [B, Sk]
            
            # Select the highest-attention text token for each beam.
            # token_idx = attn_wo_head.argmax(dim=-1)  # [B, ]
            # token_value = attn_wo_head.gather(-1, token_idx.unsqueeze(-1)).squeeze(-1)  # [B, ]
            token_idx, token_value = self._update_attn_monotonic(attn_wo_head)

            # Prepare per-beam comparison values.
            attention_phase = model_kwargs.get("attention_phase", None)
            text_last_position_array = torch.tensor(
                text_last_token_position[1] - text_last_token_position[0],
                device=attn_wo_head.device)  # [B, ]
            text_last_pos_per_beam = text_last_position_array[attention_phase]  # [B, ]
            
            # TODO: emit emotion-twist artifacts for offline analysis.
            if model_kwargs.get("save_attention_maps", False):
                self.attn_map_processor.process_emotion_twist_detect(
                    attn_wo_head=attn_wo_head,
                    method=method,
                    token_idx=token_idx,
                    attention_phase=attention_phase,
                    output_path=model_kwargs.get("output_path", None),
                    text_last_token_position=model_kwargs.get('text_last_token_position', None),
                )
            
            # 1. Pull the position from the previous step (handle step 0 / empty history).
            if self.past_attn_pos is not None and len(self.past_attn_pos) > 0:
                prev_token_idx = self.past_attn_pos[-1]
            else:
                # No history yet (first step). Seed with a sentinel that can't match the condition.
                prev_token_idx = torch.full_like(token_idx, -100)
            
            self._update_attn_cache(token_value, token_idx)

            suspect_twist_pos = (token_idx > text_last_pos_per_beam) & \
                                (prev_token_idx == token_idx - 1)
                                # bool tensor: alignment passed the last text token

            if suspect_twist_pos.any():
                # Run the emotion-twist comparison.
                indices = torch.where(suspect_twist_pos)[0]
                for i in indices:
                    i = i.item()

                    # Record the segment-switch position (in semantic tokens, prefix stripped).
                    current_total_len = all_attn.shape[-1]  # total length, including prefix
                    trunc_index = self.trunc_index  # prefix token count, stored on self
                    current_sem_len = current_total_len - trunc_index  # semantic-token position

                    # Use attention_phase as the segment index.
                    if self.segment_positions is None:
                        self.segment_positions = dict()
                    if i not in self.segment_positions:
                        self.segment_positions[i] = [current_sem_len]
                    else:
                        if len(self.segment_positions[i]) > model_kwargs["attention_phase"][i]:
                            # The switch position for this segment was already recorded; refresh it.
                            self.segment_positions[i][model_kwargs["attention_phase"][i]] = current_sem_len
                        else:
                            self.segment_positions[i].append(current_sem_len)
                    
                    model_kwargs["attention_phase"][i] += 1
                    _logger.debug(f"[Max head topk] Beam {i} enters phase {model_kwargs['attention_phase'][i]} at sem {all_attn.shape[-1]} ")
                # ===== Notify duration_processor of the new segment boundaries. =====
                if self.duration_processor is not None:
                    self.duration_processor.update_segment_positions(self.segment_positions)
                return True
            return False
        elif method == "mas":
            # Use the dominant head.
            all_attn = torch.stack(output_attentions, dim=0)  # [num_layers, beamsize, num_heads, seq_len, seq_len]

            text_last_token_position = model_kwargs.get('text_last_token_position', (34, [34]))
            text_last_token_position = model_kwargs.get('text_last_token_position', None)
            text_bos_idx = text_last_token_position[0]
            text_eos_idx = text_last_token_position[1][-1] + 1
            all_layer_attn = all_attn[:, :, :, -1, text_bos_idx:text_eos_idx] # [num_layers, beamsize, num_heads, seq_len]; last token only, text region only

            L, B, H, Sk = all_layer_attn.shape  # L: layers, H: heads, B: beams, Sk: text_len

            # ==== Pick the dominant head ====
            # 1. Per-(layer, head, beam) score (mean over the text axis).
            # Result shape: [L, B, H].
            attn_std = torch.mean(all_layer_attn, dim=-1)

            # 2. Reshape so beams come first and (layer, head) is flattened.
            # [L, B, H] -> [B, L, H]
            attn_std = attn_std.permute(1, 0, 2)
            # [B, L, H] -> [B, L*H]
            attn_std_flat = attn_std.reshape(B, -1)

            # 3. For each beam, find the index of the L*H head with the highest score.
            best_head_idx = attn_std_flat.argmax(dim=-1)  # [B, ]

            # 4. Gather the corresponding attention vectors.
            # Reshape the raw attention data to [B, L*H, Sk] for gather.
            attn_data = all_layer_attn.permute(1, 0, 2, 3).reshape(B, L*H, Sk)

            # Build the gather index: [B, 1, Sk].
            gather_idx = best_head_idx.view(B, 1, 1).expand(-1, -1, Sk)

            # Extract [B, 1, Sk] -> squeeze -> [B, Sk].
            attn_wo_head = attn_data.gather(1, gather_idx).squeeze(1)
            
            # Prepare per-beam comparison values.
            attention_phase = model_kwargs.get("attention_phase", None)
            text_last_position_array = torch.tensor(
                text_last_token_position[1] - text_last_token_position[0],
                device=attn_wo_head.device)  # [B, ]
            text_last_pos_per_beam = text_last_position_array[attention_phase]  # [B, ]

            mas_mu = self._update_mas_mu_w2(attn_wo_head)
            
            # TODO: emit emotion-twist artifacts for offline analysis.
            if model_kwargs.get("save_attention_maps", False):
                self.attn_map_processor.process_emotion_twist_detect(
                    attn_wo_head=attn_wo_head,
                    method=method,
                    token_idx=mas_mu,
                    attention_phase=attention_phase,
                    output_path=model_kwargs.get("output_path", None),
                    text_last_token_position=model_kwargs.get('text_last_token_position', None),
                )
            suspect_twist_pos = (mas_mu >= text_last_pos_per_beam + 1)  # bool tensor
            if suspect_twist_pos.any():
                indices = torch.where(suspect_twist_pos)[0]
                for i in indices:
                    i = i.item()
                    # Record the segment-switch position (in semantic tokens, prefix stripped).
                    current_total_len = all_attn.shape[-1]  # total length, including prefix
                    trunc_index = self.trunc_index  # prefix token count, stored on self
                    current_sem_len = current_total_len - trunc_index  # semantic-token position

                    # Use attention_phase as the segment index.
                    if self.segment_positions is None:
                        self.segment_positions = dict()
                    if i not in self.segment_positions:
                        self.segment_positions[i] = [current_sem_len]
                    else:
                        if len(self.segment_positions[i]) > model_kwargs["attention_phase"][i]:
                            # The switch position for this segment was already recorded; refresh it.
                            self.segment_positions[i][model_kwargs["attention_phase"][i]] = current_sem_len
                        else:
                            self.segment_positions[i].append(current_sem_len)

                    model_kwargs["attention_phase"][i] += 1
                    _logger.debug(f"[Mas] Beam {i} enters phase {model_kwargs['attention_phase'][i]} at sem {current_sem_len} ")
                # ===== Notify duration_processor of the new segment boundaries. =====
                if self.duration_processor is not None:
                    self.duration_processor.update_segment_positions(self.segment_positions)
                return True
            return False
        elif method == "wo_align":
            # randomly pick a phase
            current_phases = model_kwargs.get("attention_phase", None)
            current_phases = np.array(current_phases)
            
            n_states = emotion_final_phase + 1
            should_jump = np.random.random(current_phases.shape) < 0.05
            
            # 3. Compute the "random jump" target (the core trick).
            # Random offset in [1, n_states); `high` is exclusive.
            random_shifts = np.random.randint(1, n_states, size=current_phases.shape)

            # (current + offset) % n_states yields a new position guaranteed to differ from current.
            jump_targets = (current_phases + random_shifts) % n_states

            # 4. Apply: pick target where should_jump is True, otherwise keep the original phase.
            new_phases = np.where(should_jump, jump_targets, current_phases)
            
            new_phases_list = new_phases.tolist()
            for i in range(len(new_phases_list)):
                if new_phases_list[i] != current_phases[i]:
                    _logger.debug(f"[Wo align] Beam {i} jumps from phase {current_phases[i]} to {new_phases_list[i]}")
            model_kwargs["attention_phase"] = new_phases_list
            return False      
        else:
            raise ValueError(f"Unknown method {method} for emotion twist detection.")
                
                
    
    def _update_mas_mu(self, attn_wo_head):
        """Update the MAS anchor `mas_mu` using a 3-token lookahead window.

        Args:
            attn_wo_head (torch.Tensor): attention weights with shape [B, Sk].
        """
        if self.mas_mu is None:
            # First call: pick the maximum.
            B, _ = attn_wo_head.shape
            device = attn_wo_head.device
            self.mas_mu = torch.ones(B, device=device)  # initialize to 1.0
        else:
            B, S = attn_wo_head.shape
            sensitivity = 1.0
            look_ahead = 2     # lookahead window covers [0, 1, 2]

            # 1. Current anchor.
            current_idx = torch.round(self.mas_mu).long()

            # 2. Build the lookahead window [0, 1, 2].
            window_offsets = torch.arange(0, look_ahead + 1, device=attn_wo_head.device)
            window_indices = current_idx.unsqueeze(-1) + window_offsets.unsqueeze(0)
            window_indices = window_indices.clamp(max=S - 1)

            # 3. Gather raw weights (do NOT normalize them).
            # weights: [B, 3] -> [w_0, w_1, w_2]
            weights = attn_wo_head.gather(1, window_indices)

            # 4. Compute the "push force".
            # Rule:
            # w_0 (current position):  contributes 0 push (we don't want to move).
            # w_1 (next position):     contributes 1 * w_1.
            # w_2 (next-next position):contributes 2 * w_2.

            # window_offsets is [0, 1, 2]
            # force = w_0*0 + w_1*1 + w_2*2
            push_force = (weights * window_offsets).sum(dim=-1) # [B, ]

            # 5. Update.
            # delta = push_force.
            # If w1=0.01, w2=0.01 -> force ~ 0.03 -> the anchor barely moves.
            # If w1=0.8,  w2=0.1  -> force = 1.0  -> the anchor advances one slot.
            delta = push_force

            self.mas_mu = self.mas_mu + sensitivity * delta

            # 6. Clamp.
            self.mas_mu = self.mas_mu.clamp(max=S - 1)
        return self.mas_mu

    def _update_mas_mu_w2(self, attn_wo_head):
        """MAS anchor update with a 2-token lookahead window ([0, 1])."""
        if self.mas_mu is None:
            # First call: pick the maximum.
            B, _ = attn_wo_head.shape
            device = attn_wo_head.device
            self.mas_mu = torch.ones(B, device=device)  # initialize to 1.0
        else:
            B, S = attn_wo_head.shape
            sensitivity = 1.0
            look_ahead = 1     # only consider [0, 1]

            # 1. Current anchor (rounded to int).
            current_idx = torch.round(self.mas_mu).long()

            # 2. Build the lookahead window [0, 1].
            # window_offsets -> [0, 1]
            window_offsets = torch.arange(0, look_ahead + 1, device=attn_wo_head.device)

            # Index pair: current, current+1.
            window_indices = current_idx.unsqueeze(-1) + window_offsets.unsqueeze(0)
            window_indices = window_indices.clamp(max=S - 1)

            # 3. Gather weights.
            # Normalize attn_wo_head first via log-softmax.
            log_attn = torch.log(attn_wo_head + 1e-8)
            attn_normalized = torch.softmax(log_attn, dim=-1)

            # weights: [B, 2] -> [w_0, w_1]
            weights = attn_normalized.gather(1, window_indices)

            # 4. Compute the "push force".
            # Formula: w_0 * 0 + w_1 * 1, i.e. push = w_1 (attention of the next token).
            push_force = (weights * window_offsets).sum(dim=-1) # [B, ]

            # 5. Update.
            # delta = w_1; larger w_1 pushes the anchor up to one slot forward.
            delta = push_force

            self.mas_mu = self.mas_mu + sensitivity * delta

            # 6. Clamp.
            self.mas_mu = self.mas_mu.clamp(max=S - 1)

        return self.mas_mu
            
        
    def _update_attn_cache(self, token_value, token_idx):
        """Keep a rolling cache of the last N (token_value, token_idx) per beam.

        :param token_value: attention value chosen this step.
        :param token_idx:   text-position index chosen this step.
        """
        N = 5
        if self.past_attn_cache is None:
            self.past_attn_cache = token_value.unsqueeze(0)  # [1, ]
        else:
            # Append.
            self.past_attn_cache = torch.cat([self.past_attn_cache, token_value.unsqueeze(0)], dim=0)  # [total_beams_so_far, ]
            if self.past_attn_cache.shape[0] > N:
                self.past_attn_cache = self.past_attn_cache[1:]  # keep only the last N

        if self.past_attn_pos is None:
            self.past_attn_pos = token_idx.unsqueeze(0)  # [1, ]
        else:
            # Append.
            self.past_attn_pos = torch.cat([self.past_attn_pos, token_idx.unsqueeze(0)], dim=0)  # [total_beams_so_far, ]
            if self.past_attn_pos.shape[0] > N:
                self.past_attn_pos = self.past_attn_pos[1:]  # keep only the last N
                
    def _update_attn_monotonic(self, attn_wo_head):
        if self.past_attn_pos is None:
            # Initialize to the position with the strongest attention.
            B = attn_wo_head.shape[0]
            token_idx = torch.ones(B, dtype=torch.long, device=attn_wo_head.device)
            token_value = attn_wo_head.gather(-1, token_idx.unsqueeze(-1)).squeeze(-1)
            return token_idx, token_value
        else:
            # Decide based on the previous step.
            # 1. Fetch the previous-step position.
            B = attn_wo_head.shape[0]
            if self.past_attn_pos is not None and len(self.past_attn_pos) > 0:
                prev_token_idx = self.past_attn_pos[-1]

            # 2. Only compare the current position (prev) and the next one (prev + 1).
            Sk = attn_wo_head.shape[-1]
            
            val_curr = attn_wo_head.gather(1, prev_token_idx.unsqueeze(-1)).squeeze(-1)
            
            next_cand_idx = (prev_token_idx + 1).clamp(max=Sk - 1)
            val_next = attn_wo_head.gather(1, next_cand_idx.unsqueeze(-1)).squeeze(-1)
            
            move_mask = val_next > val_curr
            token_idx = torch.where(move_mask, next_cand_idx, prev_token_idx)
            token_value = torch.where(move_mask, val_next, val_curr)

            return token_idx, token_value
    
    @property
    def attn_map_processor(self):
        if not hasattr(self, "_attn_map_processor"):
            self._attn_map_processor = AttentionMapProcessor()
        return self._attn_map_processor

    def forward(
            self,
            input_ids=None,
            past_key_values=None,
            attention_mask=None,
            _4d_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        assert self.cached_mel_emb is not None
        assert inputs_embeds is None  # Not supported by this inference model.
        assert labels is None  # Training not supported by this inference model.
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        # Create embedding
        mel_len = self.cached_mel_emb.shape[1]
        if input_ids.shape[1] != 1:
            text_inputs = input_ids[:, mel_len:]
            text_emb = self.embeddings(text_inputs)
            text_emb = text_emb + self.text_pos_embedding(text_emb)
            # print(f"text pos emb shape: {self.text_pos_embedding(text_emb).shape}")
            if self.cached_mel_emb.shape[0] != text_emb.shape[0]:
                mel_emb = self.cached_mel_emb.repeat_interleave(
                    text_emb.shape[0] // self.cached_mel_emb.shape[0], 0
                )
            else:  # this outcome only occurs once per loop in most cases
                mel_emb = self.cached_mel_emb
            emb = torch.cat([mel_emb, text_emb], dim=1)
        else:
            emb = self.embeddings(input_ids)
            emb = emb + self.text_pos_embedding.get_fixed_embedding(
                attention_mask.shape[1] - mel_len, attention_mask.device
            )
        transformer_outputs = self.transformer(
            inputs_embeds=emb,
            past_key_values=past_key_values,
            attention_mask=attention_mask if _4d_attention_mask is None else _4d_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        
        # Fetch attention metadata and record.
        attention_phase = kwargs.get("attention_phase", None)
        output_path = kwargs.get("output_path", None)
        
        # self.attn_map_processor.process_attention_map(
        #     transformer_outputs.attentions,
        #     attention_phase,
        #     output_path,
        # )

        # Set device for model parallelism
        if self.model_parallel:
            if torch.backends.mps.is_available():
                self.to(self.transformer.first_device)
            else:
                torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + transformer_outputs[1:]

        return CausalLMOutputWithCrossAttentions(
            loss=None,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(past, beam_idx):
        """
        This function is used to re-order the :obj:`past_key_values` cache if
        :meth:`~transformers.PreTrainedModel.beam_search` or :meth:`~transformers.PreTrainedModel.beam_sample` is
        called. This is required to match :obj:`past_key_values` with the correct beam_idx at every generation step.
        """
        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past
        )


class ConditioningEncoder(nn.Module):
    def __init__(self,
                 spec_dim,
                 embedding_dim,
                 attn_blocks=6,
                 num_attn_heads=4,
                 do_checkpointing=False,
                 mean=False):
        super().__init__()
        attn = []
        self.init = nn.Conv1d(spec_dim, embedding_dim, kernel_size=1)
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim
        self.do_checkpointing = do_checkpointing
        self.mean = mean

    def forward(self, x):
        h = self.init(x)
        h = self.attn(h)
        if self.mean:
            return h.mean(dim=2)
        else:
            return h
            # return h[:, :, 0]


class LearnedPositionEmbeddings(nn.Module):
    def __init__(self, seq_len, model_dim, init=.02):
        super().__init__()
        self.emb = nn.Embedding(seq_len, model_dim)
        # Initializing this way is standard for GPT-2
        self.emb.weight.data.normal_(mean=0.0, std=init)

    def forward(self, x):
        sl = x.shape[1]
        return self.emb(torch.arange(0, sl, device=x.device))

    def get_fixed_embedding(self, ind, dev):
        return self.emb(torch.tensor([ind], device=dev)).unsqueeze(0)


def build_hf_gpt_transformer(layers, model_dim, heads, max_mel_seq_len, max_text_seq_len, checkpointing):
    """
    GPT-2 implemented by the HuggingFace library.
    """
    from transformers import GPT2Config, GPT2Model
    gpt_config = GPT2Config(vocab_size=256,  # Unused.
                            n_positions=max_mel_seq_len + max_text_seq_len,
                            n_ctx=max_mel_seq_len + max_text_seq_len,
                            n_embd=model_dim,
                            n_layer=layers,
                            n_head=heads,
                            gradient_checkpointing=checkpointing,
                            use_cache=not checkpointing)
    gpt = GPT2Model(gpt_config)
    # Override the built in positional embeddings
    del gpt.wpe
    gpt.wpe = functools.partial(null_position_embeddings, dim=model_dim)
    # Built-in token embeddings are unused.
    del gpt.wte
    return gpt, LearnedPositionEmbeddings(max_mel_seq_len, model_dim), LearnedPositionEmbeddings(max_text_seq_len, model_dim), \
        None, None


class MelEncoder(nn.Module):
    def __init__(self, channels, mel_channels=80, resblocks_per_reduction=2):
        super().__init__()
        self.channels = channels
        self.encoder = nn.Sequential(nn.Conv1d(mel_channels, channels // 4, kernel_size=3, padding=1),
                                     nn.Sequential(*[ResBlock(channels // 4) for _ in range(resblocks_per_reduction)]),
                                     nn.Conv1d(channels // 4, channels // 2, kernel_size=3, stride=2, padding=1),
                                     nn.GroupNorm(channels // 16, channels // 2),
                                     nn.ReLU(),
                                     nn.Sequential(*[ResBlock(channels // 2) for _ in range(resblocks_per_reduction)]),
                                     nn.Conv1d(channels // 2, channels, kernel_size=3, stride=2, padding=1),
                                     nn.GroupNorm(channels // 8, channels),
                                     nn.ReLU(),
                                     nn.Sequential(*[ResBlock(channels) for _ in range(resblocks_per_reduction)]),
                                     )
        self.reduction = 4

    def forward(self, x):
        for e in self.encoder:
            x = e(x)
        return x.permute(0, 2, 1)


class UnifiedVoice(nn.Module):
    def __init__(self, layers=8, model_dim=512, heads=8, max_text_tokens=120, max_mel_tokens=250, max_conditioning_inputs=1,
                 mel_length_compression=1024, number_text_tokens=256,
                 start_text_token=0, stop_text_token=1, number_mel_codes=8194, start_mel_token=8192, stop_mel_token=8193,
                 train_solo_embeddings=False, use_mel_codes_as_input=True,
                 checkpointing=True, types=1,
                 condition_num_latent=32, condition_type="perceiver", condition_module=None, emo_condition_module=None):
        """
        Args:
            layers: Number of layers in transformer stack.
            model_dim: Operating dimensions of the transformer
            heads: Number of transformer heads. Must be divisible by model_dim. Recommend model_dim//64
            max_text_tokens: Maximum number of text tokens that will be encountered by model.
            max_mel_tokens: Maximum number of MEL tokens that will be encountered by model.
            max_conditioning_inputs: Maximum number of conditioning inputs provided to the model. If (1), conditioning input can be of format (b,80,s), otherwise (b,n,80,s).
            mel_length_compression: The factor between <number_input_samples> and <mel_tokens>. Used to compute MEL code padding given wav input length.
            number_text_tokens:
            start_text_token:
            stop_text_token:
            number_mel_codes:
            start_mel_token:
            stop_mel_token:
            train_solo_embeddings:
            use_mel_codes_as_input:
            checkpointing:
            condition_type: perceiver, gst or default encoder
        """
        super().__init__()
        self.number_text_tokens = number_text_tokens
        self.start_text_token = start_text_token
        self.stop_text_token = stop_text_token
        self.number_mel_codes = number_mel_codes
        self.start_mel_token = start_mel_token
        self.stop_mel_token = stop_mel_token
        self.layers = layers
        self.heads = heads
        self.max_mel_tokens = max_mel_tokens
        self.max_text_tokens = max_text_tokens
        self.model_dim = model_dim
        self.max_conditioning_inputs = max_conditioning_inputs
        self.mel_length_compression = mel_length_compression
        self.condition_type = condition_type
        self.cond_num = condition_num_latent
        self.cond_mask_pad = nn.ConstantPad1d((self.cond_num, 0), True)
        self.emo_cond_mask_pad = nn.ConstantPad1d((1, 0), True)
        if condition_type == "perceiver":
            self.conditioning_encoder = ConditioningEncoder(1024, model_dim, num_attn_heads=heads)
            self.perceiver_encoder = PerceiverResampler(model_dim, dim_context=model_dim, num_latents=self.cond_num)
        elif condition_type == "conformer_perceiver" or condition_type == "conformer_encoder":
            self.conditioning_encoder = ConformerEncoder(input_size=1024,
                                                         output_size=condition_module['output_size'],
                                                         linear_units=condition_module['linear_units'],
                                                         attention_heads=condition_module['attention_heads'],
                                                         num_blocks=condition_module['num_blocks'],
                                                         input_layer=condition_module['input_layer'])
            if condition_type == "conformer_perceiver":
                self.perceiver_encoder = PerceiverResampler(model_dim, dim_context=condition_module['output_size'],
                                                            ff_mult=condition_module['perceiver_mult'],
                                                            heads=condition_module['attention_heads'],
                                                            num_latents=self.cond_num)
        else:
            self.conditioning_encoder = ConditioningEncoder(1024, model_dim, num_attn_heads=heads, mean=True)

        self.emo_conditioning_encoder = ConformerEncoder(input_size=1024,
                                                         output_size=emo_condition_module['output_size'],
                                                         linear_units=emo_condition_module['linear_units'],
                                                         attention_heads=emo_condition_module['attention_heads'],
                                                         num_blocks=emo_condition_module['num_blocks'],
                                                         input_layer=emo_condition_module['input_layer'])
        self.emo_perceiver_encoder = PerceiverResampler(1024, dim_context=emo_condition_module['output_size'],
                                                            ff_mult=emo_condition_module['perceiver_mult'],
                                                            heads=emo_condition_module['attention_heads'],
                                                            num_latents=1)



        self.text_embedding = nn.Embedding(self.number_text_tokens * types + 1, model_dim)
        self.emo_layer = nn.Linear(model_dim, model_dim)
        self.emovec_layer = nn.Linear(1024, model_dim)

        if use_mel_codes_as_input:
            self.mel_embedding = nn.Embedding(self.number_mel_codes, model_dim)
        else:
            self.mel_embedding = MelEncoder(model_dim, resblocks_per_reduction=1)
        self.gpt, self.mel_pos_embedding, self.text_pos_embedding, self.mel_layer_pos_embedding, self.text_layer_pos_embedding = \
            build_hf_gpt_transformer(layers, model_dim, heads, self.max_mel_tokens + 2 + self.max_conditioning_inputs,
                                     self.max_text_tokens + 2, checkpointing)
        if train_solo_embeddings:
            self.mel_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * .02, requires_grad=True)
            self.text_solo_embedding = nn.Parameter(torch.randn(1, 1, model_dim) * .02, requires_grad=True)
        else:
            self.mel_solo_embedding = 0
            self.text_solo_embedding = 0

        self.final_norm = nn.LayerNorm(model_dim)
        self.text_head = nn.Linear(model_dim, self.number_text_tokens * types + 1)
        self.mel_head = nn.Linear(model_dim, self.number_mel_codes)

        self.speed_emb = nn.Embedding(2, model_dim)
        self.speed_emb.weight.data.normal_(mean=0.0, std=0.0)

        # Initialize the embeddings per the GPT-2 scheme
        embeddings = [self.text_embedding]
        if use_mel_codes_as_input:
            embeddings.append(self.mel_embedding)
        for module in embeddings:
            module.weight.data.normal_(mean=0.0, std=.02)

    def post_init_gpt2_config(self, use_deepspeed=False, kv_cache=False, half=False):
        seq_length = self.max_mel_tokens + self.max_text_tokens + 2
        gpt_config = GPT2Config(
            vocab_size=self.number_mel_codes,
            n_positions=seq_length,
            n_ctx=seq_length,
            n_embd=self.model_dim,
            n_layer=self.layers,
            n_head=self.heads,
            gradient_checkpointing=False,
            use_cache=True,
        )
        self.inference_model = GPT2InferenceModel(
            gpt_config,
            self.gpt,
            self.mel_pos_embedding,
            self.mel_embedding,
            self.final_norm,
            self.mel_head,
            kv_cache=kv_cache,
        )
        if use_deepspeed and half and torch.cuda.is_available():
            import deepspeed
            self.ds_engine = deepspeed.init_inference(model=self.inference_model,
                                                      mp_size=1,
                                                      replace_with_kernel_inject=True,
                                                      dtype=torch.float16)
            self.inference_model = self.ds_engine.module.eval()
        elif use_deepspeed and torch.cuda.is_available():
            import deepspeed
            self.ds_engine = deepspeed.init_inference(model=self.inference_model,
                                                      mp_size=1,
                                                      replace_with_kernel_inject=True,
                                                      dtype=torch.float32)
            self.inference_model = self.ds_engine.module.eval()
        else:
            self.inference_model = self.inference_model.eval()

        # self.inference_model = PrunedGPT2InferenceModel(gpt_config, self.gpt, self.mel_pos_embedding, self.mel_embedding, self.final_norm, self.mel_head)
        self.gpt.wte = self.mel_embedding

    def build_aligned_inputs_and_targets(self, input, start_token, stop_token):
        inp = F.pad(input, (1, 0), value=start_token)
        tar = F.pad(input, (0, 1), value=stop_token)
        return inp, tar

    def set_mel_padding(self, mel_input_tokens, mel_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(mel_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = mel_lengths[b]
            if actual_end < mel_input_tokens.shape[-1]:
                mel_input_tokens[b, actual_end:] = self.stop_mel_token
        return mel_input_tokens

    def set_text_padding(self, text_input_tokens, text_lengths):
        """
        Given mel tokens that are derived from a padded audio clip and the actual lengths of each batch element in
        that audio clip, reformats the tokens with STOP_MEL_TOKEN in place of the zero padding. This is required
        preformatting to create a working TTS model.
        """
        for b in range(len(text_lengths)):
            # Due to the convolutional nature of how these tokens are generated,
            # it would be best if the model predicts a token past the actual last token.
            actual_end = text_lengths[b]
            if actual_end < text_input_tokens.shape[-1]:
                text_input_tokens[b, actual_end:] = self.stop_text_token
        return text_input_tokens

    def get_logits(self, speech_conditioning_inputs, first_inputs, first_head, second_inputs=None, second_head=None, get_attns=False, return_latent=False, attention_mask=None):
        if second_inputs is not None:
            emb = torch.cat([speech_conditioning_inputs, first_inputs, second_inputs], dim=1)
        else:
            emb = torch.cat([speech_conditioning_inputs, first_inputs], dim=1)

        gpt_out = self.gpt(inputs_embeds=emb, return_dict=True, output_attentions=get_attns, attention_mask=attention_mask)
        if get_attns:
            return gpt_out.attentions

        offset = speech_conditioning_inputs.shape[1]
        enc = gpt_out.last_hidden_state[:, offset:]
        enc = self.final_norm(enc)

        if return_latent:
            return enc[:, :first_inputs.shape[1]], enc[:, -second_inputs.shape[1]:]

        first_logits = enc[:, :first_inputs.shape[1]]
        first_logits = first_head(first_logits)
        first_logits = first_logits.permute(0, 2, 1)
        if second_inputs is not None:
            second_logits = enc[:, -second_inputs.shape[1]:]
            second_logits = second_head(second_logits)
            second_logits = second_logits.permute(0, 2, 1)
            return first_logits, second_logits
        else:
            return first_logits

    def get_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        if self.condition_type == "perceiver":
            if speech_conditioning_input.ndim == 4:
                speech_conditioning_input = speech_conditioning_input.squeeze(1)
            speech_conditioning_input = self.conditioning_encoder(speech_conditioning_input)  # (b, d, s)
            conds = self.perceiver_encoder(speech_conditioning_input.transpose(1, 2))  # (b, 32, d)
        elif self.condition_type == "conformer_perceiver":
            speech_conditioning_input, mask = self.conditioning_encoder(speech_conditioning_input.transpose(1, 2),
                                                                        cond_mel_lengths)  # (b, s, d), (b, 1, s)
            if self.condition_type == "conformer_perceiver":
                # conds_mask = torch.cat([torch.ones((mask.shape[0], self.cond_num), dtype=torch.bool), mask.squeeze(1)], dim=1)
                conds_mask = self.cond_mask_pad(mask.squeeze(1))
                conds = self.perceiver_encoder(speech_conditioning_input, conds_mask)  # (b, 32, d)
        elif self.condition_type == "gst":
            if speech_conditioning_input.ndim == 4:
                speech_conditioning_input = speech_conditioning_input.squeeze(1)
            conds = self.gst_encoder(speech_conditioning_input.transpose(1, 2))  # (b, 1, d)
        else:
            speech_conditioning_input = (
                speech_conditioning_input.unsqueeze(1)
                if len(speech_conditioning_input.shape) == 3
                else speech_conditioning_input
            )
            conds = []
            for j in range(speech_conditioning_input.shape[1]):
                conds.append(self.conditioning_encoder(speech_conditioning_input[:, j]))
            conds = torch.stack(conds, dim=1)
            conds = conds.mean(dim=1)
            conds = conds.unsqueeze(1)
        return conds


    def get_emo_conditioning(self, speech_conditioning_input, cond_mel_lengths=None):
        speech_conditioning_input, mask = self.emo_conditioning_encoder(speech_conditioning_input.transpose(1, 2),
                                                                        cond_mel_lengths)  # (b, s, d), (b, 1, s)
        conds_mask = self.emo_cond_mask_pad(mask.squeeze(1))
        conds = self.emo_perceiver_encoder(speech_conditioning_input, conds_mask)  # (b, 1, d)
        return conds.squeeze(1)
    
    def get_duration_embeddings(self, lengths: torch.Tensor, check: bool = False):
        """
        Thanks to https://github.com/JarodMica/index-tts/commit/a9f0125531124eccb8b3e8568d1b9c711a1e7564#diff-456187d8672e8d50cdf6ffbbe9e0f139242e20fad9ea52b3ce32e2cfe41677c1R7
        """
        max_index = self.mel_pos_embedding.emb.num_embeddings - 1
        clamped = lengths.clamp(max=max_index).long()
        
        import os
        if check and os.path.exists("mel_pos_embedding.csv") == False:
            # Dump all positional embeddings to a CSV for inspection.
            import pandas as pd
            all_pos_emb = self.mel_pos_embedding.emb.weight.data.cpu().numpy()
            df = pd.DataFrame(all_pos_emb)
            df.to_csv("mel_pos_embedding.csv", index=False, header=False)
        return self.mel_pos_embedding.emb(clamped)
        
    def forward(self, speech_conditioning_latent, text_inputs, text_lengths, mel_codes, mel_codes_lengths, emo_speech_conditioning_latent,
                cond_mel_lengths=None, emo_cond_mel_lengths=None, emo_vecs=None, use_speed=None, do_spk_cond=False, attention_mask=None):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode

        Args:
            speech_conditioning_input: MEL float tensor, (b,1024); or a `list` of such tensors
            text_inputs: long tensor, (b,t)
            text_lengths: long tensor, (b,)
            mel_inputs:  long tensor, (b,m)
            wav_lengths: long tensor, (b,)

        Returns:
            If return_attentions is specified, only logits are returned.
            If return_latent is specified, loss & logits are not computed or returned. Only the predicted latents are returned.
        """

        if do_spk_cond:
            speech_conditioning_latent = self.get_conditioning(speech_conditioning_latent.transpose(1,2), cond_mel_lengths)
        else:
            speech_conditioning_latent = speech_conditioning_latent

        # if emo_vec is None:
        #     emo_vec_syn_ori = self.get_emo_conditioning(emo_speech_conditioning_latent.transpose(1,2), emo_cond_mel_lengths)
        #     emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
        #     emo_vec = self.emo_layer(emo_vec_syn)

        text_inputs = self.set_text_padding(text_inputs, text_lengths)
        text_inputs = F.pad(text_inputs, (0, 1), value=self.stop_text_token)

        mel_codes = self.set_mel_padding(mel_codes, mel_codes_lengths)
        mel_codes = F.pad(mel_codes, (0, 1), value=self.stop_mel_token)

        duration_emb = self.speed_emb(torch.zeros_like(use_speed))
        duration_emb_half = self.speed_emb(torch.ones_like(use_speed))
        # conds = torch.cat((speech_conditioning_latent + emo_vec.unsqueeze(1), duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)), 1)
        # conds = torch.cat((speech_conditioning_latent, duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)), 1)
        conds_latents = []
        for i, emo_vec in enumerate(emo_vecs):
            if isinstance(speech_conditioning_latent, list):
                conds_latent = torch.cat(
                    (speech_conditioning_latent[i] + emo_vec.unsqueeze(1), 
                     duration_emb_half.unsqueeze(1), 
                     duration_emb.unsqueeze(1)), 1)
            else:
                conds_latent = torch.cat((speech_conditioning_latent + emo_vec.unsqueeze(1), duration_emb_half.unsqueeze(1), duration_emb.unsqueeze(1)), 1)
            conds_latents.append(conds_latent)
        conds = torch.cat(conds_latents, dim=1)

        text_inputs, text_targets = self.build_aligned_inputs_and_targets(text_inputs, self.start_text_token, self.stop_text_token)
        text_emb = self.text_embedding(text_inputs) + self.text_pos_embedding(text_inputs)
        mel_codes, mel_targets = self.build_aligned_inputs_and_targets(mel_codes, self.start_mel_token, self.stop_mel_token)

        mel_emb = self.mel_embedding(mel_codes)
        mel_emb = mel_emb + self.mel_pos_embedding(mel_codes)

        text_logits, mel_logits = self.get_logits(conds, text_emb, self.text_head, mel_emb, self.mel_head, get_attns=False, return_latent=True, attention_mask = attention_mask)
        return mel_logits[:, :-2]  # Despite the name, these are not logits. Strip off the two tokens added by this forward pass.

    def prepare_gpt_inputs(
        self,
        conditional_latents: torch.Tensor,
        text_inputs_list: torch.Tensor,
        full_text: bool = False,
    ):
        
        """
        Prepare the inputs for the GPT2InferenceModel to generate.
        Args:
            conds_latent: (b, 32, dim) audio conditioning embedding by `get_conditioning()`
            text_inputs: (b, L)
            full_text(`bool`): whether to use full text or only each segment
        Returns:
            input_ids: (b, s+1) the input ids for the GPT2InferenceModel.generate()
            inputs_embeds: (b, s+1, dim) the input embeddings for the GPT2InferenceModel.forward()
            attention_mask: (b, s+1) the attention mask for the GPT2InferenceModel.generate()
        """
        b = text_inputs_list[0].shape[0]
        L = sum([t.shape[1] for t in text_inputs_list])
        device = text_inputs_list[0].device
        # single_cond = conditional_latents[0].ndim == 3 and conditional_latents[0].shape[0] == 1
        # if not single_cond:
        #     assert conditional_latents.shape[0] == b, f"batch size mismatch: {conditional_latents.shape[0]} vs {b}"
        batched_mel_emb = []
        attention_masks = []
        attention_masks_full_view = []
        cond_lengths = [t.shape[1] for t in conditional_latents]
        # All conditioning entries must share the same length.
        assert all([l == cond_lengths[0] for l in cond_lengths]), f"cond lengths not equal: {cond_lengths}"
        cond_len = cond_lengths[0]
        cond_cum_lengths = [0]
        for length in cond_lengths:
            cond_cum_lengths.append(cond_cum_lengths[-1] + length)
        total_cond_len = cond_cum_lengths[-1]
        target_len = total_cond_len + L + 2 # reserve 2 slots for BOS/EOS
        _logger.debug(f"target_len: {target_len}, cond_lens: {cond_lengths}, L: {L}")
        


        # valid_mask = (text_inputs[i] != self.stop_text_token) & (text_inputs[i] != self.start_text_token)
        # text_input = text_inputs[i][valid_mask]
        text_input = torch.cat(text_inputs_list, dim=1)
        text_input = F.pad(text_input, (1, 0), value=self.start_text_token)
        text_input = F.pad(text_input, (0, 1), value=self.stop_text_token)
        text_input_pos = torch.arange(0, text_input.size(-1), device=device)
        text_emb = self.text_embedding(text_input) + self.text_pos_embedding.emb(text_input_pos)
        # concatenate [conditional latents][text embeddings]
        _logger.debug(f"conditional_latents shape: {conditional_latents[0].shape}, text_emb shape: {text_emb.shape}")
        conds_text_emb = [c.squeeze(0) for c in conditional_latents] + [
            text_emb.squeeze(0),
        ]

        switching_points = [] # emotion switching points
        progress = 0
        for idx, t in enumerate(text_inputs_list):
            progress += t.shape[1]
            switching_points.append(progress)
        # +1 for the start_mel_token
        attention_mask = torch.ones(target_len+1, dtype=torch.long, device=device)
        # check this text input is padded
        padding: int = L + 2 - text_input.size(-1)
        
        # pad left of [cond][text] -> [pad][cond][text]
        if padding > 0:
            pad = torch.zeros((padding, conditional_latents.size(-1)), dtype=text_emb.dtype, device=device) # [p, dim]
            conds_text_emb.insert(0, pad)
            attention_mask[:padding] = 0
        # if len(conditional_latents) > 1: # by default attend to the first cond and the first text segment
        for i, sp in enumerate(switching_points):
            new_attention_mask = attention_mask.clone()
            if i > 0:
                new_attention_mask[padding:padding+cond_cum_lengths[i]] = 0
            if i < len(switching_points) - 1:
                new_attention_mask[padding+cond_cum_lengths[i+1]:padding+total_cond_len] = 0
            # attention_mask[padding+cond_len:padding+cond_len*len(conditional_latents)] = 0
            attention_masks_full_view.append(new_attention_mask.clone().unsqueeze(0))
            if not full_text:
                new_attention_mask[padding+total_cond_len+sp+1:-2] = 0
            attention_masks.append(new_attention_mask.unsqueeze(0))
        
        mel_emb = torch.cat(conds_text_emb) #[s, dim]
        assert mel_emb.shape[0] == target_len, f"mel_emb.shape: {mel_emb.shape}, target_len: {target_len}"
        batched_mel_emb.append(mel_emb)
        
        input_masks = []
        dynamic_cond_mask_idx = []
        for i in range(len(conditional_latents)):
            input_mask = attention_mask.clone()
            if i > 0:
                input_mask[padding:padding+cond_len*i] = 0
            if i < len(conditional_latents) - 1:
                input_mask[padding+cond_len*(i+1):padding+cond_len*len(conditional_latents)] = 0
            for _ in range(cond_len):
                input_masks.append(input_mask.unsqueeze(0))

        dynamic_cond_mask_idx.append(len(input_masks))
        input_masks.append(attention_mask.unsqueeze(0))  # for the start token
        for i, sp in enumerate(switching_points):
            input_mask = attention_mask.clone()
            if i > 0:
                input_mask[padding:padding+cond_len*i] = 0
            if i < len(switching_points) - 1:
                input_mask[padding+cond_len*(i+1):padding+cond_len*len(conditional_latents)] = 0
            length = sp
            if i != 0:
                length -= switching_points[i-1]
            for _ in range(length):
                input_masks.append(input_mask.unsqueeze(0))
        dynamic_cond_mask_idx.append(len(input_masks))
        input_masks.append(attention_mask.unsqueeze(0))  # for the stop token
        dynamic_cond_mask_idx.append(len(input_masks))
        input_masks.append(attention_mask.unsqueeze(0))  # for the mel_start token
        input_attention_masks = torch.cat(input_masks, dim=0)
                

        # [b, s, dim]
        batched_mel_emb = torch.stack(batched_mel_emb, dim=0)
        # [b, s+1]
        # attention_mask = torch.stack(attention_masks, dim=0)
        # [b, s+1]
        fake_inputs = torch.ones(
            (
                batched_mel_emb.shape[0],
                batched_mel_emb.shape[1] + 1,  # +1 for the start_mel_token
            ),
            dtype=torch.long,
            device=device,
        )
        fake_inputs[:, -1] = self.start_mel_token
        
        text_last_token_position = padding + total_cond_len + np.array(switching_points)
        text_last_token_position = (total_cond_len, text_last_token_position)
        
        return fake_inputs, batched_mel_emb, attention_masks, attention_masks_full_view, text_last_token_position, input_attention_masks, dynamic_cond_mask_idx

    def inference_speech(self, speech_condition, text_inputs_list, emo_speech_condition=None, cond_lengths=None, emo_cond_lengths=None, emo_vecs=None, use_speed=False, input_tokens=None, num_return_sequences=1,
                         max_generate_length=None, typical_sampling=False, typical_mass=.9, input_full_attention_mask=False, target_duration_tokens=None, duration_mode="none", **hf_generate_kwargs):
        """
        Args:
            speech_condition: (b, d, frames) or (d, frames)
            text_inputs: (b, L)
            cond_mel_lengths: lengths of the conditioning mel spectrograms in shape (b,) or (1,)
            input_tokens: additional tokens for generation in shape (b, s) or (s,)
            max_generate_length: limit the number of generated tokens
            hf_generate_kwargs: kwargs for `GPT2InferenceModel.generate(**hf_generate_kwargs)`
        """

        if not isinstance(speech_condition, list) and speech_condition.ndim == 2:
            speech_condition = speech_condition.unsqueeze(0)
            
        if emo_speech_condition is None:
            emo_speech_condition = speech_condition
            
        if cond_lengths is None:
            cond_lengths = torch.tensor([speech_condition.shape[-1]], device=speech_condition.device)
            
        if emo_cond_lengths is None:
            emo_cond_lengths = torch.tensor([emo_speech_condition.shape[-1]], device=speech_condition.device) 

        # speech cond latent part
        speech_conditioning_latent_list = None
        if isinstance(speech_condition, list) and isinstance(cond_lengths, list):
            speech_conditioning_latent_list = []
            for i, speech_cond in enumerate(speech_condition):
                latent = self.get_conditioning(speech_cond.transpose(1,2), cond_lengths[i])
                speech_conditioning_latent_list.append(latent)
        else:
            speech_conditioning_latent = self.get_conditioning(speech_condition.transpose(1,2), cond_lengths)
        
        if emo_vecs is None:
            _logger.debug('compute emo vec')
            emo_vec = self.get_emo_conditioning(emo_speech_condition.transpose(1,2), emo_cond_lengths)
            emo_vec = self.emovec_layer(emo_vec)
            emo_vec = self.emo_layer(emo_vec)
            emo_vecs = [emo_vec]
        else:
            _logger.debug('Use the specified emotion vector')
        for text_inputs in text_inputs_list:
            _logger.debug(f"text input shape: {text_inputs.shape}")


        tmp = torch.zeros(text_inputs_list[0].size(0)).to(text_inputs_list[0].device)
        duration_emb =  self.speed_emb(torch.zeros_like(tmp).long())
        duration_emb_half = self.speed_emb(torch.ones_like(tmp).long())
        
        conds_latents = []
        global_duration_cursor = 0
        _logger.debug(f"[DEBUG] target_duration_tokens (model side) = {target_duration_tokens}")

        for i, emo_vec in enumerate(emo_vecs):
            tmp = torch.zeros(text_inputs.size(0), device=text_inputs.device)
            duration_free = self.speed_emb(torch.zeros_like(tmp).long())

            if target_duration_tokens is not None:
                seg_len = int(target_duration_tokens[i])
                global_duration_cursor += seg_len
                if i < 10: 
                    _logger.debug(
                        f"[DurationCtrl] seg={i} | "
                        f"seg_len={seg_len} | "
                        f"global_cursor={global_duration_cursor}"
                    )
                t = max(1, min(global_duration_cursor, self.max_mel_tokens - 1))
                duration_idx = torch.full(
                    (text_inputs.size(0),),
                    t,
                    device=text_inputs.device,
                    dtype=torch.long
                )
                duration_ctrl = self.get_duration_embeddings(duration_idx, check=True)
            else:
                duration_ctrl = self.speed_emb(torch.ones_like(tmp).long())

            if speech_conditioning_latent_list is not None:
                conds_latent = torch.cat(
                    (
                        speech_conditioning_latent_list[i] + emo_vec.unsqueeze(1),
                        duration_emb.unsqueeze(1),
                        duration_ctrl.unsqueeze(1)
                    ),
                    dim=1
                )
            else:
                conds_latent = torch.cat(
                    (
                        speech_conditioning_latent + emo_vec.unsqueeze(1),
                        duration_emb.unsqueeze(1),
                        duration_ctrl.unsqueeze(1)
                    ),
                    dim=1
                )
            conds_latents.append(conds_latent)
            
        _logger.debug(f"Emo cond length: {emo_vecs[0].shape}")
        full_text = not (hf_generate_kwargs.get("method", "eos") == "eos")
        input_ids, inputs_embeds, attention_masks, attention_masks_full_view, text_last_token_position, input_attention_masks, dynamic_cond_mask_idx = self.prepare_gpt_inputs(conds_latents, text_inputs_list, full_text=full_text)
        _logger.debug(f"attention_mask: {attention_masks}")
        self.inference_model.store_mel_emb(inputs_embeds)
        # if input_tokens is None:
        inputs = input_ids
        # else:
        #     if input_tokens.ndim == 1:
        #         input_tokens = input_tokens.unsqueeze(0)
        #     assert num_return_sequences % input_tokens.shape[0] == 0, \
        #             "The num_return_sequences must be divisible by the batch number of input_tokens"
        #     assert num_return_sequences % text_inputs.shape[0] == 0, \
        #             "The num_return_sequences must be divisible by the batch number of text_inputs"
        #     b = num_return_sequences // input_ids.shape[0]
        #     if b > 1:
        #         input_ids = input_ids.repeat(b, 1)
        #         attention_mask = attention_mask.repeat(b, 1)
        #     input_tokens = input_tokens.repeat(num_return_sequences // input_tokens.shape[0], 1)
        #     inputs = torch.cat([input_ids, input_tokens], dim=1)
        #     attention_mask = F.pad(attention_mask, (0, input_tokens.shape[1]), value=1)
        
        trunc_index = inputs.shape[1]
        # Stash trunc_index on inference_model so segment-switch logic can read it.
        self.inference_model.trunc_index = trunc_index
        logits_processor = LogitsProcessorList()
        if typical_sampling:
            # employ custom typical sampling
            if not (typical_mass > 0.0 and typical_mass < 1.0):
                raise ValueError(f"`typical_mass` has to be a float > 0 and < 1, but is {typical_mass}")
            min_tokens_to_keep = 2 if hf_generate_kwargs.get("num_beams", 1) > 1 else 1
            logits_processor.append(TypicalLogitsWarper(mass=typical_mass, min_tokens_to_keep=min_tokens_to_keep))

        # ========== Duration Control (mode: none / global / intra / both) ==========
        use_global = duration_mode in ("global", "both")
        use_intra = duration_mode in ("intra", "both")

        if target_duration_tokens is not None and (use_global or use_intra):
            # Coerce to a list.
            if not isinstance(target_duration_tokens, list):
                target_duration_tokens = [target_duration_tokens]

            _logger.info(f"[DurationCtrl] mode={duration_mode}")
            _logger.debug(f"   Num segments: {len(target_duration_tokens)}")
            _logger.debug(f"   Target per segment: {target_duration_tokens}")
            _logger.debug(f"   Total target: {sum(target_duration_tokens)} semantic tokens")

            # ---- Global steering: RemainingBudgetEOSProcessor ----
            if use_global:
                duration_processor = RemainingBudgetEOSProcessor(
                    target_tokens_per_segment=target_duration_tokens,
                    stop_token_id=self.stop_mel_token,
                    verbose=True,
                    min_ratio=0.5,
                    neutral_ratio=(0.7, 1.0),
                    max_ratio=1.2,
                    max_negative_bias=-5.0,
                    max_positive_bias=15.0,
                )
                self.inference_model.duration_processor = duration_processor
                logits_processor.append(duration_processor)
                _logger.info("[RemainingBudgetEOS] Enabled")

            # ---- Intra-segment duration control ----
            if use_intra:
                from .duration_controller import IntraSegmentDurationController

                text_tokens_per_segment = [t.shape[1] for t in text_inputs_list]
                if isinstance(speech_condition, list):
                    device = speech_condition[0].device
                else:
                    device = speech_condition.device

                intra_seg_controller = IntraSegmentDurationController(
                    get_duration_embeddings_fn=self.get_duration_embeddings,
                    target_tokens_per_segment=target_duration_tokens,
                    text_tokens_per_segment=text_tokens_per_segment,
                    device=device,
                    max_mel_tokens=self.max_mel_tokens,
                    k_p=25.0,
                    eps=0.01,
                    delta_max=10,
                    update_freq=5,
                    verbose=True,
                )
                self.inference_model.duration_controller = intra_seg_controller
                _logger.info("[IntraSegDurCtrl] Enabled")

        # Pop save_attention_maps from generate kwargs (we handle it ourselves).
        save_attention_maps = hf_generate_kwargs.pop("save_attention_maps", False)
        
        max_length = (trunc_index + self.max_mel_tokens - 1) if max_generate_length is None else trunc_index + max_generate_length
        token_gen_start_time = time.perf_counter()
        output = self.inference_model.generate(inputs, 
                                            bos_token_id=self.start_mel_token, pad_token_id=self.stop_mel_token,
                                            eos_token_id=self.stop_mel_token, attention_masks=attention_masks,
                                            max_length=max_length, logits_processor=logits_processor,
                                            num_return_sequences=num_return_sequences,
                                            text_last_token_position=text_last_token_position,
                                            input_full_attention_mask=input_full_attention_mask,
                                            input_attention_masks=input_attention_masks,
                                            dynamic_cond_mask_idx=dynamic_cond_mask_idx,
                                            save_attention_maps=save_attention_maps,
                                            **hf_generate_kwargs)
        token_generation_time = time.perf_counter() - token_gen_start_time
        print(f">> Token generation time (model_v2.generate): {token_generation_time:.4f} seconds")
        
        # Persist attention maps only when explicitly requested.
        if save_attention_maps:
            self.inference_model.attn_map_processor.save_attention_maps(input_embeds_len=inputs_embeds.shape[1],)
        # self.inference_model.hmm = None   # release the HMM instance
        
        output.sequences = output.sequences[:, trunc_index:]  # remove the input part
        _logger.debug(f"Generated output shape: {output.sequences.shape}, inputs shape: {inputs.shape}")
    
        # min_dtype = torch.finfo(mask_dtype).min
        # causal_masks = []
        # for i in range(len(output.attention_mask_ids)):
        #     output_len = len(output.attention_mask_ids[i]) + trunc_index
            
            
        #     if input_full_attention_mask:
        #         print("Using full attention mask")
        #         attention_mask_list = [torch.ones((1, output_len), dtype=torch.long, device=output.sequences.device)] * trunc_index
        #     else:
        #         print("Using input attention masks")
        #         attention_mask_list = [ torch.nn.functional.pad(input_attention_masks[i], (0, output_len - input_attention_masks[i].shape[0]), value=1).unsqueeze(0) for i in range(trunc_index)]
            
        #     for mask_id in output.attention_mask_ids[i]:
        #         attention_mask_list.append(torch.nn.functional.pad(attention_masks_full_view[mask_id], (0, output_len - attention_masks_full_view[mask_id].shape[1]), value=1))
        #     attention_mask = torch.cat(attention_mask_list, dim=0).unsqueeze(0).unsqueeze(0)
        #     print("attention_mask shape:", attention_mask.shape)

        #     mask_dtype = next(self.gpt.parameters()).dtype
        #     causal_mask = torch.full(
        #         (output_len, output_len), fill_value=min_dtype, dtype=mask_dtype, device=output.sequences.device
        #     )
        #     causal_mask = torch.triu(causal_mask, diagonal=1)
        #     # causal_mask *= torch.arange(output_len, device=output.sequences.device) > cache_position.reshape(-1, 1)
        #     causal_mask = causal_mask[None, None, :, :]
        #     padding_mask = causal_mask + attention_mask
        #     padding_mask = padding_mask == 0
        #     causal_mask = causal_mask.masked_fill(
        #         padding_mask, min_dtype
        #     )
        #     causal_masks.append(causal_mask)
        causal_mask = construct_attn_mask(attention_masks_full_view, output.attention_mask_ids, trunc_index, input_attention_masks if input_full_attention_mask==False else None, None, None, output.sequences.device, next(self.gpt.parameters()).dtype)
        _logger.debug(f"causal_mask shape: {causal_mask.shape}")

        # ========== Per-segment semantic-token statistics. ==========
        total_semantic_tokens = output.sequences.shape[1] - 1
        selected_beam = output.beam_indices[0][0]
        # Convert tensor to int for dict key access
        if isinstance(selected_beam, torch.Tensor):
            selected_beam = selected_beam.item()
        seg_lens = []
        
        # Pull positions from the HMM segment-switch record.
        if hasattr(self.inference_model, 'segment_positions') and self.inference_model.segment_positions is not None:
            # Pretty-print per-segment statistics through the logger.
            _logger.print_segment_stats(
                segment_positions=self.inference_model.segment_positions,
                total_tokens=total_semantic_tokens,
                selected_beam=selected_beam
            )

            # Collect segment lengths for the selected beam.
            positions = self.inference_model.segment_positions[selected_beam]
            all_positions = [0] + positions + [total_semantic_tokens]
            for seg_idx in range(len(all_positions) - 1):
                start_pos = all_positions[seg_idx]
                end_pos = all_positions[seg_idx + 1]
                seg_lens.append(end_pos - start_pos)

            # Reset per-run state.
            if self.inference_model.duration_processor is not None:
                self.inference_model.duration_processor = None
            if self.inference_model.duration_controller is not None:
                self.inference_model.duration_controller = None
            self.inference_model.segment_positions = None
        else:
            _logger.warning("No segment switching record found")
            _logger.debug(f"  hasattr check: {hasattr(self.inference_model, 'segment_positions')}")
            if hasattr(self.inference_model, 'segment_positions'):
                _logger.debug(f"  segment_positions value: {self.inference_model.segment_positions}")
        # ========== End of per-segment statistics. ==========

        # if isinstance(output, torch.Tensor):
        #     print(f"Generated output shape: {output[:, trunc_index:].shape}")
        #     return output[:, trunc_index:], speech_conditioning_latent, attention_mask
        # GenerateOutput
        # Optional debugging dump: write the causal_mask to mask.csv.
        # import pandas as pd
        # pd.DataFrame(causal_mask[0,0].cpu().numpy()).to_csv('mask.csv', index=False, header=False)
        if speech_conditioning_latent_list is not None:
            speech_conditioning_latent = speech_conditioning_latent_list
        return output.sequences, speech_conditioning_latent, causal_mask, seg_lens, token_generation_time

    def get_emovec(self, emo_speech_conditioning_latent, emo_cond_lengths):
        emo_vec_syn_ori = self.get_emo_conditioning(emo_speech_conditioning_latent.transpose(1,2), emo_cond_lengths)
        emo_vec_syn = self.emovec_layer(emo_vec_syn_ori)
        emo_vec = self.emo_layer(emo_vec_syn)
        return emo_vec

    def merge_emovec(self, speech_conditioning_latent, emo_speech_conditioning_latent, cond_lengths, emo_cond_lengths, alpha = 1.0):
        emo_vec = self.get_emovec(emo_speech_conditioning_latent, emo_cond_lengths)
        base_vec = self.get_emovec(speech_conditioning_latent, cond_lengths)

        out = base_vec + alpha * (emo_vec - base_vec)
        return out
