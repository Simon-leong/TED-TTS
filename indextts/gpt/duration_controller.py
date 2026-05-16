import torch

from indextts.utils.logger import get_logger

_logger = get_logger("DurationCtrl")


class IntraSegmentDurationController:
    """Intra-segment proportional duration controller.

    During autoregressive generation, this controller dynamically adjusts the
    duration embedding cursor based on the current HMM alignment position so
    that the per-segment token/text ratio tracks the requested target duration.

    Workflow:
        1. Use the HMM-provided text_pos to estimate per-segment text progress.
        2. Map text progress to an expected token count.
        3. Compare against the actual generated token count to obtain the error.
        4. A proportional control law adjusts the duration-embedding cursor:
             - lagging behind the target  -> increase the cursor (more budget)
             - ahead of the target        -> decrease the cursor (tightening)
    """

    def __init__(
        self,
        get_duration_embeddings_fn,
        target_tokens_per_segment: list[int],
        text_tokens_per_segment: list[int],
        device,
        max_mel_tokens: int,
        k_p: float = 25.0,
        eps: float = 0.01,
        delta_max: int = 10,
        update_freq: int = 5,
        verbose: bool = True,
    ):
        """
        Args:
            get_duration_embeddings_fn: callable (idx_tensor, check=True) -> embedding.
            target_tokens_per_segment:  target semantic token count for each segment.
            text_tokens_per_segment:    text token count for each segment.
            device:                     torch device.
            max_mel_tokens:             maximum mel token count (used to clamp the cursor).
            k_p:                        proportional gain.
            eps:                        dead band for the normalized error
                                        (|error / target| < eps disables the update).
            delta_max:                  maximum per-step adjustment.
            update_freq:                run the controller every N generation steps.
            verbose:                    log progress information.
        """
        self.get_dur_emb = get_duration_embeddings_fn
        self.target_per_seg = list(target_tokens_per_segment)
        self.text_per_seg = list(text_tokens_per_segment)
        self.device = device
        self.max_mel_tokens = max_mel_tokens
        self.k_p = k_p
        self.eps = eps
        self.delta_max = delta_max
        self.update_freq = update_freq
        self.verbose = verbose

        self.num_segments = len(target_tokens_per_segment)
        self.current_segment_idx = 0
        self.total_steps_generated = 0
        self.steps_in_current_segment = 0

        # Cumulative text boundaries; convert a global text_pos to an in-segment offset.
        self._cum_text = [0]
        for t in text_tokens_per_segment:
            self._cum_text.append(self._cum_text[-1] + t)

        # Cumulative targets; base the global cursor on the end-of-segment goal.
        self._cum_target = [0]
        for t in target_tokens_per_segment:
            self._cum_target.append(self._cum_target[-1] + t)

        # Current adjustment produced by the proportional law.
        self._delta = 0.0

        # Preallocated index tensor reused every step to avoid torch.full() churn.
        self._dur_idx_buf = torch.zeros(1, device=device, dtype=torch.long)

        # Cache of the most recent cursor value and the embedding it produced.
        self._cached_t = -1
        self._cached_emb = None

        if verbose:
            _logger.info("[IntraSegDurCtrl] Initialized")
            _logger.debug(f"  Num segments: {self.num_segments}")
            _logger.debug(f"  Target per segment: {self.target_per_seg}")
            _logger.debug(f"  Text per segment: {self.text_per_seg}")
            _logger.debug(f"  Total target: {self._cum_target[-1]} tokens")
            _logger.debug(f"  k_p={k_p}, eps={eps}, delta_max={delta_max}, update_freq={update_freq}")

    def switch_to_segment(self, seg_idx: int):
        """Handle an explicit segment switch (triggered by an HMM attention-phase change)."""
        if seg_idx == self.current_segment_idx:
            return
        old_seg = self.current_segment_idx
        self.current_segment_idx = min(seg_idx, self.num_segments - 1)
        self.steps_in_current_segment = 0
        self._delta = 0.0
        # Invalidate the cache so the next step recomputes the embedding.
        self._cached_t = -1
        if self.verbose:
            _logger.debug(
                f"[IntraSegDurCtrl] Segment switch: {old_seg} -> {self.current_segment_idx}"
            )

    def step(self, text_pos: int):
        """Run one proportional-control step.

        Args:
            text_pos: current HMM-aligned text position (global, 0-based).

        Returns:
            Tuple of (segment_idx, steps_in_current_segment, new_duration_embedding).
        """
        self.total_steps_generated += 1
        self.steps_in_current_segment += 1

        seg = min(self.current_segment_idx, self.num_segments - 1)

        # Apply the proportional update only every update_freq steps.
        if self.total_steps_generated % self.update_freq == 0:
            seg_text_start = self._cum_text[seg]
            seg_text_len = self.text_per_seg[seg]
            local_text_pos = max(0, text_pos - seg_text_start)
            text_progress = min(local_text_pos / max(seg_text_len, 1), 1.0)

            expected = text_progress * self.target_per_seg[seg]
            actual = self.steps_in_current_segment

            normalized_error = (expected - actual) / max(self.target_per_seg[seg], 1)

            if abs(normalized_error) > self.eps:
                raw_delta = self.k_p * normalized_error
                self._delta = max(-self.delta_max, min(raw_delta, self.delta_max))
            else:
                self._delta = 0.0

            if self.verbose and self.total_steps_generated % (self.update_freq * 10) == 0:
                _logger.debug(
                    f"[IntraSegDurCtrl] step={self.total_steps_generated} | "
                    f"seg={seg} | text_pos={text_pos} | progress={text_progress:.2f} | "
                    f"expected={expected:.0f} | actual={actual} | delta={self._delta:+.1f}"
                )

        # Compute the adjusted global cursor.
        base_cursor = self._cum_target[seg + 1]  # cumulative target at end of segment
        adjusted = base_cursor + self._delta

        t = max(1, min(int(adjusted), self.max_mel_tokens - 1))

        # Recompute the embedding only when the cursor changes.
        if t != self._cached_t:
            self._dur_idx_buf.fill_(t)
            self._cached_emb = self.get_dur_emb(self._dur_idx_buf, check=False)
            self._cached_t = t

        return seg, self.steps_in_current_segment, self._cached_emb
