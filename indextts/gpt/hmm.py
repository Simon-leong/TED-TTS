import torch
import torch.nn.functional as F
from .attn_map import AttentionMapProcessor

class StreamingHMMAligner:
    def __init__(self,
                 num_beams,
                 num_text_tokens,
                 transition_prob=0.1,
                 sigma=1.2,
                 window_size=None,
                 device='cpu',
                 debug=False,
                 enable_std_head_prune=False,
                 std_head_topk=10):
        """Initialize the aligner.

        Args:
            num_text_tokens: length N of the text sequence.
            transition_prob: per-step probability p of moving one position
                forward (e.g. 0.3 means a 30% chance to advance each step).
            sigma: standard deviation of the Gaussian smoothing kernel.
            window_size: kernel window size (defaults to 6*sigma when None).
        """
        self.Sk = num_text_tokens
        self.B = num_beams
        self.p = transition_prob
        self.device = device
        self.debug = debug
        self.enable_std_head_prune = enable_std_head_prune
        self.std_head_topk = std_head_topk

        # Initialize the latent belief state.
        self.belief = torch.zeros((self.B, self.Sk), device=self.device)
        self.belief[:, 1] = 1.0  # all probability mass starts on the first position

        # Track alignment history for debugging.
        self.history = []

        # Precompute the Gaussian kernel used in steps B and C.
        self.kernel_radius = int(3 * sigma) if window_size is None else window_size // 2
        self.gaussian_kernel = self._build_gaussian_kernel(self.kernel_radius, sigma).to(self.device)

        # Preallocate the index vector used during expectation computation.
        self.indices = torch.arange(self.Sk, device=self.device, dtype=self.belief.dtype)

    @staticmethod
    def _normalize(tensor, dim=-1, eps=1e-30):
        """Numerically stable softmax-based normalization (log-space)."""
        log_tensor = torch.log(tensor + eps)
        normed = F.softmax(log_tensor, dim=dim)
        return normed


    def _build_gaussian_kernel(self, radius, sigma):
        """Build the discrete Gaussian convolution kernel g."""
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        # Gaussian: exp(-x^2 / (2 sigma^2))
        kernel = torch.exp(- (x ** 2) / (2 * sigma ** 2))
        # Normalize so the kernel integrates to 1.
        kernel = self._normalize(kernel, dim=0)
        # Reshape for conv1d: [out_channels, in_channels, kernel_size] -> [1, 1, K]
        return kernel.view(1, 1, -1)

    def _build_transition_matrix(self):
        """Build the state transition matrix."""
        transition_matrix = torch.zeros((self.Sk, self.Sk), device=self.device)

        # Main diagonal: probability of staying in place.
        transition_matrix = torch.eye(self.Sk, device=self.device) * (1 - self.p)
        # Upper diagonal: probability of advancing one step.
        upper_diag = torch.full((self.Sk - 1,), self.p, device=self.device)
        transition_matrix += torch.diag(upper_diag, diagonal=1)

        transition_matrix[-1, -1] = 1.0
        return transition_matrix

    def _apply_gaussian(self, tensor_bn):
        """Apply Gaussian smoothing to a distribution via 1D convolution.

        Args:
            tensor_bn: [B, N]
        Returns:
            smoothed: [B, N]
        """
        B, N = tensor_bn.shape
        # conv1d expects [B, Channels, Length].
        x = tensor_bn.unsqueeze(1)

        # 'same' padding via replicate, padding both sides by `radius`.
        x_pad = F.pad(x, (self.kernel_radius, self.kernel_radius), mode='replicate')

        out = F.conv1d(x_pad, self.gaussian_kernel)

        # Padding introduces minor edge artefacts, negligible for streaming alignment.
        return out.squeeze(1)

    def _predict_prior(self):
        """Step 1: time update.

        Apply the band-diagonal transition: stay in place with probability
        (1 - p) or advance one step with probability p. The last state is
        absorbing.
        """
        # Sparse band-diagonal update; avoids the (B,N) x (N,N) dense matmul.
        # prior[:, i] = belief[:, i] * (1-p) + belief[:, i-1] * p
        # The absorbing tail gets an extra belief[:, -1] * p contribution.
        prior = self.belief * (1 - self.p)
        prior[:, 1:] += self.belief[:, :-1] * self.p
        prior[:, -1] += self.belief[:, -1] * self.p
        return prior

    def _select_best_head(self, prior, attention_heads):
        """Step 2: pick the attention head that best matches the prior.

        Args:
            attention_heads: shape [num_layers * num_heads, B, N].

        Uses a dot-product score (overlap with the prior).
        """
        # attention_heads: (K, B, N)
        # prior: (B, N)
        # scores: (K, B)
        scores = torch.sum(attention_heads * prior.unsqueeze(0), dim=-1)

        # best_idx: (B,)
        best_idx = torch.argmax(scores, dim=0)

        # Gather best_head: (B, N)
        b_indices = torch.arange(self.B, device=self.device)
        best_head = attention_heads[best_idx, b_indices, :]

        return best_head, best_idx

    def _select_best_head_loglik(self, prior, attention_heads):
        """Step 2 (log-likelihood variant): score = sum(head[j] * log(prior[j])).

        This is the negative cross-entropy between the head distribution and
        the prior.

        Returns:
            best_head_norm: normalized best head, shape (B, N).
            best_idx:       index of the selected head, shape (B,).
            best_head:      raw best head, shape (B, N).
        """
        # Guard against log(0).
        epsilon = 1e-10
        log_prior = torch.log(prior + epsilon) # (B, N)

        attn_norm = self._normalize(attention_heads, dim=-1) # (K, B, N)

        # Optional top-K pruning by per-head std along the N axis (disabled by default).
        # Keeps the top `std_head_topk` heads to reduce subsequent scoring cost.
        if self.enable_std_head_prune and attn_norm.shape[0] > self.std_head_topk:
            # head_std: (K, B)
            head_std = torch.std(attn_norm, dim=-1)
            k_keep = min(self.std_head_topk, attn_norm.shape[0])
            # topk_idx: (k_keep, B)
            topk_idx = torch.topk(head_std, k=k_keep, dim=0, largest=True, sorted=True).indices

            # Select each beam's own top-k heads.
            attn_norm_perm = attn_norm.permute(1, 0, 2)            # (B, K, N)
            attn_heads_perm = attention_heads.permute(1, 0, 2)     # (B, K, N)
            gather_idx = topk_idx.permute(1, 0).unsqueeze(-1).expand(-1, -1, attn_norm.shape[-1])

            pruned_attn_norm = torch.gather(attn_norm_perm, dim=1, index=gather_idx).permute(1, 0, 2)   # (k_keep, B, N)
            pruned_attention_heads = torch.gather(attn_heads_perm, dim=1, index=gather_idx).permute(1, 0, 2)  # (k_keep, B, N)

            candidate_indices = topk_idx
        else:
            pruned_attn_norm = attn_norm
            pruned_attention_heads = attention_heads
            candidate_indices = torch.arange(attn_norm.shape[0], device=attn_norm.device).unsqueeze(1).expand(-1, attn_norm.shape[1])

        # Score every (remaining) head: shape (L*H, B).
        # head distribution * log_prior, summed over N.
        scores = torch.sum(pruned_attn_norm * log_prior.unsqueeze(0), dim=-1)

        # Reject heads whose center of mass is below 1 (i.e. unfocused / near the start).
        center = torch.sum(self.indices.unsqueeze(0) * pruned_attn_norm, dim=-1) # (K, B)
        mask = center < 1.0
        scores = scores.masked_fill(mask, float('-inf'))

        best_idx_local = torch.argmax(scores, dim=0) # (B,)
        b_indices = torch.arange(self.B, device=self.device)
        best_idx = candidate_indices[best_idx_local, b_indices]  # map back to the original head index

        # Gather best_head: (B, N)
        best_head_norm = pruned_attn_norm[best_idx_local, b_indices, :]  # shape (B, N)
        # Compute the raw best head only in debug mode.
        if self.debug:
            avg = torch.mean(pruned_attention_heads, dim=-1)
            avg_best_idx_local = torch.argmax(avg, dim=0)
            best_head = pruned_attention_heads[avg_best_idx_local, b_indices, :]
        else:
            best_head = best_head_norm

        return best_head_norm, best_idx, best_head

    def step(self,
             attention_matrix_stack,
             attn_map_processor: AttentionMapProcessor = None,
             **model_kwargs):
        """Run one streaming alignment step.

        Args:
            attention_matrix_stack: attention of the current semantic token over
                all text tokens, shape [L*H, B, Sk] or [L, B, H, Sk].
            attn_map_processor: optional AttentionMapProcessor for recording
                per-step states.

        Returns:
            current_alignment: tensor of shape (B,).
            selected_head_idx: tensor of shape (B,).
        """
        # 0. Flatten to a common (K, B, N) layout.
        if attention_matrix_stack.ndim == 4:
            # [L, B, H, N] -> [L, H, B, N] -> [L*H, B, N]
            L, B, H, N = attention_matrix_stack.shape
            candidates = attention_matrix_stack.permute(0, 2, 1, 3).reshape(L * H, B, N)
        else:
            candidates = attention_matrix_stack

        # Ensure candidates are on the correct device
        if candidates.device != self.device:
            candidates = candidates.to(self.device)

        # 1. Predict the prior.
        prior = self._predict_prior()

        # ---------------------------------------------------
        # Step B: prediction and selection.
        # 1. Predict the observation distribution: \hat{o} = prior * Gaussian.
        #    This tolerates small uncertainty around the predicted position.
        # ---------------------------------------------------
        predicted_obs = self._apply_gaussian(prior)
        best_obs, best_idx, best_head_or = self._select_best_head_loglik(predicted_obs, candidates)

        # ---------------------------------------------------
        # Step C: Bayesian update.
        # 1. Likelihood P(o|z) under a Gaussian observation model is
        #    equivalent to Gaussian-smoothing the observation:
        #    \lambda_t = best_obs * Gaussian.
        # ---------------------------------------------------
        likelihood = self._apply_gaussian(best_obs)
        # Mask the likelihood so positions before the previous alignment are zeroed.
        last_alignment = torch.sum(self.indices.unsqueeze(0) * self.belief, dim=1) # (B,)

        # Zero out positions strictly below floor(last_alignment).
        mask = self.indices.unsqueeze(0) < last_alignment.floor().unsqueeze(1)   # (B, N)
        likelihood = likelihood.masked_fill(mask, 0.0)

        # 2. Posterior = prior * likelihood (add epsilon to avoid zero).
        posterior = prior * (likelihood + 1e-12)

        # Normalize the posterior.
        norm_posterior = self._normalize(posterior, dim=1)

        self.belief = norm_posterior

        # 4. Non-integer center of mass (expectation).
        # index: [0, 1, 2, ..., N-1]
        alignment_center = torch.sum(self.indices.unsqueeze(0) * norm_posterior, dim=1) # (B,)

        self.history.append(alignment_center.cpu().tolist())

        if attn_map_processor:
            attn_map_processor.prcess_hmm(
                alignment_center,
                best_head_or,
                best_obs,
                self.belief,
                output_path=model_kwargs.get('output_path', None),
                attn_phase=model_kwargs.get('attention_phase', None),
                text_last_token_position=model_kwargs.get('text_last_token_position', None),
            )

        return alignment_center, best_idx

    def reorder(self, beam_indices):
        """Reorder the belief tensor along the beam dimension."""
        self.belief = self.belief.index_select(0, beam_indices)
