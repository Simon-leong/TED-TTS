import torch
import os

class AttentionMapProcessor:
    def __init__(self, beam_index: int = 0):
        self.beam_index = beam_index
        self.attn_map_dict = dict()
        self.attn_phase_dict = dict()
        self.emotion_twist = dict()

        self.save_dir = "./attention_results"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def process_attention_map(self, attns: tuple,
                              attn_phase: list,
                              output_path: str = None,
                              ):
        """Collect and store the per-step attention map.

        Args:
            attns: tuple with one tensor per layer, each shaped
                (beam_size, num_heads, query_len, key_len).
            attn_phase: current per-beam attention phase tags.
            output_path: key used to group all maps produced for one sample.
        """
        all_layers_last_token = [layer[self.beam_index, :, -1, :].detach().cpu() for layer in attns]
        all_layer_attn = torch.stack(all_layers_last_token, dim=0)  # (num_layers, num_heads, key_len)

        if output_path:
            if output_path not in self.attn_map_dict:
                self.attn_map_dict[output_path] = []
                self.attn_phase_dict[output_path] = []

            self.attn_map_dict[output_path].append(all_layer_attn)
            self.attn_phase_dict[output_path].append(attn_phase)

    def process_emotion_twist_detect(self,
                                     attn_wo_head = None,
                                     method = None,
                                     token_idx = None,
                                     attention_phase = None,
                                     output_path: str = None,
                                     text_last_token_position: tuple = None):
        if output_path:
            if output_path not in self.emotion_twist:
                self.emotion_twist[output_path] = dict(
                    attn_wo_head = [],
                    text_last_token_position = None,
                    token_idx = [],
                    attention_phases = [],
                )
                print(f"Initialized emotion_twist entry for {output_path}")

            # Record the latest emotion-twist detection state.
            if attn_wo_head is not None:
                self.emotion_twist[output_path]['attn_wo_head'].append(attn_wo_head.detach().cpu())
            self.emotion_twist[output_path]['token_idx'].append(token_idx.detach().cpu())
            self.emotion_twist[output_path]['attention_phases'].append(attention_phase)
            self.emotion_twist[output_path]['text_last_token_position'] = text_last_token_position
            self.emotion_twist[output_path]['method'] = method

    def process_mas_mu(self, mu, output_path):
        if output_path in self.emotion_twist:
            if 'mu' not in self.emotion_twist[output_path]:
                self.emotion_twist[output_path]['mu'] = []
            self.emotion_twist[output_path]['mu'].append(mu.detach().cpu())

    def prcess_hmm(self, hmm_center, attn, attn_norm, belief, output_path, attn_phase, text_last_token_position):
        if output_path not in self.emotion_twist:
            self.emotion_twist[output_path] = dict()
        if output_path in self.emotion_twist:
            if 'attn_wo_head' not in self.emotion_twist[output_path]:
                self.emotion_twist[output_path]['attn_wo_head'] = []
            self.emotion_twist[output_path]['attn_wo_head'].append(attn.detach().cpu())

            if 'attn_wo_head_norm' not in self.emotion_twist[output_path]:
                self.emotion_twist[output_path]['attn_wo_head_norm'] = []
            self.emotion_twist[output_path]['attn_wo_head_norm'].append(attn_norm.detach().cpu())

            if 'belief' not in self.emotion_twist[output_path]:
                self.emotion_twist[output_path]['belief'] = []
            self.emotion_twist[output_path]['belief'].append(belief.detach().cpu())

            if 'mu' not in self.emotion_twist[output_path]:
                self.emotion_twist[output_path]['mu'] = []
            self.emotion_twist[output_path]['mu'].append(hmm_center.detach().cpu())

            if 'attention_phases' not in self.emotion_twist[output_path]:
                self.emotion_twist[output_path]['attention_phases'] = []
            self.emotion_twist[output_path]['attention_phases'].append(attn_phase)

            if 'text_last_token_position' not in self.emotion_twist[output_path]:
                self.emotion_twist[output_path]['text_last_token_position'] = text_last_token_position
            self.emotion_twist[output_path]['text_last_token_position'] = text_last_token_position
        else:
            print(f"Warning: output_path {output_path} not found in emotion_twist dict.")

    def save_attention_maps(self, input_embeds_len: int):
        """Persist collected attention maps to disk, one .pt file per output_path."""
        for output_path, attn_maps in self.emotion_twist.items():
            save_path = os.path.join(self.save_dir, os.path.basename(output_path) + ".pt")
            save_dict = dict(
                emotion_twist = self.emotion_twist.get(output_path, None),
            )

            torch.save(save_dict, save_path)
            print(f"Saved attention maps to {save_path}")

    def save_hmm_states(self,):
        """Persist collected HMM states to a single .pt file."""
        torch.save(self.emotion_twist, os.path.join(self.save_dir, "hmm_states.pt"))
        print(f"Saved HMM states to {os.path.join(self.save_dir, 'hmm_states.pt')}")
