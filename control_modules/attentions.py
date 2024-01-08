from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from diffusers.models.attention import CrossAttention

from einops import rearrange, repeat


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class CrossAttentionIPAdapter(CrossAttention):
    """The cross-attention layer with ip-adapter added"""

    def __init__(
        self,
        query_dim,
        inner_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        lam=1,
        num_tokens=4,
        ctrl_scale=1.0,
        *args,
        **kwargs
    ):
        """
        Args:
            lam : the weight scale of image prompt
            num_tokens: The context elgnth of the image features.
        """
        # TODO: Figure out the proper way to initialize the CrossAttention
        super().__init__(query_dim, *args, **kwargs)
        # inner_dim = dim_head * heads # TODO: why is it?
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5  # TODO: why is it?
        self.heads = heads
        self.lam = lam
        self.num_tokens = num_tokens
        self.ctrl_scale = ctrl_scale

        # Here are the weights for q, k ,v
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # Here are the weights for the ip dapter k` and v`
        self.to_k_ip = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v_ip = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
    
    def compute_ip_attn(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        # TODO: how is it possible that self-attention is supported here?
        context = default(context, x)
        k = self.to_k_ip(context)
        v = self.to_v_ip(context)

        # q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=h), (q, k, v))

        q = self.reshape_heads_to_batch_dim(q)
        k = self.reshape_heads_to_batch_dim(k)
        v = self.reshape_heads_to_batch_dim(v)

        sim = einsum("b i d, b j d -> b i j", q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, "b ... -> b (...)")
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, "b j -> (b h) () j", h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        out = einsum("b i j, b j d -> b i d", attn, v)  # how out compuated from attn?
        # out = rearrange(out, "(b h) n d -> b n (h d)", h=h)
        return out
    
    def forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
        ):
            context = encoder_hidden_states
            hidden_states_local = hidden_states.clone()
            hidden_states_ipadapter = hidden_states.clone()
            if context is not None:
                context_tensor = context["CONTEXT_TENSOR"]
            else:
                context_tensor = hidden_states

            (batch_size,sequence_length, _) = hidden_states.shape

            query = self.to_q(hidden_states)

            if context is not None:
                key = self.to_k_global(context_tensor)
                value = self.to_v_global(context_tensor)
            else:
                key = self.to_k(context_tensor)
                value = self.to_v(context_tensor)

            dim = query.shape[-1]

            # TODO: this is multi-head, and we used single head in IP-Adapter nad ControlNet?
            query = self.reshape_heads_to_batch_dim(query)
            key = self.reshape_heads_to_batch_dim(key)
            value = self.reshape_heads_to_batch_dim(value)

            attention_scores = torch.matmul(query, key.transpose(-1,-2))
            attention_scores = attention_scores * self.scale

            attention_probs = attention_scores.softmax(dim=-1)

            hidden_states = torch.matmul(attention_probs, value)

            if context is not None and "LOCAL" in context:
                # Perform cross attention with the local context
                query_local = self.to_q(hidden_states_local)
                key_local = self.to_k_local(context["LOCAL"])
                value_local = self.to_v_local(context["LOCAL"])

                query_local = self.reshape_heads_to_batch_dim(query_local)
                key_local = self.reshape_heads_to_batch_dim(key_local)
                value_local = self.reshape_heads_to_batch_dim(value_local)

                attention_scores_local = torch.matmul(query_local,key_local.transpose(-1,-2))
                attention_scores_local = attention_scores_local * self.scale
                attention_probs_local = attention_scores_local.softmax(dim=-1)

                # To extract the attmap of learned [w]
                index_local = context["LOCAL_INDEX"]
                index_local = (index_local.reshape(index_local.shape[0],1).repeat((1,self.heads)).reshape(-1))
                attention_probs_clone = attention_probs.clone().permute((0,2,1))
                attention_probs_mask = attention_probs_clone[torch.arange(index_local.shape[0]),index_local]
                # Normalize the attention map
                attention_probs_mask = attention_probs_mask.unsqueeze(2) / attention_probs_mask.max()

                if "LAMBDA" in context:
                    _lambda = context["LAMBDA"]
                else:
                    _lambda = 1

                attention_probs_local = attention_probs_local * attention_probs_mask * _lambda
                hidden_states += torch.matmul(attention_probs_local,value_local)
                # value_local_list.append(value_local)

            # get output from ip adapter
            ip_adapter_context = context["CTRL_EMB"] if "CTRL_EMB" in context else None
            if ip_adapter_context is not None:
                hidden_states += self.ctrl_scale * self.compute_ip_attn(hidden_states_ipadapter, ip_adapter_context, attention_mask)

            hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            hidden_states = self.to_out[1](hidden_states)

            return hidden_states
