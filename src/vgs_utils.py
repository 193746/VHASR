import torch
import torch.nn as nn

class SpeechAdapter(nn.Module):
    def __init__(self, d_model=512, num_layers=1):
        super(SpeechAdapter, self).__init__()
        attn_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=3072,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.dim = d_model
        self.self_attn = nn.TransformerEncoder(attn_layer, num_layers=num_layers)
        self.cls = nn.Parameter(torch.randn([1,1,d_model]))
        self.output_proj = nn.Linear(d_model,d_model)
    
    def get_keypadding_mask(self,audio_mask):
        bsz = audio_mask.shape[0]
        src_mask = torch.zeros([bsz, 1]).to(audio_mask.device)
        keypadding_mask = torch.cat((src_mask,~audio_mask),dim=-1)
        return keypadding_mask.bool()
    
    def forward(
        self, audio_feat: torch.Tensor, audio_mask: torch.Tensor # [4,390,512] [4,390]
    ) -> torch.Tensor:
        bsz = audio_feat.shape[0]
        cls_ = torch.cat([self.cls] * bsz, dim=0) # cls [4,1,512]
        src = torch.cat([cls_, audio_feat], dim=1) # [4,391,512]
        key_padding_mask = self.get_keypadding_mask(audio_mask)
        out = self.self_attn(src=src, src_key_padding_mask=key_padding_mask)
        out = out[:, :1].reshape(-1, self.dim)
        out = self.output_proj(out)
        return out