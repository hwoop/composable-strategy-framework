import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BertEncoder(nn.Module):
    def __init__(self, model_name: str = 'bert-base-uncased', freeze_layers: int = 8):
        super().__init__()
        
        bert_config = BertConfig.from_pretrained(model_name, output_hidden_states=True)
        self.bert_model = BertModel.from_pretrained(model_name, config=bert_config)

        if freeze_layers > 0:
            self._freeze_bert_layers(freeze_layers)


    def _freeze_bert_layers(self, num_layers_to_freeze: int):
        for name, param in self.bert_model.named_parameters():
            if "encoder.layer" in name:
                layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                if layer_num < num_layers_to_freeze:
                    param.requires_grad = False
            # MISA 원본에서는 embedding layer 등 다른 부분은 동결하지 않으므로,
            # 특정 레이어만 동결하도록 설정합니다.


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        """
        BERT 모델을 통해 텍스트 특징을 추출하고, MISA 방식에 따라 Masked Mean을 적용합니다.

        Args:
            bert_sentences (torch.Tensor): Bert Tokenizer로 인코딩된 토큰 ID 텐서.
            bert_sentence_att_mask (torch.Tensor): 어텐션 마스크 텐서.
            bert_sentence_types (torch.Tensor): 세그먼트 ID 텐서.

        Returns:
            torch.Tensor: (Batch, 768) 크기의 텍스트 표현 벡터.
        """
        bert_output = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        last_hidden_state = bert_output.last_hidden_state

        # Masked Mean 계산
        masked_output = torch.mul(attention_mask.unsqueeze(2), last_hidden_state)
        mask_len = torch.sum(attention_mask, dim=1, keepdim=True)
        
        # 분모가 0이 되는 것을 방지
        utterance_text = torch.sum(masked_output, dim=1, keepdim=False) / (mask_len + 1e-9)
        
        return utterance_text


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        batch_first: bool = True,
    ):
        super().__init__()
        assert num_layers == 2, "MisaLSTMEncoder only supports num_layers=2"
        
        self.lstm1 = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=1, 
            bidirectional=True, 
            batch_first=batch_first
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2, 
            hidden_size=hidden_size, 
            num_layers=1, 
            bidirectional=True, 
            batch_first=batch_first
        )
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        # 첫 번째 LSTM 레이어 통과
        packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_h1, (final_h1, _) = self.lstm1(packed_x)

        # Layer Normalization을 위해 패딩 복구 및 다시 압축
        padded_h1, _ = pad_packed_sequence(packed_h1, batch_first=True)
        normed_h1 = self.layer_norm(padded_h1)
        normed_h1 = self.dropout(normed_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # 두 번째 LSTM 레이어 통과
        _, (final_h2, _) = self.lstm2(packed_normed_h1)

        # 두 LSTM의 최종 은닉 상태(final_h)를 연결 (2, B, 2H)
        combined_h = torch.cat((final_h1, final_h2), dim=2)
        
        # (B, 4H) 형태의 최종 벡터로 변환
        utterance_representation = combined_h.permute(1, 0, 2).contiguous().view(batch_size, -1)
        return utterance_representation