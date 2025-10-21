import torch
from torch.utils.data._utils.collate import default_collate
from torch.nn.utils.rnn import pad_sequence
from typing import Any, List, Dict, Tuple
from transformers import BertTokenizer


class IHM_InputConverterCollator:
    def __init__(self, **kwargs):
        pass
    
    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        processed_samples = []
        for sample in samples:
            processed_samples.append(self._convert_inputs(sample))
        
        return default_collate(processed_samples)

    def _convert_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        converted = inputs
        # converted = inputs.copy()

        if 'label' in converted:
            converted['labels'] = converted.pop('label')
        
        if 'note_ids' in converted:
            notes_data = {
                'note_ids': converted.pop('note_ids'),
                'attention_masks': converted.pop('attention_masks')
            }
            converted['notes'] = notes_data
        
        if 'timeseries' in converted and converted['timeseries'] is not None:
            converted['timeseries'] = converted['timeseries'].float()
        
        if 'notes' in converted and converted['notes'] is not None:
            if isinstance(converted['notes'], dict):
                pass
            else:
                converted['notes'] = converted['notes'].float()

        return converted


class MISA_Collator:
    """
    다중 양식(multi-modal) 데이터를 배치로 구성하고 패딩을 적용하는 collator.
    - 텍스트 시퀀스 길이를 기준으로 내림차순 정렬합니다.
    - 각 양식(text, visual, acoustic)에 패딩을 적용합니다.
    - 모델 입력에 맞게 딕셔너리 형태로 반환합니다.
    """
    def __init__(self, extra_data: Dict[str, Any], use_bert: bool = False):
        self.word2id = extra_data['word2id']
        self.text_padding_value = self.word2id['<pad>']
        self.use_bert = use_bert

        if self.use_bert:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        

    def __call__(self, batch: List[Tuple[Tuple, Any]]) -> Dict[str, torch.Tensor]:
        # 텍스트(sentences)의 길이를 기준으로 배치 내림차순 정렬
        batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
        
        # 데이터 분리 및 패딩
        # sample[0] = (sentences, visual, acoustic), sample[1] = label
        texts = [torch.LongTensor(sample[0][0]) for sample in batch]
        visuals = [torch.FloatTensor(sample[0][1]) for sample in batch]
        acoustics = [torch.FloatTensor(sample[0][2]) for sample in batch]
        labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)

        # pad_sequence를 사용하여 패딩 적용
        padded_visuals = pad_sequence(visuals, batch_first=True) # 기본값 0.0으로 패딩
        padded_acoustics = pad_sequence(acoustics, batch_first=True) # 기본값 0.0으로 패딩
        
        # RNN과 같은 모델에서 사용될 수 있는 원본 시퀀스 길이 정보
        lengths = torch.LongTensor([text.shape[0] for text in texts])

        output = {
            'text': texts,
            'vision': padded_visuals,
            'audio': padded_acoustics,
            'labels': labels,
            'lengths': lengths
        }
        
        
        if self.use_bert:
            # 원본 MISA 로직과 동일하게, 패딩 길이를 가장 긴 GloVe 시퀀스에 맞춤
            # GloVe 기반 텍스트도 일단 생성
            texts_glove = [torch.LongTensor(sample[0][0]) for sample in batch]
            padded_texts_glove = pad_sequence(texts_glove, batch_first=True, padding_value=self.text_padding_value)
            
            # 실제 문장(actual_words)을 합쳐서 BERT 토크나이징 수행
            sentences = [" ".join(sample[0][3]) for sample in batch]
            
            # MISA 원본과 같이 max_length를 GloVe 시퀀스 길이에 맞춤
            max_len = padded_texts_glove.size(1)

            encoded_bert = self.tokenizer.batch_encode_plus(
                sentences,
                max_length=max_len + 2, # [CLS], [SEP] 토큰 고려
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            output['input_ids'] = encoded_bert['input_ids']
            output['attention_mask'] = encoded_bert['attention_mask']
            output['token_type_ids'] = encoded_bert['token_type_ids']
            # GloVe 기반 텍스트도 필요한 경우를 위해 함께 반환
            output['text'] = padded_texts_glove

        else: # 기존 GloVe 방식
            texts = [torch.LongTensor(sample[0][0]) for sample in batch]
            padded_texts = pad_sequence(texts, batch_first=True, padding_value=self.text_padding_value)
            output['text'] = padded_texts

        return output        
    
    

class TriDiRA_Collator:
    """
    TriDiRA 모델을 위한 데이터를 배치로 구성하고 토큰화 및 패딩을 적용합니다.
    - 텍스트는 BERT 토크나이저를 사용해 고정된 길이로 처리합니다.
    - Audio, Vision 데이터는 배치 내 최대 길이에 맞춰 패딩합니다.
    - 모델 입력에 맞는 딕셔너리 형태로 반환합니다.
    """
    def __init__(self, max_length: int = 50, **kwargs):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_length = max_length


    def __call__(self, batch: List[Tuple[Tuple, Any]]) -> Dict[str, torch.Tensor]:
        # sample[0] = (glove_ids, visual, acoustic, raw_text)
        # sample[1] = label
        visuals = [torch.FloatTensor(sample[0][1]) for sample in batch]
        acoustics = [torch.FloatTensor(sample[0][2]) for sample in batch]
        sentences = [" ".join(sample[0][3]) for sample in batch]
        labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)

        # 2. Audio/Vision 패딩
        padded_vision = pad_sequence(visuals, batch_first=True, padding_value=0.0)
        padded_audio = pad_sequence(acoustics, batch_first=True, padding_value=0.0)

        # 3. Text 토큰화 (TriDiRA 방식)
        # max_length를 고정된 값으로 사용하여 토큰화합니다.
        encoded_bert = self.tokenizer.batch_encode_plus(
            sentences,
            max_length=self.max_length,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        batch_size = encoded_bert['input_ids'].size(0)
        output = {
            'input_ids': encoded_bert['input_ids'],
            'attention_mask': encoded_bert['attention_mask'],
            'token_type_ids': encoded_bert['token_type_ids'],
            'audio': padded_audio,
            'vision': padded_vision,
            'labels': labels,
            'modality_labels': {
                't': torch.full((batch_size,), 0, dtype=torch.long),
                'a': torch.full((batch_size,), 1, dtype=torch.long),
                'v': torch.full((batch_size,), 2, dtype=torch.long),
            }
        }
        
        return output