from torchmetrics import Metric
from torchmetrics.functional import cosine_similarity
import torch

class MeanCosineSimilarity(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # .add_state()를 통해 상태 변수(state variable)를 등록
        # DDP 환경에서 자동으로 동기화
        self.add_state("sum_cosine_similarity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # (B, N, D) -> (B*N, D)
        # (B, C, H, W, D) -> (B*C*H*W, D)
        D = preds.size(-1) 
        preds = preds.reshape(-1, D)
        target = target.reshape(-1, D)
            
        cos_sim_values = cosine_similarity(preds, target, reduction='none') 
        self.sum_cosine_similarity += torch.sum(cos_sim_values)
        self.total += cos_sim_values.numel()
        

    def compute(self) -> torch.Tensor:
        return self.sum_cosine_similarity / self.total