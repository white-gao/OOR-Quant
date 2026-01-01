import torch
import torch.nn as nn
import torch.nn.functional as F
from onerec_llm.utils.time_tracker import TimeTracker

# ===================================================================
# Cross-Entropy Loss Function
# ===================================================================

class CrossEntropyLoss(nn.Module):
    """
    An efficient CrossEntropyLoss module that avoids redundant calculations.
    It first computes per-token losses and then manually applies the reduction.
    (Based on the user-provided, superior implementation).
    """
    def __init__(self,
                 ignore_index: int = -100,
                 return_token_loss: bool = False,
                 shift_labels: bool = True,
                 reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.return_token_loss = return_token_loss
        self.reduction = reduction
        self.shift_labels = shift_labels

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            logits (torch.Tensor): A single tensor of shape (..., vocab_size).
            labels (torch.Tensor): Ground truth labels.
        """
        vocab_size = logits.shape[-1]
        
        if self.shift_labels:
          logits = logits[:, :-1, :]
          labels = labels[:, 1:]

        # Reshape for cross-entropy calculation
        logits_flat = logits.float().reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)

        # Step 1: Compute per-token loss. This is the base for all other calculations.
        per_token_loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=self.ignore_index,
            reduction="none"
        )
        
        # Step 2: Manually apply reduction to get the final loss.
        loss = per_token_loss.sum()
        if self.reduction == "mean":
            # Ensure we divide by the number of valid (non-ignored) tokens
            total_elements = (labels_flat != self.ignore_index).sum()
            if total_elements > 0:
                loss /= total_elements
            else: # Handle case where all tokens are ignored
                loss.zero_()

        # Return what's requested
        if self.return_token_loss:
            return loss, per_token_loss
        
        return loss


# ===================================================================
# Memory-Efficient Chunked Loss Computer
# ===================================================================

class ChunkedLossComputer:
    """
    内存高效的分块损失计算器，用于解决大型语言模型中lm_head过大导致的显存不足问题。
    
    通过将输入序列分块计算，手动累加梯度，避免一次性为整个序列分配巨大的中间张量。
    
    注意：返回的loss已经反向传播并detach，不能用于需要梯度的操作。
    """
    def __init__(self, lm_head: nn.Module, loss_fn: nn.Module, minibatch_size: int, shift_labels: bool = True):
        """
        初始化分块损失计算器。
        
        Args:
            lm_head: 语言模型的输出层（通常是nn.Linear）
            loss_fn: 损失函数，必须返回 (avg_loss, per_token_loss) 元组
            minibatch_size: 每个分块的大小，用于控制内存使用
            shift_labels: 是否偏移标签（用于自回归模型）
        """
        if not isinstance(lm_head, nn.Module) or not isinstance(loss_fn, nn.Module):
            raise TypeError("lm_head和loss_fn必须是nn.Module的实例")
            
        self.lm_head = lm_head
        self.loss_fn = loss_fn
        self.minibatch_size = minibatch_size
        self.shift_labels = shift_labels
        self.loss_info = {}
        self.ticker = TimeTracker()

    def forward_and_backward(self, input: torch.Tensor, labels: torch.Tensor, loss_fn_args: dict = {}):
        """
        执行分块的前向和反向传播过程。
        
        Args:
            input: 输入张量，形状为 [batch_size, seq_len, hidden_dim]
            labels: 标签张量，形状为 [batch_size, seq_len]
            loss_fn_args: 传递给损失函数的额外参数
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (final_avg_loss, per_token_loss)
            
        注意：返回的loss已经反向传播并detach，不能用于需要梯度的操作。
        """
        self.ticker.tick("lm_head")
        params = list(self.lm_head.parameters())
        grad_accs = [torch.zeros_like(p) for p in params]
        grad_input_full = torch.zeros_like(input)

        total_loss_sum_for_reporting = torch.tensor(0.0, device=input.device)
        all_per_token_losses = []

        seq_len = input.size(1)
        
        # 计算总有效元素数量
        labels_to_count = labels[:, 1:] if self.shift_labels else labels
        total_elements = (labels_to_count != getattr(self.loss_fn, 'ignore_index', -100)).sum()
        
        if total_elements.item() == 0:
            return torch.tensor(0.0, device=input.device), None

        # 分块计算前向和梯度累加
        for i in range(0, seq_len, self.minibatch_size):
            start, end = i, min(i + self.minibatch_size, seq_len)
            input_chunk = input[:, start:end, :].detach().requires_grad_()
            
            logits_chunk = self.lm_head(input_chunk)

            if self.shift_labels:
                label_start, label_end = start + 1, end + 1
                labels_chunk = labels[:, label_start:label_end]
                # 确保logits和labels长度匹配
                if logits_chunk.size(1) > labels_chunk.size(1):
                    logits_chunk = logits_chunk[:, :labels_chunk.size(1), :]
            else:
                labels_chunk = labels[:, start:end]

            if labels_chunk.numel() == 0:
                continue

            logits_flat = logits_chunk.reshape(-1, self.lm_head.out_features)
            labels_flat = labels_chunk.reshape(-1)
            
            # 计算损失
            loss_chunk_avg, per_token_loss_chunk = self.loss_fn(logits_flat, labels_flat, **loss_fn_args)

            # 转换为sum loss用于反向传播
            valid_tokens_in_chunk = (labels_flat != getattr(self.loss_fn, 'ignore_index', -100)).sum()
            
            if valid_tokens_in_chunk.item() == 0:
                all_per_token_losses.append(per_token_loss_chunk.detach())
                continue
            
            loss_chunk_sum = loss_chunk_avg * valid_tokens_in_chunk

            # 手动计算梯度并累加
            tensors_to_grad = [p for p in params if p.requires_grad] + [input_chunk]
            grads = torch.autograd.grad(outputs=loss_chunk_sum, inputs=tensors_to_grad, retain_graph=False)
        
            grad_idx = 0
            for j in range(len(params)):
                if params[j].requires_grad:
                    grad_accs[j] += grads[grad_idx]
                    grad_idx += 1
            grad_input_full[:, start:end, :] = grads[grad_idx]

            total_loss_sum_for_reporting += loss_chunk_sum.detach()
            all_per_token_losses.append(per_token_loss_chunk.detach())
        
        # 应用累加的梯度
        for j, p in enumerate(params):
            if p.requires_grad:
                p.grad = grad_accs[j] / total_elements

        self.ticker.tick("llm")        
        input.backward(gradient=grad_input_full / total_elements)
        self.ticker.tick("done")
        
        final_avg_loss = (total_loss_sum_for_reporting / total_elements).detach()
        per_token_loss = torch.cat(all_per_token_losses) if all_per_token_losses else None
        final_avg_loss.requires_grad = True

        self.loss_info = {
            'loss': final_avg_loss,
            'per_token_loss': per_token_loss
        }
        return final_avg_loss, per_token_loss
