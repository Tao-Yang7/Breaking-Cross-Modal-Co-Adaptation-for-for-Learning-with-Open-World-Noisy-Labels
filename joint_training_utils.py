#!/usr/bin/env python
"""
Joint training utilities for stage2 merged model
"""
import os
import pdb
import itertools

import torch
import torch.nn.functional as F
import numpy as np
from models.models import NegaPromptCLIP
from open_clip import create_model_and_transforms
from torch.utils.data import Dataset, DataLoader

class JointTrainingDataset(Dataset):
    """
    A dataset class for joint training that loads data on demand
    instead of keeping all data in memory at once.
    """
    def __init__(self, dataset, indices):
        """
        Args:
            dataset: Original dataset object
            indices: Indices of the samples to include in this dataset
        """
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]
        sample = self.dataset[sample_idx]
        return {
            'data': sample['data'],
            'label': sample['label']
        }

def get_before_model(cfg, classnames):
    clip_model, _, preprocess = create_model_and_transforms(
        'ViT-B-16',
        pretrained='/data/tyang/aaaa_now/.cache/huggingface/hub/models--timm--vit_base_patch16_clip_224.openai/snapshots/977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d/open_clip_pytorch_model.bin'
    )
    clip_model.dtype = torch.float32
    clip_model.visual.input_resolution = 224
    clip_model.to('cuda')
    before_model = NegaPromptCLIP(cfg, classnames, clip_model).to('cuda')
    return before_model


import torch
import numpy as np


def unified_sample_selection(model, trainloader, cfg, epoch=None):
    """
    Unified sample selection for joint model - produces single partition result
    Args:
        model: Joint NegaPromptCLIP model with LoRA and prompts
        trainloader: Training data loader
        cfg: Configuration dictionary
        epoch: Current epoch number
    Returns:
        split_dict: Single sample partition dictionary
    """
    n_nega_ctx = cfg.get('NEGA_CTX', 2)

    # --- Inference stage: Collect all samples' scores and labels ---
    if epoch is not None:
        print(f"Epoch {epoch}: Running unified inference for sample identification...")
    else:
        print("Running unified inference for sample identification...")

    model.eval()  # Switch to evaluation mode

    all_score1 = []  # MCM-style: Max probability
    all_score2 = []  # Confidence on noisy label
    # all_score3 = []  # Top 5 confidence (Unused in selection logic, can be removed to save memory if needed)
    all_labels = []  # Noisy labels
    all_original_indices = []  # Original sample indices
    all_is_open = []  # ID/OOD flag (1: OOD, 0: ID)
    all_clean_labels = []  # Clean labels for ID samples (ground truth)
    all_is_clean = []  # Derived: 1 if ID and clean, 0 otherwise

    with torch.no_grad():
        for batch_idx, batch in enumerate(trainloader):
            data = batch['data']
            labels = batch['label']  # Noise label
            clean_labels = batch['clean_label']  # Clean label
            is_open = batch['is_open']  # ID/OOD flag
            original_indices = batch['index']  # Original sample indices

            # Move tensors to GPU
            data, labels, clean_labels, is_open = data.cuda(), labels.cuda(), clean_labels.cuda(), is_open.cuda()

            # --- Forward propagation ---
            output = model.forward_joint_warmup(data)

            # Parse logits
            logits = output.view(-1, int(output.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)
            B, C = logits.shape[0], logits.shape[1]
            logits_yes = logits[:, :, 0]  # [B, C]

            # Calculate probabilities
            # epoch check: ensure safe comparison even if epoch is None
            current_epoch = epoch if epoch is not None else 0
            if current_epoch > 18:
                logits_no = logits[:, :, 1:]
                logits_no_mean = logits_no.mean(dim=2)
                prob_margin = torch.softmax(logits_yes - logits_no_mean, dim=1)
            else:
                prob_margin = torch.softmax(logits_yes, dim=1)

            # Calculate scores
            score1 = prob_margin.max(dim=1).values  # [B]
            score2 = prob_margin[torch.arange(B), labels]  # [B]

            # Cache results
            all_score1.append(score1.cpu())
            all_score2.append(score2.cpu())
            all_labels.append(labels.cpu())
            all_original_indices.append(original_indices.cpu())
            # For ground truth tracking only
            all_is_open.append(is_open.cpu())
            all_clean_labels.append(clean_labels.cpu())
            is_clean = (~is_open.bool()) & (labels == clean_labels)
            all_is_clean.append(is_clean.cpu())

    # Merge all results
    all_score1_np = torch.cat(all_score1, dim=0).numpy()
    all_score2_np = torch.cat(all_score2, dim=0).numpy()

    all_labels_np = torch.cat(all_labels, dim=0).numpy().astype(int)  # Ensure labels are int
    all_clean_labels_np = torch.cat(all_clean_labels, dim=0).numpy().astype(int)
    all_original_indices_np = torch.cat(all_original_indices, dim=0).numpy().astype(int)

    # Ground truth arrays for stats
    all_is_open_np = torch.cat(all_is_open, dim=0).numpy().astype(bool)
    all_is_clean_np = torch.cat(all_is_clean, dim=0).numpy().astype(bool)
    import pandas as pd
    sample_info_df = pd.DataFrame({
        'index': all_original_indices_np,
        'score1': all_score1_np,
        'score2': all_score2_np,
        'label': all_labels_np,
        'clean_label': all_clean_labels_np,
        'is_open': all_is_open_np.astype(int),  # 转为0/1方便观察
        'is_clean': all_is_clean_np.astype(int)
    })
    save_dir = cfg.get('SAVE_DIR', './experiment_results')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"sample_stats_0.csv")
    sample_info_df.to_csv(save_path, index=False)
    print(f"Detailed sample info saved to {save_path}")
    exit(0)



    # --- Sample Partitioning ---
    print(f"Using unified sample selection on {len(all_score1_np)} samples...")

    # Dynamic parameter adjustment
    if epoch is not None and epoch > 0 and epoch % 10 == 0:
        # Note: This modifies the dictionary in place, affecting outside scope.
        # Ensure this is intended behavior.
        cfg['id_num'] = int(cfg['id_num'] * 0.9)

    top_k_per_class_id = int(cfg.get('id_num'))  # Ensure int
    top_k_per_class_clean = int(cfg.get('clean_num'))  # Ensure int
    bottom_k_global = int(cfg.get('ood_num'))  # Ensure int

    N_total = len(all_score1_np)
    final_sample_type = np.full(N_total, -1, dtype=int)  # -1: unmarked, 0: ID, 1: ID-clean, 2: OOD

    unique_labels = np.unique(all_labels_np)

    # ==========================================================
    # 1. Mark ID-clean samples (Score2 Top-K per class)
    # ==========================================================
    all_top_k_indices_by_score2 = []
    for label in unique_labels:
        class_indices = np.where(all_labels_np == label)[0]
        if len(class_indices) == 0:
            continue
        class_score2 = all_score2_np[class_indices]

        # Determine k
        k_to_select = min(top_k_per_class_clean, len(class_score2))
        if k_to_select <= 0: continue

        # Use argpartition for efficiency
        # Note: indices returned by argpartition are local to class_score2
        if k_to_select == len(class_score2):
            top_k_local = np.arange(len(class_score2))
        else:
            top_k_local = np.argpartition(class_score2, -k_to_select)[-k_to_select:]

        top_k_global = class_indices[top_k_local]
        all_top_k_indices_by_score2.extend(top_k_global)

    # [FIX 1] Convert to numpy array with explicit int type
    all_top_k_indices_by_score2 = np.array(all_top_k_indices_by_score2, dtype=int)

    # [FIX 2] Assign only if array is not empty
    if len(all_top_k_indices_by_score2) > 0:
        final_sample_type[all_top_k_indices_by_score2] = 1

    # ==========================================================
    # 2. Mark ID samples from remaining (Score1 Top-K per class)
    # ==========================================================
    all_top_k_indices_by_score1 = []
    for label in unique_labels:
        # Get indices of this class that are NOT yet marked (i.e., == -1)
        class_indices = np.where((all_labels_np == label) & (final_sample_type == -1))[0]

        if len(class_indices) == 0:
            continue

        class_score1 = all_score1_np[class_indices]
        k_to_select = min(top_k_per_class_id, len(class_score1))

        if k_to_select <= 0:
            continue

        if k_to_select == len(class_score1):
            top_k_local = np.arange(len(class_score1))
        else:
            top_k_local = np.argpartition(class_score1, -k_to_select)[-k_to_select:]

        top_k_global = class_indices[top_k_local]
        all_top_k_indices_by_score1.extend(top_k_global)

    # [FIX 3] Convert to numpy array with explicit int type
    all_top_k_indices_by_score1 = np.array(all_top_k_indices_by_score1, dtype=int)

    # [FIX 4] Assign only if array is not empty
    if len(all_top_k_indices_by_score1) > 0:
        final_sample_type[all_top_k_indices_by_score1] = 0

    # ==========================================================
    # 3. Mark OOD samples (Score1 Bottom-K global from unmarked)
    # ==========================================================
    # Strategy: Find global bottom K, then filter for those that are unmarked

    k_to_select_global = min(bottom_k_global, N_total)

    if k_to_select_global > 0:
        # argpartition gets the indices of the k smallest elements
        bottom_k_global_indices = np.argpartition(all_score1_np, k_to_select_global)[:k_to_select_global]

        # [FIX 5] Ensure indices are int
        bottom_k_global_indices = bottom_k_global_indices.astype(int)

        # Only mark those that are currently unmarked (-1)
        # We don't want to overwrite High-Confidence ID samples if they somehow ended up here (unlikely but safe)
        ood_mask = final_sample_type[bottom_k_global_indices] == -1

        valid_ood_indices = bottom_k_global_indices[ood_mask]

        if len(valid_ood_indices) > 0:
            final_sample_type[valid_ood_indices] = 2

    # --- Generate split dictionary ---
    split_dict = {
        'id': all_original_indices_np[final_sample_type == 0].tolist(),
        'id_clean': all_original_indices_np[final_sample_type == 1].tolist(),
        'noisy_ood': all_original_indices_np[final_sample_type == 2].tolist(),
        'unmarked': all_original_indices_np[final_sample_type == -1].tolist()
    }

    # Optional: Print summary stats
    print(
        f"Selection Summary: ID_Clean={len(split_dict['id_clean'])}, ID={len(split_dict['id'])}, OOD={len(split_dict['noisy_ood'])}, Unmarked={len(split_dict['unmarked'])}")

    return split_dict




def get_visual_attention(model, images):
    """获取ViT最后一层的CLS→patch self-attention权重"""
    transformer = model.clip_model.visual.transformer

    # 保存原始的forward方法
    original_forward = transformer.resblocks[-1].forward

    attention_weights = None

    def modified_forward(x, attn_mask=None):
        nonlocal attention_weights

        x = transformer.resblocks[-1].ln_1(x)
        attn_output, attn_weights = transformer.resblocks[-1].attn(
            query=x,
            key=x,
            value=x,
            need_weights=True,
            attn_mask=attn_mask
        )
        attention_weights = attn_weights
        x = x + attn_output
        x = x + transformer.resblocks[-1].mlp(transformer.resblocks[-1].ln_2(x))

        return x

    # 替换并执行前向传播
    transformer.resblocks[-1].forward = modified_forward

    with torch.no_grad():
        # 使用ytVisualWithLocal获取patch嵌入
        _, patch_embeds = model.ytVisualWithLocal(images)

    # 恢复原始forward
    transformer.resblocks[-1].forward = original_forward

    if attention_weights is not None:
        cls_to_patch_attention = attention_weights[:, 0, 1:]  # [batch_size, num_patches]
        return patch_embeds, cls_to_patch_attention

    return patch_embeds, None


def get_text_attention(model, text_features, images, labels):
    """计算文本注意力"""
    with torch.no_grad():
        _, patch_embeds = model.ytVisualWithLocal(images)
        patch_embeds_normalized = F.normalize(patch_embeds, dim=-1)

    B = patch_embeds_normalized.shape[0]
    text_attention_list = []

    for i in range(B):
        patch_embed = patch_embeds_normalized[i]  # [N_patch, D]
        true_class_text_feature = text_features[labels[i]]  # [D]
        text_similarity = F.cosine_similarity(patch_embed, true_class_text_feature, dim=-1)  # [N_patch]
        text_attention_list.append(text_similarity)

    return torch.stack(text_attention_list, dim=0)


def calculate_score1(model, data, n_nega_ctx):
    """Calculate score1 for samples"""
    try:
        # Try using get_lora_logits first
        output = model.get_lora_logits(data)
        if output is not None:
            logits = output.view(-1, int(output.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)
            logits_yes = logits[:, :, 0]  # [B, C] positive prompt + LoRA features
            prob_margin = torch.softmax(logits_yes, dim=1)  # [B, C]
            score1 = prob_margin.max(dim=1).values  # [B] max probability
            return score1
    except:
        pass

    # Fallback to using forward_joint_warmup if get_lora_logits fails or returns None
    output = model.forward_joint_warmup(data)
    logits = output.view(-1, int(output.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]  # [B, C]
    prob_margin = torch.softmax(logits, dim=1)  # [B, C]
    score1 = prob_margin.max(dim=1).values  # [B] max probability
    return score1


def calculate_attention_guided_loss(visual_attention, text_attention, temperature=1.0):
    """计算注意力引导损失"""
    # 归一化注意力权重
    visual_attention_norm = F.softmax(visual_attention / temperature, dim=1)
    text_attention_norm = F.softmax(text_attention / temperature, dim=1)

    # 文本引导视觉的注意力对齐损失
    loss = F.cross_entropy(visual_attention, text_attention_norm)

    return loss


def negative_prompt_training(model, trainloader, cfg, epoch, split_dict, prompt_optimizer, prompt_scheduler):
    """
    Negative and positive prompt training:
    1. Train negative prompts with existing NIS loss (ID samples) and OOD loss
    2. Train positive prompts with ce_loss + sup_loss (ID clean samples)
    3. Add uniform distribution loss for OOD samples on positive prompts
    """


    import torch
    import torch.nn.functional as F
    import numpy as np

    n_nega_ctx = cfg.get('NEGA_CTX', 2)
    batch_size = cfg.get('batch_size', 256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"Epoch {epoch}: Training Negative and Positive Prompts")
    print(f"{'='*60}")

    # Set to train text prompt branch
    model.train()

    # Extract indices from split_dict
    id_indices_np = np.array(split_dict['id'])
    id_clean_indices_np = np.array(split_dict['id_clean'])
    ood_indices_np = np.array(split_dict['noisy_ood'])

    # Get the underlying dataset from the trainloader
    full_dataset = trainloader.dataset

    # Create datasets for ID, ID-clean, and OOD samples
    id_dataset = JointTrainingDataset(full_dataset, id_indices_np.tolist())


    id_clean_dataset = JointTrainingDataset(full_dataset, id_clean_indices_np.tolist())
    ood_dataset = JointTrainingDataset(full_dataset, ood_indices_np.tolist())

    num_id = len(id_dataset)
    num_id_clean = len(id_clean_dataset)
    num_ood = len(ood_dataset)

    if num_id == 0 or num_ood == 0 or num_id_clean == 0:
        print("Warning: No ID, ID-clean, or OOD samples available for prompt training.")
        return 0.0, None

    # Create dataloaders
    id_dataloader = DataLoader(id_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    id_clean_dataloader = DataLoader(id_clean_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    ood_dataloader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Training parameters (already defined above)
    max_iter = max(num_id, num_ood, num_id_clean)
    total_loss = 0.0
    total_samples = 0

    # Training parameters
    max_iter = max(num_id, num_ood, num_id_clean)
    total_loss = 0.0
    total_samples = 0

    print(f"Training on {num_id} ID samples, {num_id_clean} ID-clean samples, and {num_ood} OOD samples...")

    from core.coteaching_utils import freeze_negative_prompt, unfreeze_positive_prompt, freeze_positive_prompt, unfreeze_negative_prompt


    # --------------------------
    # Phase 1: Train Negative Prompts
    # --------------------------
    print("Phase 1: Training Negative Prompts (freeze positive prompts)")
    freeze_positive_prompt(model)  # Freeze positive prompts
    unfreeze_negative_prompt(model)  # Unfreeze negative prompts

    # Iterate through dataloaders until both are exhausted, calculating loss for available batches
    id_iter = iter(id_dataloader)
    ood_iter = iter(ood_dataloader)
    batch_idx = 0

    while True:
        # Get next batches from both dataloaders
        id_batch = None
        ood_batch = None
        B_id = 0
        B_ood = 0

        try:
            id_batch = next(id_iter)
            B_id = id_batch['data'].size(0)
        except StopIteration:
            pass

        try:
            ood_batch = next(ood_iter)
            B_ood = ood_batch['data'].size(0)
        except StopIteration:
            pass

        # Stop when both dataloaders are exhausted
        if B_id == 0 and B_ood == 0:
            break

        prompt_optimizer.zero_grad()

        loss_nis = torch.tensor(0.0).to(device)
        loss_ood = torch.tensor(0.0).to(device)

        # NIS Loss for ID samples (if available)
        if B_id > 0:
            x_id_batch = id_batch['data'].to(device)
            output_id = model.get_lora_logits(x_id_batch)
            if output_id is None:
                # Fallback to forward_joint_warmup if get_lora_logits fails
                output_id = model.forward_joint_warmup(x_id_batch)
            logits_id = output_id.view(-1, int(output_id.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)
            logits_no = logits_id[:, :, 1:]  # Negative prompts
            logits_no_mean = logits_no.mean(dim=-1)  # Mean over negative prompts
            loss_nis = -(logits_no_mean.mean(dim=-1) - torch.logsumexp(logits_no_mean, dim=-1)).mean()

        # OOD Loss for OOD samples
        if B_ood > 0:
            x_ood_batch = ood_batch['data'].to(device)
            output_ood = model.get_lora_logits(x_ood_batch)
            if output_ood is None:
                # Fallback to forward_joint_warmup if get_lora_logits fails
                output_ood = model.forward_joint_warmup(x_ood_batch)
            logits_ood = output_ood.view(-1, int(output_ood.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)
            logits_yes = logits_ood[:, :, 0]   # Positive prompts
            logits_no = logits_ood[:, :, 1:]   # Negative prompts

            s_pos = torch.logsumexp(logits_yes, dim=1)  # Aggregate evidence for ID
            s_neg = torch.logsumexp(logits_no.reshape(logits_no.size(0), -1), dim=1)  # Aggregate evidence for all negative prompts

            loss_ood = F.softplus(s_pos - s_neg).mean()

        # Calculate and backpropagate negative prompt loss (no attention loss)
        neg_loss_batch = loss_nis + loss_ood
        neg_loss_batch.backward()
        prompt_optimizer.step()
        # Update stats
        total_loss += neg_loss_batch.item() * (B_id + B_ood)
        total_samples += (B_id + B_ood)

        if batch_idx % 20 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | "
                  f"NIS Loss: {loss_nis.item():.4f} | OOD Loss: {loss_ood.item():.4f} | "
                  f"Negative Loss: {neg_loss_batch.item():.4f}")
        # Stop if we've processed all samples
        if total_samples >= max_iter * batch_size:
            break

        batch_idx += 1  # Increment batch index


    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    prompt_scheduler.step()
    print(f"Epoch {epoch}: Negative and positive prompt training complete. Avg Loss: {avg_loss:.4f}")
    return avg_loss, None


def train_joint_text_branch(model, optimizer, scheduler, trainloader, run, cfg, epoch=None, sample_indices=None, model_states=None, classnames=None):

    # Initialize score storage if not present
    if not hasattr(model, 'score1_before'):
        model.score1_before = None
    if not hasattr(model, 'score1_epoch'):
        model.score1_epoch = -1
    n_nega_ctx = cfg.get('NEGA_CTX', 2)
    batch_size = cfg.get('batch_size', 256)
    total_loss = 0.0
    total_samples = 0

    # --- Sample Selection ---
    if sample_indices is None:
        split_dict = unified_sample_selection(model, trainloader, cfg, epoch)
    else:
        split_dict = sample_indices  # Use existing partition

    # --- Training Stage ---
    print(f"Epoch {epoch}: Training Text (Prompt) Branch...")

    # Set to train prompt branch only, freeze LoRA
    from core.coteaching_utils import freeze_lora, unfreeze_prompt
    freeze_lora(model)
    unfreeze_prompt(model)
    model.train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract ID and ID-clean indices
    id_indices = split_dict['id']
    id_clean_indices = split_dict['id_clean']

    # Get the underlying dataset from the trainloader
    full_dataset = trainloader.dataset

    # Create datasets for ID, ID-clean, and combined samples
    id_clean_dataset = JointTrainingDataset(full_dataset, id_clean_indices)
    id_dataset = JointTrainingDataset(full_dataset, id_indices)

    num_clean = len(id_clean_dataset)
    num_id = len(id_dataset)

    if num_clean == 0:
        print("No ID-clean samples, skipping training.")
        return 0.0, split_dict

    # Create dataloaders with shuffle enabled
    id_clean_dataloader = DataLoader(id_clean_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    id_dataloader = DataLoader(id_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # Training parameters
    alpha = cfg.get('mixup_alpha', 0.2)
    ce_weight = cfg.get('ce_weight', 1.0)
    sup_weight = cfg.get('sup_weight', 0.5)
    tau = cfg.get('tau', 1.0)

    # Attention loss settings
    attention_temperature = cfg.get('attention_temperature', 1.0)
    attention_loss_weight = cfg.get('attention_loss_weight', 0.1)

    # Precompute text features for attention loss
    text_features = model.clip_model.encode_text(model.prompt_learner.tokenized_prompts).detach()

    # --- Sample Selection for Attention Loss ---
    # Combine ID-clean and ID samples (union)
    combined_indices = sorted(list(set(id_indices).union(set(id_clean_indices))))
    num_combined = len(combined_indices)

    # Create combined dataset
    combined_dataset = JointTrainingDataset(full_dataset, combined_indices)

    # Calculate score differences using the exact model pairs specified by user
    score_difference = None

    if model_states is not None:
        # ------------------------
        # 确定当前计算需要使用的模型对
        # ------------------------
        # 获取当前epoch和cycle数
        cycle = epoch // 6  # 当前处于第几个完整循环
        remainder = epoch % 6  # 当前循环内的位置

        before_epoch = -1  # 默认使用初始模型
        if remainder == 3:  # Only calculate attention loss when remainder == 3 for text branch
            # 计算 Text 端训练前后的模型得分差: epoch3(当前cycle) - epoch-1(初始) 或 epoch9 - epoch5...
            if cycle == 0:
                before_epoch = -1  # 第一个循环使用初始模型
            else:
                before_epoch = 6 * (cycle - 1) + 5  # 后续循环使用上一循环的最终模型
        else:
            # Don't calculate attention loss for other remainders in text branch
            before_epoch = None
            score_difference = None  # Explicitly set to None to skip attention loss calculation

        # ------------------------
        # 加载历史模型计算score1_before
        # ------------------------
        if before_epoch is not None and before_epoch in model_states:
            print(f"计算 Attention Loss: 当前模型 (epoch {epoch}) - 历史模型 (epoch {before_epoch})")
            # 创建临时模型并加载历史状态（显存优化：避免deepcopy，直接创建新模型实例）
            #from models.models import NegaPromptCLIP
            # 直接创建新模型实例，只复制必要的构造参数
            batch_size_score = batch_size  # Reuse the same batch size for score calculation
            before_model = get_before_model(cfg, classnames)
            before_model.load_state_dict(model_states[before_epoch])
            before_model.to('cuda')
            before_model.eval()

            # 计算历史模型的score1_before，分batch处理以避免显存爆炸
            score1_before_list = []
            score1_after_list = []
            combined_dataloader = DataLoader(combined_dataset, batch_size=batch_size_score, shuffle=False, num_workers=4)

            with torch.no_grad():
                for batch in combined_dataloader:
                    x_batch = batch['data'].to(device)
                    score_batch = calculate_score1(before_model, x_batch, n_nega_ctx).cpu().numpy()
                    score_batch_after = calculate_score1(model, x_batch, n_nega_ctx).cpu().numpy()

                    score1_before_list.append(score_batch)
                    score1_after_list.append(score_batch_after)
                    # Clear memory
                    del x_batch
                    torch.cuda.empty_cache()

            # Concatenate all batch results
            score1_before = np.concatenate(score1_before_list)
            score1_after = np.concatenate(score1_after_list)

            # 计算score差值（当前 - 历史）
            score_difference = score1_after - score1_before  # 注意：用户明确要求当前模型得分减去历史模型得分

            # 立即释放临时模型和相关内存
            del before_model
            torch.cuda.empty_cache()

        else:
            print("不计算attention_loss")

    else:
        print("model_states is None")


    # Check if we need to calculate attention loss
    if score_difference is not None:
        # Select top 20% of samples with largest difference
        num_top = int(num_combined * 0.2)
        # Get indices of top 20% samples in the combined dataset
        top_indices_in_combined = np.argsort(-score_difference)[:num_top]

        # Get the corresponding original indices
        top_original_indices = [combined_indices[i] for i in top_indices_in_combined]
        num_top = len(top_original_indices)

        # Create dataset for top samples
        top_dataset = JointTrainingDataset(full_dataset, top_original_indices)
    else:
        # No attention loss calculation needed for this remainder
        print("Skipping attention loss calculation for this epoch based on remainder.")
        attention_loss_weight = 0.0
        x_combined_top = None  # Placeholder in case it's used somewhere else
        y_combined_top = None
        num_top = -1

    # Training loop: Iterate through ID dataloader and cycle through ID-clean dataloader if needed
    top_dataloader = DataLoader(top_dataset, batch_size=batch_size, shuffle=True, num_workers=4) if score_difference is not None else None
    top_iterator = itertools.cycle(top_dataloader) if top_dataloader else None

    for batch_idx, (id_batch, clean_batch) in enumerate(zip(id_dataloader, itertools.cycle(id_clean_dataloader))):
        optimizer.zero_grad()

        # 1. 获取当前批次的ID样本和ID-clean样本
        x_id_batch = id_batch['data'].to(device)  # ID样本
        x_clean_batch = clean_batch['data'].to(device)  # ID-clean样本
        y_clean_batch = clean_batch['label'].to(device)  # ID-clean样本的标签
        B = x_id_batch.size(0)
        if batch_idx == 0:
            print(f"B: {B}")
        # --- CE Loss: 使用所有ID-clean样本中的当前批次部分 ---
        output_clean = model.forward_joint_warmup(x_clean_batch)
        logits_clean = output_clean.view(-1, int(output_clean.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]  # [B, C]
        ce_loss = F.cross_entropy(logits_clean / tau, y_clean_batch)

        # --- Mixup operation: ID-clean samples + ID samples (if enabled) ---
        sup_loss = torch.tensor(0.0).to(device)

        if alpha > 0:
            # 打乱当前批次的ID样本作为mixup的partner
            shuffle_idx = torch.randperm(B, device=device)
            x_id_shuffled = x_id_batch[shuffle_idx]

            lam = np.random.beta(alpha, alpha)
            lam = max(lam, 1 - lam)
            x_mix = lam * x_clean_batch + (1 - lam) * x_id_shuffled

            # Forward pass on partner and mixed data
            output_id = model.forward_joint_warmup(x_id_batch)
            output_mix = model.forward_joint_warmup(x_mix)

            # 解析logits
            logits_id = output_id.view(-1, int(output_id.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]
            logits_mix = output_mix.view(-1, int(output_mix.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]

            # Calculate probabilities
            p_i = F.softmax(logits_clean / tau, dim=1)  # 来自ID-clean样本的概率
            p_j = F.softmax(logits_id / tau, dim=1)     # 来自ID样本的概率
            t_mix = (lam * p_i + (1 - lam) * p_j[shuffle_idx]).detach()  # 混合目标

            # Supervision loss
            log_probs_mix = F.log_softmax(logits_mix / tau, dim=1)
            sup_loss = -(t_mix * log_probs_mix).sum(dim=1).mean()

        # Attention loss: 分batch对二次筛选的top样本计算
        loss_attention = torch.tensor(0.0).to(device)
        # if attention_loss_weight > 0.0 and top_iterator is not None:
        #     try:
        #         top_batch = next(top_iterator)
        #         x_top_batch = top_batch['data'].to(device)
        #         y_top_batch = top_batch['label'].to(device)
        #
        #         # 计算attention loss
        #         _, visual_attention = get_visual_attention(model, x_top_batch)
        #         if visual_attention is not None:
        #             text_attention = get_text_attention(model, text_features, x_top_batch, y_top_batch)
        #             loss_attention = calculate_attention_guided_loss(visual_attention, text_attention, temperature=attention_temperature)
        #     except StopIteration:
        #         # Reset the iterator if we reach the end
        #         top_iterator = itertools.cycle(top_dataloader)
        #         top_batch = next(top_iterator)
        #         x_top_batch = top_batch['data'].to(device)
        #         y_top_batch = top_batch['label'].to(device)
        #
        #         # 计算attention loss
        #         _, visual_attention = get_visual_attention(model, x_top_batch)
        #         if visual_attention is not None:
        #             text_attention = get_text_attention(model, text_features, x_top_batch, y_top_batch)
        #             loss_attention = calculate_attention_guided_loss(visual_attention, text_attention, temperature=attention_temperature)

        # 总损失
        # loss = ce_weight * ce_loss + sup_weight * sup_loss + attention_loss_weight * loss_attention
        loss = ce_weight * ce_loss + sup_weight * sup_loss
        loss.backward()
        optimizer.step()

        # Update stats
        total_loss += loss.item() * B
        total_samples += B

        # Log batch info
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | Batch Size: {B} | "
                  f"CE Loss: {ce_loss.item():.4f} | Sup Loss: {sup_loss.item():.4f} | Attention Loss: {loss_attention.item():.4f} | Total Loss: {loss.item():.4f}")

    scheduler.step()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    print(f"Epoch {epoch}: Text Branch Training Complete. Avg Loss: {avg_loss:.4f}")

    return avg_loss, split_dict



def train_joint_visual_branch(model, optimizer, scheduler, trainloader, run, cfg, epoch=None, sample_indices=None, model_states=None, classnames=None):
    import numpy as np
    import torch
    import torch.nn.functional as F

    n_nega_ctx = cfg.get('NEGA_CTX', 2)
    batch_size = cfg.get('batch_size', 256)
    total_loss = 0.0
    total_samples = 0

    # --- Sample Selection ---
    if sample_indices is None:
        split_dict = unified_sample_selection(model, trainloader, cfg, epoch)
    else:
        split_dict = sample_indices  # Reuse same partition

    # --- Training Stage ---
    print(f"Epoch {epoch}: Training Visual (LoRA) Branch...")

    # Set to train LoRA branch only, freeze prompt
    from core.coteaching_utils import unfreeze_lora, freeze_prompt
    unfreeze_lora(model)
    freeze_prompt(model)
    model.train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Extract ID and ID-clean indices
    id_indices = split_dict['id']
    id_clean_indices = split_dict['id_clean']

    # Get the underlying dataset from the trainloader
    full_dataset = trainloader.dataset

    # Create datasets for ID, ID-clean, and combined samples
    id_clean_dataset = JointTrainingDataset(full_dataset, id_clean_indices)
    id_dataset = JointTrainingDataset(full_dataset, id_indices)

    num_clean = len(id_clean_dataset)
    num_id = len(id_dataset)

    if num_clean == 0:
        print("Warning: No ID-clean samples to train on.")
        return 0.0, split_dict

    # Create dataloaders with shuffle enabled
    id_clean_dataloader = DataLoader(id_clean_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    id_dataloader = DataLoader(id_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # Training parameters
    alpha = cfg.get('mixup_alpha', 0.2)
    ce_weight = cfg.get('ce_weight', 1.0)
    sup_weight = cfg.get('sup_weight', 0.5)
    tau = cfg.get('tau', 1.0)

    # Attention loss settings
    attention_temperature = cfg.get('attention_temperature', 0.7)
    attention_loss_weight = cfg.get('attention_loss_weight', 0.1)

    # --- Sample Selection for Attention Loss ---
    # Combine ID-clean and ID samples (union)
    combined_indices = sorted(list(set(id_indices).union(set(id_clean_indices))))
    num_combined = len(combined_indices)

    # Create combined dataset
    combined_dataset = JointTrainingDataset(full_dataset, combined_indices)



    score_difference = None

    if model_states is not None:
        # ------------------------
        # Determine the appropriate model pairs to use for calculation
        # ------------------------
        cycle = epoch // 6  # Which complete cycle are we in
        remainder = epoch % 6  # Position within the current cycle

        before_epoch = -1  # Default to using the initial model
        if remainder == 5:  # Only calculate attention loss when remainder == 5 for visual branch
            # Calculate score difference for visual branch training: epoch5 - epoch3 or epoch11 - epoch9...
            before_epoch = 6 * cycle + 3
        else:
            # Don't calculate attention loss for other remainders in visual branch
            before_epoch = None
            score_difference = None  # Explicitly set to None to skip attention loss calculation

        # ------------------------
        # Load historical model to calculate score1_before
        # ------------------------
        if before_epoch is not None and before_epoch in model_states:
            print(f"Calculating Attention Loss: Current model (epoch {epoch}) - Historical model (epoch {before_epoch})")

            batch_size_score = batch_size  # Reuse the same batch size for score calculation

            # Create new model instance with necessary parameters
            before_model = get_before_model(cfg, classnames)
            # Load historical weights
            before_model.load_state_dict(model_states[before_epoch])
            before_model.to(device)  # Ensure model is on correct device
            before_model.eval()

            # Calculate score1_before with historical model, batch by batch to avoid OOM
            score1_before_list = []
            score1_after_list = []
            combined_dataloader = DataLoader(combined_dataset, batch_size=batch_size_score, shuffle=False, num_workers=4)

            with torch.no_grad():
                for batch in combined_dataloader:
                    x_batch = batch['data'].to(device)
                    score_batch = calculate_score1(before_model, x_batch, n_nega_ctx).cpu().numpy()
                    score_batch_after = calculate_score1(model, x_batch, n_nega_ctx).cpu().numpy()

                    score1_before_list.append(score_batch)
                    score1_after_list.append(score_batch_after)
                    del x_batch
                    torch.cuda.empty_cache()
            score1_before = np.concatenate(score1_before_list)
            score1_after = np.concatenate(score1_after_list)
            # Calculate score difference (current - historical) - user explicitly requested current model score minus historical model score
            score_difference = score1_after - score1_before
            print(f"score1 difference range: {score_difference.min()} ~ {score_difference.max()}")

            # Release temporary model memory immediately
            del before_model
            torch.cuda.empty_cache()

        else:
            print("不计算attention_loss")
    else:
        print("Warning: Model states not provided, using current score as alternative")

    # Check if we need to calculate attention loss
    if score_difference is not None:
        # Select top 20% of samples with largest difference
        num_top = int(num_combined * 0.2)
        if num_top == 0:
            num_top = 1

        # Get indices of top 20% samples in the combined dataset
        top_indices_in_combined = np.argsort(-score_difference)[:num_top]

        # Get the corresponding original indices
        top_original_indices = [combined_indices[i] for i in top_indices_in_combined]
        num_top = len(top_original_indices)

        # Create dataset for top samples
        top_dataset = JointTrainingDataset(full_dataset, top_original_indices)

        if num_top == 0:
            print("No well-learned samples selected, skipping attention loss.")
            attention_loss_weight = 0.0
            x_combined_top = x_combined_cpu  # Fallback to all samples if no selection
            y_combined_top = y_combined_cpu
            num_top = num_clean
    else:
        # No attention loss calculation needed for this remainder
        print("Skipping attention loss calculation for this epoch based on remainder.")
        attention_loss_weight = 0.0
        x_combined_top = None  # Placeholder in case it's used somewhere else
        y_combined_top = None
        num_top = -1

    # Precompute text features for attention loss
    text_features = model.clip_model.encode_text(model.prompt_learner.tokenized_prompts).detach()

    # Training loop: Iterate through ID dataloader and cycle through ID-clean dataloader if needed
    top_dataloader = DataLoader(top_dataset, batch_size=batch_size, shuffle=True, num_workers=4) if score_difference is not None else None
    top_iterator = itertools.cycle(top_dataloader) if top_dataloader else None

    for batch_idx, (id_batch, clean_batch) in enumerate(zip(id_dataloader, itertools.cycle(id_clean_dataloader))):
        optimizer.zero_grad()

        # Get ID partner batch and ID-clean batch
        x_partner_b = id_batch['data'].to(device)  # ID样本
        x_clean_b = clean_batch['data'].to(device)  # ID-clean样本
        y_clean_b = clean_batch['label'].to(device)  # ID-clean样本的标签
        B = x_partner_b.size(0)

        if B == 0:
            continue  # No more samples

        # --- Forward pass for ID-clean sample ---
        output_clean = model.forward_joint_warmup(x_clean_b)
        logits_clean = output_clean.view(-1, int(output_clean.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]  # [B, C]

        # CE loss
        ce_loss = F.cross_entropy(logits_clean / tau, y_clean_b)

        # --- Mixup operation (if enabled) ---
        sup_loss = torch.tensor(0.0).to(device)

        if alpha > 0:
            shuffle_idx = torch.randperm(B, device=device)
            x_partner_shuffled = x_partner_b[shuffle_idx]

            lam = np.random.beta(alpha, alpha)
            lam = max(lam, 1 - lam)
            x_mix = lam * x_clean_b + (1 - lam) * x_partner_shuffled

            # Forward pass on mixed data
            output_partner = model.forward_joint_warmup(x_partner_b)
            output_mix = model.forward_joint_warmup(x_mix)

            logits_partner = output_partner.view(-1, int(output_partner.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]
            logits_mix = output_mix.view(-1, int(output_mix.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]

            # Soft targets
            p_i = F.softmax(logits_clean / tau, dim=1)
            p_j = F.softmax(logits_partner / tau, dim=1)
            t_mix = (lam * p_i + (1 - lam) * p_j[shuffle_idx]).detach()

            # Supervision loss
            log_probs_mix = F.log_softmax(logits_mix / tau, dim=1)
            sup_loss = -(t_mix * log_probs_mix).sum(dim=1).mean()

        # # Attention loss: 分batch对二次筛选的top样本计算
        loss_attention = torch.tensor(0.0).to(device)
        # if attention_loss_weight > 0.0 and top_iterator is not None:
        #     try:
        #         top_batch = next(top_iterator)
        #         x_top_batch = top_batch['data'].to(device)
        #         y_top_batch = top_batch['label'].to(device)
        #
        #         # 计算attention loss
        #         _, visual_attention = get_visual_attention(model, x_top_batch)
        #         if visual_attention is not None:
        #             text_attention = get_text_attention(model, text_features, x_top_batch, y_top_batch)
        #             loss_attention = calculate_attention_guided_loss(visual_attention, text_attention, temperature=attention_temperature)
        #     except StopIteration:
        #         # Reset the iterator if we reach the end
        #         top_iterator = itertools.cycle(top_dataloader)
        #         top_batch = next(top_iterator)
        #         x_top_batch = top_batch['data'].to(device)
        #         y_top_batch = top_batch['label'].to(device)
        #
        #         # 计算attention loss
        #         _, visual_attention = get_visual_attention(model, x_top_batch)
        #         if visual_attention is not None:
        #             text_attention = get_text_attention(model, text_features, x_top_batch, y_top_batch)
        #             loss_attention = calculate_attention_guided_loss(visual_attention, text_attention, temperature=attention_temperature)
        # loss = ce_weight * ce_loss + sup_weight * sup_loss + attention_loss_weight * loss_attention
        loss = ce_weight * ce_loss + sup_weight * sup_loss
        loss.backward()
        optimizer.step()

        # Update stats
        total_loss += loss.item() * B
        total_samples += B

    scheduler.step()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    print(f"Epoch {epoch}: Visual Branch Training Complete. Avg Loss: {avg_loss:.4f}")

    return avg_loss, split_dict


def train_joint_all(model, optimizer, scheduler, trainloader, run, cfg, epoch=None, sample_indices=None,
                    model_states=None, classnames=None):
    """
    联合训练函数：同时优化 LoRA (Visual) + Positive Prompts + Negative Prompts
    结合了 CE Loss, Mixup Sup Loss, NIS Loss 和 OOD Loss
    """
    import torch
    import torch.nn.functional as F
    import numpy as np
    import itertools
    from torch.utils.data import DataLoader
    from core.coteaching_utils import unfreeze_lora, unfreeze_prompt, unfreeze_negative_prompt

    # ==========================
    # 1. 模型设置：全参数解冻
    # ==========================
    print(f"Epoch {epoch}: [Joint Training] Updating All Parameters (LoRA + Pos/Neg Prompts)...")

    # 解冻所有部分
    unfreeze_lora(model)
    unfreeze_prompt(model)  # 解冻 Positive
    unfreeze_negative_prompt(model)  # 解冻 Negative
    model.train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 配置参数
    n_nega_ctx = cfg.get('NEGA_CTX', 2)
    batch_size = cfg.get('batch_size', 256)
    alpha = cfg.get('mixup_alpha', 0.2)
    tau = cfg.get('tau', 1.0)

    # 损失权重 (根据经验或原有配置)
    ce_weight = cfg.get('ce_weight', 1.0)
    sup_weight = cfg.get('sup_weight', 0.5)
    # 联合训练时，可能需要调整 OOD/NIS 的权重，防止分类任务被压制
    ood_weight = cfg.get('ood_weight', 1.0)
    nis_weight = cfg.get('nis_weight', 1.0)

    # ==========================
    # 2. 数据准备
    # ==========================
    # --- Sample Selection ---
    if sample_indices is None:
        split_dict = unified_sample_selection(model, trainloader, cfg, epoch)
    else:
        split_dict = sample_indices

    full_dataset = trainloader.dataset

    # 提取索引
    id_indices = split_dict['id']
    id_clean_indices = split_dict['id_clean']
    ood_indices = split_dict['noisy_ood']  # 使用 split 出来的 OOD/Noisy 数据作为 OOD 源

    # 构建 Dataset
    id_dataset = JointTrainingDataset(full_dataset, id_indices)
    id_clean_dataset = JointTrainingDataset(full_dataset, id_clean_indices)
    ood_dataset = JointTrainingDataset(full_dataset, ood_indices)

    num_id = len(id_dataset)
    num_clean = len(id_clean_dataset)
    num_ood = len(ood_dataset)

    if num_clean == 0:
        print("Warning: No ID-clean samples. Skipping epoch.")
        return 0.0, split_dict

    print(f"Joint Training Samples - ID: {num_id}, Clean: {num_clean}, OOD/Noisy: {num_ood}")

    # 构建 DataLoader
    # 主循环基于 ID Dataset，其他使用 cycle 循环
    id_dataloader = DataLoader(id_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    id_clean_dataloader = DataLoader(id_clean_dataset, batch_size=batch_size, shuffle=True, num_workers=4,
                                     drop_last=True)

    # OOD Loader (如果存在)
    ood_iterator = None
    if num_ood > 0:
        ood_dataloader = DataLoader(ood_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
        ood_iterator = itertools.cycle(ood_dataloader)

    # ==========================
    # 3. Attention Loss 准备 (可选)
    # ==========================
    # 注意：在联合训练中，原本基于 Cycle 的 score_difference 逻辑可能不再适用
    # 这里为了简化对比，如果 model_states 包含上一轮(epoch-1)，则计算，否则跳过
    attention_loss_weight = cfg.get('attention_loss_weight', 0.0)  # 联合训练通常设为 0 或者保持原样
    top_iterator = None

    # (此处省略了复杂的 score difference 计算逻辑，以保证联合训练的纯粹性)
    # 如果你坚持要在联合训练中加 Attention Loss，逻辑与原代码相同，
    # 但建议将 compare epoch 设为 fixed logic (e.g., epoch - 1)

    # ==========================
    # 4. 训练循环
    # ==========================
    total_loss = 0.0
    total_samples = 0

    # 使用 zip 和 cycle 组合所有数据流
    # 主迭代器是 id_dataloader (通常最大)
    clean_iterator = itertools.cycle(id_clean_dataloader)

    for batch_idx, id_batch in enumerate(id_dataloader):
        optimizer.zero_grad()

        # ----------------------
        # Data Loading
        # ----------------------
        # A. ID Samples (用于 Mixup Partner 和 NIS Loss)
        x_id = id_batch['data'].to(device)
        B = x_id.size(0)

        # B. Clean Samples (用于 CE Loss 和 Mixup Source)
        try:
            clean_batch = next(clean_iterator)
        except StopIteration:
            clean_iterator = itertools.cycle(id_clean_dataloader)
            clean_batch = next(clean_iterator)

        x_clean = clean_batch['data'].to(device)
        y_clean = clean_batch['label'].to(device)

        # C. OOD Samples (用于 OOD Loss)
        x_ood = None
        if ood_iterator is not None:
            try:
                ood_batch = next(ood_iterator)
                x_ood = ood_batch['data'].to(device)
            except StopIteration:
                # 理论上 cycle 不会 stop，但为了安全
                pass

        # ----------------------
        # Loss Part 1: Classification (CE + Sup Mixup)
        # 来自 train_joint_text_branch / visual_branch
        # ----------------------

        # 1.1 CE Loss on Clean Data
        output_clean = model.forward_joint_warmup(x_clean)
        # 提取 Positive Logits: [B, Class, 1+N_ctx] -> 取第0个(Positive)
        logits_clean = output_clean.view(-1, int(output_clean.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]
        loss_ce = F.cross_entropy(logits_clean / tau, y_clean)

        # 1.2 Mixup Sup Loss
        loss_sup = torch.tensor(0.0).to(device)
        if alpha > 0:
            # Mixup Clean with ID (Partner)
            # 确保 batch size 匹配 (cycle 的 clean batch 可能和 id batch 大小不同，取最小值)
            min_B = min(x_clean.size(0), x_id.size(0))
            x_clean_cut = x_clean[:min_B]
            x_id_cut = x_id[:min_B]

            shuffle_idx = torch.randperm(min_B, device=device)
            x_id_shuffled = x_id_cut[shuffle_idx]

            lam = np.random.beta(alpha, alpha)
            lam = max(lam, 1 - lam)
            x_mix = lam * x_clean_cut + (1 - lam) * x_id_shuffled

            # Forward Mix
            output_mix = model.forward_joint_warmup(x_mix)
            logits_mix = output_mix.view(-1, int(output_mix.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]

            # Forward Partner (ID) for Soft Target
            with torch.no_grad():
                output_partner = model.forward_joint_warmup(x_id_cut)
                logits_partner = output_partner.view(-1, int(output_partner.shape[1] / (1 + n_nega_ctx)),
                                                     1 + n_nega_ctx)[:, :, 0]

            # Soft Targets
            p_clean = F.softmax(logits_clean[:min_B] / tau, dim=1)
            p_partner = F.softmax(logits_partner / tau, dim=1)
            t_mix = (lam * p_clean + (1 - lam) * p_partner[shuffle_idx]).detach()

            # Loss
            log_probs_mix = F.log_softmax(logits_mix / tau, dim=1)
            loss_sup = -(t_mix * log_probs_mix).sum(dim=1).mean()

        # ----------------------
        # Loss Part 2: Negative Prompts (NIS + OOD)
        # 来自 negative_prompt_training
        # ----------------------

        loss_nis = torch.tensor(0.0).to(device)
        loss_ood = torch.tensor(0.0).to(device)

        # 2.1 NIS Loss (利用 ID 数据让 Negative Prompt 输出均匀)
        # 我们可以复用上面的 output_partner (x_id 的输出)，但需要取 Negative 部分
        # 如果上面没计算 output_partner (alpha=0的情况)，需要重新计算
        if alpha <= 0:
            output_id_full = model.forward_joint_warmup(x_id)
        else:
            # 如果 alpha > 0, output_partner 是 x_id_cut 的输出，可能比 B 小
            # 为了简单，我们对整个 batch x_id 算一次 (或者复用 x_id_cut)
            # 这里选择重新对 x_id 算一次以利用完整 batch，或者复用 x_id_cut 节省显存
            # 为了显存安全，我们复用 x_id_cut (如果 mixup 开启)
            output_id_full = model.forward_joint_warmup(x_id)  # 重新算，保证对齐

        logits_id_struct = output_id_full.view(-1, int(output_id_full.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)
        logits_neg_id = logits_id_struct[:, :, 1:]  # [B, C, N_neg]
        logits_neg_mean = logits_neg_id.mean(dim=-1)  # Mean over negative prompts
        # NIS Logic: Maximize entropy of negative prompts on ID data
        loss_nis = -(logits_neg_mean.mean(dim=-1) - torch.logsumexp(logits_neg_mean, dim=-1)).mean()

        # 2.2 OOD Loss (Push OOD samples to Negative Prompts)
        if x_ood is not None:
            output_ood = model.forward_joint_warmup(x_ood)
            logits_ood_struct = output_ood.view(-1, int(output_ood.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)

            logits_pos_ood = logits_ood_struct[:, :, 0]  # Positive
            logits_neg_ood = logits_ood_struct[:, :, 1:]  # Negative

            s_pos = torch.logsumexp(logits_pos_ood, dim=1)
            s_neg = torch.logsumexp(logits_neg_ood.reshape(logits_neg_ood.size(0), -1), dim=1)

            # 这里的逻辑是：让 OOD 样本在 Negative Prompts 上的响应(s_neg) 大于 Positive (s_pos)
            loss_ood = F.softplus(s_pos - s_neg).mean()

        # ----------------------
        # Total Loss & Update
        # ----------------------
        loss = (ce_weight * loss_ce) + \
               (sup_weight * loss_sup) + \
               (nis_weight * loss_nis) + \
               (ood_weight * loss_ood)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        total_samples += B

        if batch_idx % 20 == 0:
            print(f"Epoch {epoch} | Batch {batch_idx} | "
                  f"CE: {loss_ce.item():.3f} | Sup: {loss_sup.item():.3f} | "
                  f"NIS: {loss_nis.item():.3f} | OOD: {loss_ood.item():.3f} | "
                  f"Total: {loss.item():.3f}")

    scheduler.step()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    print(f"Epoch {epoch}: Joint Training Complete. Avg Loss: {avg_loss:.4f}")

    return avg_loss, split_dict