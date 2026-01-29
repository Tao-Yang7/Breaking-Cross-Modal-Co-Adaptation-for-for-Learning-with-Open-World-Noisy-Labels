#!/usr/bin/env python
"""
Joint training utilities for stage2 merged model
"""
import os
import pdb

import torch
import torch.nn.functional as F
import numpy as np
from models.models import NegaPromptCLIP
from open_clip import create_model_and_transforms

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
    batch_size = cfg.get('batch_size', 256)

    # --- Inference stage: Collect all samples' scores and labels ---
    print(f"Epoch {epoch}: Running unified inference for sample identification...")
    model.eval()  # Switch to evaluation mode

    all_score1 = []  # MCM-style: Max probability
    all_score2 = []  # Confidence on noisy label
    all_score3 = []  # Top 5 confidence
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

            # --- Forward propagation using forward_joint_warmup (required by user) ---
            output  = model.forward_joint_warmup(data)

            # Parse logits the same way as stage2
            logits = output.view(-1, int(output.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)
            B, C = logits.shape[0], logits.shape[1]
            logits_yes = logits[:, :, 0]  # [B, C] - positive prompt + LoRA features

            # Calculate probabilities
            if epoch > 1:
                logits_no = logits[:, :, 1:]
                logits_no_mean = logits_no.mean(dim=2)
                prob_margin = torch.softmax(logits_yes - logits_no_mean, dim=1)  # [B, C]
            else:
                prob_margin = torch.softmax(logits_yes, dim=1)  # [B, C]

            # Calculate scores
            score1 = prob_margin.max(dim=1).values  # [B] - max probability
            score2 = prob_margin[torch.arange(B), labels]  # [B] - probability on noisy label

            # Calculate score3: Top5 max probability
            _, top5_indices = torch.topk(logits_yes, k=5, dim=1)
            batch_indices = torch.arange(B).unsqueeze(1).repeat(1, 5).view(-1)
            top5_probs = prob_margin[batch_indices, top5_indices.view(-1)].view(B, 5)
            score3 = top5_probs.max(dim=1).values  # [B]

            # Cache results
            all_score1.append(score1.cpu())
            all_score2.append(score2.cpu())
            all_score3.append(score3.cpu())
            all_labels.append(labels.cpu())
            all_original_indices.append(original_indices.cpu())
            all_is_open.append(is_open.cpu())
            all_clean_labels.append(clean_labels.cpu())
            # Calculate is_clean (ID and clean)
            is_clean = (~is_open.bool()) & (labels == clean_labels)
            all_is_clean.append(is_clean.cpu())

    # Merge all results
    all_score1_np = torch.cat(all_score1, dim=0).numpy()
    all_score2_np = torch.cat(all_score2, dim=0).numpy()
    all_score3_np = torch.cat(all_score3, dim=0).numpy()
    all_labels_np = torch.cat(all_labels, dim=0).numpy()
    all_original_indices_np = torch.cat(all_original_indices, dim=0).numpy()
    all_is_open_np = torch.cat(all_is_open, dim=0).numpy().astype(bool)  # Ground truth OOD flag
    all_is_clean_np = torch.cat(all_is_clean, dim=0).numpy().astype(bool)  # Ground truth ID-clean flag

    # --- Sample Partitioning (single partition) ---
    print(f"Using unified sample selection on {len(all_score1_np)} samples...")

    # Parameters for sample selection (same as original)
    # top_k_per_class_id = 80
    # top_k_per_class_clean = 100
    # bottom_k_global = 3000

    top_k_per_class_id = 100
    top_k_per_class_clean = 80
    bottom_k_global = 3000

    N_total = len(all_score1_np)
    final_sample_type = np.full(N_total, -1, dtype=int)  # -1: unmarked, 0: ID, 1: ID-clean, 2: OOD

    unique_labels = np.unique(all_labels_np)

    # --- Modified: First mark ID-clean samples (Score2 Top-K per class) ---
    all_top_k_indices_by_score2 = []  # ID-clean indices
    for label in unique_labels:
        class_indices = np.where(all_labels_np == label)[0]
        if len(class_indices) == 0:
            continue
        class_score2 = all_score2_np[class_indices]
        k_to_select = min(top_k_per_class_clean, len(class_score2))
        top_k_local_by_score2 = np.argpartition(class_score2, -k_to_select)[-k_to_select:]
        top_k_global_by_score2 = class_indices[top_k_local_by_score2]
        all_top_k_indices_by_score2.extend(top_k_global_by_score2)
    all_top_k_indices_by_score2 = np.array(all_top_k_indices_by_score2)

    # Mark ID-clean first
    final_sample_type[all_top_k_indices_by_score2] = 1  # Mark as ID-clean

    # --- Modified: Then mark ID samples from non-ID-clean (Score1 Top-K per class) ---
    all_top_k_indices_by_score1 = []  # ID indices
    for label in unique_labels:
        # Get all class indices, excluding already marked ID-clean samples
        class_indices = np.where((all_labels_np == label) & (final_sample_type == -1))[0]
        if len(class_indices) == 0:
            continue
        class_score1 = all_score1_np[class_indices]
        k_to_select = min(top_k_per_class_id, len(class_score1))
        if k_to_select <= 0:
            continue  # No samples to select
        top_k_local_by_score1 = np.argpartition(class_score1, -k_to_select)[-k_to_select:]
        top_k_global_by_score1 = class_indices[top_k_local_by_score1]
        all_top_k_indices_by_score1.extend(top_k_global_by_score1)
    all_top_k_indices_by_score1 = np.array(all_top_k_indices_by_score1)

    # Mark ID from remaining samples
    final_sample_type[all_top_k_indices_by_score1] = 0  # Mark as ID

    # Mark OOD samples (Score2 Bottom-K global, not already marked)
    k_to_select_global = min(bottom_k_global, N_total)
    bottom_k_global_indices = np.argpartition(all_score1_np, k_to_select_global)[:k_to_select_global]
    bottom_k_global_indices = bottom_k_global_indices[np.argsort(all_score1_np[bottom_k_global_indices])]
    # Only mark unmarked samples
    ood_mask = final_sample_type[bottom_k_global_indices] == -1
    final_sample_type[bottom_k_global_indices[ood_mask]] = 2  # Mark as OOD

    # --- Statistics ---
    id_count = (final_sample_type == 0).sum()
    clean_count = (final_sample_type == 1).sum()
    ood_count = (final_sample_type == 2).sum()
    unmarked_count = (final_sample_type == -1).sum()

    # --- Calculate real proportions for each split ---
    print("\n--- Real Sample Proportions in Split Results (Ground Truth) ---")

    # 1. ID group (final_sample_type == 0)
    id_indices = np.where(final_sample_type == 0)[0]
    id_real_clean = all_is_clean_np[id_indices].sum()
    id_real_noisy = (~all_is_clean_np[id_indices] & ~all_is_open_np[id_indices]).sum()
    id_real_ood = all_is_open_np[id_indices].sum()
    if id_indices.size > 0:
        print(f"ID group: {id_indices.size} samples")
        print(f"  Real ID-Clean: {id_real_clean} ({id_real_clean/id_indices.size*100:.2f}%)")
        print(f"  Real ID-Noisy: {id_real_noisy} ({id_real_noisy/id_indices.size*100:.2f}%)")
        print(f"  Real OOD: {id_real_ood} ({id_real_ood/id_indices.size*100:.2f}%)")

    # 2. ID-Clean group (final_sample_type == 1)
    id_clean_indices = np.where(final_sample_type == 1)[0]
    id_clean_real_clean = all_is_clean_np[id_clean_indices].sum()
    id_clean_real_noisy = (~all_is_clean_np[id_clean_indices] & ~all_is_open_np[id_clean_indices]).sum()
    id_clean_real_ood = all_is_open_np[id_clean_indices].sum()
    if id_clean_indices.size > 0:
        print(f"ID-Clean group: {id_clean_indices.size} samples")
        print(f"  Real ID-Clean: {id_clean_real_clean} ({id_clean_real_clean/id_clean_indices.size*100:.2f}%)")
        print(f"  Real ID-Noisy: {id_clean_real_noisy} ({id_clean_real_noisy/id_clean_indices.size*100:.2f}%)")
        print(f"  Real OOD: {id_clean_real_ood} ({id_clean_real_ood/id_clean_indices.size*100:.2f}%)")

    # 3. Noisy-OOD group (final_sample_type == 2)
    noisy_ood_indices = np.where(final_sample_type == 2)[0]
    noisy_ood_real_clean = all_is_clean_np[noisy_ood_indices].sum()
    noisy_ood_real_noisy = (~all_is_clean_np[noisy_ood_indices] & ~all_is_open_np[noisy_ood_indices]).sum()
    noisy_ood_real_ood = all_is_open_np[noisy_ood_indices].sum()
    if noisy_ood_indices.size > 0:
        print(f"Noisy-OOD group: {noisy_ood_indices.size} samples")
        print(f"  Real ID-Clean: {noisy_ood_real_clean} ({noisy_ood_real_clean/noisy_ood_indices.size*100:.2f}%)")
        print(f"  Real ID-Noisy: {noisy_ood_real_noisy} ({noisy_ood_real_noisy/noisy_ood_indices.size*100:.2f}%)")
        print(f"  Real OOD: {noisy_ood_real_ood} ({noisy_ood_real_ood/noisy_ood_indices.size*100:.2f}%)")
    print("\n" + "=" * 60 + "\n")

    # --- Generate split dictionary ---
    split_dict = {
        'id': all_original_indices_np[final_sample_type == 0].tolist(),
        'id_clean': all_original_indices_np[final_sample_type == 1].tolist(),
        'noisy_ood': all_original_indices_np[final_sample_type == 2].tolist(),
        'unmarked': all_original_indices_np[final_sample_type == -1].tolist()
    }

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

    # First pass to collect all data and labels
    print(f"Collecting training data...")
    all_data_list = []
    all_labels_list = []
    all_indices_list = []
    all_clean_labels_list = []
    all_is_open_list = []

    with torch.no_grad():
        for batch in trainloader:
            all_data_list.append(batch['data'].cpu())
            all_labels_list.append(batch['label'].cpu())
            all_indices_list.append(batch['index'].cpu())
            all_clean_labels_list.append(batch['clean_label'].cpu())
            all_is_open_list.append(batch['is_open'].cpu())

    all_data = torch.cat(all_data_list, dim=0)
    all_labels = torch.cat(all_labels_list, dim=0)
    all_indices = torch.cat(all_indices_list, dim=0).numpy()

    # Create masks for ID, ID-clean, and OOD samples
    original_id_mask = np.isin(all_indices, id_indices_np)
    original_id_clean_mask = np.isin(all_indices, id_clean_indices_np)
    original_ood_mask = np.isin(all_indices, ood_indices_np)
    data_indices = np.arange(len(all_data))

    id_mask = data_indices[original_id_mask]
    id_clean_mask = data_indices[original_id_clean_mask]
    ood_mask = data_indices[original_ood_mask]

    if len(id_mask) == 0 or len(ood_mask) == 0 or len(id_clean_mask) == 0:
        print("Warning: No ID, ID-clean, or OOD samples available for prompt training.")
        return 0.0

    # Filter data
    x_id = all_data[id_mask].to(device)
    x_id_clean = all_data[id_clean_mask].to(device)
    y_id_clean = all_labels[id_clean_mask].to(device)
    x_ood = all_data[ood_mask].to(device)

    num_id = len(x_id)
    num_id_clean = len(x_id_clean)
    num_ood = len(x_ood)

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

    for iter_idx in range(0, max(num_id, num_ood), batch_size):
        prompt_optimizer.zero_grad()

        # Get ID batch
        start_id = iter_idx
        end_id = min(iter_idx + batch_size, num_id)
        x_id_batch = x_id[start_id:end_id]
        B_id = x_id_batch.size(0)

        # Get OOD batch
        start_ood = iter_idx % num_ood
        end_ood = (start_ood + batch_size) % num_ood
        if start_ood < end_ood:
            x_ood_batch = x_ood[start_ood:end_ood]
        else:
            x_ood_batch = torch.cat([x_ood[start_ood:], x_ood[:end_ood]], dim=0) if num_ood > 0 else None
        B_ood = x_ood_batch.size(0) if x_ood_batch is not None else 0

        if B_id == 0 and B_ood == 0:
            continue

        loss_nis = torch.tensor(0.0).to(device)
        loss_ood = torch.tensor(0.0).to(device)

        # NIS Loss for ID samples
        if B_id > 0:
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

        if (iter_idx // batch_size) % 10 == 0:
            print(f"Epoch {epoch} | Iter {iter_idx // batch_size} | "
                  f"NIS Loss: {loss_nis.item():.4f} | OOD Loss: {loss_ood.item():.4f} | "
                  f"Negative Loss: {neg_loss_batch.item():.4f}")


    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

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

    # Get all data for efficient training (same as original)
    all_data = []
    all_labels = []

    # First pass to collect all data and labels
    with torch.no_grad():
        for batch in trainloader:
            all_data.append(batch['data'].cpu())
            all_labels.append(batch['label'].cpu())

    all_data = torch.cat(all_data, dim=0)  # [N, C, H, W]
    all_labels = torch.cat(all_labels, dim=0)  # [N]

    # Extract ID and ID-clean indices
    id_indices_np = np.array(split_dict['id'])
    id_clean_indices_np = np.array(split_dict['id_clean'])

    # Filter ID-clean samples
    id_clean_mask = np.isin(np.arange(len(all_data)), id_clean_indices_np)
    x_clean_cpu = all_data[id_clean_mask]
    y_clean_cpu = all_labels[id_clean_mask]
    num_clean = len(x_clean_cpu)

    # Filter ID samples (non-ID-clean)
    id_mask = np.isin(np.arange(len(all_data)), id_indices_np)
    x_id_cpu = all_data[id_mask]
    num_id = len(x_id_cpu)

    if num_clean == 0:
        print("No ID-clean samples, skipping training.")
        return 0.0, split_dict

    # Shuffle samples
    perm_clean = torch.randperm(num_clean, device='cpu')
    x_clean_cpu = x_clean_cpu[perm_clean]
    y_clean_cpu = y_clean_cpu[perm_clean]

    perm_id = torch.randperm(num_id, device='cpu')
    x_id_cpu = x_id_cpu[perm_id]

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
    # Create masks for combined samples
    id_mask = np.isin(np.arange(len(all_data)), id_indices_np)
    id_clean_mask = np.isin(np.arange(len(all_data)), id_clean_indices_np)
    combined_mask = id_mask | id_clean_mask

    x_combined_cpu = all_data[combined_mask]
    y_combined_cpu = all_labels[combined_mask]
    num_combined = len(x_combined_cpu)

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
            with torch.no_grad():
                for i in range(0, num_combined, batch_size_score):
                    end_idx = min(i + batch_size_score, num_combined)
                    x_batch = x_combined_cpu[i:end_idx].to(device)
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
        # Get indices of top 20% samples
        top_indices = np.argsort(-score_difference)[:num_top]

        # Filter data to only include top 20% samples from the combined dataset
        x_combined_top = x_combined_cpu.contiguous()[top_indices]
        y_combined_top = y_combined_cpu.contiguous()[top_indices]
        num_top = len(x_combined_top)
    else:
        # No attention loss calculation needed for this remainder
        print("Skipping attention loss calculation for this epoch based on remainder.")
        attention_loss_weight = 0.0
        x_combined_top = None  # Placeholder in case it's used somewhere else
        y_combined_top = None
        num_top = -1

    # Training loop with max_iter based on num_id (ID样本数量更多，循环利用ID-clean样本)
    max_iter = num_id

    # 同时循环处理ID样本和top样本进行attention_loss计算
    for iter_idx in range(0, max_iter, batch_size):
        optimizer.zero_grad()

        # 1. 首先获取当前批次的ID样本
        start_id = iter_idx
        end_id = min(iter_idx + batch_size, num_id)
        x_id_batch = x_id_cpu[start_id:end_id].to(device)  # ID样本
        B = x_id_batch.size(0)

        # 2. 循环利用ID-clean样本进行配对 (循环索引避免越界)
        clean_indices = torch.arange(start_id, end_id) % num_clean
        x_clean_batch = x_clean_cpu[clean_indices].to(device)  # ID-clean样本
        y_clean_batch = y_clean_cpu[clean_indices].to(device)  # ID-clean样本的标签

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
        if attention_loss_weight > 0.0:
            # 获取当前批次的top样本
            start_top = iter_idx % num_top
            end_top = start_top + B

            # 如果end_top不超过num_top，则计算attention loss
            if end_top <= num_top:
                x_top_batch = x_combined_top[start_top:end_top].to(device)
                y_top_batch = y_combined_top[start_top:end_top].to(device)

                # 计算attention loss
                _, visual_attention = get_visual_attention(model, x_top_batch)
                if visual_attention is not None:
                    text_attention = get_text_attention(model, text_features, x_top_batch, y_top_batch)
                    loss_attention = calculate_attention_guided_loss(visual_attention, text_attention, temperature=attention_temperature)

        # 总损失
        loss = ce_weight * ce_loss + sup_weight * sup_loss + attention_loss_weight * loss_attention
        loss.backward()
        optimizer.step()

        # Update stats
        total_loss += loss.item() * B
        total_samples += B

        # Log batch info
        if (iter_idx // batch_size) % 10 == 0:
            print(f"Epoch {epoch} | Iter {iter_idx // batch_size} | Batch Size: {B} | "
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
    id_indices_np = np.array(split_dict['id'])
    id_clean_indices_np = np.array(split_dict['id_clean'])

    # Get all data for training
    all_data = []
    all_labels = []

    with torch.no_grad():
        for batch in trainloader:
            all_data.append(batch['data'].cpu())
            all_labels.append(batch['label'].cpu())

    all_data = torch.cat(all_data, dim=0)  # [N, C, H, W]
    all_labels = torch.cat(all_labels, dim=0)  # [N]

    # Filter training samples
    id_clean_mask = np.isin(np.arange(len(all_data)), id_clean_indices_np)
    x_clean_cpu = all_data[id_clean_mask].cpu()
    y_clean_cpu = all_labels[id_clean_mask].cpu()

    id_mask = np.isin(np.arange(len(all_data)), id_indices_np)
    x_id_cpu = all_data[id_mask].cpu()

    num_clean = len(x_clean_cpu)
    num_id = len(x_id_cpu)

    if num_clean == 0:
        print("Warning: No ID-clean samples to train on.")
        return 0.0, split_dict

    # Shuffle samples
    perm_clean = torch.randperm(num_clean, device='cpu')
    x_clean_cpu = x_clean_cpu[perm_clean]
    y_clean_cpu = y_clean_cpu[perm_clean]

    perm_id = torch.randperm(num_id, device='cpu')
    x_id_cpu = x_id_cpu[perm_id]

    # Training parameters
    alpha = cfg.get('mixup_alpha', 0.2)
    ce_weight = cfg.get('ce_weight', 1.0)
    sup_weight = cfg.get('sup_weight', 0.5)
    tau = cfg.get('tau', 1.0)

    # Attention loss settings
    attention_temperature = cfg.get('attention_temperature', 1.0)
    attention_loss_weight = cfg.get('attention_loss_weight', 0.1)

    # --- Sample Selection for Attention Loss ---
    # Combine ID-clean and ID samples (union)
    # Create masks for combined samples
    id_mask = np.isin(np.arange(len(all_data)), id_indices_np)
    id_clean_mask = np.isin(np.arange(len(all_data)), id_clean_indices_np)
    combined_mask = id_mask | id_clean_mask

    x_combined_cpu = all_data[combined_mask]
    y_combined_cpu = all_labels[combined_mask]
    num_combined = len(x_combined_cpu)



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
            with torch.no_grad():
                for i in range(0, num_combined, batch_size_score):
                    end_idx = min(i + batch_size_score, num_combined)
                    x_batch = x_combined_cpu[i:end_idx].to(device)

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

        # Get indices of top 20% samples
        top_indices = np.argsort(-score_difference)[:num_top]

        # Filter data to only include top 20% samples from the combined dataset
        x_combined_top = x_combined_cpu.contiguous()[top_indices]
        y_combined_top = y_combined_cpu.contiguous()[top_indices]
        num_top = len(x_combined_top)

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

    # Training loop with ID samples
    max_iter = num_id

    for iter_idx in range(0, max_iter, batch_size):
        optimizer.zero_grad()

        # Get ID partner batch
        partner_start = iter_idx
        partner_end = min(iter_idx + batch_size, num_id)
        x_partner_b = x_id_cpu[partner_start:partner_end].to(device)  # ID样本
        B = x_partner_b.size(0)

        if B == 0:
            continue  # No more samples

        # Get ID-clean batch (top 20% samples)
        start_clean = iter_idx % num_clean
        end_clean = start_clean + B
        end_clean = min(end_clean, num_clean)

        # 循环利用ID-clean样本
        if end_clean <= num_clean:
            x_clean_b = x_clean_cpu[start_clean:end_clean].to(device)
            y_clean_b = y_clean_cpu[start_clean:end_clean].to(device)
        else:
            # 循环取样本
            batch1_end = num_clean
            batch1_size = batch1_end - start_clean
            batch2_size = B - batch1_size

            x_clean_b1 = x_clean_cpu[start_clean:batch1_end].to(device)
            y_clean_b1 = y_clean_cpu[start_clean:batch1_end].to(device)

            x_clean_b2 = x_clean_cpu[:batch2_size].to(device)
            y_clean_b2 = y_clean_cpu[:batch2_size].to(device)

            x_clean_b = torch.cat([x_clean_b1, x_clean_b2], dim=0)
            y_clean_b = torch.cat([y_clean_b1, y_clean_b2], dim=0)

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
            t_mix = lam * p_i + (1 - lam) * p_j[shuffle_idx]

            # Supervision loss
            log_probs_mix = F.log_softmax(logits_mix / tau, dim=1)
            sup_loss = -(t_mix * log_probs_mix).sum(dim=1).mean()

        # Attention loss: 分batch对二次筛选的top样本计算
        loss_attention = torch.tensor(0.0).to(device)
        if attention_loss_weight > 0.0:
            # 获取当前批次的top样本用于计算attention_loss
            start_top = iter_idx % num_top
            end_top = start_top + B

            # 如果end_top不超过num_top，则计算attention loss
            if end_top <= num_top:
                x_top_batch = x_combined_top[start_top:end_top].to(device)
                y_top_batch = y_combined_top[start_top:end_top].to(device)

                # 计算attention loss
                _, visual_attention = get_visual_attention(model, x_top_batch)
                if visual_attention is not None:
                    text_attention = get_text_attention(model, text_features, x_top_batch, y_top_batch)
                    loss_attention = calculate_attention_guided_loss(visual_attention, text_attention, temperature=attention_temperature)
        loss = ce_weight * ce_loss + sup_weight * sup_loss + attention_loss_weight * loss_attention
        loss.backward()
        optimizer.step()

        # Update stats
        total_loss += loss.item() * B
        total_samples += B

    scheduler.step()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    print(f"Epoch {epoch}: Visual Branch Training Complete. Avg Loss: {avg_loss:.4f}")

    return avg_loss, split_dict
