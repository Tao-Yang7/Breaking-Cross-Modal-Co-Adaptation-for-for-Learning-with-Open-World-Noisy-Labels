import pdb


def agreeing_to_differ(logits, tau=1.0):
    """
    ATD OOD 判定，严格按照 CLIPN 论文公式 4 和 8-9 实现
    logits: [B, n_class, 1+n_neg_ctx]
        logits[:,:,0]   -> yes prompt (正文本)
        logits[:,:,1:]  -> no prompts (否定文本)
    tau: 温度参数
    Returns:
        is_id: [B] 1=ID, 0=OOD
        p_yes: [B, n_class] 正文本概率
        p_no: [B, n_class] 否定文本概率（公式4）
        p_C_plus_1: [B] 未知类别概率（公式8）
    """
    B, C, n_total = logits.shape
    n_neg_ctx = n_total - 1

    # 1. 分割 yes / no logits
    logits_yes = logits[:, :, 0]          # [B, C]
    logits_no  = logits[:, :, 1:]         # [B, C, n_neg_ctx]

    # 2. 计算公式4中的 p_no_ij
    # p_no_ij = exp(<fi, gno_j>/tau) / (exp(<fi, gj>/tau) + exp(<fi, gno_j>/tau))
    # logits 已经是 <fi, g>/tau
    logits_yes_exp = logits_yes.unsqueeze(-1)              # [B, C, 1]
    p_no = torch.exp(logits_no).sum(dim=-1) / (torch.exp(logits_yes) + torch.exp(logits_no).sum(dim=-1))

    # 3. ID 类概率
    p_yes = F.softmax(logits_yes, dim=-1)                 # [B, C]

    # 4. 计算未知类别概率 p_{C+1} (公式8)
    p_C_plus_1 = 1 - torch.sum((1 - p_no) * p_yes, dim=-1)  # [B]

    # 5. OOD 判定 (公式9)
    max_p_id, _ = torch.max(p_yes, dim=-1)
    is_id = (p_C_plus_1 <= max_p_id).long()  # 1=ID, 0=OOD
    return is_id, p_yes, p_no, p_C_plus_1



import torch
from typing import List, Tuple

def compute_grad_norms_on_params(losses: List[torch.Tensor],
                                 params: List[torch.nn.Parameter],
                                 retain_graph: bool = True,
                                 to_cpu: bool = True
                                ) -> Tuple[List[float], List[torch.Tensor]]:
    """
    计算每个 scalar loss 在给定 params 上的梯度 L2 范数，并返回扁平化的 grad 向量（detached）。
    losses: [L1, L2, ...] 每个为 scalar torch.Tensor（未加权）
    params: 你想监控的参数 list（例如 list(model.prompt_learner.parameters()))
    retain_graph: 如果后续还要 total_loss.backward()，设置 True（会占用额外显存）
    to_cpu: 是否把返回的 flat grad vectors 转到 CPU（便于打印/保存）
    返回:
      grad_norms: list of floats (L2 norm)
      grad_vecs: list of 1D torch.Tensor (detached, on CPU if to_cpu=True)
    """
    params = [p for p in params if p.requires_grad]
    grad_norms = []
    grad_vecs = []
    device = params[0].device if len(params) else torch.device('cpu')

    for L in losses:
        grads = torch.autograd.grad(L, params, retain_graph=retain_graph, create_graph=False)
        # 拼接扁平向量并计算平方和
        flat_list = []
        sqsum = 0.0
        for g in grads:
            if g is None:
                continue
            g_det = g.detach()
            flat_list.append(g_det.reshape(-1))
            # 把平方和搬到 CPU 累计，避免显存增长（若在大量参数上）
            sqsum += float((g_det ** 2).sum().cpu())
        if len(flat_list) > 0:
            flat_vec = torch.cat(flat_list)
            if to_cpu:
                flat_vec_out = flat_vec.cpu().detach()
            else:
                flat_vec_out = flat_vec.detach()
            grad_norm = sqsum ** 0.5
        else:
            flat_vec_out = torch.zeros(0)
            grad_norm = 0.0
        grad_norms.append(grad_norm)
        grad_vecs.append(flat_vec_out)

    return grad_norms, grad_vecs


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def eval_isid_sklearn(is_id_pred, is_ood_true):
    """
    评估ID/OOD检测性能
    is_id_pred   : 预测 (1=ID, 0=OOD) 
    is_ood_true  : 真实 (1=OOD, 0=ID) 
    """
    # 转换为numpy数组
    if isinstance(is_id_pred, list):
        is_id_pred = np.concatenate(is_id_pred)
    if isinstance(is_ood_true, list):
        is_ood_true = np.concatenate(is_ood_true)
    
    is_id_pred = np.asarray(is_id_pred).ravel()
    is_ood_true = np.asarray(is_ood_true).ravel()
    
    # 转换预测：1=ID, 0=OOD → 0=ID, 1=OOD
    is_ood_pred = 1 - is_id_pred
    
    if is_ood_pred.shape[0] != is_ood_true.shape[0]:
        raise ValueError(f"长度不一致: pred={len(is_ood_pred)}, true={len(is_ood_true)}")

    # 计算各项指标（OOD为正类）
    metrics = {
        "accuracy": accuracy_score(is_ood_true, is_ood_pred),
        "precision_macro": precision_score(is_ood_true, is_ood_pred, average="macro"),
        "recall_macro": recall_score(is_ood_true, is_ood_pred, average="macro"), 
        "f1_macro": f1_score(is_ood_true, is_ood_pred, average="macro"),
        # ID类指标 (label=0)
        "precision_id": precision_score(is_ood_true, is_ood_pred, pos_label=0),
        "recall_id": recall_score(is_ood_true, is_ood_pred, pos_label=0),
        "f1_id": f1_score(is_ood_true, is_ood_pred, pos_label=0),
        # OOD类指标 (label=1) 
        "precision_ood": precision_score(is_ood_true, is_ood_pred, pos_label=1),
        "recall_ood": recall_score(is_ood_true, is_ood_pred, pos_label=1),
        "f1_ood": f1_score(is_ood_true, is_ood_pred, pos_label=1),
    }

    for k, v in metrics.items():
        print(f"{k:15s}: {v:.4f}")

    return metrics
def eval_clean_detector(is_clean_pred, is_clean_true):
    """
    评估干净/噪声样本检测性能
    参数:
        is_clean_pred: [N] 预测 (bool 或 0/1)，1=干净
        is_clean_true: [N] 真实 (bool 或 0/1)，1=干净
    返回:
        metrics: dict，包含 accuracy, precision, recall, f1, confusion_matrix
    """
    is_clean_pred = np.concatenate(is_clean_pred)
    is_clean_true = np.concatenate(is_clean_true)
    y_pred = np.asarray(is_clean_pred).astype(int).ravel()
    y_true = np.asarray(is_clean_true).astype(int).ravel()

    if y_pred.shape[0] != y_true.shape[0]:
        raise ValueError(f"长度不一致: pred={len(y_pred)}, true={len(y_true)}")

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1),  # 1=干净
        "recall": recall_score(y_true, y_pred, pos_label=1),
        "f1": f1_score(y_true, y_pred, pos_label=1),

    }

    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k:12s}: {v:.4f}")
    # print("confusion_matrix:\n", np.array(metrics["confusion_matrix"]))

    return metrics


def gl_mcm_score(logits_global, logits_local, logits_global_neg=None, pooling="max", n_nega_ctx = 2, temperature = 1):
    """
    logits_global: [B, C*(1+n_nega)]
    logits_local: [B, grid*grid, C*(1+n_nega)]
    logits_neg: 全局负提示的相似度
    """
    lambda_local = 0
    
    # 第二种
    # logits = logits_global.view(-1, int(logits_global.shape[1]/(1+n_nega_ctx)), 1+n_nega_ctx)
    # B, C = logits.shape[0], logits.shape[1]
    # logits_no, logits_yes = logits[:, :, 1:], logits[:, :, 0] #[B, C, n_nega], [B, C]
    # probs_g = torch.exp(logits_yes) / (torch.exp(logits_yes) + torch.exp(logits_no).sum(dim=-1)) #[B, C]
    # logits_l = logits_local.view(B, -1, int(logits_global.shape[1]/(1+n_nega_ctx)), 1+n_nega_ctx)
    # logits_no_l, logits_yes_l = logits_l[:, :, :, 1:], logits_l[:, :, :, 0]
    # probs_l = torch.exp(logits_yes_l) / (torch.exp(logits_yes_l) + torch.exp(logits_no_l).sum(dim=-1))

    # probs_g, probs_l = compute_probs_allneg(logits_global, logits_local, n_nega_ctx=n_nega_ctx, tau=1.0)
    # probs_g = probs_g.cpu().numpy()
    # probs_l = probs_l.cpu().numpy()

    # 第一种
    # probs_g = F.softmax(logits_global, dim=1).data.cpu().numpy()              # [B, C*(1+n_nega)]
    # probs_l = F.softmax(logits_local, dim=-1).data.cpu().numpy()              # [B, grid*grid, C*(1+n_nega)]

    # 第三种
    B, spatial_dim, total_logits = logits_local.shape
    C = total_logits // (1 + n_nega_ctx)  # 计算类别数量
    # 重新reshape logits: [B, C, 1+n_nega]
    # 其中第0个位置是positive logits，后面n_nega个位置是negative logits
    logits_reshaped = logits_global.view(B, C, 1 + n_nega_ctx)
    logits_l = logits_local.view(B, spatial_dim, C, 1 + n_nega_ctx)

    # 分离positive和negative logits
    pos_logits, neg_logits = logits_reshaped[:, :, 0], logits_reshaped[:, :, 1:]  
    pos_logits_l, neg_logits_l = logits_l[:, :, :, 0], logits_l[:, :, :, 1:]

    diff_logits = pos_logits - neg_logits.mean(dim=-1)
    diff_logits_l = pos_logits_l - neg_logits_l.mean(dim=-1)

    # 对每个样本的C个类别进行softmax归一化
    probs_g = F.softmax(diff_logits / temperature, dim=-1)
    probs_l = F.softmax(diff_logits_l / temperature, dim = -1)

    
    probs_g = probs_g.detach().cpu().numpy()
    probs_l = probs_l.detach().cpu().numpy()

    # A negative sign has already been added here
    scores = -np.max(probs_g, axis = 1) - np.max(probs_l, axis = (1, 2)) * lambda_local

    return scores





"""
10.12 之前的函数实在是太乱了，写一个新的，首先看下没有ID-noisy数据的时候效果怎么样。
"""


def warmup_positive_prompt_only(
    net,
    data,
    labels,
    modify_to_ori,
    n_nega_ctx,
    temp=None,
    confidence_threshold=0.9,
    
):
    """
    第一阶段：仅用正提示词训练 + 筛选高置信 clean 样本

    Args:
        net: 模型
        data: 输入图像 [B, C, H, W]
        labels: 标签 [B]（可能包含 OOD 的错误标签）
        modify_to_ori: 传给 net 的标志
        logit_scale: CLIP 的 logit_scale（标量）
        temp: 温度参数，若为 None 则从 logit_scale 推导
        confidence_threshold: 置信度阈值，用于筛选 clean 样本

    Returns:
        loss: 仅基于 proxy-clean 样本的正提示分类损失（用于反向传播）
        clean_mask: [B] bool tensor，True 表示高置信 clean 样本
        mcm_score: [B] 置信度分数
    """
    # 前向传播
    output, text_features = net(data, modify_to_ori)
    # reshape logits: [B, C, 1 + n_nega_ctx]
    n_ctx = 1 + net.n_nega_ctx  # 假设 net 有属性 n_nega_ctx
    logits = output.view(-1, output.shape[1] // n_ctx, n_ctx)  # [B, C, n_ctx]
    logits_pos = logits[:, :, 0]  # [B, C]
    logit_scale = net.logit_scale

    # 设置温度
    if temp is None:
        temp = 1.0 / logit_scale.exp().item()  # 通常 ~0.01

    # Step 1: 计算 MCM 置信度（ID 置信度）
    prob = F.softmax(logits_pos / temp, dim=1)  # [B, C]
    mcm_score = prob.max(dim=1)[0]             # [B]

    # Step 2: 筛选高置信样本（proxy-clean）
    clean_mask = mcm_score > confidence_threshold  # [B]
    
    logits_pos_clean = logits_pos[clean_mask]  # [N_clean, C]
    labels_clean = labels[clean_mask]          # [N_clean]，对应 proxy-clean 的标签
    loss = F.cross_entropy(logits_pos_clean / temp, labels_clean)
    
    return loss, clean_mask, mcm_score

def evaluate_sample_types(all_id_mask, all_id_clean_mask, all_batch_is_clean, all_batch_is_open, all_batch_labels):
    """
    评估 ID-clean, ID-noisy, OOD 三类样本的 Acc 和 Recall。
    
    注意：
        - all_id_mask: 模型预测的 ID 掩码 (bool, [N])
        - all_id_clean_mask: 在预测为 ID 的样本中，预测为 clean 的掩码 (bool, [N_id])
        - final_is_clean_full: 真实 clean 标签 (bool, [N])，仅对真实 ID 有效
        - final_is_open_full: 真实 OOD 标签 (bool, [N])，1 表示 OOD
    """
    # 合并批次数据

    

    final_is_clean_full = torch.cat(all_batch_is_clean, dim=0).numpy()  # [N]
    final_is_open_full = torch.cat(all_batch_is_open, dim=0).numpy()    # [N], 1 = OOD

    N = len(final_is_clean_full)

    # === 构建真实三分类标签 (0: clean, 1: noisy, 2: ood) ===
    y_true = np.full(N, -1, dtype=int)
    true_id_mask = ~final_is_open_full  # 真实 ID 样本

    y_true[true_id_mask & final_is_clean_full] = 0   # clean
    y_true[true_id_mask & (~final_is_clean_full)] = 1  # noisy
    y_true[final_is_open_full] = 2                     # ood

    # === 构建预测三分类标签 ===
    y_pred = np.full(N, -1, dtype=int)

    # 先初始化：所有预测为 OOD 的样本 → pred = 2
    pred_ood_mask = ~all_id_mask  # 模型预测为 OOD
    y_pred[pred_ood_mask] = 2

    # 对预测为 ID 的样本，进一步划分 clean/noisy
    pred_id_indices = np.where(all_id_mask)[0]  # 长度 = N_id
    # all_id_clean_mask 长度 = N_id，对应 pred_id_indices 位置
    pred_clean_in_id = all_id_clean_mask  # bool array of length N_id

    # 预测为 ID-clean
    y_pred[pred_id_indices[pred_clean_in_id]] = 0
    # 预测为 ID-noisy
    y_pred[pred_id_indices[~pred_clean_in_id]] = 1

    # === 安全检查 ===
    assert not np.any(y_true == -1), "y_true contains unlabeled samples!"
    assert not np.any(y_pred == -1), "y_pred contains unlabeled samples!"

    # === 计算每类的 Accuracy 和 Recall ===
    print("--- Sample Type Evaluation (3-class) ---")
    class_names = ["ID-Clean", "ID-Noisy", "OOD"]
    for cls in [0, 1, 2]:
        cls_mask = (y_true == cls)
        if cls_mask.sum() == 0:
            print(f"{class_names[cls]:<10} - No samples in ground truth.")
            continue

        # Accuracy for this class: among samples predicted as cls, how many are correct?
        pred_as_cls = (y_pred == cls)
        if pred_as_cls.sum() > 0:
            acc = (y_pred[pred_as_cls] == y_true[pred_as_cls]).mean()
        else:
            acc = 0.0  # 或 float('nan')

        # Recall for this class: among true cls samples, how many are predicted as cls?
        recall = (y_pred[cls_mask] == cls).mean()

        print(f"{class_names[cls]:<10} - Accuracy: {acc:.4f}, Recall: {recall:.4f}")

    # (可选) 打印整体 accuracy
    # overall_acc = (y_pred == y_true).mean()
    # print(f"\nOverall 3-class Accuracy: {overall_acc:.4f}")

        # === 补充：整体 ID vs OOD 的二分类指标 ===
    print("\n--- Binary ID vs OOD Evaluation ---")
    
    # 真实标签：1 = ID, 0 = OOD
    y_true_id_binary = (~final_is_open_full).astype(int)  # [N], 1 for ID, 0 for OOD
    # 预测标签：1 = ID, 0 = OOD
    y_pred_id_binary = all_id_mask.astype(int)           # [N], 1 for predicted ID

    # ID 类别 (positive class = ID)
    tp_id = ((y_pred_id_binary == 1) & (y_true_id_binary == 1)).sum()
    fp_id = ((y_pred_id_binary == 1) & (y_true_id_binary == 0)).sum()
    fn_id = ((y_pred_id_binary == 0) & (y_true_id_binary == 1)).sum()
    tn_id = ((y_pred_id_binary == 0) & (y_true_id_binary == 0)).sum()

    # ID metrics
    precision_id = tp_id / (tp_id + fp_id) if (tp_id + fp_id) > 0 else 0.0
    recall_id = tp_id / (tp_id + fn_id) if (tp_id + fn_id) > 0 else 0.0

    # OOD metrics (positive class = OOD)
    tp_ood = tn_id  # 预测 OOD 且真实 OOD
    fp_ood = fn_id  # 预测 OOD 但真实 ID（即漏检）
    fn_ood = fp_id  # 预测 ID 但真实 OOD（即误报）
    # tn_ood = tp_id

    precision_ood = tp_ood / (tp_ood + fp_ood) if (tp_ood + fp_ood) > 0 else 0.0
    recall_ood = tp_ood / (tp_ood + fn_ood) if (tp_ood + fn_ood) > 0 else 0.0

    print(f"ID (vs OOD)   - Precision (Acc): {precision_id:.4f}, Recall: {recall_id:.4f}")
    print(f"OOD (vs ID)   - Precision (Acc): {precision_ood:.4f}, Recall: {recall_ood:.4f}")


"""
10.15 修改全局 GMM
"""

def analyze_per_class_our_method(
    run, all_score1_np, all_score2_np, all_labels_np, all_batch_is_open, all_batch_is_clean,
    top_k_per_class_id=16, top_k_per_class_clean=16, bottom_k_global=500, epoch=None,
):
    """
    分析我们方法的划分效果：
    1. 选取每个类别 Score1 得分最高的 top_k_per_class 个样本 (标记为 ID)，统计其真实标签分布。
    2. 选取每个类别 Score2 得分最高的 top_k_per_class 个样本 (标记为 ID-clean)，统计其真实标签分布。
    3. 选取全局 Score2 得分最低的 bottom_k_global 个样本 (标记为 OOD)，统计其真实标签分布。

    Args:
        all_score1_np (np.ndarray): [N_total], Score1 分数
        all_score2_np (np.ndarray): [N_total], Score2 分数
        all_labels_np (np.ndarray): [N_total], 所有样本的类别标签
        all_batch_is_open (np.ndarray): [N_total], 所有样本的 OOD 标签 (1=OOD, 0=ID)
        all_batch_is_clean (np.ndarray): [N_total], 所有样本的 clean 标签 (1=ID-clean, 0=ID-noisy or OOD)
        top_k_per_class_id (int): 每个类别中标记为ID的样本数 (Score1 Top-K)
        top_k_per_class_clean (int): 每个类别中标记为ID-clean的样本数 (Score2 Top-K)
        bottom_k_global (int): 全局提取的最低分样本数
        epoch (int, optional): 当前epoch数，用于日志记录
    """
    N_total = len(all_score1_np)
    if len(all_batch_is_open) != N_total or len(all_batch_is_clean) != N_total:
        raise ValueError(f"Length mismatch: all_score1_np ({N_total}), all_batch_is_open ({len(all_batch_is_open)}), all_batch_is_clean ({len(all_batch_is_clean)})")

    unique_labels = np.unique(all_labels_np)

    # 存储结果
    results_score1_topk = {"total": 0, "clean": 0, "noisy": 0, "ood": 0}
    results_score2_topk = {"total": 0, "clean": 0, "noisy": 0, "ood": 0}
    results_score2_bottomk = {"total": 0, "clean": 0, "noisy": 0, "ood": 0}

    # --- 1. 分析 Score1 Top-K (ID) ---
    all_top_k_indices_by_score1 = []
    for label in unique_labels:
        class_indices = np.where(all_labels_np == label)[0]
        if len(class_indices) == 0:
            continue
        class_score1 = all_score1_np[class_indices]
        k_to_select = min(top_k_per_class_id, len(class_score1))
        if k_to_select <= 0:
            continue
        top_k_local_by_score1 = np.argpartition(class_score1, -k_to_select)[-k_to_select:]
        top_k_global_by_score1 = class_indices[top_k_local_by_score1]
        top_k_global_by_score1 = top_k_global_by_score1[np.argsort(all_score1_np[top_k_global_by_score1])[::-1]]
        all_top_k_indices_by_score1.extend(top_k_global_by_score1)

    all_top_k_indices_by_score1 = np.array(all_top_k_indices_by_score1)
    if len(all_top_k_indices_by_score1) > 0:
        true_open_score1 = all_batch_is_open[all_top_k_indices_by_score1] # 0=ID, 1=OOD
        true_clean_score1 = all_batch_is_clean[all_top_k_indices_by_score1] # 0=not_clean, 1=ID-clean
        is_clean = (true_clean_score1 == 1) & (true_open_score1 == 0)
        is_noisy = (true_clean_score1 == 0) & (true_open_score1 == 0)
        is_ood = (true_open_score1 == 1)
        results_score1_topk["total"] = len(all_top_k_indices_by_score1)
        results_score1_topk["clean"] = is_clean.sum()
        results_score1_topk["noisy"] = is_noisy.sum()
        results_score1_topk["ood"] = is_ood.sum()

    # --- 2. 分析 Score2 Top-K (ID-clean) ---
    all_top_k_indices_by_score2 = []
    for label in unique_labels:
        class_indices = np.where(all_labels_np == label)[0]
        if len(class_indices) == 0:
            continue
        class_score2 = all_score2_np[class_indices]
        k_to_select = min(top_k_per_class_clean, len(class_score2))
        if k_to_select <= 0:
            continue
        top_k_local_by_score2 = np.argpartition(class_score2, -k_to_select)[-k_to_select:]
        top_k_global_by_score2 = class_indices[top_k_local_by_score2]
        top_k_global_by_score2 = top_k_global_by_score2[np.argsort(all_score2_np[top_k_global_by_score2])[::-1]]
        all_top_k_indices_by_score2.extend(top_k_global_by_score2)

    all_top_k_indices_by_score2 = np.array(all_top_k_indices_by_score2)
    if len(all_top_k_indices_by_score2) > 0:
        true_open_score2 = all_batch_is_open[all_top_k_indices_by_score2] # 0=ID, 1=OOD
        true_clean_score2 = all_batch_is_clean[all_top_k_indices_by_score2] # 0=not_clean, 1=ID-clean
        is_clean = (true_clean_score2 == 1) & (true_open_score2 == 0)
        is_noisy = (true_clean_score2 == 0) & (true_open_score2 == 0)
        is_ood = (true_open_score2 == 1)
        results_score2_topk["total"] = len(all_top_k_indices_by_score2)
        results_score2_topk["clean"] = is_clean.sum()
        results_score2_topk["noisy"] = is_noisy.sum()
        results_score2_topk["ood"] = is_ood.sum()

    # --- 3. 分析 Score2 Bottom-K (OOD) ---
    k_to_select_global = min(bottom_k_global, N_total)
    bottom_k_global_indices = np.argpartition(all_score2_np, k_to_select_global)[:k_to_select_global]
    bottom_k_global_indices = bottom_k_global_indices[np.argsort(all_score2_np[bottom_k_global_indices])]
    
    if len(bottom_k_global_indices) > 0:
        true_open_bottom = all_batch_is_open[bottom_k_global_indices] # 0=ID, 1=OOD
        true_clean_bottom = all_batch_is_clean[bottom_k_global_indices] # 0=not_clean, 1=ID-clean
        is_clean = (true_clean_bottom == 1) & (true_open_bottom == 0)
        is_noisy = (true_clean_bottom == 0) & (true_open_bottom == 0)
        is_ood = (true_open_bottom == 1)
        results_score2_bottomk["total"] = len(bottom_k_global_indices)
        results_score2_bottomk["clean"] = is_clean.sum()
        results_score2_bottomk["noisy"] = is_noisy.sum()
        results_score2_bottomk["ood"] = is_ood.sum()


    # --- 打印结果 ---
    def print_result_section(title, results):
        total = results['total']
        if total == 0:
            print(f"   - No samples selected.")
            return
        clean_c = results['clean']
        noisy_c = results['noisy']
        ood_c = results['ood']
        print(f"   - Total samples: {total}")
        print(f"   - True ID-clean: {clean_c} ({clean_c/total*100:.2f}%)")
        print(f"   - True ID-noisy: {noisy_c} ({noisy_c/total*100:.2f}%)")
        print(f"   - True OOD: {ood_c} ({ood_c/total*100:.2f}%)")


    print("--- Analysis: Our Method Sample Selection (ID-clean/ID-noisy/OOD Distribution) ---")
    print(f"1. Per-Class Top-{top_k_per_class_id} by Score1 (Marked as ID):")
    print_result_section("Score1-TopK", results_score1_topk)

    print(f"\n2. Per-Class Top-{top_k_per_class_clean} by Score2 (Marked as ID-clean):")
    print_result_section("Score2-TopK", results_score2_topk)

    print(f"\n3. Global Bottom-{bottom_k_global} by Score2 (Marked as OOD):")
    print_result_section("Score2-BottomK", results_score2_bottomk)

    # --- Wandb 日志记录 ---
    # if run is not None:
    #     # Marked as ID (Score1 Top-K)
    #     if results_score1_topk["total"] > 0:
    #         run.log({
    #             "Marked as ID/Total": results_score1_topk["total"],
    #             "Marked as ID/True ID-clean": results_score1_topk["clean"],
    #             "Marked as ID/True ID-noisy": results_score1_topk["noisy"],
    #             "Marked as ID/True OOD": results_score1_topk["ood"],
    #             "Marked as ID/True ID-clean Ratio": results_score1_topk["clean"] / results_score1_topk["total"] * 100,
    #             "Marked as ID/True ID-noisy Ratio": results_score1_topk["noisy"] / results_score1_topk["total"] * 100,
    #             "Marked as ID/True OOD Ratio": results_score1_topk["ood"] / results_score1_topk["total"] * 100,
    #         }, step=epoch)
    #
    #     # Marked as ID-clean (Score2 Top-K)
    #     if results_score2_topk["total"] > 0:
    #         run.log({
    #             "Marked as ID-clean/Total": results_score2_topk["total"],
    #             "Marked as ID-clean/True ID-clean": results_score2_topk["clean"],
    #             "Marked as ID-clean/True ID-noisy": results_score2_topk["noisy"],
    #             "Marked as ID-clean/True OOD": results_score2_topk["ood"],
    #             "Marked as ID-clean/True ID-clean Ratio": results_score2_topk["clean"] / results_score2_topk["total"] * 100,
    #             "Marked as ID-clean/True ID-noisy Ratio": results_score2_topk["noisy"] / results_score2_topk["total"] * 100,
    #             "Marked as ID-clean/True OOD Ratio": results_score2_topk["ood"] / results_score2_topk["total"] * 100,
    #         }, step=epoch)
    #
    #     # Marked as OOD (Score2 Bottom-K)
    #     if results_score2_bottomk["total"] > 0:
    #         run.log({
    #             "Marked as OOD/Total": results_score2_bottomk["total"],
    #             "Marked as OOD/True ID-clean": results_score2_bottomk["clean"],
    #             "Marked as OOD/True ID-noisy": results_score2_bottomk["noisy"],
    #             "Marked as OOD/True OOD": results_score2_bottomk["ood"],
    #             "Marked as OOD/True ID-clean Ratio": results_score2_bottomk["clean"] / results_score2_bottomk["total"] * 100,
    #             "Marked as OOD/True ID-noisy Ratio": results_score2_bottomk["noisy"] / results_score2_bottomk["total"] * 100,
    #             "Marked as OOD/True OOD Ratio": results_score2_bottomk["ood"] / results_score2_bottomk["total"] * 100,
    #         }, step=epoch)





def analyze_score_distribution(score1, score2, is_clean, is_open, epoch=None, save_path="."):
    """
    分析 score1 和 score2 在 ID-clean、ID-noisy、OOD 三类数据上的分布

    参数:
        score1: 一维数组，对应 "s+ - s-" 分数
        score2: 一维数组，对应 "MCM-style" 分数
        is_clean: 一维布尔数组，True 表示 ID-clean 样本
        is_open: 一维布尔数组，True 表示 OOD 样本
        epoch: 可选，当前 epoch 数（用于图片命名）
        save_path: 图片保存路径

    返回:
        stats_list: 包含所有统计指标的列表
    """
    # --------------------------
    # 1. 数据校验（确保输入合法）
    # --------------------------
    assert len(score1) == len(score2) == len(is_clean) == len(is_open), \
        "score1、score2、is_clean、is_open 的长度必须一致！"
    assert isinstance(is_clean, np.ndarray) and is_clean.dtype == bool, \
        "is_clean 必须是布尔类型的 numpy 数组！"
    assert isinstance(is_open, np.ndarray) and is_open.dtype == bool, \
        "is_open 必须是布尔类型的 numpy 数组！"

    # --------------------------
    # 2. 数据类别划分
    # --------------------------
    # 生成三类数据的掩码
    mask_id_clean = is_clean  # ID-clean：is_clean=True
    mask_ood = is_open  # OOD：is_open=True
    mask_id_noisy = ~mask_id_clean & ~mask_ood  # ID-noisy：非干净且非OOD

    # 筛选每类数据的 score1 和 score2
    data = {
        "ID-clean": {
            "score1": score1[mask_id_clean],
            "score2": score2[mask_id_clean]
        },
        "ID-noisy": {
            "score1": score1[mask_id_noisy],
            "score2": score2[mask_id_noisy]
        },
        "OOD": {
            "score1": score1[mask_ood],
            "score2": score2[mask_ood]
        }
    }

    # 打印每类数据量
    print("=== 各类数据样本数量 ===")
    for data_type in data:
        print(f"{data_type}: {len(data[data_type]['score1'])} 个")

    # --------------------------
    # 3. 计算统计指标（内部辅助函数）
    # --------------------------
    def _calculate_stats(score_array, data_type, score_name):
        if len(score_array) == 0:
            return {
                "data_type": data_type,
                "score_name": score_name,
                "mean": "无数据",
                "std": "无数据",
                "median": "无数据",
                "q25": "无数据",
                "q75": "无数据"
            }
        return {
            "data_type": data_type,
            "score_name": score_name,
            "mean": np.round(np.mean(score_array), 4),
            "std": np.round(np.std(score_array), 4),
            "median": np.round(np.median(score_array), 4),
            "q25": np.round(np.percentile(score_array, 25), 4),
            "q75": np.round(np.percentile(score_array, 75), 4)
        }

    # 批量计算统计指标
    stats_list = []
    for data_type in data:
        stats_list.append(_calculate_stats(
            data[data_type]["score1"], data_type, "score1 (s+ - s-)"
        ))
        stats_list.append(_calculate_stats(
            data[data_type]["score2"], data_type, "score2 (MCM-style)"
        ))

    # 打印统计结果
    print("\n=== 统计指标汇总 ===")
    print(f"{'数据类型':<10} {'指标名':<20} {'均值':<8} {'标准差':<8} {'中位数':<8} {'下四分位':<8} {'上四分位':<8}")
    print("-" * 80)
    for stats in stats_list:
        print(
            f"{stats['data_type']:<10} {stats['score_name']:<20} {stats['mean']:<8} {stats['std']:<8} {stats['median']:<8} {stats['q25']:<8} {stats['q75']:<8}")

    # --------------------------
    # 4. 可视化分布（直方图 + 箱线图）
    # --------------------------
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['figure.figsize'] = (12, 5)
    colors = {
        "ID-clean": "#2E86AB",
        "ID-noisy": "#A23B72",
        "OOD": "#F18F01"
    }

    # 创建子图
    fig, axes = plt.subplots(1, 2, tight_layout=True)

    # 绘制 score1 和 score2 的分布
    for ax_idx, score_name in enumerate(["score1", "score2"]):
        ax = axes[ax_idx]
        plot_title = f"score1 (s+ - s-)" if score_name == "score1" else "score2 (MCM-style)"

        # 收集当前 score 的所有数据（用于统一 bins 范围）
        all_scores = []
        for data_type in data:
            all_scores.extend(data[data_type][score_name])
        if not all_scores:  # 避免空数据报错
            ax.set_title(f"{plot_title} (无数据)")
            continue

        # 统一 bins 范围
        bins = np.linspace(min(all_scores), max(all_scores), 30)

        # 绘制箱线图
        box_data = []
        box_labels = []
        for data_type in data:
            scores = data[data_type][score_name]
            if len(scores) > 0:
                box_data.append(scores)
                box_labels.append(data_type)

        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                        widths=0.6, showfliers=False,
                        medianprops={"color": "white", "linewidth": 2})
        for patch, label in zip(bp['boxes'], box_labels):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.5)

        # 绘制直方图（密度模式）
        for data_type in data:
            scores = data[data_type][score_name]
            if len(scores) > 0:
                ax.hist(scores, bins=bins, alpha=0.3, color=colors[data_type],
                        label=data_type, density=True)

        # 子图配置
        ax.set_title(plot_title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Score Value", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    # 保存图片
    if epoch is not None:
        filename = f"score_distribution_epoch_{epoch}.png"
    else:
        filename = "score_distribution.png"

    full_path = f"{save_path}/{filename}"
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    plt.savefig(full_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n分布对比图已保存至：{full_path}")

    return stats_list

def train_dividemix(net, optimizer, scheduler, trainloader, run, epoch=None, proto=None, **options):
    n_nega_ctx = options['NEGA_CTX']
    batch_size = 512  # 可调
    total_loss = 0.0
    total_samples = 0

    # --- 1. 推断阶段：收集所有样本的 score1, score2 和标签 ---
    print(f"Epoch {epoch}: Running inference for sample identification...")
    net.eval()  # 切换到评估模式
    all_score1 = []  # MCM-style
    all_score2 = []  # s+ - s-
    all_score3 = []  # Top 5 (s+ - s-)
    all_labels = []  # 存储POMP变换后的标签（按你的需求保留）
    all_batch_data = []
    all_batch_labels = []
    all_batch_is_clean = []  # 用于评估
    all_batch_is_open = []  # 用于评估

    with torch.no_grad():  # 禁用梯度计算以节省内存
        all_top5_acc = []  # 收集ID样本的top5准确率
        all_top1_acc = []  # 新增：收集ID样本的top1准确率
        for batch_idx, batch in enumerate(trainloader):

            data = batch['data']
            labels = batch['label']  # 噪声标签
            clean_labels = batch['clean_label']  # 真实标签（用于准确率计算）
            is_open = batch['is_open']  # 区分ID/OOD
            if options['use_gpu']:
                # 所有张量统一移到GPU
                data, labels, clean_labels, is_open = data.cuda(), labels.cuda(), clean_labels.cuda(), is_open.cuda()

            # POMP标签变换（保持原逻辑）
            if options['POMP']:
                ori_to_modify, modify_to_ori = label_transform(
                    labels.cpu().numpy(), options['POMP_k'], options['num_classes'] - 1
                )
                modified_labels = torch.tensor(
                    [ori_to_modify[label.item()] for label in labels]
                ).cuda()
                labels = modified_labels
            else:
                ori_to_modify, modify_to_ori = None, None

            # 前向传播与logits解析（保持原逻辑）
            if options['stage'] == 1:
                output, text_features = net(data, modify_to_ori)
            else:
                output, text_features, output_global_neg, global_neg_features, logits_local, logits_local_neg = net(
                    data, modify_to_ori)

            logits = output.view(-1, int(output.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)
            B, C = logits.shape[0], logits.shape[1]
            logits_no, logits_yes = logits[:, :, 1:], logits[:, :, 0]  # logits_yes: [B, C]

            # 计算logits_margin和prob_margin（保持原逻辑）
            mean_logits_no = logits_no.mean(dim=2)  # [B, C]

            logits_margin = logits_yes - mean_logits_no  # [B, C]
            # logits_margin = logits_yes
            T = 0.1
            prob_margin = torch.softmax(logits_margin / T, dim=1)  # [B, C]：所有类别的概率

            # 计算score1：prob_margin的最大值（保持原逻辑）
            score1 = prob_margin.max(dim=1).values

            # score2：基于噪声标签的概率（保持原逻辑）
            score2 = prob_margin[torch.arange(B), labels]  # [B]

            # 计算score3：logits_yes前5类中prob_margin的最大值（保持原逻辑）
            _, top5_indices = torch.topk(logits_yes, k=5, dim=1, largest=True, sorted=True)
            batch_indices = torch.arange(B).unsqueeze(1).repeat(1, 5).view(-1)
            top5_flat_indices = top5_indices.view(-1)
            top5_probs = prob_margin[batch_indices, top5_flat_indices].view(B, 5)
            score3 = top5_probs.max(dim=1).values  # [B]



            # --------------------------
            # 新增：计算top1准确率（仅ID样本）
            # 逻辑：logits_yes的第1名（top1）是否等于真实标签clean_label
            # --------------------------
            # id_mask = is_open == False
            # if id_mask.sum() > 0:
            #     # 提取ID样本的真实标签和top5索引（top5的第0位即top1）
            #     id_clean_labels = clean_labels[id_mask]
            #     id_top5_indices = top5_indices[id_mask]  # [num_id, 5]
            #     id_top1_indices = id_top5_indices[:, 0]  # 取第1名的索引（[num_id]）
            #
            #     # top5准确率（原逻辑保留）
            #     correct_top5 = (id_top5_indices == id_clean_labels.unsqueeze(1)).any(dim=1).float()
            #     batch_top5_acc = correct_top5.sum() / id_mask.sum()
            #     all_top5_acc.append(batch_top5_acc.cpu().item())
            #
            #     # 新增：top1准确率
            #     correct_top1 = (id_top1_indices == id_clean_labels).float()  # 第1名是否等于真实标签
            #     batch_top1_acc = correct_top1.sum() / id_mask.sum()
            #     all_top1_acc.append(batch_top1_acc.cpu().item())

            # 数据缓存（保持原逻辑）
            all_score1.append(score1.cpu().numpy())
            all_score2.append(score2.cpu().numpy())
            all_score3.append(score3.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_batch_data.append(data.cpu().numpy())
            all_batch_labels.append(labels.cpu().numpy())
            all_batch_is_clean.append(batch['is_clean'].cpu().bool().numpy())
            all_batch_is_open.append(is_open.cpu().bool().numpy())

        # 输出平均准确率（含top1和top5）
        # if all_top5_acc:
        #     avg_top5_acc = sum(all_top5_acc) / len(all_top5_acc)
        #     print(f"所有ID样本的平均top5准确率: {avg_top5_acc:.4f}")
        # else:
        #     print("无ID样本，未计算top5准确率")
        #
        # if all_top1_acc:
        #     avg_top1_acc = sum(all_top1_acc) / len(all_top1_acc)
        #     print(f"所有ID样本的平均top1准确率: {avg_top1_acc:.4f}")  # 新增：top1结果
        # else:
        #     print("无ID样本，未计算top1准确率")

    # 合并所有批次的分数和标签（保持你要求的分数顺序）
    all_score1_np = np.concatenate(all_score1, axis=0)
    all_score2_np = np.concatenate(all_score2, axis=0)
    all_score3_np = np.concatenate(all_score3, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)
    all_batch_is_clean = np.concatenate(all_batch_is_clean, axis=0)
    all_batch_is_open = np.concatenate(all_batch_is_open, axis=0)
    all_data_np = np.concatenate(all_batch_data, axis=0)

    # """"""
    # # --------------------------
    # # 关键：保存原始数据（单文件）
    # # --------------------------
    # # 1. 定义保存路径
    # save_data_dir = "/data3/tyang/score_anal/11-4"
    # os.makedirs(save_data_dir, exist_ok=True)  # 确保目录存在
    #
    # # 2. 定义文件名（含epoch，方便对应不同训练阶段的数据）
    #
    # data_filename = "11-5-check.npz"
    # data_full_path = f"{save_data_dir}/{data_filename}"
    #
    # # 3. 保存所有数组（字典形式，key对应数组名）
    # np.savez(
    #     data_full_path,
    #     score1=all_score1_np,
    #     score2=all_score2_np,
    #     score3=all_score3_np,
    #     labels=all_labels_np,
    #     is_clean=all_batch_is_clean,
    #     is_open=all_batch_is_open,
    #     batch_data=all_data_np
    # )
    # print(f"原始数据已保存至：{data_full_path}")
    # #
    # exit(0)
    # """"""


    # --- 2. 样本划分 ---
    print(f"Using Score1 & Score2 for sample identification on {len(all_score1_np)} samples...")
    top_k_per_class = 256  # 16-shot
    bottom_k_global = 500  # 全局最低分样本数

    # 初始化标签掩码
    N_total = len(all_score1_np)
    final_sample_type = np.full(N_total, -1, dtype=int)  # -1:未标记, 0:ID, 1:ID-clean, 2:OOD

    # --- 1. 标记ID样本（Score1 Top-K）---
    unique_labels = np.unique(all_labels_np)
    all_top_k_indices_by_score1 = []
    for label in unique_labels:
        class_indices = np.where(all_labels_np == label)[0]
        if len(class_indices) == 0:
            continue
        class_score1 = all_score1_np[class_indices]
        k_to_select = min(top_k_per_class, len(class_score1))
        if k_to_select <= 0:
            continue
        # 取Top-K索引
        top_k_local_by_score1 = np.argpartition(class_score1, -k_to_select)[-k_to_select:]
        top_k_global_by_score1 = class_indices[top_k_local_by_score1]
        top_k_global_by_score1 = top_k_global_by_score1[np.argsort(all_score1_np[top_k_global_by_score1])[::-1]]
        all_top_k_indices_by_score1.extend(top_k_global_by_score1)
    all_top_k_indices_by_score1 = np.array(all_top_k_indices_by_score1)
    final_sample_type[all_top_k_indices_by_score1] = 0  # 标记为ID

    # --- 2. 标记ID-clean样本（Score2 Top-K，覆盖ID）---
    all_top_k_indices_by_score2 = []
    for label in unique_labels:
        class_indices = np.where(all_labels_np == label)[0]
        if len(class_indices) == 0:
            continue
        class_score2 = all_score2_np[class_indices]
        k_to_select = min(top_k_per_class_clean, len(class_score2))
        if k_to_select <= 0:
            continue
        top_k_local_by_score2 = np.argpartition(class_score2, -k_to_select)[-k_to_select:]
        top_k_global_by_score2 = class_indices[top_k_local_by_score2]
        top_k_global_by_score2 = top_k_global_by_score2[np.argsort(all_score2_np[top_k_global_by_score2])[::-1]]
        all_top_k_indices_by_score2.extend(top_k_global_by_score2)
    all_top_k_indices_by_score2 = np.array(all_top_k_indices_by_score2)
    final_sample_type[all_top_k_indices_by_score2] = 1  # 覆盖为ID-clean

    # --- 3. 标记OOD样本（Score2 Bottom-K，仅标记未被ID/ID-clean覆盖的样本）---
    k_to_select_global = min(bottom_k_global, N_total)
    bottom_k_global_indices = np.argpartition(all_score2_np, k_to_select_global)[:k_to_select_global]
    bottom_k_global_indices = bottom_k_global_indices[np.argsort(all_score2_np[bottom_k_global_indices])]
    # 修复：仅标记未被ID/ID-clean覆盖的样本（避免类型冲突）
    ood_mask = final_sample_type[bottom_k_global_indices] == -1
    # final_sample_type[bottom_k_global_indices[ood_mask]] = 2  # 标记为OOD

    # --- 4. 统计结果 ---
    id_count = (final_sample_type == 0).sum()
    clean_count = (final_sample_type == 1).sum()
    ood_count = (final_sample_type == 2).sum()
    unmarked_count = (final_sample_type == -1).sum()
    print("--- Final Sample Identification Summary ---")
    print(f"Total Samples: {N_total}")
    print(f"ID samples (from Score1 Top-K): {id_count}")
    print(f"ID-Clean samples (from Score2 Top-K, overrides ID): {clean_count}")
    print(f"OOD samples (from Score2 Bottom-K): {ood_count}")
    print(f"Unmarked samples: {unmarked_count}")
    print("\n" + "=" * 60 + "\n")
    analyze_per_class_our_method(
        all_score1_np, all_score2_np, all_labels_np, all_batch_is_open, all_batch_is_clean,
        top_k_per_class=top_k_per_class, bottom_k_global=bottom_k_global,
    )
    # --- 5. 训练准备 ---
    marked_indices = np.where(final_sample_type != -1)[0]
    marked_labels = final_sample_type[marked_indices]

    print(f"Epoch {epoch}: Running training on {len(marked_indices)} marked samples with adaptive losses...")
    net.train()  # 切换回训练模式
    id_clean_mask = (final_sample_type == 1)
    id_mask = (final_sample_type == 0)

    x_clean_np = all_data_np[id_clean_mask]
    y_clean_np = all_labels_np[id_clean_mask]
    x_id_np = all_data_np[id_mask]

    num_clean = len(x_clean_np)
    num_id = len(x_id_np)

    if num_clean == 0:
        print("No ID-clean samples, skipping DivideMix training.")
        return 0.0

    device = next(net.parameters()).device
    num_classes = options['num_classes']

    # === Step 2: 分块训练（修复索引逻辑）===
    # 按最小长度遍历，避免冗余迭代
    max_iter = max(num_clean, num_id) if num_id > 0 else num_clean
    for start in range(0, max_iter, batch_size):
        optimizer.zero_grad()

        # --- 取当前块的干净样本 ---
        end_clean = min(start + batch_size, num_clean)
        x_clean_b = torch.from_numpy(x_clean_np[start:end_clean]).float().to(device)
        y_clean_b = torch.from_numpy(y_clean_np[start:end_clean]).long().to(device)
        B_clean = x_clean_b.size(0)

        if B_clean == 0:
            continue

        loss_ce = 0.0
        loss_mse = 0.0

        # === 1. CE loss on ID-clean samples ===
        logits_clean, _, _, _, _, _ = net(x_clean_b, modify_to_ori)
        logits_clean = logits_clean.view(-1, int(logits_clean.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]
        loss_ce = F.cross_entropy(logits_clean, y_clean_b)

        # === 2. MSE loss on Mixup(clean, ID) ===
        if num_id > 0:
            # 修复：用取模索引循环填充ID样本，确保长度与干净样本一致
            id_indices = np.mod(np.arange(start, start + B_clean), num_id)  # 循环索引
            x_id_b = torch.from_numpy(x_id_np[id_indices]).float().to(device)
            B_id = x_id_b.size(0)

            # 生成伪标签（使用模型当前输出）
            with torch.no_grad():
                net.eval()
                logits_id, _, _, _, _, _ = net(x_clean_b, modify_to_ori)
                logits_id = logits_id.view(-1, int(logits_id.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]
                p_id = F.softmax(logits_id, dim=1)
                y_id_pseudo = sharpen(p_id, T=0.5)  # 软伪标签
                net.train()

            # Mixup: clean + ID
            mixed_x, mixed_y = mixup_data(
                x_clean_b,
                x_id_b,
                F.one_hot(y_clean_b, num_classes=num_classes).float(),
                y_id_pseudo
            )

            # 前向传播 + MSE loss
            mixed_logits, _, _, _, _, _ = net(mixed_x, modify_to_ori)
            mixed_logits = mixed_logits.view(-1, int(mixed_logits.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]
            pred = F.softmax(mixed_logits, dim=1)
            loss_mse = F.mse_loss(pred, mixed_y)
        else:
            loss_mse = torch.tensor(0.0, device=device)
            print("Warning: No ID samples for Mixup, training with only clean samples.")  # 增加警告

        # === 总损失（添加权重参数）===
        loss = options.get('ce_weight', 1.0) * loss_ce + options.get('mse_weight', 1.0) * loss_mse
        # loss = options.get('ce_weight', 1.0) * loss_ce
        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B_clean
        total_samples += B_clean

    # 调度器在epoch结束后调用
    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    print(
        f"Epoch {epoch}: DivideMix-style avg loss = {avg_loss:.6f} (CE: {loss_ce.item():.6f}, MSE: {loss_mse.item():.6f})")
    return avg_loss


# --- 辅助函数 ---
def sharpen(p, T=0.5):
    p_sharp = p ** (1.0 / T)
    return p_sharp / p_sharp.sum(dim=1, keepdim=True)


def mixup_data(x1, x2, y1, y2, alpha=0.75):
    lam = np.random.beta(alpha, alpha)
    mixed_x = lam * x1 + (1 - lam) * x2
    mixed_y = lam * y1 + (1 - lam) * y2
    return mixed_x, mixed_y







import torch
import torch.nn.functional as F
from torch.distributions import Beta
import numpy as np
# 注意：移除GMM相关依赖（如sklearn.mixture.GaussianMixture）


"""

11 - 10, 样本筛选的逻辑
"""

import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

# ----------------------------
# 1) compute_marked_indices：做完整推断 + selection（Per-class Top-K / Bottom-K）
# ----------------------------
def compute_marked_indices(net, trainloader, options, n_nega_ctx, top_k_per_class=64, bottom_k_global=500, device='cuda'):
    net.eval()
    all_score1 = []
    all_score2 = []
    all_score3 = []
    all_labels = []
    all_clean_labels = []
    all_is_open = []
    all_is_clean = []
    all_batch_data = []
    all_pseudo_label = []

    with torch.no_grad():
        for batch in trainloader:
            data = batch['data']
            labels = batch['label']           # noisy label
            clean_labels = batch['clean_label']
            is_open = batch['is_open']

            if options.get('use_gpu', True):
                data = data.cuda()
                labels = labels.cuda()
                clean_labels = clean_labels.cuda()
                is_open = is_open.cuda()

            # forward (与你原始逻辑一致)
            if options.get('stage', 1) == 1:
                output, text_features = net(data)
            else:
                output, text_features, output_global_neg, global_neg_features, logits_local, logits_local_neg = net(data)

            logits = output.view(-1, int(output.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)
            B, C = logits.shape[0], logits.shape[1]
            logits_no, logits_yes = logits[:, :, 1:], logits[:, :, 0]
            mean_logits_no = logits_no.mean(dim=2)
            logits_margin = logits_yes - mean_logits_no
            T = 0.1
            prob_margin = torch.softmax(logits_margin / T, dim=1)

            score1, pseudo_label = prob_margin.max(dim=1)                # [B]
            score2 = prob_margin[torch.arange(B), labels]                # [B]
            _, top5_indices = torch.topk(logits_yes, k=5, dim=1, largest=True, sorted=True)
            batch_idx = torch.arange(B).unsqueeze(1).repeat(1,5).view(-1)
            top5_flat = top5_indices.view(-1)
            top5_probs = prob_margin[batch_idx, top5_flat].view(B,5)
            score3 = top5_probs.max(dim=1).values

            all_score1.append(score1.cpu().numpy())
            all_score2.append(score2.cpu().numpy())
            all_score3.append(score3.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_clean_labels.append(clean_labels.cpu().numpy())
            all_is_open.append(is_open.cpu().numpy())
            all_is_clean.append(batch['is_clean'].cpu().numpy())
            all_batch_data.append(data.cpu().numpy())
            all_pseudo_label.append(pseudo_label.cpu().numpy())

    # 合并
    all_score1_np = np.concatenate(all_score1, axis=0)
    all_score2_np = np.concatenate(all_score2, axis=0)
    all_score3_np = np.concatenate(all_score3, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)
    all_clean_labels_np = np.concatenate(all_clean_labels, axis=0)
    all_is_open_np = np.concatenate(all_is_open, axis=0)
    all_is_clean_np = np.concatenate(all_is_clean, axis=0)
    all_data_np = np.concatenate(all_batch_data, axis=0)
    all_pseudo_label_np = np.concatenate(all_pseudo_label, axis=0)

    N_total = len(all_score1_np)
    final_sample_type = np.full(N_total, -1, dtype=int)  # -1 unmarked, 0 ID, 1 ID-clean, 2 OOD

    # Per-class Top-K by score1 -> ID
    unique_labels = np.unique(all_labels_np)
    all_top_k_indices_by_score1 = []
    for label in unique_labels:
        class_indices = np.where(all_labels_np == label)[0]
        if len(class_indices) == 0: continue
        class_score1 = all_score1_np[class_indices]
        k_to_select = min(top_k_per_class, len(class_score1))
        if k_to_select <= 0: continue
        top_k_local = np.argpartition(class_score1, -k_to_select)[-k_to_select:]
        top_k_global = class_indices[top_k_local]
        top_k_global = top_k_global[np.argsort(all_score1_np[top_k_global])[::-1]]
        all_top_k_indices_by_score1.extend(top_k_global)
    all_top_k_indices_by_score1 = np.array(all_top_k_indices_by_score1)
    final_sample_type[all_top_k_indices_by_score1] = 0

    # Per-class Top-K by score2 -> ID-clean (覆盖)
    all_top_k_indices_by_score2 = []
    for label in unique_labels:
        class_indices = np.where(all_labels_np == label)[0]
        if len(class_indices) == 0: continue
        class_score2 = all_score2_np[class_indices]
        k_to_select = min(top_k_per_class_clean, len(class_score2))
        if k_to_select <= 0: continue
        top_k_local = np.argpartition(class_score2, -k_to_select)[-k_to_select:]
        top_k_global = class_indices[top_k_local]
        top_k_global = top_k_global[np.argsort(all_score2_np[top_k_global])[::-1]]
        all_top_k_indices_by_score2.extend(top_k_global)
    all_top_k_indices_by_score2 = np.array(all_top_k_indices_by_score2)
    final_sample_type[all_top_k_indices_by_score2] = 1

    # Bottom-K global by score2 -> OOD (仅未被覆盖的)
    k_to_select_global = min(bottom_k_global, N_total)
    bottom_k_global_indices = np.argpartition(all_score2_np, k_to_select_global)[:k_to_select_global]
    bottom_k_global_indices = bottom_k_global_indices[np.argsort(all_score2_np[bottom_k_global_indices])]
    ood_mask = final_sample_type[bottom_k_global_indices] == -1
    final_sample_type[bottom_k_global_indices[ood_mask]] = 2

    # 统计
    id_count = (final_sample_type == 0).sum()
    clean_count = (final_sample_type == 1).sum()
    ood_count = (final_sample_type == 2).sum()
    unmarked_count = (final_sample_type == -1).sum()

    marked_indices = np.where(final_sample_type != -1)[0]

    analyze_per_class_our_method(
        all_score1_np, all_score2_np, all_labels_np, all_is_open_np, all_is_clean_np,
        top_k_per_class=top_k_per_class, bottom_k_global=bottom_k_global,
    )
    selection_info = {
        'Total': N_total,
        'ID': id_count,
        'ID_clean': clean_count,
        'OOD': ood_count,
        'Unmarked': unmarked_count,
        'final_sample_type': final_sample_type
    }

    caches = {
        'all_data_np': all_data_np,
        'all_labels_np': all_labels_np,
        'all_clean_labels_np': all_clean_labels_np,
        'all_is_open_np': all_is_open_np,
        'all_is_clean_np': all_is_clean_np,
        'all_pseudo_label_np': all_pseudo_label_np,
    }

    return marked_indices, selection_info, caches

# ----------------------------
# 2) compute_iou：计算两个被标记集合的 IoU（top-k overlap）
# ----------------------------
def compute_iou(set_prev, set_curr):
    if set_prev is None:
        return None
    prev = set(set_prev.tolist()) if isinstance(set_prev, np.ndarray) else set(set_prev)
    curr = set(set_curr.tolist()) if isinstance(set_curr, np.ndarray) else set(set_curr)
    if len(prev) == 0 and len(curr) == 0:
        return 1.0
    inter = len(prev & curr)
    union = len(prev | curr)
    return inter / union if union > 0 else 0.0

"""
11-10， 主函数的辅助函数
"""
from .test_clip import test_clip, my_test_nega_clip, my_test_noOOD, my_test_lora, my_test_resnet
def print_results(results, indent=0):
    prefix = '  ' * indent
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{prefix}{key}:")
            print_results(value, indent + 1)
        else:
            print(f"{prefix}{key}: {value}")

def sharpen(p, T=0.5, eps=1e-8):
    """Sharpen probability vector p along class dim (row-wise)."""
    p_pow = p.clamp(min=eps) ** (1.0 / T)
    return p_pow / p_pow.sum(dim=1, keepdim=True).clamp(min=eps)

def mixup_pairs(x1, x2, t1, t2, alpha=0.2, device='cuda'):
    """Per-row beta mixup. Returns mixed (x, t, lam) where t are soft labels."""
    B = x1.size(0)
    lam = torch.distributions.Beta(alpha, alpha).sample((B,)).to(device)
    lam = torch.max(lam, 1 - lam)  # ensure >= 0.5 like you used
    lam_x = lam.view(-1, 1, 1, 1)
    x_mix = lam_x * x1 + (1.0 - lam_x) * x2
    lam_t = lam.view(-1, 1)
    t_mix = lam_t * t1 + (1.0 - lam_t) * t2
    return x_mix, t_mix, lam

@torch.no_grad()
def compute_soft_targets(net, x, y_noisy, all_pos_feat, tau=0.07, T=0.5, eps=1e-8):
    """DivideMix式soft targets: 对clean样本，结合noisy label与模型预测；对ID样本，取sharpen(p)。"""
    img_feat, _ = net.ytVisualWithLocal(x.type(net.dtype))
    img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + eps)
    logits = img_feat @ all_pos_feat.T / tau
    probs = F.softmax(logits, dim=1)

    B, C = probs.size()
    y_onehot = torch.zeros(B, C, device=x.device).scatter_(1, y_noisy.view(-1,1), 1)
    # clean: y = 0.5*y_noisy + 0.5*probs; noisy: sharpen(probs)
    y_soft = 0.5 * y_onehot + 0.5 * probs
    q_soft = sharpen(probs, T=T)
    return y_soft, q_soft

def compute_prompt_logits(img_feats, all_pos_feat, tau=0.07):
    """
    img_feats: [B, D] (normalized)
    all_pos_feat: [C, D] (normalized)
    returns logits [B, C] before softmax
    """
    return torch.matmul(img_feats, all_pos_feat.t()) / tau

def compute_accuracy_on_all(model, caches, device='cuda', n_nega_ctx=1, options=None, batch_size=512):
    """
    在整个训练集（由 caches['all_data_np'] 提供）上计算准确率，但**排除 OOD 样本**（all_is_open_np == 1）。
    返回 (acc_clean_all, acc_noisy_all)：
      - acc_clean_all: 使用 caches['all_clean_labels_np'] 作为真值，仅在 ID 样本上统计
      - acc_noisy_all: 使用 caches['all_labels_np']（训练标签/伪标签）作为真值，仅在 ID 样本上统计

    要求：caches 必须包含 'all_data_np', 'all_clean_labels_np', 'all_labels_np', 'all_is_open_np'。
    """
    model.eval()
    device = device if device is not None else ('cuda' if options and options.get('use_gpu', True) else 'cpu')

    all_data_np = caches.get('all_data_np', None)
    all_clean_labels_np = caches.get('all_clean_labels_np', None)
    all_labels_np = caches.get('all_labels_np', None)
    all_is_open_np = caches.get('all_is_open_np', None)  # 0 for ID, 1 for OOD

    if any(x is None for x in [all_data_np, all_clean_labels_np, all_labels_np, all_is_open_np]):
        raise ValueError("caches must contain 'all_data_np','all_clean_labels_np','all_labels_np','all_is_open_np'")

    # 只保留 ID（in-distribution）样本：all_is_open_np == 0
    id_mask = (np.array(all_is_open_np) == 0)
    if id_mask.sum() == 0:
        return 0.0, 0.0

    data_id = all_data_np[id_mask]
    clean_labels_id = all_clean_labels_np[id_mask]
    noisy_labels_id = all_labels_np[id_mask]

    N = len(data_id)
    bs = options.get('acc_batch_size', batch_size) if options is not None else batch_size

    correct_clean = 0
    correct_noisy = 0
    total = 0

    with torch.no_grad():
        for start in range(0, N, bs):
            end = min(start + bs, N)
            x_batch = torch.from_numpy(data_id[start:end]).float().to(device)
            y_clean = torch.from_numpy(clean_labels_id[start:end]).long().to(device)
            y_noisy = torch.from_numpy(noisy_labels_id[start:end]).long().to(device)

            # 前向（兼容 stage）
            if options is not None and options.get('stage', 1) == 1:
                output, _ = model(x_batch)
            else:
                output, _, _, _, _, _ = model(x_batch)

            # logits 解析与预测：与训练/测试中完全一致
            logits = output.view(-1, int(output.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)
            logits_yes = logits[:,:,0]
            mean_logits_no = logits[:,:,1:].mean(dim=2)
            logits_margin = logits_yes - mean_logits_no
            preds = torch.softmax(logits_margin / 0.1, dim=1).argmax(dim=1)

            correct_clean += (preds == y_clean).sum().item()
            correct_noisy += (preds == y_noisy).sum().item()
            total += preds.size(0)

    acc_clean_all = correct_clean / total if total > 0 else 0.0
    acc_noisy_all = correct_noisy / total if total > 0 else 0.0
    return acc_clean_all, acc_noisy_all

def default_augment_fn(x_batch):
    # x_batch: torch.Tensor [B, C, H, W], device-agnostic
    # simple random horizontal flip per image with p=0.5
    B = x_batch.size(0)
    flipped = x_batch.flip(dims=[3])  # horizontal flip
    mask = (torch.rand(B, device=x_batch.device) < 0.5).float().view(B, 1, 1, 1)
    return mask * flipped + (1.0 - mask) * x_batch

def save_nega_prompt_model(model, save_path):
    """
    保存 NegaPromptCLIP 模型，特别处理关键参数
    """
    state_dict = {}
    for k, v in model.state_dict().items():
        state_dict[k] = v.cpu()  # 移动到CPU

    torch.save({
        'model_state_dict': state_dict,
        # 专门保存关键参数，便于直接访问
        'ctx_positive': model.prompt_learner.ctx_positive.cpu(),
        'ctx_negative': model.prompt_learner.ctx_negative.cpu(),
    }, save_path)


import numpy as np
import torch
import torch.nn.functional as F
#     """
#     修正版：
#     - 不做增强（使用原始样本一次前向计算 p_i/p_j）
#     - 在 torch.no_grad() 下计算 p_i/p_j（目标不带梯度）
#     - 确保 lam_vec 与 t_mix 被 detach
#     - 避免原地操作
#     """
#     device = 'cuda' if options.get('use_gpu', True) else 'cpu'
#     net.train()

#     batch_size = options.get('batch_size', 512)
#     alpha = options.get('mixup_alpha', 0.2)
#     tau = options.get('logit_temperature', 1)

#     all_data_np = caches['all_data_np']
#     all_labels_np = caches['all_labels_np']
#     final_sample_type = caches['final_sample_type']

#     id_clean_mask = (final_sample_type == 1)
#     id_mask = (final_sample_type == 0)
#     non_ood_mask = id_clean_mask | id_mask

#     x_clean_np = all_data_np[id_clean_mask]
#     y_clean_np = all_labels_np[id_clean_mask]
#     x_combined_np = all_data_np[non_ood_mask]
#     y_combined_np = all_labels_np[non_ood_mask]

#     num_clean = len(x_clean_np)
#     num_combined = len(x_combined_np)
#     if num_clean == 0:
#         print(f"Epoch {epoch} 警告：无足够非OOD样本参与训练（clean: {num_clean}, combined: {num_combined}）")
#         return 0.0

#     perm_clean = np.random.permutation(num_clean)
#     x_clean_np = x_clean_np[perm_clean]
#     y_clean_np = y_clean_np[perm_clean]

#     perm_combined = np.random.permutation(num_combined)
#     x_combined_np = x_combined_np[perm_combined]
#     y_combined_np = y_combined_np[perm_combined]

#     total_loss = 0.0
#     total_samples = 0

#     max_iter = max(num_clean, num_combined)
#     for start in range(0, max_iter, batch_size):
#         optimizer.zero_grad()

#         end_clean = min(start + batch_size, num_clean)
#         x_clean_b = torch.from_numpy(x_clean_np[start:end_clean]).float().to(device)
#         y_clean_b = torch.from_numpy(y_clean_np[start:end_clean]).float().to(device)
#         B_clean = x_clean_b.size(0)
#         if B_clean == 0:
#             continue

#         combined_indices = [(i % num_combined) for i in range(start, start + B_clean)]
#         x_partner_b = torch.from_numpy(x_combined_np[combined_indices]).float().to(device)

#         shuffle_idx = torch.randperm(B_clean, device=device)
#         x_partner_shuffled = x_partner_b[shuffle_idx]

#         # ---------- 计算 p_i, p_j（作为目标）: 不做增强、用 no_grad（确保目标不带梯度） ----------
#         with torch.no_grad():
#             logits_i, *_ = net(x_clean_b.type(net.dtype))
#             p_i = F.softmax(logits_i / tau, dim=1)  # [B_clean, C], no grad

#             logits_j, *_ = net(x_partner_b.type(net.dtype))
#             p_j = F.softmax(logits_j / tau, dim=1)  # [B_clean, C], no grad
#         # -------------------------------------------------------------------------------

#         # Mixup：生成混合样本和混合目标
#         x_mix, _, lam_vec = mixup_pairs(
#             x_clean_b, x_partner_shuffled, p_i, p_j[shuffle_idx], alpha=alpha, device=device
#         )
#         # lam_vec 可能形状为 [B] 或 [B,1]，标准化为 [B,1]，并确保在 device 上
#         lam_vec = lam_vec.view(-1, 1).to(device).detach()

#         # 混合目标：确保 t_mix 被 detach（双保险）
#         t_mix = lam_vec * p_i + (1.0 - lam_vec) * p_j[shuffle_idx]
#         t_mix = t_mix.detach()

#         # 前向（混合样本部分必须保留梯度）
#         logits_mix, *_ = net(x_mix.type(net.dtype))
#         log_probs_mix = F.log_softmax(logits_mix / tau, dim=1)

#         logits_clean, *_ = net(x_clean_b.type(net.dtype))
#         ce_loss = F.cross_entropy(logits_clean.float() / tau, y_clean_b.long().to(device))
#         # 监督损失（交叉熵/等价的 KL）
#         sup_loss = -(t_mix * log_probs_mix).sum(dim=1).mean()

#         # 反向传播与更新
#         # loss = sup_loss + 0.5 * ce_loss
#         loss = ce_loss
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item() * B_clean
#         total_samples += B_clean

#         if (start // batch_size) % 100 == 0:
#             print(f"Epoch {epoch} | Iter {start//batch_size} | Batch Size: {B_clean} | 监督损失: {sup_loss.item():.4f}")

#     if scheduler is not None:
#         scheduler.step()

#     avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
#     print(f"Epoch {epoch} 训练结束 | 平均监督损失: {avg_loss:.6f}")
#     return avg_loss


"""
11-11 模型的第一阶段，即pos prompt的预热 (交叉熵)
"""
def train_stage1(net, optimizer, scheduler, trainloader, run, epoch=None,  **options):
    print("start training: stage1")
    import pdb
    from utils import AverageMeter
    losses = AverageMeter()
    loss_all = 0
    n_nega_ctx = options['NEGA_CTX']
    for batch_idx, batch in enumerate(trainloader):
        data = batch['data']
        labels = batch['label']
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()
             
        with torch.set_grad_enabled(True):
            # output = net.get_lora_logits(data)
            output = net(data)
            # output.shape = [batch_size, nclass * 1+n_nega_ctx]
            output_posi = output.view(-1, int(output.shape[1]/(1+n_nega_ctx)), 1+n_nega_ctx)[:, :, 0]
            
            loss_positive = F.cross_entropy(output_posi, labels)
            loss = loss_positive

            net.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    
        losses.update(loss.item(), labels.size(0))
        # print(f"Epoch Loss Averages: L_NPD={loss_NPD_meter.avg:.4f}, L_NND={loss_NND_meter.avg:.4e}, L_ITB={loss_ITB_meter.avg:.4f}")
        # if (batch_idx+1) % options['print_freq'] == 0:
        #     print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
        #           .format(batch_idx+1, len(trainloader), losses.val, losses.avg))
        
        loss_all += losses.avg
    run.log({'loss': loss_all}, step=epoch)

    return loss_all


def train_merge_stage1(net, optimizer, scheduler, trainloader, run, epoch=None, proto=None, **options):
    print("start training: stage1")
    import pdb
    from utils import AverageMeter
    losses = AverageMeter()
    loss_all = 0
    n_nega_ctx = options['NEGA_CTX']

    for batch_idx, batch in enumerate(trainloader):
        data = batch['data']
        labels = batch['label']
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()

        with torch.set_grad_enabled(True):
            output, text_features = net(data)
            # output.shape = [batch_size, nclass * 1+n_nega_ctx]
            output_posi = output.view(-1, int(output.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]

            loss_positive = F.cross_entropy(output_posi, labels)
            loss = loss_positive

            net.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        losses.update(loss.item(), labels.size(0))
        # print(f"Epoch Loss Averages: L_NPD={loss_NPD_meter.avg:.4f}, L_NND={loss_NND_meter.avg:.4e}, L_ITB={loss_ITB_meter.avg:.4f}")
        if (batch_idx + 1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})" \
                  .format(batch_idx + 1, len(trainloader), losses.val, losses.avg))

        loss_all += losses.avg
    run.log({'loss': loss_all}, step=epoch)

    return loss_all


"""
11-11 模型的第二阶段，即neg prompt的初始化 (L_NIS),注意用的是筛选过后的样本做的
"""

def train_stage2(net, optimizer, scheduler, trainloader, run, epoch=None, proto=None, **options):
    n_nega_ctx = options['NEGA_CTX']
    batch_size = 512  # 可调
    total_loss = 0.0
    total_samples = 0

    # --- 1. 推断阶段：收集所有样本的分数和标签 ---
    print(f"Epoch {epoch}: Running inference for sample identification...")
    net.eval()
    all_score1, all_score2, all_score3 = [], [], []
    all_labels, all_data, all_is_clean, all_is_open = [], [], [], []

    with torch.no_grad():
        for batch in trainloader:

            data = batch['data']
            labels = batch['label']
            clean_labels = batch['clean_label']
            is_open = batch['is_open']
            if options['use_gpu']:
                data, labels, clean_labels, is_open = [x.cuda() for x in [data, labels, clean_labels, is_open]]

            # POMP标签变换
            ori_to_modify, modify_to_ori = None, None
            if options['POMP']:
                ori_to_modify, modify_to_ori = label_transform(
                    labels.cpu().numpy(), options['POMP_k'], options['num_classes'] - 1
                )
                labels = torch.tensor([ori_to_modify[l.item()] for l in labels]).cuda()

            # 前向传播
            if options['stage'] == 1:
                output, text_features = net(data, modify_to_ori)
            else:
                output, *rest = net(data, modify_to_ori)  # 简化非核心变量存储

            # 解析logits并计算分数
            logits = output.view(-1, int(output.shape[1]/(1+n_nega_ctx)), 1+n_nega_ctx)
            B, C = logits.shape[0], logits.shape[1]
            logits_yes = logits[:, :, 0]  # [B, C]
            prob_margin = torch.softmax(logits_yes / 0.1, dim=1)  # T=0.1

            # 计算分数
            score1 = prob_margin.max(dim=1).values
            score2 = prob_margin[torch.arange(B), labels]
            _, top5 = torch.topk(logits_yes, 5, dim=1)
            score3 = prob_margin[torch.arange(B).unsqueeze(1).repeat(1,5).view(-1), top5.view(-1)].view(B,5).max(dim=1).values

            # 缓存数据
            [lst.append(arr.cpu().numpy()) for lst, arr in zip(
                [all_score1, all_score2, all_score3, all_labels, all_data, all_is_clean, all_is_open],
                [score1, score2, score3, labels, data, batch['is_clean'], is_open]
            )]

    # 合并数组
    all_score1_np = np.concatenate(all_score1)
    all_score2_np = np.concatenate(all_score2)
    all_labels_np = np.concatenate(all_labels)
    all_data_np = np.concatenate(all_data)
    all_is_clean = np.concatenate(all_is_clean).astype(bool)
    all_is_open = np.concatenate(all_is_open).astype(bool)

    # --- 2. 样本划分 ---
    print(f"Using Score1 & Score2 for identification on {len(all_score1_np)} samples...")
    # 为ID和ID-clean样本分别设置top_k_per_class值
    top_k_per_class_id = 64  # Marked as ID的top_k值
    top_k_per_class_clean = 64  # Marked as ID-clean的top_k值，用户可根据需要修改
    bottom_k_global = 500
    N_total = len(all_score1_np)
    final_sample_type = np.full(N_total, -1, dtype=int)  # -1:未标记, 0:ID, 1:ID-clean, 2:OOD

    # 标记ID样本（Score1 Top-K）
    for label in np.unique(all_labels_np):
        idx = np.where(all_labels_np == label)[0]
        if len(idx) < 1:
            continue
        top_k = min(top_k_per_class_id, len(idx))
        final_sample_type[idx[np.argpartition(all_score1_np[idx], -top_k)[-top_k:]]] = 0

    # 标记ID-clean样本（Score2 Top-K，覆盖ID）
    for label in np.unique(all_labels_np):
        idx = np.where(all_labels_np == label)[0]
        if len(idx) < 1:
            continue
        top_k = min(top_k_per_class_clean, len(idx))
        final_sample_type[idx[np.argpartition(all_score2_np[idx], -top_k)[-top_k:]]] = 1

    # 标记OOD样本（Score2 Bottom-K，未被覆盖的样本）
    bottom_k = np.argpartition(all_score2_np, bottom_k_global)[:bottom_k_global]
    final_sample_type[bottom_k[final_sample_type[bottom_k] == -1]] = 2

    # 统计结果
    id_count = (final_sample_type == 0).sum()
    clean_count = (final_sample_type == 1).sum()
    ood_count = (final_sample_type == 2).sum()
    print(f"--- Sample Summary ---\nTotal: {N_total}, ID: {id_count}, Clean: {clean_count}, OOD: {ood_count}, Unmarked: {N_total - id_count - clean_count - ood_count}")
    print("="*60 + "\n")
    analyze_per_class_our_method(
        run, all_score1_np, all_score2_np, all_labels_np, all_is_open, all_is_clean,
        top_k_per_class_id=top_k_per_class_id,
        top_k_per_class_clean=top_k_per_class_clean,
        bottom_k_global=bottom_k_global,
        epoch=epoch
    )

    # --- 3. 训练ID样本 ---
    print(f"Epoch {epoch}: Training on {id_count + clean_count} marked samples...")
    net.train()
    id_clean_mask = (final_sample_type == 1)
    id_mask = (final_sample_type == 0)

    x_clean_np = all_data_np[id_clean_mask]
    y_clean_np = all_labels_np[id_clean_mask]
    x_id_np = all_data_np[id_mask | id_clean_mask]
    y_id_np = all_labels_np[id_mask]  # 修复原代码中y_id_np引用错误

    num_id = len(x_id_np)
    if num_id == 0:
        print("No ID-clean samples, skipping training.")
        return 0.0

    device = next(net.parameters()).device
    num_classes = options['num_classes']


    # 遍历ID样本批次计算损失
    for start in range(0, num_id, batch_size):
        end = min(start + batch_size, num_id)
        batch_data = torch.from_numpy(x_id_np[start:end]).float().to(device)
        B = batch_data.shape[0]

        loss_nega_to_posi = 0.0
        loss_nega_to_nega = 0.0
        loss_nis = 0.0
        # 前向传播
        logits, text_features, *_ = net(batch_data, modify_to_ori=None)
        logits = logits.view(-1, int(output.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)

        ensemble_feat = text_features.view(int(text_features.shape[0] / (1 + n_nega_ctx)), 1 + n_nega_ctx,
                                                    -1)
        positive_text_features, negative_text_features = ensemble_feat[:, 0:, ], ensemble_feat[:, 1:, :]  # 分离正负特征 [80, 1, 512], [80, 2, 512]

        for i in range(negative_text_features.shape[0]):
            negative_features = negative_text_features[i, :, :].float()
            negative_features_mean = torch.mean(negative_features, dim=0, keepdim=True)
            negative_features_normed = F.normalize(negative_features, dim=-1)
            negative_mean_normed = F.normalize(negative_features_mean, dim=-1)
            cos_sim = negative_features_normed @ negative_mean_normed.t()
            loss_nega_to_nega += torch.mean(cos_sim)
        loss_nega_to_nega = loss_nega_to_nega / negative_text_features.shape[0]

        all_class_dis = 0
        for i in range(negative_text_features.shape[0]):
            positive_feature = positive_text_features[i:i + 1, :].float()
            negative_feature = negative_text_features[i, :, :].float()
            positive_feature_norm = positive_feature / positive_feature.norm(dim=-1, keepdim=True)
            negative_feature_norm = negative_feature / negative_feature.norm(dim=-1, keepdim=True)
            dot_product = positive_feature_norm @ negative_feature_norm.t()
            mean_cosine_dis = (1 - dot_product).mean()
            all_class_dis += mean_cosine_dis
        loss_nega_to_posi += all_class_dis / negative_text_features.shape[0]

        # 计算NIS损失
        logits_no = logits[:, :, 1:]
        logits_no_mean = logits_no.mean(dim=-1)
        loss_nis = -(logits_no_mean.mean(dim=-1) - torch.logsumexp(logits_no_mean, dim=-1)).mean()

        # 总损失与反向传播
        loss = loss_nega_to_posi + 0.1 * loss_nega_to_nega + loss_nis
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * B
        total_samples += B

    if scheduler is not None:
        scheduler.step()
    avg_loss = total_loss / total_samples if total_samples else 0.0
    print(f"ID样本平均损失：{avg_loss:.6f}")
    return avg_loss

"""
11-11,有效果的训练函数,用的prompt-guided 的无监督策略 + 对ID-clean做交叉熵的监督损失
"""
def train_stage3(net, optimizer, scheduler, trainloader, run, epoch=None, proto=None, **options):
    n_nega_ctx = options['NEGA_CTX']
    batch_size = 512  # 可调
    total_loss = 0.0
    total_samples = 0

    # --- 1. 推断阶段：收集所有样本的 score1, score2 和标签 ---
    print(f"Epoch {epoch}: Running inference for sample identification...")
    net.eval()  # 切换到评估模式
    all_score1 = []  # MCM-style
    all_score2 = []  # s+ - s-
    all_score3 = []  # Top 5 (s+ - s-)
    all_labels = []  # 存储POMP变换后的标签
    all_batch_data = []
    all_batch_labels = []
    all_batch_is_clean = []  # 用于评估
    all_batch_is_open = []  # 用于评估

    with torch.no_grad():  # 
        all_top5_acc = []  # 收集ID样本的top5准确率
        all_top1_acc = []  # 新增：收集ID样本的top1准确率
        for batch_idx, batch in enumerate(trainloader):
            data = batch['data']
            labels = batch['label']  # 噪声标签
            clean_labels = batch['clean_label']  # 真实标签（用于准确率计算）
            is_open = batch['is_open']  # 区分ID/OOD
            if options['use_gpu']:
                # 所有张量统一移到GPU
                data, labels, clean_labels, is_open = data.cuda(), labels.cuda(), clean_labels.cuda(), is_open.cuda()

            # POMP标签变换（保持原逻辑）

            ori_to_modify, modify_to_ori = None, None

            # 前向传播与logits解析（保持原逻辑）
            output, text_features = net(data, modify_to_ori)

            logits = output.view(-1, int(output.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)
            B, C = logits.shape[0], logits.shape[1]
            logits_no, logits_yes = logits[:, :, 1:], logits[:, :, 0]  # logits_yes: [B, C]

            # 计算logits_margin和prob_margin（保持原逻辑）
            mean_logits_no = logits_no.mean(dim=2)  # [B, C]

            """
            11-24, 只用正提示看看。
            """
            # logits_margin = logits_yes - mean_logits_no  # [B, C]
            logits_margin = logits_yes
            T = 1
            prob_margin = torch.softmax(logits_margin / T, dim=1)  # [B, C]：所有类别的概率

            # 计算score1：prob_margin的最大值（保持原逻辑）
            score1 = prob_margin.max(dim=1).values

            # score2：基于噪声标签的概率（保持原逻辑）
            score2 = prob_margin[torch.arange(B), labels]  # [B]

            # 计算score3：logits_yes前5类中prob_margin的最大值（保持原逻辑）
            _, top5_indices = torch.topk(logits_yes, k=5, dim=1, largest=True, sorted=True)
            batch_indices = torch.arange(B).unsqueeze(1).repeat(1, 5).view(-1)
            top5_flat_indices = top5_indices.view(-1)
            top5_probs = prob_margin[batch_indices, top5_flat_indices].view(B, 5)
            score3 = top5_probs.max(dim=1).values  # [B]

            # 优化：直接缓存torch张量，避免numpy转换开销
            all_score1.append(score1.cpu())
            all_score2.append(score2.cpu())
            all_score3.append(score3.cpu())
            all_labels.append(labels.cpu())
            all_batch_data.append(data.cpu())
            all_batch_labels.append(labels.cpu())
            all_batch_is_clean.append(batch['is_clean'].cpu().bool())
            all_batch_is_open.append(is_open.cpu().bool())

    # 合并所有批次的分数和标签（使用torch.cat替代numpy.concatenate）
    all_score1_np = torch.cat(all_score1, dim=0).numpy()
    all_score2_np = torch.cat(all_score2, dim=0).numpy()
    all_score3_np = torch.cat(all_score3, dim=0).numpy()
    all_labels_np = torch.cat(all_labels, dim=0).numpy()
    all_batch_is_clean = torch.cat(all_batch_is_clean, dim=0).numpy()
    all_batch_is_open = torch.cat(all_batch_is_open, dim=0).numpy()
    all_data = torch.cat(all_batch_data, dim=0)  # 保持为torch张量，避免后续numpy转换

    # --- 2. 样本划分 ---
    print(f"Using Score1 & Score2 for sample identification on {len(all_score1_np)} samples...")
    # 为ID和ID-clean样本分别设置top_k_per_class值
    top_k_per_class_id = 64  # Marked as ID的top_k值
    top_k_per_class_clean = 64  # Marked as ID-clean的top_k值，用户可根据需要修改
    bottom_k_global = 1000  # 全局最低分样本数

    # 初始化标签掩码
    N_total = len(all_score1_np)
    final_sample_type = np.full(N_total, -1, dtype=int)  # -1:未标记, 0:ID, 1:ID-clean, 2:OOD

    # --- 1. 标记ID样本（Score1 Top-K）---
    unique_labels = np.unique(all_labels_np)
    all_top_k_indices_by_score1 = []
    for label in unique_labels:
        class_indices = np.where(all_labels_np == label)[0]
        if len(class_indices) == 0:
            continue
        class_score1 = all_score1_np[class_indices]
        k_to_select = min(top_k_per_class_id, len(class_score1))
        if k_to_select <= 0:
            continue
        # 取Top-K索引
        top_k_local_by_score1 = np.argpartition(class_score1, -k_to_select)[-k_to_select:]
        top_k_global_by_score1 = class_indices[top_k_local_by_score1]
        top_k_global_by_score1 = top_k_global_by_score1[np.argsort(all_score1_np[top_k_global_by_score1])[::-1]]
        all_top_k_indices_by_score1.extend(top_k_global_by_score1)
    all_top_k_indices_by_score1 = np.array(all_top_k_indices_by_score1)
    final_sample_type[all_top_k_indices_by_score1] = 0  # 标记为ID

    # --- 2. 标记ID-clean样本（Score2 Top-K，覆盖ID）---
    all_top_k_indices_by_score2 = []
    for label in unique_labels:
        class_indices = np.where(all_labels_np == label)[0]
        if len(class_indices) == 0:
            continue
        class_score2 = all_score2_np[class_indices]
        k_to_select = min(top_k_per_class_clean, len(class_score2))
        if k_to_select <= 0:
            continue
        top_k_local_by_score2 = np.argpartition(class_score2, -k_to_select)[-k_to_select:]
        top_k_global_by_score2 = class_indices[top_k_local_by_score2]
        top_k_global_by_score2 = top_k_global_by_score2[np.argsort(all_score2_np[top_k_global_by_score2])[::-1]]
        all_top_k_indices_by_score2.extend(top_k_global_by_score2)
    all_top_k_indices_by_score2 = np.array(all_top_k_indices_by_score2)
    final_sample_type[all_top_k_indices_by_score2] = 1  # 覆盖为ID-clean

    # --- 3. 标记OOD样本（Score2 Bottom-K，仅标记未被ID/ID-clean覆盖的样本）---
    k_to_select_global = min(bottom_k_global, N_total)
    bottom_k_global_indices = np.argpartition(all_score2_np, k_to_select_global)[:k_to_select_global]
    bottom_k_global_indices = bottom_k_global_indices[np.argsort(all_score2_np[bottom_k_global_indices])]
    # 修复：仅标记未被ID/ID-clean覆盖的样本（避免类型冲突）
    ood_mask = final_sample_type[bottom_k_global_indices] == -1
    final_sample_type[bottom_k_global_indices[ood_mask]] = 2  # 标记为OOD

    # --- 4. 统计结果 ---
    id_count = (final_sample_type == 0).sum()
    clean_count = (final_sample_type == 1).sum()
    ood_count = (final_sample_type == 2).sum()
    unmarked_count = (final_sample_type == -1).sum()
    print("--- Final Sample Identification Summary ---")
    print(f"Total Samples: {N_total}")
    print(f"ID samples (from Score1 Top-K): {id_count}")
    print(f"ID-Clean samples (from Score2 Top-K, overrides ID): {clean_count}")
    print(f"OOD samples (from Score2 Bottom-K): {ood_count}")
    print(f"Unmarked samples: {unmarked_count}")
    print("\n" + "=" * 60 + "\n")
    analyze_per_class_our_method(
        run, all_score1_np, all_score2_np, all_labels_np, all_batch_is_open, all_batch_is_clean,
        top_k_per_class_id=top_k_per_class_id,
        top_k_per_class_clean=top_k_per_class_clean,
        bottom_k_global=bottom_k_global,
        epoch=epoch
    )

    # --- 5. 训练准备 ---
    marked_indices = np.where(final_sample_type != -1)[0]
    marked_labels = final_sample_type[marked_indices]

    # --- 5. 训练准备（修复DataLoader问题，保留分batch加载GPU）---
    print(f"Epoch {epoch}: Running training on {len(marked_indices)} marked samples with adaptive losses....")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    net.train()  # 切换回训练模式
    id_clean_mask = (final_sample_type == 1)
    id_mask = (final_sample_type == 0)
    non_ood_mask = id_clean_mask | id_mask  # 非OOD样本掩码（ID-clean + ID）

    # 1. 在CPU上完成所有数据处理，避免一次性占用大量GPU显存
    all_data_cpu = all_data.float().cpu()  # 保持在CPU
    all_labels_cpu = torch.from_numpy(all_labels_np).long().cpu()  # 保持在CPU

    # 2. 在CPU上进行样本筛选（torch掩码，避免numpy转换）
    id_clean_mask_tensor = torch.tensor(id_clean_mask, dtype=torch.bool, device='cpu')
    non_ood_mask_tensor = torch.tensor(non_ood_mask, dtype=torch.bool, device='cpu')

    x_clean_cpu = all_data_cpu[id_clean_mask_tensor]
    y_clean_cpu = all_labels_cpu[id_clean_mask_tensor]
    x_combined_cpu = all_data_cpu[non_ood_mask_tensor]
    y_combined_cpu = all_labels_cpu[non_ood_mask_tensor]

    # 释放CPU内存（关键：删除大张量）
    del all_labels_np, all_data, all_data_cpu, all_labels_cpu, id_clean_mask_tensor, non_ood_mask_tensor

    num_clean = len(x_clean_cpu)
    num_combined = len(x_combined_cpu)

    if num_clean == 0:
        print("No ID-clean samples, skipping DivideMix training.")
        return 0.0

    # 3. CPU上打乱样本（保持原逻辑，避免DataLoader重复shuffle）
    perm_clean = torch.randperm(num_clean, device='cpu')
    x_clean_cpu = x_clean_cpu[perm_clean]
    y_clean_cpu = y_clean_cpu[perm_clean]

    perm_combined = torch.randperm(num_combined, device='cpu')
    x_combined_cpu = x_combined_cpu[perm_combined]
    y_combined_cpu = y_combined_cpu[perm_combined]

    # 训练参数
    num_classes = options['num_classes']
    tau = options.get('logit_temperature', 1)
    alpha = options.get('mixup_alpha', 0.2)
    ce_weight = options.get('ce_weight', 1.0)
    sup_weight = options.get('sup_weight', 0.5)  # 启用监督损失权重

    # === Step 2: 分batch训练（修复迭代逻辑，分batch加载GPU）===
    print(f"start iter loop with batch size {batch_size}! ")
    max_iter = max(num_clean, num_combined)

    for iter_idx in range(0, max_iter, batch_size):  # 按batch_size步长迭代
        optimizer.zero_grad()

        # --- 取当前块的干净样本（CPU上切片，仅当前batch）---
        start_clean = iter_idx
        end_clean = min(iter_idx + batch_size, num_clean)
        x_clean_b = x_clean_cpu[start_clean:end_clean]  # CPU张量
        y_clean_b = y_clean_cpu[start_clean:end_clean]  # CPU张量
        B_clean = x_clean_b.size(0)
        if B_clean == 0:
            continue

        # --- 获取混合样本对（CPU上切片，仅当前batch）---
        combined_start = iter_idx
        combined_end = iter_idx + B_clean
        # 循环索引，避免越界（保持原配对逻辑）
        combined_indices = torch.arange(combined_start, combined_end) % num_combined
        x_partner_b = x_combined_cpu[combined_indices]  # CPU张量

        # --- 仅将当前batch移到GPU（核心优化）---
        x_clean_b = x_clean_b.to(device, non_blocking=True)
        y_clean_b = y_clean_b.to(device, non_blocking=True)
        x_partner_b = x_partner_b.to(device, non_blocking=True)

        # 打乱混合样本（GPU上执行）
        shuffle_idx = torch.randperm(B_clean, device=device)
        x_partner_shuffled = x_partner_b[shuffle_idx]

        # --- 计算相似度向量 p_i/p_j ---
        with torch.no_grad():
            logits_i, _ = net(x_clean_b, modify_to_ori)
            logits_j, _ = net(x_partner_b, modify_to_ori)

            # 解析logits（复用逻辑）
            logits_i = logits_i.view(-1, int(logits_i.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]
            logits_j = logits_j.view(-1, int(logits_j.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]
            p_i = F.softmax(logits_i / tau, dim=1)
            p_j = F.softmax(logits_j / tau, dim=1)

        # --- Mixup操作 ---
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        lam = max(lam, 1 - lam)
        x_mix = lam * x_clean_b + (1 - lam) * x_partner_shuffled

        # 混合目标分布
        # 选择是否detach t_mix，KL散度版本不需要detach
        if 'loss_type' in options and options['loss_type'] == 'kl_div':
            # KL散度版本，不detach t_mix
            t_mix = lam * p_i + (1 - lam) * p_j[shuffle_idx]
        else:
            # 默认交叉熵版本，detach t_mix
            t_mix = (lam * p_i + (1 - lam) * p_j[shuffle_idx]).detach()

        # --- 计算损失函数 ---
        # 1. 干净样本CE损失
        logits_clean, _ = net(x_clean_b, modify_to_ori)

        logits_clean = logits_clean.view(-1, int(logits_clean.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]
        ce_loss = F.cross_entropy(logits_clean / tau, y_clean_b)

        # 2. 混合样本监督损失
        logits_mix, _ = net(x_mix, modify_to_ori)

        logits_mix = logits_mix.view(-1, int(logits_mix.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]

        # 根据损失类型选择损失函数
        if 'loss_type' in options and options['loss_type'] == 'kl_div':
            # KL散度版本
            log_probs_mix = F.log_softmax(logits_mix / tau, dim=1)
            sup_loss = F.kl_div(log_probs_mix, t_mix, reduction='batchmean')
        else:
            # 默认交叉熵版本
            log_probs_mix = F.log_softmax(logits_mix / tau, dim=1)
            sup_loss = -(t_mix * log_probs_mix).sum(dim=1).mean()

        # 总损失
        # loss = ce_weight * ce_loss + sup_weight * sup_loss
        loss = ce_weight * ce_loss + sup_weight * sup_loss

        # 反向传播
        loss.backward()
        optimizer.step()

        # 累计损失
        total_loss += loss.item() * B_clean
        total_samples += B_clean

        # 打印批次信息（修复：用iter_idx替代start）
        if (iter_idx // batch_size) % 15 == 0:
            print(f"Epoch {epoch} | Iter {iter_idx // batch_size} | Batch Size: {B_clean} | "
                  f"CE损失: {ce_loss.item():.4f} | 监督损失: {sup_loss.item():.4f} | 总损失: {loss.item():.4f}")

        # 释放当前batch的GPU显存（避免碎片化）
        del x_clean_b, y_clean_b, x_partner_b, x_partner_shuffled, x_mix
        del logits_i, logits_j, logits_clean, logits_mix, t_mix
        torch.cuda.empty_cache() if options['use_gpu'] else None

    # 调度器在epoch结束后调用
    if scheduler is not None:
        scheduler.step()

    # 释放CPU内存
    del x_clean_cpu, y_clean_cpu, x_combined_cpu, y_combined_cpu

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    print(
        f"Epoch {epoch}: DivideMix-style avg loss = {avg_loss:.6f} (CE: {ce_loss.item():.6f}, Sup: {sup_loss.item():.6f})")
    return avg_loss


"""
train_prompt_branch: train_stage3的扩展版本，支持外部传入sample_indices，并返回样本划分结果
参数说明:
- sample_indices: 如果为None，使用函数内部的样本划分逻辑；否则，使用传入的样本索引作为ID-clean样本
- 返回值: (avg_loss, split_dict)，其中split_dict格式为{'id_clean': [...], 'id': [...], 'noisy_ood': [...]}
"""
def train_prompt_branch(net, optimizer, scheduler, trainloader, run, cfg, epoch=None, sample_indices=None):
    n_nega_ctx = cfg.get('NEGA_CTX', 2)
    batch_size = cfg.get('batch_size', 256)  # 可调
    total_loss = 0.0
    total_samples = 0

    # --- 1. 推断阶段：收集所有样本的 score1, score2 和标签 ---
    print(f"prompt branch   Epoch {epoch}: Running inference for sample identification...")
    net.eval()  # 切换到评估模式
    all_score1 = []  # MCM-style
    all_score2 = []  # s+ - s-
    all_score3 = []  # Top 5 (s+ - s-)
    all_labels = []  # 存储POMP变换后的标签
    all_batch_data = []
    all_batch_labels = []
    all_batch_is_clean = []  # 用于评估
    all_batch_is_open = []  # 用于评估
    all_original_indices = []  # 新增：存储样本在原始数据集中的唯一索引

    # id_clean = sample_indices['id_clean']
    # id_clean_tensor = torch.tensor(list(id_clean), dtype=torch.long)  # CPU
    # a = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(trainloader):
            data = batch['data']
            labels = batch['label']  # 噪声标签
            clean_labels = batch['clean_label']  # 真实标签（用于准确率计算）
            is_open = batch['is_open']  # 区分ID/OOD
            is_clean = batch['is_clean']
            # 获取样本原始索引（来自dataset.__getitem__返回的'index'字段）
            original_indices = batch['index']  # 样本在原始数据集中的唯一索引
            # mask = torch.isin(original_indices, id_clean_tensor)  # bool tensor [B]
            # a += is_clean[mask].sum().item()


            # 所有张量统一移到GPU
            data, labels, clean_labels, is_open = data.cuda(), labels.cuda(), clean_labels.cuda(), is_open.cuda()

            # 前向传播与logits解析（保持原逻辑）
            output = net.get_original_logits(data)

            logits = output.view(-1, int(output.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)
            B, C = logits.shape[0], logits.shape[1]
            logits_no, logits_yes = logits[:, :, 1:], logits[:, :, 0]  # logits_yes: [B, C]

            # 计算logits_margin和prob_margin（保持原逻辑）
            # mean_logits_no = logits_no.mean(dim=2)  # [B, C]
            # logits_margin = logits_yes - mean_logits_no  # [B, C]

            logits_margin = logits_yes
            T = 1
            prob_margin = torch.softmax(logits_margin / T, dim=1)  # [B, C]：所有类别的概率


            # vals, idxs = prob_margin.topk(2, dim=1)  # vals: [B,2], idxs: [B,2]
            # # 如果第一大类就是 noisy label，则取第二大，否则取第一大
            # first_is_label = (idxs[:, 0] == labels)
            # score1 = torch.where(first_is_label, vals[:, 1], vals[:, 0])  # [B]

            ## 计算score1
            score1 = prob_margin.max(dim=1).values

            # score2：基于噪声标签的概率（保持原逻辑）
            score2 = prob_margin[torch.arange(B), labels]  # [B]

            # 计算score3：logits_yes前5类中prob_margin的最大值（保持原逻辑）
            _, top5_indices = torch.topk(logits_yes, k=5, dim=1, largest=True, sorted=True)
            batch_indices = torch.arange(B).unsqueeze(1).repeat(1, 5).view(-1)
            top5_flat_indices = top5_indices.view(-1)
            top5_probs = prob_margin[batch_indices, top5_flat_indices].view(B, 5)
            score3 = top5_probs.max(dim=1).values  # [B]

            # 优化：直接缓存torch张量，避免numpy转换开销
            all_score1.append(score1.cpu())
            all_score2.append(score2.cpu())
            all_score3.append(score3.cpu())
            all_labels.append(labels.cpu())
            all_batch_data.append(data.cpu())
            all_batch_labels.append(labels.cpu())
            all_batch_is_clean.append(batch['is_clean'].cpu().bool())
            all_batch_is_open.append(is_open.cpu().bool())
            all_original_indices.append(original_indices.cpu())  # 缓存样本原始索引
    # 合并所有批次的分数和标签（使用torch.cat替代numpy.concatenate）
    all_score1_np = torch.cat(all_score1, dim=0).numpy()
    all_score2_np = torch.cat(all_score2, dim=0).numpy()
    all_score3_np = torch.cat(all_score3, dim=0).numpy()
    all_labels_np = torch.cat(all_labels, dim=0).numpy()
    all_batch_is_clean = torch.cat(all_batch_is_clean, dim=0).numpy()
    all_batch_is_open = torch.cat(all_batch_is_open, dim=0).numpy()
    all_data = torch.cat(all_batch_data, dim=0)  # 保持为torch张量，避免后续numpy转换

    # --- 2. 样本划分 ---
    print(f"Using Score1 & Score2 for sample identification on {len(all_score1_np)} samples...")
    # 为ID和ID-clean样本分别设置top_k_per_class值
    top_k_per_class_id = 64  # Marked as ID的top_k值
    top_k_per_class_clean = 64  # Marked as ID-clean的top_k值，用户可根据需要修改
    bottom_k_global = 1000  # 全局最低分样本数

    # 初始化标签掩码
    N_total = len(all_score1_np)
    final_sample_type = np.full(N_total, -1, dtype=int)  # -1:未标记, 0:ID, 1:ID-clean, 2:OOD

    # 检查是否使用外部传入的sample_indices
    if sample_indices is None:
        # --- 2.1 使用内部的样本划分逻辑 ---

        # 标记ID样本（Score1 Top-K）
        unique_labels = np.unique(all_labels_np)
        all_top_k_indices_by_score1 = []
        for label in unique_labels:
            class_indices = np.where(all_labels_np == label)[0]
            if len(class_indices) == 0:
                continue
            class_score1 = all_score1_np[class_indices]
            k_to_select = min(top_k_per_class_id, len(class_score1))
            if k_to_select <= 0:
                continue
            # 取Top-K索引
            top_k_local_by_score1 = np.argpartition(class_score1, -k_to_select)[-k_to_select:]
            top_k_global_by_score1 = class_indices[top_k_local_by_score1]
            top_k_global_by_score1 = top_k_global_by_score1[np.argsort(all_score1_np[top_k_global_by_score1])[::-1]]
            all_top_k_indices_by_score1.extend(top_k_global_by_score1)
        all_top_k_indices_by_score1 = np.array(all_top_k_indices_by_score1)
        final_sample_type[all_top_k_indices_by_score1] = 0  # 标记为ID

        # 标记ID-clean样本（Score2 Top-K，覆盖ID）
        all_top_k_indices_by_score2 = []
        for label in unique_labels:
            class_indices = np.where(all_labels_np == label)[0]
            if len(class_indices) == 0:
                continue
            class_score2 = all_score2_np[class_indices]
            k_to_select = min(top_k_per_class_clean, len(class_score2))
            if k_to_select <= 0:
                continue
            top_k_local_by_score2 = np.argpartition(class_score2, -k_to_select)[-k_to_select:]
            top_k_global_by_score2 = class_indices[top_k_local_by_score2]
            top_k_global_by_score2 = top_k_global_by_score2[np.argsort(all_score2_np[top_k_global_by_score2])[::-1]]
            all_top_k_indices_by_score2.extend(top_k_global_by_score2)
        all_top_k_indices_by_score2 = np.array(all_top_k_indices_by_score2)
        final_sample_type[all_top_k_indices_by_score2] = 1  # 覆盖为ID-clean

    else:
        # --- 2.2 使用外部传入的sample_indices ---
        # 如果sample_indices是字典，使用train_lora_branch返回的完整划分结果
        if isinstance(sample_indices, dict):
            print(f"Using external sample split:")
            print(f"  - ID samples: {len(sample_indices['id'])}")
            print(f"  - ID-clean samples: {len(sample_indices['id_clean'])}")
            print(f"  - OOD samples: {len(sample_indices['noisy_ood'])}")

            # 将外部传入的全局索引映射到当前epoch加载的样本位置
            all_original_indices_np = torch.cat(all_original_indices, dim=0).numpy()  # 合并所有样本的原始全局索引
            # 创建全局索引到当前位置的映射字典
            global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(all_original_indices_np)}

            # 映射ID样本索引
            all_top_k_indices_by_score1 = np.array([global_to_local[global_idx] for global_idx in sample_indices['id'] if global_idx in global_to_local])
            # 映射ID-clean样本索引
            all_top_k_indices_by_score2 = np.array([global_to_local[global_idx] for global_idx in sample_indices['id_clean'] if global_idx in global_to_local])
            print(f"映射后ID样本数量: {len(all_top_k_indices_by_score1)}")
            print(f"映射后ID-clean样本数量: {len(all_top_k_indices_by_score2)}")

            # 标记样本类型
            final_sample_type[all_top_k_indices_by_score1] = 0  # 标记为ID
            final_sample_type[all_top_k_indices_by_score2] = 1  # 标记为ID-clean

            # 实时验证：打印样本类型标记统计（可选）
            print(f"ID 样本标记后数量: {np.sum(final_sample_type == 0)}")
            print(f"ID-clean 样本标记后数量: {np.sum(final_sample_type == 1)}")
            print(f"标记率: {(np.sum(final_sample_type != -1) / len(final_sample_type)) * 100:.2f}%")

            # 精确验证：标记为ID-clean样本的真实类型比例
            print("\n=== 开始精确验证样本类型比例 ===")
            # 获取所有被标记为ID-clean的样本索引
            id_clean_marked_indices = np.where(final_sample_type == 1)[0]
            # 使用已收集的 all_is_open 和 all_is_clean 直接计算真实类型
            real_types = []
            for idx in id_clean_marked_indices:
                if idx < len(all_batch_is_open) and idx < len(all_batch_is_clean):
                    is_open = all_batch_is_open[idx]
                    is_clean = all_batch_is_clean[idx]
                    # 根据is_open和is_clean确定真实样本类型
                    if is_open:
                        real_types.append(2)  # OOD样本
                    else:
                        real_types.append(0 if is_clean else 1)  # ID-clean或ID-noisy样本
                else:
                    real_types.append(-1)  # 无效类型

            # 统计各真实类型的数量
            real_types_of_marked = np.array(real_types)
            count_id_clean = np.sum(real_types_of_marked == 0)
            count_id_noisy = np.sum(real_types_of_marked == 1)
            count_ood = np.sum(real_types_of_marked == 2)

            # 计算比例
            total_valid_marked = len(real_types_of_marked) - np.sum(real_types_of_marked == -1)
            if total_valid_marked > 0:
                print(f"=== 标记为ID-clean样本的真实类型比例 ===")
                print(f"真实ID-clean比例: {(count_id_clean / total_valid_marked) * 100:.2f}%")
                print(f"真实ID-noisy比例: {(count_id_noisy / total_valid_marked) * 100:.2f}%")
                print(f"真实OOD比例: {(count_ood / total_valid_marked) * 100:.2f}%")
                print(f"总计: {total_valid_marked}个有效样本")

            # 同时验证标记为ID的样本的真实类型比例
            id_marked_indices = np.where(final_sample_type == 0)[0]
            real_types_id = []
            for idx in id_marked_indices:
                if idx < len(all_batch_is_open) and idx < len(all_batch_is_clean):
                    is_open = all_batch_is_open[idx]
                    is_clean = all_batch_is_clean[idx]
                    # 根据is_open和is_clean确定真实样本类型
                    if is_open:
                        real_types_id.append(2)  # OOD样本
                    else:
                        real_types_id.append(0 if is_clean else 1)  # ID-clean或ID-noisy样本
                else:
                    real_types_id.append(-1)  # 无效类型

            # 统计各真实类型的数量
            real_types_of_id_marked = np.array(real_types_id)
            count_id_marked_clean = np.sum(real_types_of_id_marked == 0)
            count_id_marked_noisy = np.sum(real_types_of_id_marked == 1)
            count_id_marked_ood = np.sum(real_types_of_id_marked == 2)

            # 计算比例
            total_id_valid_marked = len(real_types_of_id_marked) - np.sum(real_types_of_id_marked == -1)
            if total_id_valid_marked > 0:
                print(f"\n=== 标记为ID样本的真实类型比例 ===")
                print(f"真实ID-clean比例: {(count_id_marked_clean / total_id_valid_marked) * 100:.2f}%")
                print(f"真实ID-noisy比例: {(count_id_marked_noisy / total_id_valid_marked) * 100:.2f}%")
                print(f"真实OOD比例: {(count_id_marked_ood / total_id_valid_marked) * 100:.2f}%")
                print(f"总计: {total_id_valid_marked}个有效样本")
            print("=== 样本类型比例验证结束 ===")

        # 兼容旧版：如果sample_indices是列表，仅作为ID-clean样本
        else:
            print("传入的样本划分有问题")
            exit(0)
            # print(f"Using external sample indices ({len(sample_indices)} samples) as ID-clean...")
            # final_sample_type[sample_indices] = 1  # 标记为ID-clean
            #
            # # 对于ID样本，仍使用Score1 Top-K的划分
            # unique_labels = np.unique(all_labels_np)
            # all_top_k_indices_by_score1 = []
            # for label in unique_labels:
            #     class_indices = np.where(all_labels_np == label)[0]
            #     if len(class_indices) == 0:
            #         continue
            #     class_score1 = all_score1_np[class_indices]
            #     k_to_select = min(top_k_per_class_id, len(class_score1))
            #     if k_to_select <= 0:
            #         continue
            #     # 取Top-K索引
            #     top_k_local_by_score1 = np.argpartition(class_score1, -k_to_select)[-k_to_select:]
            #     top_k_global_by_score1 = class_indices[top_k_local_by_score1]
            #     top_k_global_by_score1 = top_k_global_by_score1[np.argsort(all_score1_np[top_k_global_by_score1])[::-1]]
            #     all_top_k_indices_by_score1.extend(top_k_global_by_score1)
            # all_top_k_indices_by_score1 = np.array(all_top_k_indices_by_score1)
            # # 将不是ID-clean的ID样本标记为ID
            # for idx in all_top_k_indices_by_score1:
            #     if final_sample_type[idx] != 1:  # 不覆盖ID-clean
            #         final_sample_type[idx] = 0  # 标记为ID

    # --- 2.3 标记OOD样本（无论是否使用外部sample_indices，都执行此步骤）---
    k_to_select_global = min(bottom_k_global, N_total)
    bottom_k_global_indices = np.argpartition(all_score2_np, k_to_select_global)[:k_to_select_global]
    bottom_k_global_indices = bottom_k_global_indices[np.argsort(all_score2_np[bottom_k_global_indices])]
    # 修复：仅标记未被ID/ID-clean覆盖的样本（避免类型冲突）
    ood_mask = final_sample_type[bottom_k_global_indices] == -1
    final_sample_type[bottom_k_global_indices[ood_mask]] = 2  # 标记为OOD

    # --- 4. 统计结果 ---
    id_count = (final_sample_type == 0).sum()
    clean_count = (final_sample_type == 1).sum()
    ood_count = (final_sample_type == 2).sum()
    unmarked_count = (final_sample_type == -1).sum()
    print("--- Final Sample Identification Summary ---")
    print(f"Total Samples: {N_total}")
    print(f"ID samples (from Score1 Top-K): {id_count}")
    print(f"ID-Clean samples: {clean_count}")
    print(f"OOD samples (from Score2 Bottom-K): {ood_count}")
    print(f"Unmarked samples: {unmarked_count}")
    print("\n" + "=" * 60 + "\n")

    # 分析样本划分情况（保持原逻辑）
    analyze_per_class_our_method(
        run, all_score1_np, all_score2_np, all_labels_np, all_batch_is_open, all_batch_is_clean,
        top_k_per_class_id=top_k_per_class_id,
        top_k_per_class_clean=top_k_per_class_clean,
        bottom_k_global=bottom_k_global,
        epoch=epoch
    )

    # --- 5. 训练准备 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    net.train()  # 切换回训练模式
    id_clean_mask = (final_sample_type == 1)
    id_mask = (final_sample_type == 0)
    non_ood_mask = id_clean_mask | id_mask  # 非OOD样本掩码（ID-clean + ID）

    # 1. 在CPU上完成所有数据处理，避免一次性占用大量GPU显存
    all_data_cpu = all_data.float().cpu()  # 保持在CPU
    all_labels_cpu = torch.from_numpy(all_labels_np).long().cpu()  # 保持在CPU

    # 2. 在CPU上进行样本筛选（torch掩码，避免numpy转换）
    id_clean_mask_tensor = torch.tensor(id_clean_mask, dtype=torch.bool, device='cpu')
    non_ood_mask_tensor = torch.tensor(non_ood_mask, dtype=torch.bool, device='cpu')

    x_clean_cpu = all_data_cpu[id_clean_mask_tensor]
    y_clean_cpu = all_labels_cpu[id_clean_mask_tensor]
    x_combined_cpu = all_data_cpu[non_ood_mask_tensor]
    y_combined_cpu = all_labels_cpu[non_ood_mask_tensor]

    # 释放CPU内存（关键：删除大张量）
    del all_labels_np, all_data, all_data_cpu, all_labels_cpu, id_clean_mask_tensor, non_ood_mask_tensor

    num_clean = len(x_clean_cpu)
    num_combined = len(x_combined_cpu)

    if num_clean == 0:
        print("No ID-clean samples, skipping training.")
        # 返回空的划分结果
        return 0.0, {'id_clean': [], 'id': [], 'noisy_ood': []}

    # 3. CPU上打乱样本（保持原逻辑，避免DataLoader重复shuffle）
    perm_clean = torch.randperm(num_clean, device='cpu')
    x_clean_cpu = x_clean_cpu[perm_clean]
    y_clean_cpu = y_clean_cpu[perm_clean]

    perm_combined = torch.randperm(num_combined, device='cpu')
    x_combined_cpu = x_combined_cpu[perm_combined]
    y_combined_cpu = y_combined_cpu[perm_combined]

    # 训练参数
    tau = 1
    alpha = cfg.get('mixup_alpha', 0.2)
    ce_weight = cfg.get('ce_weight', 1.0)
    sup_weight = cfg.get('sup_weight', 0.5)  # 启用监督损失权重

    # === Step 2: 分batch训练（修复迭代逻辑，分batch加载GPU）===
    print(f"start iter loop with batch size {batch_size}! ")
    max_iter = max(num_clean, num_combined)

    for iter_idx in range(0, max_iter, batch_size):  # 按batch_size步长迭代
        optimizer.zero_grad()

        # --- 取当前块的干净样本（CPU上切片，仅当前batch）---
        start_clean = iter_idx
        end_clean = min(iter_idx + batch_size, num_clean)
        x_clean_b = x_clean_cpu[start_clean:end_clean]  # CPU张量
        y_clean_b = y_clean_cpu[start_clean:end_clean]  # CPU张量
        B_clean = x_clean_b.size(0)
        if B_clean == 0:
            continue

        # --- 获取混合样本对（CPU上切片，仅当前batch）---
        combined_start = iter_idx
        combined_end = iter_idx + B_clean
        # 循环索引，避免越界（保持原配对逻辑）
        combined_indices = torch.arange(combined_start, combined_end) % num_combined
        x_partner_b = x_combined_cpu[combined_indices]  # CPU张量

        # --- 仅将当前batch移到GPU（核心优化）---
        x_clean_b = x_clean_b.to(device, non_blocking=True)
        y_clean_b = y_clean_b.to(device, non_blocking=True)
        x_partner_b = x_partner_b.to(device, non_blocking=True)

        # 打乱混合样本（GPU上执行）
        shuffle_idx = torch.randperm(B_clean, device=device)
        x_partner_shuffled = x_partner_b[shuffle_idx]

        # --- 计算相似度向量 p_i/p_j ---
        with torch.no_grad():

            logits_i = net.get_original_logits(x_clean_b)
            logits_j = net.get_original_logits(x_partner_b)

            # 解析logits（复用逻辑）
            logits_i = logits_i.view(-1, int(logits_i.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]
            logits_j = logits_j.view(-1, int(logits_j.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]
            p_i = F.softmax(logits_i / tau, dim=1)
            p_j = F.softmax(logits_j / tau, dim=1)

        # --- Mixup操作 ---
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        lam = max(lam, 1 - lam)
        x_mix = lam * x_clean_b + (1 - lam) * x_partner_shuffled

        # 混合目标分布
        # 选择是否detach t_mix，KL散度版本不需要detach
        if 'loss_type' in cfg and cfg['loss_type'] == 'kl_div':
            # KL散度版本，不detach t_mix
            t_mix = lam * p_i + (1 - lam) * p_j[shuffle_idx]
        else:
            # 默认交叉熵版本，detach t_mix
            t_mix = (lam * p_i + (1 - lam) * p_j[shuffle_idx]).detach()

        # --- 计算损失函数 ---
        # 1. 干净样本CE损失

        logits_clean = net.get_original_logits(x_clean_b)
        logits_clean = logits_clean.view(-1, int(logits_clean.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]
        ce_loss = F.cross_entropy(logits_clean / tau, y_clean_b)

        # 2. 混合样本监督损失
        logits_mix = net.get_original_logits(x_mix)

        logits_mix = logits_mix.view(-1, int(logits_mix.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]

        # 根据损失类型选择损失函数
        if 'loss_type' in cfg and cfg['loss_type'] == 'kl_div':
            # KL散度版本
            log_probs_mix = F.log_softmax(logits_mix / tau, dim=1)
            sup_loss = F.kl_div(log_probs_mix, t_mix, reduction='batchmean')
        else:
            # 默认交叉熵版本
            log_probs_mix = F.log_softmax(logits_mix / tau, dim=1)
            sup_loss = -(t_mix * log_probs_mix).sum(dim=1).mean()

        # 总损失
        # loss = ce_weight * ce_loss + sup_weight * sup_loss
        loss = ce_weight * ce_loss + sup_weight * sup_loss
        # 反向传播
        loss.backward()
        optimizer.step()

        # 累计损失
        total_loss += loss.item() * B_clean
        total_samples += B_clean

        # 打印批次信息（修复：用iter_idx替代start）
        if (iter_idx // batch_size) % 15 == 0:
            print(f"Epoch {epoch} | Iter {iter_idx // batch_size} | Batch Size: {B_clean} | "
                  f"CE损失: {ce_loss.item():.4f} | 监督损失: {sup_loss.item():.4f} | 总损失: {loss.item():.4f}")

        # 释放当前batch的GPU显存（避免碎片化）
        del x_clean_b, y_clean_b, x_partner_b, x_partner_shuffled, x_mix
        del logits_i, logits_j, logits_clean, logits_mix, t_mix
        torch.cuda.empty_cache()

    # 调度器在epoch结束后调用
    if scheduler is not None:
        scheduler.step()

    # 释放CPU内存
    del x_clean_cpu, y_clean_cpu, x_combined_cpu, y_combined_cpu

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    print(
        f"Epoch {epoch}: DivideMix-style avg loss = {avg_loss:.6f} (CE: {ce_loss.item():.6f}, Sup: {sup_loss.item():.6f})")

    # 创建要返回的样本划分结果字典
    split_dict = {
        'id_clean': np.where(final_sample_type == 1)[0].tolist(),  # ID-clean样本索引
        'id': np.where(final_sample_type == 0)[0].tolist(),         # ID样本索引
        'noisy_ood': np.where((final_sample_type == 2) | (final_sample_type == -1))[0].tolist()  # OOD和未标记样本
    }
    avg_loss = 0
    # 返回平均损失和划分结果
    return avg_loss, split_dict

def freeze_lora(model):
    """冻结视觉端的 LoRA 参数（根据NegaPromptCLIP的结构）"""
    for name, param in model.named_parameters():
        if 'clean' in name:
            param.requires_grad = False
    print("✓ Frozen all LoRA parameters")


def unfreeze_lora(model):
    """解冻视觉端的 LoRA 参数（根据NegaPromptCLIP的结构）"""
    for name, param in model.named_parameters():
        if "clean" in name:
            param.requires_grad = True
    print("✓ Unfrozen all LoRA parameters")

def freeze_prompt(model):
    """冻结文本端的正负提示词参数（根据NegaPromptCLIP的结构）"""
    for name, param in model.named_parameters():
        if "prompt_learner" in name:
            param.requires_grad = False
    print("✓ Frozen all prompt parameters")


def unfreeze_prompt(model):
    """解冻文本端的正负提示词参数（根据NegaPromptCLIP的结构）"""
    for name, param in model.named_parameters():
        if "prompt_learner" in name:
            param.requires_grad = True
            print(name)
    print("✓ Unfrozen all prompt parameters")
"""
11-10, 现在主要的训练循环
"""
from datetime import datetime

def training_main_loop(model, trainloader, testloader, run, options):

    lora_params = [p for name, p in model.named_parameters() if ("A_clean" in name or "B_clean" in name)]
    text_params = [p for name, p in model.named_parameters() if "prompt_learner" in name]
    optimizer_text = torch.optim.SGD(text_params, lr=5e-4, momentum=0.9)
    optimizer_lora = torch.optim.SGD(lora_params, lr=options['lr'], momentum=0.9)

    scheduler_text = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_text, T_max=40, eta_min=1e-5
    )

    scheduler_lora = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_lora, T_max=40, eta_min=3e-5
    )




    # load_path = f"/data/tyang/checkpoints/WebFG/web-aircraft/lr=0.01_20260110_212438.pth"
    # checkpoint = torch.load(load_path, map_location='cuda')
    # model.load_state_dict(checkpoint['model_state_dict'])


    print("开始测试")
    max_epoch = options['max_epoch']
    results = my_test_nega_clip(model, testloader)
    exit(0)
    # best_acc = results['ID']['acc']
    best_acc = 0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"/data/tyang/checkpoints/now/lora_lr={options['lr']}_{timestamp}.pth"


    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"参数名: {name}, 维度: {tuple(param.shape)}")

    print("冻结prompt, 训练lora")
    unfreeze_lora(model)
    freeze_prompt(model)
    for epoch in range(1, max_epoch):
        print("==> Epoch {}/{}".format(epoch, max_epoch))
        avg_loss = train_stage1(model, optimizer_lora, scheduler_lora, trainloader, run, epoch=epoch, **options)
        results = my_test_lora(model, testloader)
        run.log({'acc': results['ID']['acc']})
        run.log({'visual_loss': avg_loss})
        if results['ID']['acc'] > best_acc:
            best_acc = results['ID']['acc']
            print(f"epoch: {epoch} best_acc: {best_acc}")
            save_nega_prompt_model(model, path)
            print(path)
    print(f"best_acc: {best_acc}")

    # 训练提示词部分
    # print("冻结lora, 训练prompt")
    # freeze_lora(model)
    # unfreeze_prompt(model)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"参数名: {name}")
    # for epoch in range(1, max_epoch):
    #     print("==> Epoch {}/{}".format(epoch, max_epoch))
    #     avg_loss = train_stage1(model, optimizer_text, scheduler_text, trainloader, run, epoch=epoch, **options)
    #     results = my_test_lora(model, testloader)
    #     run.log({'acc': results['ID']['acc']})
    #     run.log({'text_loss': avg_loss})
    #     if results['ID']['acc'] > best_acc and options['stage'] == 1:
    #         best_acc = results['ID']['acc']
    #         print(f"epoch: {epoch} best_acc: {best_acc}")
    #         save_nega_prompt_model(model, path)


    #     #
    #     # if options['stage'] == 2:
    #     #     if results['OOD']['auroc'] > best_auroc:
    #     #         best_auroc = results['OOD']['auroc']
    #     #         print(f"epoch: {epoch} best_auroc: {best_auroc}")
    #     #         path = "/data/tyang/checkpoints/sym0.8/clip_stage2_best.pth"
    #     #         save_nega_prompt_model(model, path)
    #     #         print(f"path : {path} best_auroc: {best_auroc}")
    #
    #     if options['stage'] == 3:
    #         if results['ID']['acc'] > best_acc:
    #             best_acc = results['ID']['acc']
    #             print(f"epoch: {epoch} best_acc: {best_acc}")
    #             run.log({'best acc': best_acc}, step=epoch)
    #             path = "/data/tyang/impo_checkpoints/sym0.8/only_text/new_best.pth"
    #             save_nega_prompt_model(model, path)
    #             print(f"path : {path} best_acc: {best_acc}")
    #     #     if (epoch % 2) == 0:
    #     #         path = f"/data/tyang/checkpoints/sym0.2/clip_stage3_{epoch}.pth"
    #     #         save_nega_prompt_model(model, path)
    #     #         print(f"path : {path} best_acc: {best_acc}")
    #
    #
    #     print_results(results)
    #
    #
    #     # 保存模型
    #     best_auroc = 0
    #     if options['stage'] == 2:
    #         if results['OOD']['auroc'] > best_auroc:
    #             best_auroc = results['OOD']['auroc']
    #             print(f"epoch: {epoch} best_auroc: {best_auroc}")
    #             run.log({'best auroc': best_auroc}, step=epoch)
    #             path = "/data3/tyang/data/sym0.2/11-11stage2.pth"
    #             save_nega_prompt_model(model, path)
    #             print(f"path : {path} best_auroc: {best_auroc}")
    #
    #     try:
    #         # 记录 train loss 与 test 结果
    #         run.log({'train_loss': avg_loss, 'epoch': epoch, **results})
    #     except Exception:
    #         pass
    #
    # print("Training finished.")


def training_main_loop1(model, trainloader, testloader, run, options):
    # ====================================================
    # 1. 智能参数分组 (ResNet 架构适配版)
    # ====================================================
    visual_params = []
    text_params = []

    # 遍历所有“已解冻”的参数 (在 main_worker 的 configure_resnet_finetuning 中解冻的)
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # 跳过冻结的层

        if "prompt_learner" in name or "ctx" in name:
            text_params.append(param)
        else:
            # 剩下的就是视觉端参数 (例如 layer4, attnpool)
            visual_params.append(param)

    print(f"\n[Optimizer Setup] Visual Params: {len(visual_params)} tensors")
    print(f"[Optimizer Setup] Text Params: {len(text_params)} tensors")

    # ====================================================
    # 2. 定义优化器 (关键：使用参数组设置不同的 LR)
    # ====================================================
    # 视觉端 LR: 必须小 (建议 1e-5 到 5e-5)
    # 文本端 LR: 可以大 (建议 1e-3 到 2e-3)

    # 注意：绝对不要用 options['lr']=0.01 来训练 ResNet，会直接崩掉
    visual_lr = 1e-5
    text_lr = 1e-3

    param_groups = [
        {'params': visual_params, 'lr': visual_lr, 'name': 'visual'},
        {'params': text_params, 'lr': text_lr, 'name': 'text'}
    ]

    # 这里我们将两组参数合并到一个优化器中，方便 step
    optimizer = torch.optim.SGD(param_groups, momentum=0.9, weight_decay=5e-4)

    # 学习率调度器 (作用于所有组)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=options['max_epoch'], eta_min=1e-6
    )

    # ====================================================
    # 3. 训练准备
    # ====================================================
    print("开始测试")
    max_epoch = options['max_epoch']

    results = my_test_resnet(model, testloader)
    best_acc = results['ID']['acc']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存路径包含架构信息
    path = f"/data/tyang/checkpoints/WebFG/web-aircraft/ResNet_ft_{timestamp}.pth"

    # 打印一下要训练的参数，再次确认
    freeze_prompt(model)
    print("-" * 30)
    print("Learnable Parameters Check:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f" -> {name}: {tuple(param.shape)}")
    print("-" * 30)

    # ResNet 微调不需要像 LoRA 那样反复 freeze/unfreeze
    # 因为我们在外面已经配置好了 requires_grad，这里直接跑就行

    # ====================================================
    # 4. 训练循环
    # ====================================================
    for epoch in range(1, max_epoch + 1):
        print(f"==> Epoch {epoch}/{max_epoch}")

        # 这里的 train_stage1 会自动更新 optimizer 中的所有参数组
        # 包括 ResNet High Layers 和 Prompt Learner
        avg_loss = train_stage1(model, optimizer, scheduler, trainloader, run, epoch=epoch, **options)

        # 测试 (如果是 ResNet，应该调用 my_test_noOOD 或者通用的 test_split_prompt)
        if hasattr(model, 'module'):
            # 处理 DataParallel
            is_lora = False  # ResNet模式没有LoRA
        else:
            is_lora = False

        # 直接复用测试函数，只要 forward_test 没变就行
        results = my_test_resnet(model, testloader)

        acc = results['ID']['acc']
        run.log({'acc': acc})
        run.log({'visual_loss': avg_loss})
        run.log({'lr_visual': optimizer.param_groups[0]['lr']})  # 记录一下视觉端的LR

        if acc > best_acc:
            best_acc = acc
            print(f"Epoch: {epoch} | New Best Acc: {best_acc:.2f}%")
            save_nega_prompt_model(model, path)
            print(f"Model saved to {path}")

    print(f"Final Best Acc: {best_acc:.2f}%")



# 注意：这个文件应该放在core目录下，或者与train_clip.py合并

import numpy as np

import os
import json
from datetime import datetime

import os
import json
from datetime import datetime

import os
import json
from datetime import datetime

def train_new_stage3(net, optimizer, scheduler, trainloader, run, epoch=None, proto=None, **options):
    import numpy as np

    n_nega_ctx = options['NEGA_CTX']
    batch_size = 512  # 可调
    total_loss = 0.0
    total_samples = 0

    # --- 1. 推断阶段：收集所有样本的 score1, score2 和标签 ---
    print(f"Epoch {epoch}: Running inference for sample identification...")
    net.eval()  # 切换到评估模式
    all_score1 = []  # MCM-style
    all_score2 = []  # s+ - s-
    all_score3 = []  # Top 5 (s+ - s-)
    all_labels = []  # 存储POMP变换后的标签
    all_batch_data = []
    all_batch_labels = []
    all_batch_is_clean = []  # 用于评估
    all_batch_is_open = []  # 用于评估
    all_predicted_classes = []  # 存储模型预测的类别
    all_original_indices = []  # 存储样本在原始数据集中的唯一索引
    all_clean_labels = []  # 存储样本的真实干净标签

    with torch.no_grad():  #
        all_top5_acc = []  # 收集ID样本的top5准确率
        all_top1_acc = []  # 收集ID样本的top1准确率
        for batch_idx, batch in enumerate(trainloader):
            data = batch['data']
            labels = batch['label']  # 噪声标签
            clean_labels_batch = batch['clean_label']  # 真实干净标签
            is_open = batch['is_open']  # 区分ID/OOD
            is_clean = batch['is_clean']  # 是否是干净样本
            original_indices = batch['index']  # 获取真实的原始样本索引

            if options['use_gpu']:
                # 所有张量统一移到GPU
                data, labels, clean_labels_batch, is_open, is_clean = data.cuda(), labels.cuda(), clean_labels_batch.cuda(), is_open.cuda(), is_clean.cuda()

            # POMP标签变换（保持原逻辑）
            if options['POMP']:
                ori_to_modify, modify_to_ori = label_transform(
                    labels.cpu().numpy(), options['POMP_k'], options['num_classes'] - 1
                )
                modified_labels = torch.tensor(
                    [ori_to_modify[label.item()] for label in labels]
                ).cuda()
                labels = modified_labels
            else:
                ori_to_modify, modify_to_ori = None, None

            # 前向传播与logits解析（保持原逻辑）
            if options['stage'] == 1:
                output, text_features = net(data, modify_to_ori)
            else:
                output, text_features, output_global_neg, global_neg_features, logits_local, logits_local_neg = net(
                    data, modify_to_ori)

            logits = output.view(-1, int(output.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)
            B, C = logits.shape[0], logits.shape[1]
            logits_no, logits_yes = logits[:, :, 1:], logits[:, :, 0]  # logits_yes: [B, C]

            # 计算logits_margin和prob_margin（保持原逻辑）
            mean_logits_no = logits_no.mean(dim=2)  # [B, C]

            logits_margin = logits_yes - mean_logits_no  # [B, C]
            T = 0.1
            prob_margin = torch.softmax(logits_margin / T, dim=1)  # [B, C]：所有类别的概率

            # 计算score1和对应的预测类别
            max_scores, predicted_classes = prob_margin.max(dim=1)  # score1是最大值，predicted_classes是对应索引
            score1 = max_scores

            # score2：基于噪声标签的概率（保持原逻辑）
            score2 = prob_margin[torch.arange(B), labels]  # [B]

            # 保存预测类别
            all_predicted_classes.append(predicted_classes.cpu())
            # 保存原始样本索引
            all_original_indices.append(original_indices.cpu())  # 原始索引
            # 保存真实干净标签
            all_clean_labels.append(clean_labels_batch.cpu())

            # 优化：直接缓存torch张量，避免numpy转换开销
            all_score1.append(score1.cpu())
            all_score2.append(score2.cpu())
            all_score3.append(score3.cpu())
            all_labels.append(labels.cpu())
            all_batch_data.append(data.cpu())
            all_batch_labels.append(labels.cpu())
            all_batch_is_clean.append(is_clean.cpu().bool())
            all_batch_is_open.append(is_open.cpu().bool())

    # 合并所有批次的分数和标签（使用torch.cat替代numpy.concatenate）
    all_score1_np = torch.cat(all_score1, dim=0).numpy()
    all_score2_np = torch.cat(all_score2, dim=0).numpy()
    all_score3_np = torch.cat(all_score3, dim=0).numpy()
    all_labels_np = torch.cat(all_labels, dim=0).numpy()
    all_batch_is_clean = torch.cat(all_batch_is_clean, dim=0).numpy()
    all_batch_is_open = torch.cat(all_batch_is_open, dim=0).numpy()
    all_predicted_classes_np = torch.cat(all_predicted_classes, dim=0).numpy()  # 模型预测的类别
    all_original_indices_np = torch.cat(all_original_indices, dim=0).numpy()  # 原始样本索引
    all_clean_labels_np = torch.cat(all_clean_labels, dim=0).numpy()  # 真实干净标签
    all_data = torch.cat(all_batch_data, dim=0)  # 保持为torch张量，避免后续numpy转换

    # --- 记录所有样本的score1和预测类别 ---
    print(f"Recording all samples' score1, score2 and predictions for epoch {epoch}...")

    # 创建所有样本数据
    all_samples_data = []
    for idx in range(len(all_original_indices_np)):
        sample_info = {
            'epoch': epoch,
            'sample_index': int(all_original_indices_np[idx]),  # 使用原始样本索引
            'score1': float(all_score1_np[idx]),
            'score2': float(all_score2_np[idx]),  # 添加score2字段
            'predicted_class': int(all_predicted_classes_np[idx]),
            'clean_label': int(all_clean_labels_np[idx]),  # 真实干净标签
            'is_open': bool(all_batch_is_open[idx]),  # 添加is_open字段
            'is_clean': bool(all_batch_is_clean[idx])  # 添加is_clean字段
        }
        all_samples_data.append(sample_info)

    # 保存为JSON文件
    output_dir = options.get('output_dir', './all_samples_records')
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"all_samples_epoch_{epoch}_{timestamp}.json")

    with open(filename, 'w') as f:
        json.dump(all_samples_data, f, indent=2)

    print(f"All samples record saved to: {filename}")

    # --- 2. 样本划分 ---
    print(f"Using Score1 & Score2 for sample identification on {len(all_score1_np)} samples...")
    # 为ID和ID-clean样本分别设置top_k_per_class值
    top_k_per_class_id = 64  # Marked as ID的top_k值
    top_k_per_class_clean = 64  # Marked as ID-clean的top_k值，用户可根据需要修改
    bottom_k_global = 1000  # 全局最低分样本数

    # 初始化标签掩码
    N_total = len(all_score1_np)
    final_sample_type = np.full(N_total, -1, dtype=int)  # -1:未标记, 0:ID, 1:ID-clean, 2:OOD

    # --- 1. 标记ID样本（Score1 Top-K）---
    unique_labels = np.unique(all_labels_np)
    all_top_k_indices_by_score1 = []
    for label in unique_labels:
        class_indices = np.where(all_labels_np == label)[0]
        if len(class_indices) == 0:
            continue
        class_score1 = all_score1_np[class_indices]
        k_to_select = min(top_k_per_class_id, len(class_score1))
        if k_to_select <= 0:
            continue
        # 取Top-K索引
        top_k_local_by_score1 = np.argpartition(class_score1, -k_to_select)[-k_to_select:]
        top_k_global_by_score1 = class_indices[top_k_local_by_score1]
        top_k_global_by_score1 = top_k_global_by_score1[np.argsort(all_score1_np[top_k_global_by_score1])[::-1]]
        all_top_k_indices_by_score1.extend(top_k_global_by_score1)
    all_top_k_indices_by_score1 = np.array(all_top_k_indices_by_score1)
    final_sample_type[all_top_k_indices_by_score1] = 0  # 标记为ID

    # --- 2. 标记ID-clean样本（Score2 Top-K，覆盖ID）---
    all_top_k_indices_by_score2 = []
    for label in unique_labels:
        class_indices = np.where(all_labels_np == label)[0]
        if len(class_indices) == 0:
            continue
        class_score2 = all_score2_np[class_indices]
        k_to_select = min(top_k_per_class_clean, len(class_score2))
        if k_to_select <= 0:
            continue
        top_k_local_by_score2 = np.argpartition(class_score2, -k_to_select)[-k_to_select:]
        top_k_global_by_score2 = class_indices[top_k_local_by_score2]
        top_k_global_by_score2 = top_k_global_by_score2[np.argsort(all_score2_np[top_k_global_by_score2])[::-1]]
        all_top_k_indices_by_score2.extend(top_k_global_by_score2)
    all_top_k_indices_by_score2 = np.array(all_top_k_indices_by_score2)
    final_sample_type[all_top_k_indices_by_score2] = 1  # 覆盖为ID-clean

    # --- 3. 标记OOD样本（Score2 Bottom-K，仅标记未被ID/ID-clean覆盖的样本）---
    k_to_select_global = min(bottom_k_global, N_total)
    bottom_k_global_indices = np.argpartition(all_score2_np, k_to_select_global)[:k_to_select_global]
    bottom_k_global_indices = bottom_k_global_indices[np.argsort(all_score2_np[bottom_k_global_indices])]
    # 修复：仅标记未被ID/ID-clean覆盖的样本（避免类型冲突）
    ood_mask = final_sample_type[bottom_k_global_indices] == -1
    final_sample_type[bottom_k_global_indices[ood_mask]] = 2  # 标记为OOD

    # --- 4. 统计结果 ---
    id_count = (final_sample_type == 0).sum()
    clean_count = (final_sample_type == 1).sum()
    ood_count = (final_sample_type == 2).sum()
    unmarked_count = (final_sample_type == -1).sum()
    print("--- Final Sample Identification Summary ---")
    print(f"Total Samples: {N_total}")
    print(f"ID samples (from Score1 Top-K): {id_count}")
    print(f"ID-Clean samples (from Score2 Top-K, overrides ID): {clean_count}")
    print(f"OOD samples (from Score2 Bottom-K): {ood_count}")
    print(f"Unmarked samples: {unmarked_count}")
    print("\n" + "=" * 60 + "\n")
    analyze_per_class_our_method(
        run, all_score1_np, all_score2_np, all_labels_np, all_batch_is_open, all_batch_is_clean,
        top_k_per_class_id=top_k_per_class_id,
        top_k_per_class_clean=top_k_per_class_clean,
        bottom_k_global=bottom_k_global,
        epoch=epoch
    )
    # --- 5. 训练准备 ---
    marked_indices = np.where(final_sample_type != -1)[0]

    # --- 5. 训练准备（修复DataLoader问题，保留分batch加载GPU）---
    print(f"Epoch {epoch}: Running training on {len(marked_indices)} marked samples with adaptive losses....")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)

    net.train()  # 切换回训练模式
    id_clean_mask = (final_sample_type == 1)
    id_mask = (final_sample_type == 0)
    non_ood_mask = id_clean_mask | id_mask  # 非OOD样本掩码（ID-clean + ID）

    # 1. 在CPU上完成所有数据处理，避免一次性占用大量GPU显存
    all_data_cpu = all_data.float().cpu()  # 保持在CPU
    all_labels_cpu = torch.from_numpy(all_labels_np).long().cpu()  # 保持在CPU

    # 2. 在CPU上进行样本筛选（torch掩码，避免numpy转换）
    id_clean_mask_tensor = torch.tensor(id_clean_mask, dtype=torch.bool, device='cpu')
    non_ood_mask_tensor = torch.tensor(non_ood_mask, dtype=torch.bool, device='cpu')

    x_clean_cpu = all_data_cpu[id_clean_mask_tensor]
    y_clean_cpu = all_labels_cpu[id_clean_mask_tensor]
    x_combined_cpu = all_data_cpu[non_ood_mask_tensor]
    y_combined_cpu = all_labels_cpu[non_ood_mask_tensor]

    # 释放CPU内存（关键：删除大张量）
    del all_labels_np, all_data, all_data_cpu, all_labels_cpu, id_clean_mask_tensor, non_ood_mask_tensor, \
        all_score1_np, all_score2_np, all_score3_np, all_batch_is_open, all_batch_is_clean, \
        all_predicted_classes_np, all_original_indices_np, all_clean_labels_np  # 清理所有样本数据

    num_clean = len(x_clean_cpu)
    num_combined = len(x_combined_cpu)

    if num_clean == 0:
        print("No ID-clean samples, skipping DivideMix training.")
        return 0.0

    # 3. CPU上打乱样本（保持原逻辑，避免DataLoader重复shuffle）
    perm_clean = torch.randperm(num_clean, device='cpu')
    x_clean_cpu = x_clean_cpu[perm_clean]
    y_clean_cpu = y_clean_cpu[perm_clean]

    perm_combined = torch.randperm(num_combined, device='cpu')
    x_combined_cpu = x_combined_cpu[perm_combined]
    y_combined_cpu = y_combined_cpu[perm_combined]

    # 训练参数
    num_classes = options['num_classes']
    tau = options.get('logit_temperature', 1)
    alpha = options.get('mixup_alpha', 0.2)
    ce_weight = options.get('ce_weight', 1.0)
    sup_weight = options.get('sup_weight', 0.5)  # 启用监督损失权重

    # === Step 2: 分batch训练（修复迭代逻辑，分batch加载GPU）===
    print(f"start iter loop with batch size {batch_size}! ")
    max_iter = max(num_clean, num_combined)

    for iter_idx in range(0, max_iter, batch_size):  # 按batch_size步长迭代
        optimizer.zero_grad()

        # --- 取当前块的干净样本（CPU上切片，仅当前batch）---
        start_clean = iter_idx
        end_clean = min(iter_idx + batch_size, num_clean)
        x_clean_b = x_clean_cpu[start_clean:end_clean]  # CPU张量
        y_clean_b = y_clean_cpu[start_clean:end_clean]  # CPU张量
        B_clean = x_clean_b.size(0)
        if B_clean == 0:
            continue

        # --- 获取混合样本对（CPU上切片，仅当前batch）---
        combined_start = iter_idx
        combined_end = iter_idx + B_clean
        # 循环索引，避免越界（保持原配对逻辑）
        combined_indices = torch.arange(combined_start, combined_end) % num_combined
        x_partner_b = x_combined_cpu[combined_indices]  # CPU张量

        # --- 仅将当前batch移到GPU（核心优化）---
        x_clean_b = x_clean_b.to(device, non_blocking=True)
        y_clean_b = y_clean_b.to(device, non_blocking=True)
        x_partner_b = x_partner_b.to(device, non_blocking=True)

        # 打乱混合样本（GPU上执行）
        shuffle_idx = torch.randperm(B_clean, device=device)
        x_partner_shuffled = x_partner_b[shuffle_idx]

        # --- 计算相似度向量 p_i/p_j ---
        with torch.no_grad():
            if options['stage'] == 1:
                logits_i, _ = net(x_clean_b, modify_to_ori)
                logits_j, _ = net(x_partner_b, modify_to_ori)
            else:
                logits_i, _, _, _, _, _ = net(x_clean_b, modify_to_ori)
                logits_j, _, _, _, _, _ = net(x_partner_b, modify_to_ori)

            # 解析logits（复用逻辑）
            logits_i = logits_i.view(-1, int(logits_i.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]
            logits_j = logits_j.view(-1, int(logits_j.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]
            p_i = F.softmax(logits_i / tau, dim=1)
            p_j = F.softmax(logits_j / tau, dim=1)

        # --- Mixup操作 ---
        lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
        lam = max(lam, 1 - lam)
        x_mix = lam * x_clean_b + (1 - lam) * x_partner_shuffled

        # 混合目标分布
        # 选择是否detach t_mix，KL散度版本不需要detach
        if 'loss_type' in options and options['loss_type'] == 'kl_div':
            # KL散度版本，不detach t_mix
            t_mix = lam * p_i + (1 - lam) * p_j[shuffle_idx]
        else:
            # 默认交叉熵版本，detach t_mix
            t_mix = (lam * p_i + (1 - lam) * p_j[shuffle_idx]).detach()

        # --- 计算损失函数 ---
        # 1. 干净样本CE损失
        if options['stage'] == 1:
            logits_clean, _ = net(x_clean_b, modify_to_ori)
        else:
            logits_clean, _, _, _, _, _ = net(x_clean_b, modify_to_ori)
        logits_clean = logits_clean.view(-1, int(logits_clean.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]
        ce_loss = F.cross_entropy(logits_clean / tau, y_clean_b)

        # 2. 混合样本监督损失
        if options['stage'] == 1:
            logits_mix, _ = net(x_mix, modify_to_ori)
        else:
            logits_mix, _, _, _, _, _ = net(x_mix, modify_to_ori)
        logits_mix = logits_mix.view(-1, int(logits_mix.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)[:, :, 0]

        # 根据损失类型选择损失函数
        if 'loss_type' in options and options['loss_type'] == 'kl_div':
            # KL散度版本
            log_probs_mix = F.log_softmax(logits_mix / tau, dim=1)
            sup_loss = F.kl_div(log_probs_mix, t_mix, reduction='batchmean')
        else:
            # 默认交叉熵版本
            log_probs_mix = F.log_softmax(logits_mix / tau, dim=1)
            sup_loss = -(t_mix * log_probs_mix).sum(dim=1).mean()

        # 总损失
        loss = ce_weight * ce_loss + sup_weight * sup_loss

        # 反向传播
        loss.backward()
        optimizer.step()

        # 累计损失
        total_loss += loss.item() * B_clean
        total_samples += B_clean

        # 打印批次信息
        if (iter_idx // batch_size) % 15 == 0:
            print(f"Epoch {epoch} | Iter {iter_idx // batch_size} | Batch Size: {B_clean} | "
                  f"CE损失: {ce_loss.item():.4f} | 监督损失: {sup_loss.item():.4f} | 总损失: {loss.item():.4f}")

        # 释放当前batch的GPU显存（避免碎片化）
        del x_clean_b, y_clean_b, x_partner_b, x_partner_shuffled, x_mix
        del logits_i, logits_j, logits_clean, logits_mix, t_mix
        torch.cuda.empty_cache() if options['use_gpu'] else None

    # 调度器在epoch结束后调用
    if scheduler is not None:
        scheduler.step()

    # 释放CPU内存
    del x_clean_cpu, y_clean_cpu, x_combined_cpu, y_combined_cpu

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    print(
        f"Epoch {epoch}: DivideMix-style avg loss = {avg_loss:.6f} (CE: {ce_loss.item():.6f}, Sup: {sup_loss.item():.6f})")
    return avg_loss
