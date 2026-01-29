import os
import os.path as osp
import numpy as np
from tqdm import tqdm
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import sklearn.metrics as sk
from core import evaluation
from transformers import CLIPTokenizer
from transformers import CLIPModel
import time
import pdb
import matplotlib.pyplot as plt


def test_clip(net, criterion, testloader, outloader, epoch=None, **options):
    correct, total = 0, 0

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []
    known_number = {}
    correct_number = {}
    all_results = {}
    with torch.no_grad():
        for data, labels in testloader:
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(False):
                logits, _ = net(data)
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
                for i in range(len(labels.data)):
                    if labels.data[i].item() not in known_number.keys():
                        known_number[labels.data[i].item()] = 0
                        correct_number[labels.data[i].item()] = 0
                        all_results[labels.data[i].item()] = {}
                    if predictions[i].item() not in all_results[labels.data[i].item()].keys():
                        all_results[labels.data[i].item()][predictions[i].item()] = 0
                    all_results[labels.data[i].item()][predictions[i].item()] += 1
                    known_number[labels.data[i].item()] += 1
                    if predictions[i] == labels.data[i]:
                        correct_number[labels.data[i].item()] += 1

                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(outloader):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(False):
                logits, _ = net(data)
                ood_score = logits.data.cpu().numpy()
    # # class_acc
    # class_acc = {}
    # for key in known_number.keys():
    #     class_acc[key] = correct_number[key] / known_number[key]
    # print('class_acc: ', class_acc  )
    # # print all_result
    # for key in all_results.keys():
    #     print('class ', key)
    #     print(all_results[key])
    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)

    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']

    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.

    return results


################################################################################################yt test

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import torch
import torch.nn.functional as F
from tqdm import tqdm


def compute_fpr95(ood_labels, ood_scores):
    # 返回 FPR@95TPR（如果 TPR 没有精确到 0.95，则取最近点）
    fpr, tpr, thresholds = roc_curve(ood_labels, ood_scores)
    target = 0.95
    idx = np.argmin(np.abs(tpr - target))
    return fpr[idx], fpr, tpr, thresholds


def ensure_local_layout(logits_local, n_classes):
    """
    将 logits_local 规范到 [B, H, W, C] 形式。
    支持输入形状：
      - [B, H, W, C] (直接返回)
      - [B, C, H, W] -> permute -> [B, H, W, C]
    """
    if logits_local is None:
        return None
    if logits_local.ndim == 4:
        if logits_local.shape[-1] == n_classes:
            # already [B, H, W, C]
            return logits_local
        if logits_local.shape[1] == n_classes:
            # [B, C, H, W] -> [B, H, W, C]
            return logits_local.permute(0, 2, 3, 1)
    raise ValueError(f"Unrecognized logits_local shape {logits_local.shape} for n_classes={n_classes}")


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
    logits_yes = logits[:, :, 0]  # [B, C]
    logits_no = logits[:, :, 1:]  # [B, C, n_neg_ctx]

    # 2. 计算公式4中的 p_no_ij
    # p_no_ij = exp(<fi, gno_j>/tau) / (exp(<fi, gj>/tau) + exp(<fi, gno_j>/tau))
    # logits 已经是 <fi, g>/tau
    logits_yes_exp = logits_yes.unsqueeze(-1)  # [B, C, 1]
    p_no = torch.exp(logits_no).sum(dim=-1) / (torch.exp(logits_yes) + torch.exp(logits_no).sum(dim=-1))

    # 3. ID 类概率
    p_yes = F.softmax(logits_yes, dim=-1)  # [B, C]

    # 4. 计算未知类别概率 p_{C+1} (公式8)
    p_C_plus_1 = 1 - torch.sum((1 - p_no) * p_yes, dim=-1)  # [B]

    # 5. OOD 判定 (公式9)
    max_p_id, _ = torch.max(p_yes, dim=-1)
    is_id = (p_C_plus_1 <= max_p_id).long()  # 1=ID, 0=OOD
    # pdb.set_trace()
    return is_id, p_yes, p_no, p_C_plus_1


import torch


def compute_probs_allneg(logits_global, logits_local=None, n_nega_ctx=4, tau=1.0):
    """
    logits_global: [B, C*(1+n_nega_ctx)]  （跟你原来的一致）
    logits_local : 可选，[B, M*C*(1+n_nega_ctx)] 或原始 local shape，按下面 view 恢复
    n_nega_ctx   : 每类 negative prompt 数量
    tau          : 温度
    返回: probs_g [B, C], probs_l [B, M, C] 或 None
    """
    # --- global ---
    B = logits_global.shape[0]
    total = logits_global.shape[1]
    C = total // (1 + n_nega_ctx)
    # restore shape [B, C, 1+n_nega_ctx]
    g = logits_global.view(B, C, 1 + n_nega_ctx)
    logits_yes = g[:, :, 0]  # [B, C]
    logits_no = g[:, :, 1:]  # [B, C, n_nega_ctx]

    # flatten all neg across classes -> [B, C*n_nega_ctx]
    neg_flat = logits_no.reshape(B, -1)

    # concat pos (C) and all neg (C*n_nega_ctx) -> do stable softmax-like
    concat = torch.cat([logits_yes / tau, neg_flat / tau], dim=1)  # [B, C + C*n_nega_ctx]
    denom = torch.logsumexp(concat, dim=1, keepdim=True)  # [B, 1]
    probs_g = torch.exp(logits_yes / tau - denom)  # [B, C]

    # --- local (optional) ---
    if logits_local is None:
        return probs_g, None

    # logits_local expected to be viewable to [B, M, C, 1+n_nega_ctx]
    l = logits_local.view(B, -1, C, 1 + n_nega_ctx)  # M inferred as -1
    logits_yes_l = l[:, :, :, 0]  # [B, M, C]
    logits_no_l = l[:, :, :, 1:]  # [B, M, C, n_nega_ctx]

    B2, M, C2 = logits_yes_l.shape
    # flatten neg for each local context: [B, M, C*n_nega_ctx]
    neg_flat_l = logits_no_l.reshape(B2, M, -1)
    # concat along class dim
    concat_l = torch.cat([logits_yes_l / tau, neg_flat_l / tau], dim=2)  # [B, M, C + C*n_nega_ctx]
    denom_l = torch.logsumexp(concat_l, dim=2, keepdim=True)  # [B, M, 1]
    probs_l = torch.exp(logits_yes_l / tau - denom_l)  # [B, M, C]

    return probs_g, probs_l


def gl_mcm_score(logits_global, logits_local, logits_global_neg, pooling="max", n_nega_ctx=2, temperature=1):
    """
    logits_global: [B, C*(1+n_nega)]
    logits_local: [B, grid*grid, C*(1+n_nega)]
    logits_neg: 全局负提示的相似度

    10.14
    原来的正负提示词做法
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

    # # 第三种
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
    probs_l = F.softmax(diff_logits_l / temperature, dim=-1)

    probs_g = probs_g.detach().cpu().numpy()
    probs_l = probs_l.detach().cpu().numpy()

    # # A negative sign has already been added here
    scores = -np.max(probs_g, axis=1) - np.max(probs_l, axis=(1, 2)) * lambda_local

    return scores


# def gl_mcm_score(logits_global, logits_local, logits_global_neg, pooling="max", n_nega_ctx = 2, temperature = 1):
#     """
#     logits_global: [B, C*(1+n_nega)]
#     logits_local: [B, grid*grid, C*(1+n_nega)]
#     logits_neg: 全局负提示的相似度

#     10.13 只用global_neg 和 pos
#         pos - global_neg
#     """
#     lambda_local = 0

#     # 第三种
#     B, spatial_dim, total_logits = logits_local.shape
#     C = total_logits // (1 + n_nega_ctx)  # 计算类别数量

#     logits_reshaped = logits_global.view(B, C, 1 + n_nega_ctx)
#     logits_l = logits_local.view(B, spatial_dim, C, 1 + n_nega_ctx)

#     # 分离positive和negative logits
#     pos_logits = logits_reshaped[:, :, 0]  # [B, C]
#     pos_logits_l, neg_logits_l = logits_l[:, :, :, 0], logits_l[:, :, :, 1:]

#     #
#     global_neg_mean = logits_global_neg.mean(dim=-1).detach().cpu().numpy()


#     # 对每个样本的C个类别进行softmax归一化
#     probs_g = F.softmax(pos_logits / temperature, dim=-1)
#     probs_l = F.softmax(pos_logits_l / temperature, dim = -1)

#     # A negative sign has already been added here
#     probs_g = probs_g.detach().cpu().numpy()
#     probs_l = probs_l.detach().cpu().numpy()

#     scores = -np.max(probs_g, axis = 1) - np.max(probs_l, axis = (1, 2)) * lambda_local + global_neg_mean

#     return scores

def compute_auroc(scores, is_open):
    """
    scores: [N] torch.Tensor or np.array, ID 分数 (越大越 ID)
    is_open: [N] np.array, 0=ID, 1=OOD
    """
    # scores = scores.detach().cpu().numpy()
    labels = is_open.astype(int)  # 0=ID, 1=OOD
    auroc = roc_auc_score(labels, scores)
    return auroc


def fpr_at_95tpr(scores, is_open):
    # scores = scores.detach().cpu().numpy()
    labels = is_open.astype(int)

    fpr, tpr, thresholds = roc_curve(labels, scores)  # OOD=1
    # 找到最接近 95% 的 tpr
    idx = np.argmin(np.abs(tpr - 0.95))
    return fpr[idx]


def my_test_nega_clip1(
        net,
        criterion,
        testloader,
        epoch=None,
        **options):
    net.eval()
    all_softmax = []  # 保存 softmax logits（或 OOD 分数）
    all_preds = []  # 保存预测类别
    all_labels = []  # 保存真实标签
    all_is_open = []  # 保存是否 OOD (0=ID, 1=OOD)
    all_scores = []
    all_is_clean = []
    with torch.no_grad():
        prompts = net.prompt_learner()
        tokenized_prompts = net.tokenized_prompts

        text_features = net.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        global_neg_features = net.text_encoder.encode_global_neg(net.global_neg_ctx,
                                                                 net.global_neg_tokenized_prompts)

        global_neg_features = global_neg_features / global_neg_features.norm(dim=-1, keepdim=True)

        torch.cuda.empty_cache()
        tqdm_object = tqdm(testloader, total=len(testloader))

        for batch_idx, batch in enumerate(tqdm_object):
            """
            10.13 修改测试代码    
            """
            images = batch['data'].cuda()
            clean_labels = batch['clean_label'].cuda()  # 真实clean label（numpy or tensor）
            is_open = batch['is_open'].cuda()  # 是否 OOD (bool array)
            is_clean = batch['is_clean'].cuda()
            # [B, C * 3]
            logits_global, logits_local, _ = net.forward_test(images, text_features)
            logits = logits_global
            logits_neg_global = net.forward_global(images, global_neg_features)

            # softmax -> probabilities
            softmax_logits = F.softmax(logits_global, dim=1)  # [B, C]
            n_nega_ctx = options['NEGA_CTX']
            softmax_logits = softmax_logits.view(-1, int(softmax_logits.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)
            logits = logits.view(-1, int(logits.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)

            softmax_logits_posi = softmax_logits[:, :, 0]
            softmax_logits_negas = softmax_logits[:, :, 1:]
            logits_posi = logits[:, :, 0]
            logits_negas = logits[:, :, 1:]
            predictions = softmax_logits_posi.data.max(1)[1]

            scoers = gl_mcm_score(logits_global, logits_local, logits_neg_global, n_nega_ctx=n_nega_ctx)
            predictions_corrected = predictions.clone()

            all_preds.append(predictions.detach().cpu().numpy())
            all_labels.append(clean_labels.cpu().numpy())
            all_is_open.append(is_open.cpu().numpy())
            # pdb.set_trace()
            all_scores.append(scoers)

            # pdb.set_trace()

        all_is_open = np.concatenate(all_is_open)
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

        id_mask = (all_is_open == 0)  # 只看ID样本
        precision = precision_score(all_labels[id_mask], all_preds[id_mask], average='macro')
        recall = recall_score(all_labels[id_mask], all_preds[id_mask], average='macro')
        f1 = f1_score(all_labels[id_mask], all_preds[id_mask], average='macro')
        acc = accuracy_score(all_labels[id_mask], all_preds[id_mask])
        print("=" * 60)
        print("ID Classification: Acc=%.4f, P=%.4f, R=%.4f, F1=%.4f" % (acc, precision, recall, f1))

        auroc = compute_auroc(all_scores, all_is_open)
        fpr95 = fpr_at_95tpr(all_scores, all_is_open)
        print(f"GL-MCM OOD Detection: AUROC={auroc:.4f}, FPR95={fpr95:.4f}")
        print('=' * 60)
        # OOD_res = OOD_correct / total_OOD
        # ID_clean_res = ID_clean_correct / total_ID_clean
        # print(ID_clean_correct / (total - total_OOD))
        results = {
            'ID': {
                'acc': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            },
            'OOD': {
                'auroc': auroc,
                'fpr95': fpr95,
            }

        }
    return results, all_scores


def my_test_nega_clip(
        net, testloader,
):
    net.eval()
    all_softmax = []  # 保存 softmax logits（或 OOD 分数）
    all_preds = []  # 保存预测类别
    all_labels = []  # 保存真实标签
    all_is_open = []  # 保存是否 OOD (0=ID, 1=OOD)
    all_scores = []
    all_is_clean = []
    with torch.no_grad():
        prompts = net.prompt_learner()

        tokenized_prompts = net.tokenized_prompts
        text_features = net.text_encoder(prompts, tokenized_prompts)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        torch.cuda.empty_cache()
        tqdm_object = tqdm(testloader, total=len(testloader))

        for batch_idx, batch in enumerate(tqdm_object):
            images = batch['data'].cuda()
            clean_labels = batch['clean_label'].cuda()  # 真实clean label（numpy or tensor）
            is_open = batch['is_open'].cuda()  # 是否 OOD (bool array)
            is_clean = batch['is_clean'].cuda()
            # [B, C * 3]
            logits_global, logits_local, _ = net.forward_test(images, text_features)
            logits = logits_global

            # softmax -> probabilities
            softmax_logits = F.softmax(logits_global, dim=1)  # [B, C]

            n_nega_ctx = 2
            softmax_logits = softmax_logits.view(-1, int(softmax_logits.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)
            logits = logits.view(-1, int(logits.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)

            softmax_logits_posi = softmax_logits[:, :, 0]
            predictions = softmax_logits_posi.data.max(1)[1]

            scoers = gl_mcm_score(logits_global, logits_local, n_nega_ctx)

            all_preds.append(predictions.detach().cpu().numpy())
            all_labels.append(clean_labels.cpu().numpy())
            all_is_open.append(is_open.cpu().numpy())
            all_scores.append(scoers)

        all_is_open = np.concatenate(all_is_open)
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

        id_mask = (all_is_open == 0)  # 只看ID样本
        precision = precision_score(all_labels[id_mask], all_preds[id_mask], average='macro')
        recall = recall_score(all_labels[id_mask], all_preds[id_mask], average='macro')
        f1 = f1_score(all_labels[id_mask], all_preds[id_mask], average='macro')
        acc = accuracy_score(all_labels[id_mask], all_preds[id_mask])
        print("=" * 60)
        print("ID Classification: Acc=%.4f, P=%.4f, R=%.4f, F1=%.4f" % (acc, precision, recall, f1))

        auroc = compute_auroc(all_scores, all_is_open)
        fpr95 = fpr_at_95tpr(all_scores, all_is_open)
        print(f"GL-MCM OOD Detection: AUROC={auroc:.4f}, FPR95={fpr95:.4f}")
        print('=' * 60)
        # OOD_res = OOD_correct / total_OOD
        # ID_clean_res = ID_clean_correct / total_ID_clean
        # print(ID_clean_correct / (total - total_OOD))
        results = {
            'ID': {
                'acc': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            },
            'OOD': {
                'auroc': auroc,
                'fpr95': fpr95,
            }

        }
    return results


def my_test_clip(
        net,
        testloader, ):
    net.eval()
    all_softmax = []  # 保存 softmax logits（或 OOD 分数）
    all_preds = []  # 保存预测类别
    all_labels = []  # 保存真实标签
    all_is_open = []  # 保存是否 OOD (0=ID, 1=OOD)
    all_scores = []
    all_is_clean = []
    with torch.no_grad():
        torch.cuda.empty_cache()
        tqdm_object = tqdm(testloader, total=len(testloader))

        for batch_idx, batch in enumerate(tqdm_object):
            images = batch['data'].cuda()
            clean_labels = batch['clean_label'].cuda()  # 真实clean label（numpy or tensor）
            is_open = batch['is_open'].cuda()  # 是否 OOD (bool array)
            is_clean = batch['is_clean'].cuda()
            # [B, C * 3]
            logits_global = net.get_original_logits(images)

            # softmax -> probabilities
            softmax_logits = F.softmax(logits_global, dim=1)  # [B, C]

            n_nega_ctx = 2
            softmax_logits = softmax_logits.view(-1, int(softmax_logits.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)
            softmax_logits_posi = softmax_logits[:, :, 0]
            predictions = softmax_logits_posi.data.max(1)[1]

            all_preds.append(predictions.detach().cpu().numpy())
            all_labels.append(clean_labels.cpu().numpy())
            all_is_open.append(is_open.cpu().numpy())

        all_is_open = np.concatenate(all_is_open)
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

        id_mask = (all_is_open == 0)  # 只看ID样本
        precision = precision_score(all_labels[id_mask], all_preds[id_mask], average='macro')
        recall = recall_score(all_labels[id_mask], all_preds[id_mask], average='macro')
        f1 = f1_score(all_labels[id_mask], all_preds[id_mask], average='macro')
        acc = accuracy_score(all_labels[id_mask], all_preds[id_mask])
        print("=" * 60)
        print("ID Classification: Acc=%.4f, P=%.4f, R=%.4f, F1=%.4f" % (acc, precision, recall, f1))
        print('=' * 60)
        # OOD_res = OOD_correct / total_OOD
        # ID_clean_res = ID_clean_correct / total_ID_clean
        # print(ID_clean_correct / (total - total_OOD))
        results = {
            'ID': {
                'acc': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            },

        }
    return results
def my_test_noOOD(
        net,
        testloader,
        **options):
    net.eval()
    all_preds = []  # 保存预测类别
    all_labels = []  # 保存真实标签
    all_is_open = []  # 保存是否 OOD (0=ID, 1=OOD)
    with torch.no_grad():
        torch.cuda.empty_cache()
        tqdm_object = tqdm(testloader, total=len(testloader))

        for batch_idx, batch in enumerate(tqdm_object):
            images = batch['data'].cuda()
            clean_labels = batch['clean_label'].cuda()  # 真实clean label（numpy or tensor）
            logits_clean = net.get_lora_logits(images)

            n_nega_ctx = 2
            softmax_logits = F.softmax(logits_clean, dim=1)  # [B, C]
            softmax_logits = softmax_logits.view(-1, int(softmax_logits.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)
            softmax_logits_posi = softmax_logits[:, :, 0]

            predictions = softmax_logits_posi.data.max(1)[1]
            all_preds.append(predictions.detach().cpu().numpy())
            all_labels.append(clean_labels.cpu().numpy())

        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        print("=" * 60)
        print("ID Classification: Acc=%.4f, P=%.4f, R=%.4f, F1=%.4f" % (acc, precision, recall, f1))
        print('=' * 60)
        results = {
            'ID': {
                'acc': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            },

        }
    return results


def my_test_lora(
        net,
        testloader,
        **options):
    net.eval()
    all_preds = []  # 保存预测类别
    all_labels = []  # 保存真实标签
    all_is_open = []  # 保存是否 OOD (0=ID, 1=OOD)
    with torch.no_grad():
        torch.cuda.empty_cache()
        tqdm_object = tqdm(testloader, total=len(testloader))

        for batch_idx, batch in enumerate(tqdm_object):
            images = batch['data'].cuda()
            clean_labels = batch['clean_label'].cuda()  # 真实clean label（numpy or tensor）
            is_open = batch['is_open'].cuda()  # 是否 OOD (bool array)
            # [B, C * 3]

            logits_clean = net.get_lora_logits(images)

            n_nega_ctx = 2
            # softmax -> probabilities
            softmax_logits = F.softmax(logits_clean, dim=1)  # [B, C]

            softmax_logits = softmax_logits.view(-1, int(softmax_logits.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)
            # logits = logits_clean.view(-1, int(logits_clean.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)

            softmax_logits_posi = softmax_logits[:, :, 0]

            predictions = softmax_logits_posi.data.max(1)[1]

            all_preds.append(predictions.detach().cpu().numpy())
            all_labels.append(clean_labels.cpu().numpy())
            all_is_open.append(is_open.cpu().numpy())

        all_is_open = np.concatenate(all_is_open)
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        id_mask = (all_is_open == 0)  # 只看ID样本
        precision = precision_score(all_labels[id_mask], all_preds[id_mask], average='macro')
        recall = recall_score(all_labels[id_mask], all_preds[id_mask], average='macro')
        f1 = f1_score(all_labels[id_mask], all_preds[id_mask], average='macro')
        acc = accuracy_score(all_labels[id_mask], all_preds[id_mask])
        print("=" * 60)
        print("ID Classification: Acc=%.4f, P=%.4f, R=%.4f, F1=%.4f" % (acc, precision, recall, f1))
        print('=' * 60)
        # OOD_res = OOD_correct / total_OOD
        # ID_clean_res = ID_clean_correct / total_ID_clean
        # print(ID_clean_correct / (total - total_OOD))
        results = {
            'ID': {
                'acc': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            },

        }
    return results


### lora权重 + a phot of a {}的默认嵌入
def my_test_resnet(
        net,
        testloader,
        **options):
    net.eval()
    all_preds = []  # 保存预测类别
    all_labels = []  # 保存真实标签
    all_is_open = []  # 保存是否 OOD (0=ID, 1=OOD)
    with torch.no_grad():
        torch.cuda.empty_cache()
        tqdm_object = tqdm(testloader, total=len(testloader))

        for batch_idx, batch in enumerate(tqdm_object):
            images = batch['data'].cuda()
            clean_labels = batch['clean_label'].cuda()  # 真实clean label（numpy or tensor）
            logits_clean = net(images)

            n_nega_ctx = 2
            # softmax -> probabilities
            softmax_logits = F.softmax(logits_clean, dim=1)  # [B, C]

            softmax_logits = softmax_logits.view(-1, int(softmax_logits.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)
            # logits = logits_clean.view(-1, int(logits_clean.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)

            softmax_logits_posi = softmax_logits[:, :, 0]

            predictions = softmax_logits_posi.data.max(1)[1]
            all_preds.append(predictions.detach().cpu().numpy())
            all_labels.append(clean_labels.cpu().numpy())

        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        print("=" * 60)
        print("ID Classification: Acc=%.4f, P=%.4f, R=%.4f, F1=%.4f" % (acc, precision, recall, f1))
        print('=' * 60)
        results = {
            'ID': {
                'acc': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            },

        }
    return results


################################################################################################yt test


def test_nega_clip(net, criterion, testloader, outloader, epoch=None, **options):
    correct, total = 0, 0
    _pred_k, _pred_u, _labels = [], [], []
    logits_posi_id, logits_nega_id, logits_posi_ood, logits_nega_ood = [], [], [], []
    net.eval()
    with torch.no_grad():
        if torch.cuda.device_count() > 1:
            prompts = net.module.prompt_learner()
            tokenized_prompts = net.module.tokenized_prompts
            text_features = net.module.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        else:
            prompts = net.prompt_learner()
            tokenized_prompts = net.tokenized_prompts
            text_features = net.text_encoder(prompts, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        torch.cuda.empty_cache()
        # breakpoint()
        tqdm_object = tqdm(testloader, total=len(testloader))
        for batch_idx, (data, labels) in enumerate(tqdm_object):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            if torch.cuda.device_count() > 1:
                logits, _ = net.module.forward_test(data, text_features)
                logits /= net.module.logit_scale.exp()
            else:
                logits, _ = net.forward_test(data, text_features)
                logits /= net.logit_scale.exp()
            predictions, ood_score, logits_posi, logits_negas = get_ood_score(logits, options)
            _pred_k.append(ood_score)
            correct += (predictions == labels.data).sum()
            total += labels.size(0)
            _labels.append(labels.data.cpu().numpy())
            logits_posi_id.append(logits_posi.data.cpu().numpy())
            logits_nega_id.append(logits_negas.data.cpu().numpy())
        acc = float(correct) * 100. / float(total)
        print('Acc: {:.5f}'.format(acc))
        tqdm_object = tqdm(outloader, total=len(outloader))
        for batch_idx, (data, labels) in enumerate(tqdm_object):
            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()

            with torch.set_grad_enabled(False):
                if torch.cuda.device_count() > 1:
                    logits, _ = net.module.forward_test(data, text_features)
                    logits /= net.module.logit_scale.exp()
                else:
                    logits, _ = net.forward_test(data, text_features)
                    logits /= net.logit_scale.exp()
                predictions, ood_score, logits_posi, logits_negas = get_ood_score(logits, options)
                _pred_u.append(ood_score)
                logits_posi_ood.append(logits_posi.data.cpu().numpy())
                logits_nega_ood.append(logits_negas.data.cpu().numpy())

    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)

    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']

    # save _pred_k, -pred_u
    # score_dic = {}
    # score_dic['pred_k'] = _pred_k
    # score_dic['pred_u'] = _pred_u
    # score_dic['logits_posi_id'] = np.concatenate(logits_posi_id, 0)
    # score_dic['logits_nega_id'] = np.concatenate(logits_nega_id, 0)
    # score_dic['logits_posi_ood'] = np.concatenate(logits_posi_ood, 0)
    # score_dic['logits_nega_ood'] = np.concatenate(logits_nega_ood, 0)
    # np.save('savescores/' + options['dataset'] + '_ score_dic.npy', score_dic)
    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    auroc, aupr, fpr95 = compute_fpr(x1, x2)
    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.
    results['FPR95'] = fpr95 * 100.
    results['AUPR'] = aupr * 100.
    return results


def compute_fpr(pred_k, pred_u):
    x1 = pred_k
    x2 = pred_u
    pos = np.array(x1[:]).reshape((-1, 1))
    neg = np.array(x2[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr95 = fpr_and_fdr_at_recall(labels, examples)

    # fpr,tpr,thresh = roc_curve(labels, examples, pos_label=1)
    # fpr95 = float(interpolate.interp1d(tpr, fpr)(0.95))
    return auroc, aupr, fpr95


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))  # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_ood_score(logits, options):
    n_nega_ctx = options['NEGA_CTX']
    softmax_logits = F.softmax(logits, dim=1)
    softmax_logits = softmax_logits.view(-1, int(softmax_logits.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)
    logits = logits.view(-1, int(logits.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)

    softmax_logits_posi = softmax_logits[:, :, 0]
    softmax_logits_negas = softmax_logits[:, :, 1:]
    logits_posi = logits[:, :, 0]
    logits_negas = logits[:, :, 1:]
    predictions = softmax_logits_posi.data.max(1)[1]

    if options['open_score'] == 'msp':
        ood_score = softmax_logits_posi.data.cpu().numpy()
    elif options['open_score'] == 'maxlogit':
        ood_score = logits_posi.data.cpu().numpy()
    elif options['open_score'] == 'energy_oe':
        energy = torch.log(torch.sum(torch.exp(logits_posi), dim=1)).unsqueeze(1).cpu().numpy()
        ood_score = energy
    elif options['open_score'] == 'nega':
        ood_score = softmax_logits_negas.data.max(2)[0].cpu().numpy()
    elif options['open_score'] == 'posi_nega':
        nega_dis = torch.Tensor(softmax_logits_posi.shape[0]).cuda()
        for i in range(softmax_logits_posi.shape[0]):
            nega_dis[i] = torch.max(softmax_logits_negas[i, predictions[i], :])
        nega_dis = nega_dis.view(-1, 1)
        nega_dis = nega_dis.repeat(1, softmax_logits_posi.shape[1])
        posi_minus_nega = softmax_logits_posi - nega_dis
        ood_score = posi_minus_nega.data.cpu().numpy()
    elif options['open_score'] == 'posi_minus_closest_radius':
        _, min_loc = torch.min(softmax_logits_negas, dim=2)
        index1 = torch.arange(min_loc.shape[1])
        index1 = index1.repeat(min_loc.shape[0]).cuda()
        index2 = min_loc.flatten().cuda()
        right_radius = radius[index1, index2].view(min_loc.shape[0], min_loc.shape[1]).cuda()
        posi_minus_radius = right_radius - softmax_logits_posi
        ood_score = posi_minus_radius.data.cpu().numpy()
    elif options['open_score'] == 'posi_radius':
        # right_radius(logits_posi.shape[0] * right_radius.shape[0]) is repeated by radius_mean\
        right_radius = radius_mean.expand((softmax_logits_posi.shape[0], -1)).cuda()
        posi_minus_radius = right_radius - softmax_logits_posi
        ood_score = posi_minus_radius.data.cpu().numpy()
    return predictions, ood_score, logits_posi, logits_negas


"""
11-18 测试delora
"""


def evaluate_sample_filtering(model, dataloader, options, save_selected_indices_path=None):
    """
    评估样本筛选效果
    Args:
        model: 训练完成的 DeLoRA 模型
        dataloader: 测试用的 DataLoader (要求shuffle=False以保证索引一致性)
        options: 包含参数配置的字典
        save_selected_indices_path: 保存筛选出的样本索引的文件路径 (可选)
    Returns:
        results: 评估结果字典
    """
    model.eval()

    # 统计模型预测为干净的样本中，真实类别的数量
    model_predicted_clean_total = 0
    real_id_clean_count = 0
    real_id_noisy_count = 0
    real_ood_count = 0

    # 用于保存筛选出的样本索引
    selected_indices = []
    global_idx = 0  # 全局样本索引（用于唯一标识每个样本）

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):

            if batch_idx % 50 == 0:
                print(batch_idx)

            data = batch['data']
            labels = batch['label']

            # 访问真实类别信息
            is_open = batch['is_open']  # 1表示OOD，0表示ID
            is_clean = batch['is_clean']  # 1表示ID-clean，0表示ID-noisy + OOD

            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
                is_open = is_open.cuda()
                is_clean = is_clean.cuda()

            # 使用 p_i_c 区分干净/噪声样本
            # 获取干净 LoRA 和噪声 LoRA 的 logits
            logits_clean, logits_noisy = model.get_clean_noisy_logits(data)

            # 计算 CE_clean 和 CE_noisy
            CE_clean = F.cross_entropy(logits_clean, labels, reduction='none')
            CE_noisy = F.cross_entropy(logits_noisy, labels, reduction='none')

            # Calculate p_i^c (probability that sample is clean)
            exp_neg_CE_clean = torch.exp(-CE_clean)
            exp_neg_CE_noisy = torch.exp(-CE_noisy)
            denominator = exp_neg_CE_clean + exp_neg_CE_noisy + 1e-7  # Avoid division by zero
            p_i_c = exp_neg_CE_clean / denominator
            p_i_c = torch.clamp(p_i_c, 1e-7, 1 - 1e-7)  # Avoid log(0)
            # 模型判断为干净的样本
            predicted_is_clean = (p_i_c > 0.994).bool()
            model_predicted_clean_batch = predicted_is_clean.sum().item()

            # 记录筛选出的样本索引
            if predicted_is_clean.sum().item() > 0:
                # 获取当前batch中被选中的样本在batch内的索引
                batch_selected_indices = torch.where(predicted_is_clean)[0].tolist()
                # 转换为全局索引并保存
                global_selected_indices = [global_idx + idx for idx in batch_selected_indices]
                selected_indices.extend(global_selected_indices)

            if model_predicted_clean_batch == 0:
                # 更新全局索引
                global_idx += data.size(0)
                continue

            # 提取模型预测为干净的样本的真实类别信息
            batch_is_open = is_open[predicted_is_clean]
            batch_is_clean = is_clean[predicted_is_clean]

            # 计算真实类别：ID-clean, ID-noisy, OOD
            # ID-clean: is_open == 0 and is_clean == 1
            # ID-noisy: is_open == 0 and is_clean == 0
            # OOD: is_open == 1
            real_id_clean_batch = ((batch_is_open == 0) & (batch_is_clean == 1)).sum().item()
            real_id_noisy_batch = ((batch_is_open == 0) & (batch_is_clean == 0)).sum().item()
            real_ood_batch = (batch_is_open == 1).sum().item()

            # 累加统计
            model_predicted_clean_total += model_predicted_clean_batch
            real_id_clean_count += real_id_clean_batch
            real_id_noisy_count += real_id_noisy_batch
            real_ood_count += real_ood_batch

            print(
                f"real_id_clean_ratio: {real_id_clean_count / model_predicted_clean_total if model_predicted_clean_total > 0 else 0.0}"
                f"real_id_noisy_ratio: {real_id_noisy_count / model_predicted_clean_total if model_predicted_clean_total > 0 else 0.0}"
                f"real_ood_ratio: {real_ood_count / model_predicted_clean_total if model_predicted_clean_total > 0 else 0.0}")

            print(f"model_predicted_clean_total: {model_predicted_clean_total}")

            # 更新全局索引
            global_idx += data.size(0)

    # 计算比例
    real_id_clean_ratio = real_id_clean_count / model_predicted_clean_total if model_predicted_clean_total > 0 else 0.0
    real_id_noisy_ratio = real_id_noisy_count / model_predicted_clean_total if model_predicted_clean_total > 0 else 0.0
    real_ood_ratio = real_ood_count / model_predicted_clean_total if model_predicted_clean_total > 0 else 0.0

    # 整理结果
    results = {
        'model_predicted_clean_total': model_predicted_clean_total,
        'real_id_clean_count': real_id_clean_count,
        'real_id_clean_ratio': real_id_clean_ratio,
        'real_id_noisy_count': real_id_noisy_count,
        'real_id_noisy_ratio': real_id_noisy_ratio,
        'real_ood_count': real_ood_count,
        'real_ood_ratio': real_ood_ratio,
        'selected_indices': selected_indices  # 添加筛选出的样本索引
    }

    # 保存筛选出的样本索引
    if save_selected_indices_path is not None:
        import json
        with open(save_selected_indices_path, 'w') as f:
            json.dump(selected_indices, f)
        print(f"\n筛选出的样本索引已保存到: {save_selected_indices_path}")

    model.train()
    return results


def analyze_sample_filtering(model, dataloader, options, output_path):
    """
    分析样本筛选效果并保存结果
    Args:
        model: 训练完成的 DeLoRA 模型
        dataloader: 测试用的 DataLoader
        options: 包含参数配置的字典
        output_path: 结果输出路径
    """
    results = evaluate_sample_filtering(model, dataloader, options)

    # 打印结果
    print("=== 样本筛选效果评估 ===")
    print(f"模型预测为干净的样本总数: {results['model_predicted_clean_total']}")
    print(f"\n真实类别统计 (模型预测为干净的样本中):")
    print(f"ID-clean: {results['real_id_clean_count']} ({results['real_id_clean_ratio']:.2%})")
    print(f"ID-noisy: {results['real_id_noisy_count']} ({results['real_id_noisy_ratio']:.2%})")
    print(f"OOD: {results['real_ood_count']} ({results['real_ood_ratio']:.2%})")

    # 保存结果到文件
    # import json
    # with open(output_path, 'w') as f:
    #     json.dump(results, f, indent=2)
    # print(f"\n结果已保存到: {output_path}")

    return results


def compare_sample_selection(method1_indices_path, method2_indices_path, method1_name="Method 1",
                             method2_name="Method 2"):
    """
    比较两种样本筛选方法的重合度
    Args:
        method1_indices_path: 第一种方法保存的样本索引文件路径
        method2_indices_path: 第二种方法保存的样本索引文件路径
        method1_name: 第一种方法的名称 (默认"Method 1")
        method2_name: 第二种方法的名称 (默认"Method 2")
    Returns:
        overlap_results: 重合度分析结果字典
    """
    import json

    # 加载两种方法的样本索引
    with open(method1_indices_path, 'r') as f:
        method1_indices = set(json.load(f))

    with open(method2_indices_path, 'r') as f:
        method2_indices = set(json.load(f))

    # 计算统计指标
    total1 = len(method1_indices)
    total2 = len(method2_indices)

    # 交集：两种方法都选中的样本
    intersection = method1_indices.intersection(method2_indices)
    total_intersection = len(intersection)

    # 并集：至少被一种方法选中的样本
    union = method1_indices.union(method2_indices)
    total_union = len(union)

    # 计算重合度指标
    jaccard_coefficient = total_intersection / total_union if total_union > 0 else 0.0
    method1_recall_in_method2 = total_intersection / total1 if total1 > 0 else 0.0
    method2_recall_in_method1 = total_intersection / total2 if total2 > 0 else 0.0

    # 计算仅被一种方法选中的样本
    only_method1 = method1_indices - method2_indices
    only_method2 = method2_indices - method1_indices
    total_only_method1 = len(only_method1)
    total_only_method2 = len(only_method2)

    # 打印结果
    print(f"=== 样本筛选重合度分析 ===\n")
    print(f"{method1_name} 筛选样本数量: {total1}")
    print(f"{method2_name} 筛选样本数量: {total2}")
    print(f"\n两种方法共同筛选的样本数量: {total_intersection}")
    print(f"仅 {method1_name} 筛选的样本数量: {total_only_method1}")
    print(f"仅 {method2_name} 筛选的样本数量: {total_only_method2}")

    print(f"\nJaccard系数 (重合度): {jaccard_coefficient:.4f}")
    print(f"{method1_name} 筛选的样本在 {method2_name} 中的召回率: {method1_recall_in_method2:.4f}")
    print(f"{method2_name} 筛选的样本在 {method1_name} 中的召回率: {method2_recall_in_method1:.4f}")

    # 整理结果
    overlap_results = {
        method1_name: {
            'total_selected': total1,
            'only_selected': total_only_method1,
            'recall_in_other': method1_recall_in_method2
        },
        method2_name: {
            'total_selected': total2,
            'only_selected': total_only_method2,
            'recall_in_other': method2_recall_in_method1
        },
        'intersection': total_intersection,
        'union': total_union,
        'jaccard_coefficient': jaccard_coefficient
    }

    return overlap_results


def evaluate_top_k_sample_filtering(model, dataloader, cfg, K=64, save_selected_indices_path=None):
    """
    评估样本筛选效果：对每个类别取score1最大的K个样本作为clean样本
    Args:
        model: 训练完成的 DeLoRA 模型
        dataloader: 测试用的 DataLoader (要求shuffle=False以保证索引一致性)
        options: 包含参数配置的字典
        K: 每个类别选取的clean样本数量 (默认64)
        save_selected_indices_path: 保存筛选出的样本索引的文件路径 (可选)
    Returns:
        results: 评估结果字典
    """
    model.eval()

    # 存储所有样本的信息：(global_idx, class_label, score1, is_open, is_clean)
    sample_info = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):

            data = batch['data']
            labels = batch['label']  # 样本的类别标签
            real_indices = batch['index']  # 获取真实的全局索引
            # 访问真实类别信息
            is_open = batch['is_open']  # 1表示OOD，0表示ID
            is_clean = batch['is_clean']  # 1表示ID-clean，0表示ID-noisy + OOD
            data, labels = data.cuda(), labels.cuda()
            is_open = is_open.cuda()
            is_clean = is_clean.cuda()

            # 使用 logits_clean 区分干净/噪声样本
            # 获取干净 LoRA 的 logits
            logits_clean= model.get_lora_logits(data)

            # 解析logits_clean，与train_prompt_branch中的逻辑一致
            # logits.shape: [B, C, 1 + n_nega_ctx]
            n_nega_ctx = 2  # 获取负提示词数量，默认值为2
            logits = logits_clean.view(-1, int(logits_clean.shape[1] / (1 + n_nega_ctx)), 1 + n_nega_ctx)
            B, C = logits.shape[0], logits.shape[1]
            logits_yes = logits[:, :, 0]  # logits_yes: [B, C] - 正提示词的logits

            # 和train_prompt_branch中的logits_margin = logits_yes一致
            logits_margin = logits_yes  # 这里的logits_margin相当于用户要求的logits_clean
            T = 1
            prob_margin = torch.softmax(logits_margin / T, dim=1)  # [B, C]：所有类别的概率

            # # 计算score1：prob_margin的最大值 - MCM-style (和train_prompt_branch一致)
            # vals, idxs = prob_margin.topk(2, dim=1)  # vals: [B,2], idxs: [B,2]
            # # 如果第一大类就是 noisy label，则取第二大，否则取第一大
            # first_is_label = (idxs[:, 0] == labels)
            # score1 = torch.where(first_is_label, vals[:, 1], vals[:, 0])  # [B]
            score1 = prob_margin.max(dim=1).values
            # 计算score2：基于噪声标签的概率 - 用于筛选ID-clean样本（和train_prompt_branch一致）
            score2 = prob_margin[torch.arange(B), labels]  # [B]

            # 收集样本信息
            # 将所有数据转移到CPU进行处理
            class_labels = labels.cpu().numpy()
            score1_np = score1.cpu().numpy()
            score2_np = score2.cpu().numpy()
            is_open_np = is_open.cpu().numpy()
            is_clean_np = is_clean.cpu().numpy()
            real_indices_np = real_indices.cpu().numpy()  # 转移真实索引到CPU

            # 添加到样本信息列表
            for i in range(len(class_labels)):
                sample_info.append({
                    'global_idx': real_indices_np[i],  # 使用真实的全局索引
                    'class_label': class_labels[i],
                    'score1': score1_np[i],
                    'score2': score2_np[i],
                    'is_open': is_open_np[i],
                    'is_clean': is_clean_np[i]
                })

    # 按类别分组样本
    class_to_samples = {}
    for sample in sample_info:
        label = sample['class_label']
        if label not in class_to_samples:
            class_to_samples[label] = []
        class_to_samples[label].append(sample)

    # 样本筛选逻辑参考train_prompt_branch：
    # 1. 用score1选Top-K的ID样本
    # 2. 用score2选Top-K的ID-clean样本，覆盖ID样本
    # 默认每个类别选取K个样本

    import numpy as np

    # 保存筛选出的样本索引，按类型分类
    selected_indices = {
        'id_samples': [],  # ID样本（仅score1 Top-K，排除已选ID-clean）
        'id_clean_samples': [],  # ID-clean样本（仅score2 Top-K）
        'ood_samples': []  # OOD样本（暂时为空，按用户需求）
    }

    # 先筛选 ID-clean 样本：每个类别中 score2 最高的 K 个样本
    unique_labels = sorted(class_to_samples.keys())

    for label in unique_labels:
        samples = class_to_samples[label]
        if len(samples) == 0:
            continue

        # 按 score2 降序排序
        samples_sorted_by_score2 = sorted(samples, key=lambda x: x['score2'], reverse=True)

        # 取前 K 个作为 ID-clean
        id_clean_candidates = samples_sorted_by_score2[:K]

        # 保存索引
        for sample in id_clean_candidates:
            selected_indices['id_clean_samples'].append(sample['global_idx'])

    # 再筛选 ID 样本：每个类别中 score1 最高的 K 个样本，排除已选的 ID-clean 样本
    id_clean_set = set(selected_indices['id_clean_samples'])

    for label in unique_labels:
        samples = class_to_samples[label]
        if len(samples) == 0:
            continue

        # 过滤掉已选的 ID-clean 样本
        remaining_samples = [s for s in samples if s['global_idx'] not in id_clean_set]

        # 按 score1 降序排序
        remaining_samples_sorted_by_score1 = sorted(remaining_samples, key=lambda x: x['score1'], reverse=True)

        # 取前 K 个作为 ID
        id_candidates = remaining_samples_sorted_by_score1[:K]

        # 保存索引
        for sample in id_candidates:
            selected_indices['id_samples'].append(sample['global_idx'])

    # 统计真实分布（仅用于事后评估，不参与筛选逻辑）
    id_clean_set = set(selected_indices['id_clean_samples'])
    id_set = set(selected_indices['id_samples'])
    all_selected_set = id_clean_set | id_set

    total_top_k = len(all_selected_set)
    real_id_clean_count = 0
    real_id_noisy_count = 0
    real_ood_count = 0

    # 遍历所有样本，统计真实分布
    for sample in sample_info:
        if sample['global_idx'] not in all_selected_set:
            continue

        is_open = sample['is_open']
        is_clean = sample['is_clean']

        if is_open == 1:
            real_ood_count += 1
        else:
            if is_clean == 1:
                real_id_clean_count += 1
            else:
                real_id_noisy_count += 1

    # 重新创建 final_sample_type 用于评估函数
    N_total = len(sample_info)
    final_sample_type = np.full(N_total, -1, dtype=int)

    # 标记ID样本
    for idx in selected_indices['id_samples']:
        final_sample_type[idx] = 0

    # 标记ID-clean样本
    for idx in selected_indices['id_clean_samples']:
        final_sample_type[idx] = 1
    # 计算比例
    real_id_clean_ratio = real_id_clean_count / total_top_k if total_top_k > 0 else 0.0
    real_id_noisy_ratio = real_id_noisy_count / total_top_k if total_top_k > 0 else 0.0
    real_ood_ratio = real_ood_count / total_top_k if total_top_k > 0 else 0.0

    # 整理结果
    results = {
        'total_top_k_samples': total_top_k,
        'samples_per_class': K,
        'real_id_clean_count': real_id_clean_count,
        'real_id_clean_ratio': real_id_clean_ratio,
        'real_id_noisy_count': real_id_noisy_count,
        'real_id_noisy_ratio': real_id_noisy_ratio,
        'real_ood_count': real_ood_count,
        'real_ood_ratio': real_ood_ratio,
        'selected_indices': selected_indices  # 添加筛选出的样本索引
    }

    # 保存筛选出的样本索引
    if save_selected_indices_path is not None:
        import json
        with open(save_selected_indices_path, 'w') as f:
            json.dump(selected_indices, f, indent=2)
        print(f"\n筛选出的样本索引已保存到: {save_selected_indices_path}")

    model.train()

    # 调用样本选择准确性评估函数
    # print(f"Evaluating sample selection accuracy for epoch {cfg.get('epoch', 'N/A')}...")
    selection_accuracy = evaluate_sample_selection_accuracy(sample_info, final_sample_type)
    print(selection_accuracy)

    return results


def evaluate_sample_selection_accuracy(sample_info, final_sample_type=None):
    """
    评估样本筛选结果的准确性，计算真实ID-clean、ID、OOD的比例
    逻辑与analyze_per_class_our_method保持一致：
    1. 在筛选出来的样本中，看真实ID, OOD, ID-clean的比例
    2. 直接使用真实标签信息，不需要重新遍历数据集

    Args:
        sample_info: 从evaluate_top_k_sample_filtering中获取的样本信息列表
        final_sample_type (np.ndarray, optional): [N_total], 样本类型数组 (-1:未标记, 0:ID, 1:ID-clean, 2:OOD)
    Returns:
        accuracy_dict: 包含各类型样本比例的字典
    """
    import numpy as np

    if not sample_info:
        raise ValueError("sample_info must not be empty")

    # 统计样本总数
    N_total = len(sample_info)

    # 将样本信息转换为numpy数组以便处理
    global_indices = np.array([sample['global_idx'] for sample in sample_info])
    is_open_all = np.array([sample['is_open'] for sample in sample_info])  # 1=OOD, 0=ID
    is_clean_all = np.array([sample['is_clean'] for sample in sample_info])  # 1=ID-clean, 0=ID-noisy or OOD

    # 如果提供了final_sample_type，则使用它来确定样本类型
    if final_sample_type is not None:
        # 检查长度是否一致
        if len(final_sample_type) != N_total:
            raise ValueError(
                f"final_sample_type length ({len(final_sample_type)}) must match sample_info length ({N_total})")

        # 筛选出ID, ID-clean, OOD样本
        id_mask = (final_sample_type == 0)
        id_clean_mask = (final_sample_type == 1)
        ood_mask = (final_sample_type == 2)
    else:
        # 如果没有提供final_sample_type，默认使用所有样本
        id_mask = np.ones(N_total, dtype=bool)
        id_clean_mask = np.ones(N_total, dtype=bool)
        ood_mask = np.ones(N_total, dtype=bool)

    # --- 1. 统计ID样本（score1 top-k）---
    true_open_id = is_open_all[id_mask]  # 0=ID, 1=OOD
    true_clean_id = is_clean_all[id_mask]  # 0=not_clean, 1=ID-clean
    is_real_clean_id = (true_clean_id == 1) & (true_open_id == 0)
    is_real_noisy_id = (true_clean_id == 0) & (true_open_id == 0)
    is_real_ood_id = (true_open_id == 1)

    id_stats = {
        "total": id_mask.sum(),
        "clean": is_real_clean_id.sum(),
        "noisy": is_real_noisy_id.sum(),
        "ood": is_real_ood_id.sum()
    }

    # --- 2. 统计ID-clean样本（score2 top-k）---
    true_open_clean = is_open_all[id_clean_mask]  # 0=ID, 1=OOD
    true_clean_clean = is_clean_all[id_clean_mask]  # 0=not_clean, 1=ID-clean
    is_real_clean_clean = (true_clean_clean == 1) & (true_open_clean == 0)
    is_real_noisy_clean = (true_clean_clean == 0) & (true_open_clean == 0)
    is_real_ood_clean = (true_open_clean == 1)

    clean_stats = {
        "total": id_clean_mask.sum(),
        "clean": is_real_clean_clean.sum(),
        "noisy": is_real_noisy_clean.sum(),
        "ood": is_real_ood_clean.sum()
    }

    # --- 3. 统计OOD样本---
    ood_stats = None
    if ood_mask.any():
        true_open_ood = is_open_all[ood_mask]  # 0=ID, 1=OOD
        true_clean_ood = is_clean_all[ood_mask]  # 0=not_clean, 1=ID-clean
        is_real_clean_ood = (true_clean_ood == 1) & (true_open_ood == 0)
        is_real_noisy_ood = (true_clean_ood == 0) & (true_open_ood == 0)
        is_real_ood_ood = (true_open_ood == 1)

        ood_stats = {
            "total": ood_mask.sum(),
            "clean": is_real_clean_ood.sum(),
            "noisy": is_real_noisy_ood.sum(),
            "ood": is_real_ood_ood.sum()
        }

    # 计算比例
    def calc_ratios(stats):
        if stats["total"] == 0:
            return {
                "clean_ratio": 0.0,
                "noisy_ratio": 0.0,
                "ood_ratio": 0.0
            }
        return {
            "clean_ratio": stats["clean"] / stats["total"],
            "noisy_ratio": stats["noisy"] / stats["total"],
            "ood_ratio": stats["ood"] / stats["total"]
        }

    id_ratios = calc_ratios(id_stats)
    clean_ratios = calc_ratios(clean_stats)
    ood_ratios = calc_ratios(ood_stats) if ood_stats is not None else None

    # 构建返回字典
    result = {
        "id": {"stats": id_stats, "ratios": id_ratios},
        "id_clean": {"stats": clean_stats, "ratios": clean_ratios},
    }

    if ood_stats is not None:
        result["ood"] = {"stats": ood_stats, "ratios": ood_ratios}

    # 生成美观的格式化输出
    formatted_output = ""

    # 1. Per-Class Top-64 by Score1 (Marked as ID):
    formatted_output += "1. Per-Class Top-64 by Score1 (Marked as ID):\n"
    formatted_output += f"   - Total samples: {id_stats['total']}\n"
    formatted_output += f"   - True ID-clean: {id_stats['clean']} ({id_ratios['clean_ratio'] * 100:.2f}%)\n"
    formatted_output += f"   - True ID-noisy: {id_stats['noisy']} ({id_ratios['noisy_ratio'] * 100:.2f}%)\n"
    formatted_output += f"   - True OOD: {id_stats['ood']} ({id_ratios['ood_ratio'] * 100:.2f}%)\n\n"

    # 2. Per-Class Top-64 by Score2 (Marked as ID-clean):
    formatted_output += "2. Per-Class Top-64 by Score2 (Marked as ID-clean):\n"
    formatted_output += f"   - Total samples: {clean_stats['total']}\n"
    formatted_output += f"   - True ID-clean: {clean_stats['clean']} ({clean_ratios['clean_ratio'] * 100:.2f}%)\n"
    formatted_output += f"   - True ID-noisy: {clean_stats['noisy']} ({clean_ratios['noisy_ratio'] * 100:.2f}%)\n"
    formatted_output += f"   - True OOD: {clean_stats['ood']} ({clean_ratios['ood_ratio'] * 100:.2f}%)\n\n"

    # 如果有OOD样本统计，也显示出来
    if ood_stats is not None and ood_stats["total"] > 0:
        formatted_output += "3. Per-Class Top-k by Score3 (Marked as OOD):\n"
        formatted_output += f"   - Total samples: {ood_stats['total']}\n"
        formatted_output += f"   - True ID-clean: {ood_stats['clean']} ({ood_ratios['clean_ratio'] * 100:.2f}%)\n"
        formatted_output += f"   - True ID-noisy: {ood_stats['noisy']} ({ood_ratios['noisy_ratio'] * 100:.2f}%)\n"
        formatted_output += f"   - True OOD: {ood_stats['ood']} ({ood_ratios['ood_ratio'] * 100:.2f}%)\n\n"

    # 返回格式化的字符串而不是字典，这样调用者print时就会显示美观的格式
    return formatted_output



def my_test_visual_and_text(model_1, model_2,  testloader,):
    model_1.eval()
    model_2.eval()
    device = 'cuda'
    all_preds_visual = []  # 保存预测类别
    all_preds_text = []
    all_preds = []
    all_labels = []  # 保存真实标签
    all_is_open = []  # 保存是否 OOD (0=ID, 1=OOD)
    with torch.no_grad():
        torch.cuda.empty_cache()
        tqdm_object = tqdm(testloader, total=len(testloader))

        for batch in testloader:
            images = batch['data'].to(device)  # [B, C, H, W]
            clean_labels = batch.get('clean_label', None)
            is_open = batch['is_open']
            if clean_labels is not None:
                clean_labels = clean_labels.to(device)

            ## 视觉端 logits -> probs -> reshape -> positive-context slice
            logits_visual = model_1.get_lora_logits(images)  # 假设返回 [B, C*(1+n_nega_ctx)]
            n_nega_ctx = 2
            probs_visual = torch.softmax(logits_visual, dim=1)  # [B, C*(1+n_nega_ctx)]
            B = probs_visual.shape[0]
            C_total = probs_visual.shape[1]
            C = int(C_total / (1 + n_nega_ctx))
            probs_visual = probs_visual.view(B, C, 1 + n_nega_ctx)  # [B, C, 1+n_nega_ctx]
            probs_pos_visual = probs_visual[:, :, 0]  # [B, C] 正提示对应的概率

            # predictions_visual: long indices [B], score_visual: float probs [B]
            score_visual, predictions_visual = probs_pos_visual.max(dim=1)  # both shape [B]

            ## 文本端 logits -> probs -> reshape -> positive-context slice
            logits_text = model_2.get_original_logits(images)
            probs_text = torch.softmax(logits_text, dim=1)
            C_total_t = probs_text.shape[1]
            C_t = int(C_total_t / (1 + n_nega_ctx))
            probs_text = probs_text.view(B, C_t, 1 + n_nega_ctx)
            probs_pos_text = probs_text[:, :, 0]  # [B, C]

            score_text, predictions_text = probs_pos_text.max(dim=1)  # both shape [B]

            # ---- 逐样本决策掩码 ----
            # use_text_mask[i] == True 表示第 i 个样本采用 text 端预测，否则采用 visual 端
            use_text_mask = (score_text > score_visual)  # boolean tensor shape [B]

            # 向量化方式：并行选择最终预测与最终置信度
            final_predictions = torch.where(use_text_mask, predictions_text, predictions_visual)

            # all_preds_visual.append(predictions_visual.detach().cpu().numpy())
            # all_preds_text.append(predictions_text.detach().cpu().numpy())
            all_preds.append(final_predictions.detach().cpu().numpy())
            all_labels.append(clean_labels.cpu().numpy())
            all_is_open.append(is_open.cpu().numpy())

        all_is_open = np.concatenate(all_is_open)
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

        id_mask = (all_is_open == 0)  # 只看ID样本
        precision = precision_score(all_labels[id_mask], all_preds[id_mask], average='macro')
        recall = recall_score(all_labels[id_mask], all_preds[id_mask], average='macro')
        f1 = f1_score(all_labels[id_mask], all_preds[id_mask], average='macro')
        acc = accuracy_score(all_labels[id_mask], all_preds[id_mask])
        print("=" * 60)
        print("ID Classification: Acc=%.4f, P=%.4f, R=%.4f, F1=%.4f" % (acc, precision, recall, f1))
        print('=' * 60)
        results = {
            'ID': {
                'acc': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            },

        }
    return results


def my_test_visual_and_text_v2(model_1, model_2, testloader):
    model_1.eval()
    model_2.eval()
    device = 'cuda'
    all_preds_visual = []  # 保存预测类别
    all_preds_text = []
    all_labels = []  # 保存真实标签
    all_is_open = []  # 保存是否 OOD (0=ID, 1=OOD)
    with torch.no_grad():
        torch.cuda.empty_cache()

        for batch in testloader:
            images = batch['data'].to(device)  # [B, C, H, W]
            clean_labels = batch.get('clean_label', None)
            is_open = batch['is_open']
            if clean_labels is not None:
                clean_labels = clean_labels.to(device)

            ## 视觉端 logits -> probs -> reshape -> positive-context slice
            logits_visual = model_1.get_lora_logits(images)
            n_nega_ctx = 2
            probs_visual = torch.softmax(logits_visual, dim=1)
            B = probs_visual.shape[0]
            C_total = probs_visual.shape[1]
            C = int(C_total / (1 + n_nega_ctx))
            probs_visual = probs_visual.view(B, C, 1 + n_nega_ctx)
            probs_pos_visual = probs_visual[:, :, 0]

            score_visual, predictions_visual = probs_pos_visual.max(dim=1)

            ## 文本端 logits -> probs -> reshape -> positive-context slice
            logits_text = model_2.get_original_logits(images)
            probs_text = torch.softmax(logits_text, dim=1)
            C_total_t = probs_text.shape[1]
            C_t = int(C_total_t / (1 + n_nega_ctx))
            probs_text = probs_text.view(B, C_t, 1 + n_nega_ctx)
            probs_pos_text = probs_text[:, :, 0]

            score_text, predictions_text = probs_pos_text.max(dim=1)

            # ---- 计算至少一个预测正确的 acc ----
            # 对每个样本，只要 text 或 visual 正确就算正确
            final_correct = ((predictions_text == clean_labels) | (predictions_visual == clean_labels))

            all_preds_text.append(predictions_text.detach().cpu().numpy())
            all_preds_visual.append(predictions_visual.detach().cpu().numpy())
            all_labels.append(clean_labels.cpu().numpy())
            all_is_open.append(is_open.cpu().numpy())

            # 可选：记录每个 batch 的最终 correct
            # final_correct.cpu().numpy()  # 布尔数组

        # 拼接所有 batch
        all_labels = np.concatenate(all_labels)
        all_is_open = np.concatenate(all_is_open)
        all_preds_text = np.concatenate(all_preds_text)
        all_preds_visual = np.concatenate(all_preds_visual)

        # 只统计 ID 样本
        id_mask = (all_is_open == 0)
        # 至少一个预测正确的样本
        final_correct_id = ((all_preds_text[id_mask] == all_labels[id_mask]) |
                            (all_preds_visual[id_mask] == all_labels[id_mask]))
        acc = final_correct_id.mean()  # True=1, False=0

        print("=" * 60)
        print("ID Classification (text OR visual correct): Acc=%.4f" % acc)
        print("=" * 60)

        results = {
            'ID': {
                'acc': acc
            }
        }
    return results
