#!/usr/bin/env python
"""
Stage2 merged model co-teaching training script
This script implements alternating training of visual (LoRA) and text (prompt) branches for stage2 merged model
"""
import os
import argparse
import logging
import pdb
import pickle
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F

# Add project root to path
import sys
sys.path.append("/data/tyang/aaaa_now/compare")

# Import necessary modules from existing codebase
from models.models import NegaPromptCLIP
from data.cifar100 import build_dataset_loader, class_names
from core import my_test_nega_clip, my_test_lora, my_test_clip, my_test_noOOD
from open_clip import create_model_and_transforms

# Import new joint training utilities
from new.joint_training_utils import (
    unified_sample_selection,
    train_joint_text_branch,
    train_joint_visual_branch,
    negative_prompt_training,
    train_joint_all
)
from data.stanford.my_stanford_cars import *

# Import attention guided loss related modules (optional - decoupled for easy removal)
try:
    from new.attention_guided_loss import AttentionGuidedLoss
    from new.attention_guided_branches import modified_train_prompt_branch, modified_train_lora_branch

    ATTENTION_LOSS_AVAILABLE = False
except ImportError as e:
    ATTENTION_LOSS_AVAILABLE = False
    logging.warning(f"Attention-guided loss modules not found: {e}. Attention loss will be disabled.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
# os.environ["WANDB_MODE"] = "disabled"



def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Stage2 merged model co-teaching training")

    # Core parameters matching Fine-tuning.py
    parser.add_argument("--dataset", type=str, default="cifar100", help="Dataset name")
    parser.add_argument("--clip_backbone", type=str, default="ViT-B/16", help="CLIP backbone model")
    parser.add_argument("--max_epoch", type=int, default=200, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Stage2 specific parameters
    parser.add_argument('--delora', action='store_true', help='Enable DeLoRA training (optional)')
    parser.add_argument('--partial', type=int, default=1,
                        help='Number of top transformer layers to apply PEFT (default: all layers)')
    parser.add_argument('--rank', type=int, default=2, help='rank of LoRA')

    parser.add_argument("--closeset-ratio", type=float, default=0.8, help="Close set ratio")
    parser.add_argument("--id_noise_ratio", type=float, default=0.2)
    parser.add_argument("--ood_ratio", type=float, default=0.2)
    parser.add_argument("--noise_type", type=str, default='symmetric')
    parser.add_argument("--NEGA_CTX", type=int, default=2, help="Number of negative contexts")
    # parser.add_argument("--pretrained-model", type=str, default="/data/tyang/checkpoints/new_sym0.8/merge_new_stage1.pth", help="Path to pre-trained model weights (cross-entropy warmed up)")
    # 12-17
    # parser.add_argument("--pretrained-model", type=str, default=f"/data/tyang/checkpoints/cifar80N/sym0.8_lr0.01.pth", help="Path to pre-trained model weights (cross-entropy warmed up)")

    # parser.add_argument("--pretrained-model", type=str, default=f"/data/tyang/checkpoints/crop_cifar100/id0.2_ood0.2/lr=0.01_20251229_165910.pth", help="Path to pre-trained model weights (cross-entropy warmed up)")
    # parser.add_argument("--pretrained-model", type=str, default=f"/data/tyang/checkpoints/crop_cifar100/id0.4_ood0.2/lr=0.01_20251229_170610.pth", help="Path to pre-trained model weights (cross-entropy warmed up)")
    # parser.add_argument("--pretrained-model", type=str, default=f"/data/tyang/checkpoints/crop_cifar100/id0.2_ood0.6/lr=0.01_20251230_220408.pth", help="Path to pre-trained model weights (cross-entropy warmed up)")
    # parser.add_argument("--pretrained-model", type=str, default=f"/data/tyang/checkpoints/crop_cifar100/id0.4_ood0.4/lr=0.01_20251231_112643.pth", help="Path to pre-trained model weights (cross-entropy warmed up)")
    # parser.add_argument("--pretrained-model", type=str, default=f"/data/tyang/checkpoints/place365/id0.2_ood0.2/lr=0.01_20260102_214628.pth", help="Path to pre-trained model weights (cross-entropy warmed up)")
    # parser.add_argument("--pretrained-model", type=str, default=f"/data/tyang/checkpoints/place365/id0.2_ood0.4/lr=0.01_20260103_162136.pth", help="Path to pre-trained model weights (cross-entropy warmed up)")
    # parser.add_argument("--pretrained-model", type=str, default=f"/data/tyang/checkpoints/place365/id0.2_ood0.6/lr=0.01_20260103_213058.pth", help="Path to pre-trained model weights (cross-entropy warmed up)")
    # parser.add_argument("--pretrained-model", type=str, default=f"/data/tyang/checkpoints/place365/id0.4_ood0.4/lr=0.01_20260104_114251.pth", help="Path to pre-trained model weights (cross-entropy warmed up)")
    # parser.add_argument("--pretrained-model", type=str, default=f"/data/tyang/checkpoints/WebFG/web-aircraft/lr=0.01_20260110_212438.pth", help="Path to pre-trained model weights (cross-entropy warmed up)")


    #1-28
    parser.add_argument("--pretrained-model", type=str, default=f"/data/tyang/checkpoints/cifar80N/sym0.2/lr=0.01_20251223_093556.pth", help="Path to pre-trained model weights (cross-entropy warmed up)")


    # Attention guided loss parameters (optional)
    if ATTENTION_LOSS_AVAILABLE:
        parser.add_argument("--attention_temperature", type=float, default=1.0, help="Temperature for soft attention mask generation")
        parser.add_argument("--attention_margin", type=float, default=0.05, help="Margin for guidance decision")

    # load parameters
    args = parser.parse_args()


    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)


    # Load dataset
    logger.info(f"Loading {args.dataset} dataset with closeset_ratio={args.closeset_ratio}...")
    # 1217
    # CIFAR80N

    # StanfordCars

    if args.dataset == "cifar80N":
        from data.cifar100 import build_dataset_loader, class_names

        _, trainloader, testloader = build_dataset_loader(
            closeset_ratio=args.closeset_ratio,
            batch_size=args.batch_size,
            noise_type=args.noise_type
        )
        classnames = class_names[:80]
    elif args.dataset == "stanford_cars":
        trainloader, testloader = get_stanford_cars(
        batch_size=args.batch_size,
        num_shots=16,
        noise_type='symmetric',
        noise_rate=args.closeset_ratio,
        seed=42
        )
        classnames = get_stanford_cars_classnames()
    elif args.dataset == "crop_cifar100":
        from data.crop_cifar100.mixed_cifar100_imagenet32 import build_dataset_loader, class_names, ID_class_names
        _, trainloader, testloader = build_dataset_loader(args.batch_size, id_noise_ratio=args.id_noise_ratio, ood_noise_ratio=args.ood_ratio)
        classnames = class_names
    elif args.dataset == "place365":
        from data.place365_cifar100.extended_mixed_dataset import get_cifar100_loaders
        import torchvision.transforms as transforms
        from data.crop_cifar100.mixed_cifar100_imagenet32 import class_names, ID_class_names
        cifar_path = "/data/tyang/aaaa_now/compare/data/cifar100"
        place365_path = "/data/tyang/aaaa_now/compare/data/place365"
        trainloader, testloader = get_cifar100_loaders(
            data_root=cifar_path,
            place365_root=place365_path,
            batch_size=256,
            ood_ratio=args.ood_ratio,  # 您想要的参数
            id_noise_ratio=args.id_noise_ratio,  # 您想要的参数
            num_workers=4
        )
        classnames = class_names
    else:
        from data.webfg import build_webfg_dataloader, get_webfg_class_names
        dataset_name = args.dataset

        # 创建训练集和测试集加载器
        dataset, trainloader, testloader = build_webfg_dataloader(
            dataset_name,
            batch_size=512,
            num_workers=8
        )
        classnames = get_webfg_class_names(dataset_name)

    # Initialize original model
    logger.info(f"Initializing stage2 NegaPromptCLIP with backbone {args.clip_backbone}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, _, preprocess = create_model_and_transforms(
        'ViT-B-16',
        pretrained='/data/tyang/aaaa_now/.cache/huggingface/hub/models--timm--vit_base_patch16_clip_224.openai/snapshots/977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d/open_clip_pytorch_model.bin'
    )
    id_num = 160
    clean_num = 75
    ood_num= 14000

    id_num = 30
    clean_num = 30
    ood_num = 1200
    cfg = {
        'NEGA_CTX': args.NEGA_CTX,
        'stage': 2,  # Stage2
        'delora': args.delora,
        'N_CTX': 16,  # Number of trainable prompt tokens
        'CTX_INIT': 'a photo of a "{}"',
        'CSC': 0,
        'batch_size': args.batch_size,
        'rank': args.rank,
        'partial': args.partial,
        'id_num': id_num,
        'clean_num': clean_num,
        'ood_num': ood_num,
    }
    clip_model.dtype = torch.float32
    clip_model.visual.input_resolution = 224
    clip_model.to(device)
    # Freeze base CLIP model parameters
    for params in clip_model.parameters():
        params.requires_grad_(False)
    model = NegaPromptCLIP(cfg, classnames, clip_model).to(device)

    model.get_ctx_posi(model.prompt_learner.ctx_positive)



    # Load pre-trained model weights if provided
    if args.pretrained_model is not None and os.path.exists(args.pretrained_model):
        logger.info(f"Loading pre-trained model weights from: {args.pretrained_model}")
        try:
            checkpoint = torch.load(args.pretrained_model, map_location=device)

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)

            logger.info("✓ Pre-trained weights loaded successfully!")
        except Exception as e:
            logger.warning(f"Warning: Failed to load pre-trained weights: {e}")
    elif args.pretrained_model is not None:
        logger.warning(f"Warning: Pre-trained model path not found: {args.pretrained_model}")
        exit(0)



    # Initialize optimizers - separate for visual and text branches
    logger.info("Initializing separate optimizers for visual and text branches...")

    # 遍历image_encoder的所有参数
    lora_params = []
    for name, p in model.named_parameters():
        # 选择所有LoRA相关参数，包括"A_clean", "B_clean"
        if "A_clean" in name or "B_clean" in name:
            lora_params.append(p)

    lora_optimizer = torch.optim.SGD(lora_params, lr=5e-4, momentum=0.9)  # Match original settings
    lora_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lora_optimizer, T_max=50, eta_min=2e-5)

    # Text branch optimizer: prompt parameters only

    positive_params = [p for name, p in model.named_parameters() if "prompt_learner.ctx_positive" in name]
    negative_prompt_params = [p for name, p in model.named_parameters() if "prompt_learner.ctx_negative" in name]

    positive_prompt_optimizer = torch.optim.SGD(positive_params, lr=1e-3, momentum=0.9)  # Match original settings
    positive_prompt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(positive_prompt_optimizer, T_max=80, eta_min=5e-4)

    negative_prompt_optimizer = torch.optim.SGD(negative_prompt_params, lr=1e-3, momentum=0.9)  # Match original settings
    negative_prompt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(negative_prompt_optimizer, T_max=80, eta_min=5e-4)

    for name, p in model.named_parameters():
        if "prompt_learner.ctx_positive" in name:
            print(name)
        if "prompt_learner.ctx_negative" in name:
            print(name)

    # Initialize best accuracy trackers
    best_acc_visual = -1

    # Initialize model state storage for attention loss calculation
    # 只保存需要的模型状态，每个状态仅使用一次
    model_states = {}  # 存储不同阶段的模型状态，用于计算score1差值
    # 保存初始导入的预热模型状态（epoch -1）
    import copy
    model_states[-1] = copy.deepcopy(model.state_dict())

    # Initialize attention guided loss (if available)
    attention_guided_loss = None
    if ATTENTION_LOSS_AVAILABLE:
        logging.info("Initializing AttentionGuidedLoss...")
        attention_guided_loss = AttentionGuidedLoss(
            temperature=args.attention_temperature,
            margin=args.attention_margin
        )
        logging.info("✓ AttentionGuidedLoss initialized successfully!")

    # 测试一下模型，看看现在的权重的效果
    results_visual = my_test_nega_clip(model, testloader) # cifar80N才有的测试OOD 测试的结果。
    # results_visual = my_test_noOOD(model, testloader)
    best_acc_visual = results_visual['ID']['acc']

    #得到模型的划分结果：
    single_split_dict = None
    single_split_dict = unified_sample_selection(model, trainloader, cfg, epoch=-1)

    exit(0)
    # with open('split_dict.pkl', 'wb') as f:
    #     pickle.dump(single_split_dict, f)
    # with open('split_dict.pkl', 'rb') as f:
    #     single_split_dict = pickle.load(f)

    # for name, params in model.named_parameters():
    #     if params.requires_grad:
    #         print(name)

    ##### initialize wandb
    import wandb
    run = wandb.init(project="crop_cifar100_with_imagenet32", dir='.', reinit=True, name=f"id{args.id_noise_ratio}_ood_{args.ood_ratio}",
        config={
        "dataset": args.dataset,
        "noise_type": "symmetric",
        "noise_rate": args.closeset_ratio,
        "id_num": id_num,
        "clean_num": clean_num,
        "ood_num": ood_num
    })
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = f"/data/tyang/checkpoints/place365/id{args.id_noise_ratio}_ood{args.ood_ratio}/acc_best_{timestamp}_s2.pth"

    # ==========================================
    # Modification 1: 定义联合优化器 (Joint Optimizer)
    # ==========================================

    # 收集所有需要更新的参数
    # 1. Visual LoRA 参数
    lora_params = [p for name, p in model.named_parameters() if 'lora' in name]  # 假设lora参数名包含lora，具体根据你实际情况
    # 2. Positive Prompt 参数
    positive_params = [p for name, p in model.named_parameters() if "prompt_learner.ctx_positive" in name]
    # 3. Negative Prompt 参数
    negative_prompt_params = [p for name, p in model.named_parameters() if "prompt_learner.ctx_negative" in name]

    # 定义优化器，使用参数组区分学习率
    # Visual: 5e-4, Prompts: 1e-3 (保持和你原文一致)
    joint_optimizer = torch.optim.SGD([
        {'params': lora_params, 'lr': 5e-4},
        {'params': positive_params, 'lr': 1e-3},
        {'params': negative_prompt_params, 'lr': 1e-3}
    ], momentum=0.9)

    # 定义联合调度器
    joint_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        joint_optimizer, T_max=args.max_epoch, eta_min=2e-5
    )
    # start training
    for epoch in range(args.max_epoch):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Epoch {epoch}/{args.max_epoch} - JOINT TRAINING MODE")
        logger.info(f"{'=' * 60}")

        # ==========================================
        # Modification 3: 移除 Cycle 逻辑，统一训练
        # ==========================================

        # 每一轮都训练所有参数
        avg_loss, _ = train_joint_all(
            model=model,
            optimizer=joint_optimizer,  # 使用新的联合优化器
            scheduler=joint_scheduler,
            trainloader=trainloader,
            run=run,
            cfg=cfg,
            epoch=epoch,
            sample_indices=single_split_dict,
            model_states=model_states,
            classnames=classnames
        )

        # Test logic (保持不变)
        results = my_test_noOOD(model, testloader)
        run.log({"acc": results['ID']['acc']})

        if results['ID']['acc'] > best_acc_visual:
            best_acc_visual = results['ID']['acc']
            logger.info(f"  New best accuracy: {best_acc_visual:.4f}")
            print(path)
            torch.save(model.state_dict(), path)

        logger.info(f"  Joint training epoch complete. Avg Loss: {avg_loss:.4f}")


    exit(0)



    # start training
    for epoch in range(args.max_epoch):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{args.max_epoch}")
        logger.info(f"{'='*60}")

        epoch_cycle = epoch % 6  # 6 epoch cycle

        # --------------------------
        # Epochs 0-1: Train prompts with negative_prompt_training
        # --------------------------
        if epoch_cycle < 2:

            logger.info(f"Epoch {epoch} ({epoch_cycle}/6 in cycle): Training prompts with negative_prompt_training")
            logger.info(f"  Using unified sample partition")

            avg_loss_text, _ = negative_prompt_training(model, trainloader, cfg, epoch, single_split_dict,
                                                     negative_prompt_optimizer, negative_prompt_scheduler)

            # Test prompt branch performance
            # results = my_test_nega_clip(model, testloader)
            results = my_test_noOOD(model, testloader)
            # results = my_test_noOOD(model, testloader)
            run.log({"acc": results['ID']['acc']})
            # run.log({"auroc": results['OOD']['auroc']})
            if results['ID']['acc'] > best_acc_visual:
                best_acc_visual = results['ID']['acc']
                logger.info(f"  New best accuracy: {best_acc_visual:.4f}")

                print(path)
                torch.save(model.state_dict(), path)

            logger.info(f"  Prompt training complete. Avg Loss: {avg_loss_text:.4f}")

        # --------------------------
        # Epochs 2-3: Train prompts with train_joint_text_branch
        # --------------------------
        elif epoch_cycle < 4:
            logger.info(f"Epoch {epoch} ({epoch_cycle}/6 in cycle): Training prompts with train_joint_text_branch")
            logger.info(f"  Using unified sample partition")

            avg_loss_text, _ = train_joint_text_branch(
                model=model,
                optimizer=positive_prompt_optimizer,
                scheduler=positive_prompt_scheduler,
                trainloader=trainloader,
                run=None,  # No wandb logging by default
                cfg=cfg,
                epoch=epoch,
                sample_indices=single_split_dict,  # Single unified partition
                model_states=model_states,  # 传递模型状态用于注意力损失计算
                classnames=classnames,
            )

            # Test prompt branch performance
            # results = my_test_nega_clip(model, testloader)
            results = my_test_noOOD(model, testloader)
            run.log({"acc": results['ID']['acc']})
            # run.log({"auroc": results['OOD']['auroc']})
            if results['ID']['acc'] > best_acc_visual:
                best_acc_visual = results['ID']['acc']
                logger.info(f"  New best accuracy: {best_acc_visual:.4f}")

                torch.save(model.state_dict(), path)

            logger.info(f"  Joint text branch training complete. Avg Loss: {avg_loss_text:.4f}")

        # --------------------------
        # Epochs 4-5: Train LoRA with train_joint_visual_branch
        # --------------------------
        else:
            logger.info(f"Epoch {epoch} ({epoch_cycle}/6 in cycle): Training LoRA with train_joint_visual_branch")
            logger.info(f"  Using unified sample partition")
            # Train visual branch using joint training function with forward_joint_warmup
            avg_loss_visual, _ = train_joint_visual_branch(
                model=model,
                optimizer=lora_optimizer,
                scheduler=lora_scheduler,
                trainloader=trainloader,
                run=None,  # No wandb logging by default
                cfg=cfg,
                epoch=epoch,
                sample_indices=single_split_dict,  # Single unified partition
                model_states=model_states,  # 传递模型状态用于注意力损失计算
                classnames=classnames
            )

            # results = my_test_nega_clip(model, testloader)
            results = my_test_noOOD(model, testloader)
            run.log({"acc": results['ID']['acc']})
            # run.log({"auroc": results['OOD']['auroc']})
            if results['ID']['acc'] > best_acc_visual:
                best_acc_visual = results['ID']['acc']
                logger.info(f"  New best visual branch accuracy: {best_acc_visual:.4f}")

                print(path)
                torch.save(model.state_dict(), path)

            logger.info(f"  LoRA training complete. Avg Loss: {avg_loss_visual:.4f}")

        # ---- 保存模型状态用于attention loss计算 ----
        import copy
        # 保存text训练结束时的模型状态（每个循环的第4个epoch结束后：epoch3, epoch9, epoch15...）
        if epoch % 6 == 3:
            # 清除之前的epoch3状态，只保留最新的
            old_epoch3 = epoch - 6
            if old_epoch3 in model_states:
                del model_states[old_epoch3]
            # 保存当前epoch3状态
            model_states[epoch] = copy.deepcopy(model.state_dict())
            logging.info(f"✓ 保存模型状态 (epoch {epoch}) 用于后续visual端attention loss计算")

        # 保存visual训练结束时的模型状态（每个循环的第6个epoch结束后：epoch5, epoch11, epoch17...）
        elif epoch % 6 == 5:
            # 清除之前的epoch5状态，只保留最新的
            old_epoch5 = epoch - 6
            if old_epoch5 in model_states:
                del model_states[old_epoch5]
            # 保存当前epoch5状态
            model_states[epoch] = copy.deepcopy(model.state_dict())
            logging.info(f"✓ 保存模型状态 (epoch {epoch}) 用于后续text端attention loss计算")

        # Recompute unified partition after every N epochs (optional: can be removed if not needed)
        if (epoch + 1) % 1 == 0:
            logger.info(f"\nRecomputing unified sample partition at epoch {epoch + 1}...")
            single_split_dict = unified_sample_selection(model, trainloader, cfg, epoch=epoch)


    print(f"\n{'=' * 60}")
    print(f"Best Visual (LoRA) Branch Accuracy: {best_acc_visual:.4f}")

if __name__ == "__main__":
    main()
