# CoGBD: A Consistency-Guided Framework for Universal Graph Backdoor Defense

## Required Dependencies

Please make sure the following dependencies are installed before running the code:

- **Python** >= 3.9
- **Pytorch** >= 2.7.1
- **torch_geometric** >= 2.6.1
- **torch-scatter** >= 2.1.2
- **torch-sparse** >= 0.6.18
- **torch-cluster** >= 1.6.3
- **torch-spline-conv** >= 1.2.2
- **scikit-learn** == 1.6.1
- **numpy** == 1.26.3
- **ogb** == 1.3.6
- **deeprobust** == 0.2.11

## Quick Start

To reproduce the paper experiments directly, run:

```bash
bash script/run.sh
```

---

### Run on SPEAR

**Cora:**

```bash
python -u main.py \
  --top_num 100 --weight_ood_view 0.9 \
  --dataset Cora --trigger SPEAR \
  --homo_loss_weight 0.1 --target_loss_weight 1 \
  --vs_number 40 --test_model GCN --defense_mode none \
  --epochs 200 --trojan_epochs 200 \
  --device_id 5 --alpha_int 30 --hidden 32 \
  --ae_lr 0.0002 --shadow_lr 0.0002 --trojan_lr 0.0002 \
  --unlearn_mode entropy \
  --a 0 --b 0 --conf_tau 0.1 --lambda_unlearn 0 \
  >> results/SPEAR_Cora.out
```

**ogbn-arxiv:**

```bash
python -u main.py \
  --top_num 650 --weight_ood_view 0.9 \
  --dataset ogbn-arxiv --trigger SPEAR \
  --homo_loss_weight 0 --target_loss_weight 1 \
  --vs_number 565 --test_model GCN --defense_mode none \
  --epochs 800 --trojan_epochs 800 \
  --device_id 5 --alpha_int 5 --hidden 80 --outter_size 256 \
  --ae_lr 0.001 --shadow_lr 0.001 --trojan_lr 0.001 \
  --a 1.0 --b 0.5 --conf_tau 0.5 --lambda_unlearn 0.1 \
  --unlearn_mode entropy \
  >> results/SPEAR_OGB.out
```

---

### Run on UGBA

**ogbn-arxiv:**

```bash
python -u main.py \
  --top_num 1000 --weight_ood_view 0.5 \
  --trigger UGBA \
  --homo_loss_weight 200 --homo_boost_thrd 0.8 --target_loss_weight 1 \
  --inner 5 --dataset ogbn-arxiv --vs_number 565 \
  --test_model GCN --defense_mode none \
  --epochs 800 --trojan_epochs 800 \
  --device_id 4 --hidden 64 --outter_size 256 \
  --ae_lr 0.001 --shadow_lr 0.001 --trojan_lr 0.001 \
  --a 1.0 --b 1.0 --conf_tau 1.0 --lambda_unlearn 0.1 \
  --unlearn_mode entropy \
  >> results/UGBA_ogbn-arxiv.out
```

---

### Run on GTA

**ogbn-arxiv:**

```bash
python -u main.py \
  --top_num 250 --weight_ood_view 0.5 \
  --trigger GTA --dataset ogbn-arxiv \
  --inner 1 --vs_number 565 \
  --test_model GCN --defense_mode none \
  --epochs 800 --trojan_epochs 800 \
  --device_id 4 --hidden 32 \
  --train_lr 0.05 --shadow_lr 0.05 --trojan_lr 0.05 \
  --a 4 --b 0.125 --conf_tau 1.0 --lambda_unlearn 0.1 \
  --unlearn_mode entropy \
  >> results/GTA_ogbn-arxiv.out
```

---

### Run on DPGBA

**Cora:**

```bash
python -u main.py \
  --weight_ood_view 0.1 --top_num 100 --prune_thr 0.1 \
  --trigger DPGBA --dataset Cora --target_class 2 \
  --k 50 --defense_mode none --vs_number 40 --inner 1 \
  --hidden 64 \
  --ae_lr 0.05 --train_lr 0.05 --shadow_lr 0.05 --trojan_lr 0.05 --detector_lr 0.05 \
  --test_model GCN \
  --weight_target 1 --weight_targetclass 3 --weight_ood 1 \
  --epochs 200 --range 1.0 --trojan_epochs 301 --outter_size 4096 \
  --device_id 4 \
  --a 8 --b 1 --conf_tau 0.1 --lambda_unlearn 2.0 \
  --unlearn_mode entropy \
  >> results/DPGBA_Cora.out
```

---

## Notes

- `--trigger` selects the attack: `SPEAR` | `UGBA` | `GTA` | `DPGBA` | `GCBA`.
- CoGBD defense hyperparameters: `--a`, `--b`, `--conf_tau`, `--lambda_unlearn`, `--ae_thr`, `--rec_epochs`.
