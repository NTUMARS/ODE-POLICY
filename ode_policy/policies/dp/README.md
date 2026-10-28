# Diffusion Policy

## 1. Install

```bash
cd ode_policy/policies/dp
pip install -r requirements.txt
```

Register for a Weights & Biases (wandb) account to obtain an API key.

## 2. Collect and process data

```bash
./ode_policy/collect_demo.sh
```

## 3. Train and eval

```bash
./ode_policy/policies/dp/dp_run.sh
```

### 3.1 Train only

```bash
train_enable=True
eval_enable=False
```

### 3.2 Eval only

```bash
train_enable=False
eval_enable=True
```
