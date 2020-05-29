#  Growing Neural Cellular Automata Model

This is a model that learn to generate an image from one initial pixel . This model is based to  the way that real multi-cellular organisms growth. Based to [this|https://distill.pub/2020/growing-ca] paper.

## Setup

**Step 1**: Clone repository.

```bash
git clone https://github.com/adrianmarino/growing-neural-cellular-automata.git
```

**Step 2**: Create project environment.

```bash
conda env create --file environment.yml
```

## Test model

**Step 1**: Activate environment.

```bash
conda activate ca-growth
```

**Step 1**: Test an already trained config

```bash
./ca-growth --action test --config-name lizard-16x16
```

## Training a model

**Step 1**: Activate environment.

```bash
conda activate ca-growth
```

**Step 1**: Test an already trained config

```bash
./ca-growth --action train --config-name lizard-16x16
```