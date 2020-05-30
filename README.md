#  Growing Neural Cellular Automata Model

This is a model that learn to generate an image from one initial pixel. This model is based to the way that real multi-cellular organisms growth. Based to [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca) paper.


### Demo video

<p align="center">
    <a href="http://www.youtube.com/watch?v=lqLZOWkb81Q" target="_tab"/>
    <img src="http://img.youtube.com/vi/lqLZOWkb81Q/0.jpg" 
        title="Click to see video" 
        alt="Click to see video"/>
    </a>
</p>


### Requeriments

* [anaconda](https://www.anaconda.com/download/#linux)

### Setup

**Step 1**: Clone repository.

```bash
git clone https://github.com/adrianmarino/growing-neural-cellular-automata.git
```

**Step 2**: Create project environment.

```bash
conda env create --file environment.yml
```

### See all `ca-growth` parameters

```bash
./ca-growth --help

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Configuration file name
  --action {train,test}
                        Specify train or test model
  --show-output         Show output evolution
  --hide-output         Hide output evolution
  --show-loss-graph     Show loss graph
  --hide-loss-graph     Hide loss graph
```

### Test model

**Step 1**: Activate environment.

```bash
conda activate ca-growth
```

**Step 1**: Test an already trained config.

```bash
./ca-growth --action test --config lizard-16x16
```

### Training a model

**Step 1**: Activate environment.

```bash
conda activate ca-growth
```

**Step 2**: Train model.

```bash
./ca-growth --action train --config lizard-16x16
```


### Colab

To run training or test from console in colab you must hide graphs like next:


```bash
ca-growth --action train \
    --config lizard-16x16 \
    --hide-output \
    --hide-loss-graph
```
