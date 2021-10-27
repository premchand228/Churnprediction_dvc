# DVC - DL - TF - AIOPS demo

download data --> [source](Churn Data freely avilable
)

## commands - 

### create a new env
```bash
conda create --prefix ./env python=3.7 -y
```

### activate new env
```bash
source activate ./env
```

### init DVC
```bash
git init
dvc init
```

### create empty files - 
```bash
mkdir -p src/utils config
touch src/__init__.py src/utils/__init__.py param.yaml dvc.yaml config/config.yaml src/stage_01_load_save.py src/utils/all_utils.py setup.py .gitignore
```

### install src 
```bash
pip install -e .
```
### create config.yaml
''' you shall put all configurations and metadata info here and read during execution

### create params.yaml
''' yiu shall put all params chganges here
# create dvc.yaml 
``` This file will create pipe lines based on stages created.