```bash
cd /Users/baeseojun/proj
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/dataset/validate_data.py
.venv/bin/python src/dataset/prepare_pools.py
.venv/bin/python src/visualization/create_pool_examples.py
.venv/bin/python src/inference/download_model.py configs/model.json
.venv/bin/python src/inference/validate_e1_inputs.py configs/model.json
.venv/bin/python src/inference/run_e1.py configs/model.json
.venv/bin/python src/inference/run_e1_fp32.py configs/model.json
.venv/bin/python src/visualization/plot_e1.py
```
