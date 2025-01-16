# BDG—ODE

## Requirements

Before running the code, ensure that you have the required dependencies installed. The key libraries used in this project include:

- Python 3.x
- NumPy
- PyTorch
- TorchDiffeq
- Matplotlib
- NetworkX
- PIL (Python Imaging Library)
- Pandas

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

## Usage

To run the training script, use the following command format:

```bash
python main.py --T <Terminal_Time> --sampled_time <Time_Sampling_Type> --baseline <Model_Type> --gpu <GPU_ID> --weight_decay <Weight_Decay> --lr <Learning_Rate> --sparse --niters <Number_of_Iterations> --dump --dataset <Dataset>
```

### Example Command

The following command trains the model using dataset `f3` with specific hyperparameters:

```bash
python main.py --T 5 --sampled_time irregular --baseline BDG --gpu -1 --weight_decay 1e-3 --lr 0.01 --sparse --niters 1500 --dump --dataset f3
```

### Parameters

- `--T`: Terminal time (the length of the time interval for the dynamics).
- `--sampled_time`: Time sampling method, options include `irregular` or `equal`.
- `--baseline`: The baseline model to use for training (e.g., `BDG`).
- `--gpu`: The GPU ID to use for training. Use `-1` for CPU.
- `--weight_decay`: The L2 regularization weight for the optimizer.
- `--lr`: Learning rate for the optimizer.
- `--sparse`: Use sparse matrices for the adjacency matrix (optional).
- `--niters`: Number of iterations (epochs) for training.
- `--dump`: Save the training results in the results directory (optional).
- `--dataset`: **Dataset selection**. Options are `f1`, `f2`, and `f3`.


### Output

If the `--dump` option is used, the results will be saved in the `results/cognitive/` directory.