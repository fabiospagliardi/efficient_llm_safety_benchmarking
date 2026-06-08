import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyro
import torch
from py_irt.config import IrtConfig
from py_irt.dataset import Dataset
from py_irt.training import IrtModelTrainer

from two_param_logistic import TwoParamLogistic


def main():

    # Read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0,
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=1000,
    )
    args = parser.parse_args()

    # Make reproducible
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pyro.set_rng_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(True)

    # Process input and output paths
    input_path = Path(args.input_path).expanduser()
    print(input_path)
    params_dir = Path(__file__).resolve().with_name("fit_results")
    params_dir.mkdir(parents=True, exist_ok=True)
    out_csv = params_dir / f"{input_path.stem.replace('data', 'parameters')}.csv" # return final without extention

    # Load input data
    data = Dataset.from_jsonlines(str(input_path))
    config = IrtConfig(
        model_type=TwoParamLogistic,
        priors="hierarchical",
    )

    # Fit IRT model
    trainer = IrtModelTrainer(
        config=config, 
        data_path=None, 
        dataset=data
    )
    trainer.train(epochs=args.epochs, device=args.device)

    # Extract IRT parameters
    discriminations = [np.exp(i) for i in trainer.best_params["disc"]]  # Since parameter in log space
    difficulties = list(trainer.best_params["diff"])
    item_ids = trainer.best_params["item_ids"].values()

    # Fetch variational posterior distributions
    # Note: reflects param store state at end of training, not necessarily the best epoch
    posterior = trainer.irt_model.export_posterior()

    irt_model = pd.DataFrame({
        "item_ids": item_ids,
        "a": discriminations,
        "b": difficulties,
        # LogNormal posterior for disc: loc/scale are in log-space
        "a_posterior_loc": posterior["disc"]["loc"],
        "a_posterior_scale": posterior["disc"]["scale"],
        # Normal posterior for diff: loc == point estimate, only scale adds info
        "b_posterior_scale": posterior["diff"]["scale"],
    })
    irt_model.to_csv(out_csv, index=None)

    # Need to save abilities too
    file_name = params_dir / f"{input_path.stem.replace('data', 'abilities')}.csv"
    fitted_abilities = list(trainer.best_params["ability"])
    llms = list(trainer.best_params["subject_ids"].values())
    ability_result = pd.DataFrame({
        "model": llms,
        "ability": fitted_abilities,
        # Normal posterior for ability: loc == point estimate, only scale adds info
        "ability_posterior_scale": posterior["ability"]["scale"],
    })
    ability_result.to_csv(file_name, index=False)


if __name__ == "__main__":
    trainer = main()
