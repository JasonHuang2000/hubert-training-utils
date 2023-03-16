import fire
import yaml


def main(
    exp_dir: str,
    preprocess_dir: str,
    label_dir: str,
    label_rate: float,
    source_file: str,
    output_file: str,
):
    with open(source_file, "r") as f:
        config = yaml.safe_load(f)

    config["hydra"]["run"]["dir"] = exp_dir
    config["task"]["data"] = f"{preprocess_dir}/tsv"
    config["task"]["label_dir"] = f"{preprocess_dir}/{label_dir}"
    config["task"]["labels"] = '["km"]'
    config["model"]["label_rate"] = label_rate

    with open(output_file, "w") as f:
        yaml.safe_dump(config, f)

if __name__ == "__main__":
    fire.Fire(main)

