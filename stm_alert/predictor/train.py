import hydra


@hydra.main(config_path="config", config_name="default")
def main(cfg):
    """Main training function

    Args:
        cfg: Hydra configuration object
    """


if __name__ == "__main__":
    main()
