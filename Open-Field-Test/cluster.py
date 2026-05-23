import vame
config_path = "Open-Field-Test\config.yaml"

config_data = vame.read_config(config_path)

vame.segment_session(
    config=config_data,
    overwrite_segmentation=True,
    overwrite_embeddings=False
)