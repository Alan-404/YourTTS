
class CheckpointManager:
    def __init__(self, saved_folder: str, n_saved_checkpoints: int = 3) -> None:
        self.saved_folder = saved_folder
        self.n_saved_checkpoints = n_saved_checkpoints