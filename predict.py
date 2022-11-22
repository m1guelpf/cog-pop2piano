import note_seq
from omegaconf import OmegaConf
from cog import BasePredictor, Input, Path, BaseModel
from transformer_wrapper import TransformerWrapper


class ModelOutput(BaseModel):
    mixed_audio: Path
    midi_audio: Path


model = "dpipqxiy"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        config = OmegaConf.load("config.yaml")
        self.wrapper = TransformerWrapper(config)
        self.wrapper = self.wrapper.load_from_checkpoint(
            "model.ckpt", config=config
        ).to("cuda")
        self.wrapper.eval()

    def predict(
        self,
        audio: Path = Input(description="Audio file from a pop song."),
        composer: str = Input(description="Composer to use for the piano part.", choices=[
            "composer1", "composer2", "composer3", "composer4", "composer5",
            "composer6", "composer7", "composer8", "composer9", "composer10",
            "composer11", "composer12", "composer13", "composer14", "composer15",
            "composer16", "composer17", "composer18", "composer19", "composer20",
            "composer21"
        ], default="composer1"),
    ) -> ModelOutput:
        """Run a single prediction on the model"""
        _, _, mix_path, midi_path = self.wrapper.generate(
            audio_path=audio,
            composer=composer,
            model=model,
            save_midi=True,
            save_mix=True,
            save_rendered_midi=True,
            midi_path="midi.mid",
            mix_path="mix.wav",
            rendered_midi_path="rendered_midi.wav",
        )

        return ModelOutput(mixed_audio=mix_path, midi_audio=midi_path)
