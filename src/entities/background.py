from ..utils import GameConfig
from .entity import Entity


class Background(Entity):
    def __init__(self, config: GameConfig) -> None:
        super().__init__(
            config,
            config.images.background,
            0,
            0,
            config.window.width,
            config.window.height,
        )

    def get_state(self):
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}
