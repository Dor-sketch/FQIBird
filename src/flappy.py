import asyncio
import sys
import copy

import pygame
from pygame.locals import K_ESCAPE, K_SPACE, K_UP, KEYDOWN, QUIT

from .entities import (
    Background,
    Floor,
    GameOver,
    Pipes,
    Player,
    PlayerMode,
    Score,
    WelcomeMessage,
)
from .utils import GameConfig, Images, Sounds, Window

class Flappy:
    def __init__(self):
        print("Flappy Bird Game")
        pygame.init()
        self.best_score = 0
        self.frames = 300000000000000
        self.counter = 300
        pygame.display.set_caption("Flappy Bird")
        window = Window(288, 512)
        screen = pygame.display.set_mode((window.width, window.height))
        images = Images()

        self.config = GameConfig(
            screen=screen,
            clock=pygame.time.Clock(),
            fps=self.frames,
            window=window,
            images=images,
            sounds=Sounds(),
        )

    def get_state(self):
        import numpy as np

        # Get the state of each component of the game
        state = {
            "player": np.array(self.player.get_state()).reshape(-1),
            "pipes": np.array(self.pipes.get_state()).reshape(-1),
            "time": np.array([pygame.time.get_ticks() - self.start_time]).reshape(-1),
            "score": np.array([self.score.score]).reshape(-1),
            "floor": np.array(self.floor.get_state()).reshape(-1),
        }

        # Stack all values into a single Numpy array
        con = np.concatenate(list(state.values()), axis=0)
        return con

    async def start(self, agent):
        self.start_time = pygame.time.get_ticks()
        fake_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_SPACE})
        pygame.event.post(fake_event)

        while True:
            self.background = Background(self.config)
            self.floor = Floor(self.config)
            self.player = Player(self.config)
            self.welcome_message = WelcomeMessage(self.config)
            self.game_over_message = GameOver(self.config)
            self.pipes = Pipes(self.config)
            self.score = Score(self.config)
            await self.splash()
            async for state, action, game in self.play(agent):
                yield state, action, game

            await self.game_over()

    async def splash(self):
        """Shows welcome splash screen animation of flappy bird"""
        self.frames = 30
        self.counter -= 1
        if self.counter < 20:
            self.config.fps = 300
        if self.counter < 0:
            self.config.fps = 30
        self.player.set_mode(PlayerMode.SHM)
        fake_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_SPACE})
        self.start_time = pygame.time.get_ticks()
        pygame.event.post(fake_event)

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    return

            self.background.tick()
            self.floor.tick()
            self.player.tick()
            self.welcome_message.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    def check_quit_event(self, event):
        if event.type == QUIT or (
            event.type == KEYDOWN and event.key == K_ESCAPE
        ):
            pygame.quit()
            sys.exit()

    def is_tap_event(self, event):
        m_left, _, _ = pygame.mouse.get_pressed()
        space_or_up = event.type == KEYDOWN and (
            event.key == K_SPACE or event.key == K_UP
        )
        screen_tap = event.type == pygame.FINGERDOWN
        return m_left or space_or_up or screen_tap

    def apply_action(self, action):
        if action == 1:
            fake_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_SPACE})
            pygame.event.post(fake_event)

    def copy(self):
        new_flappy = copy.copy(self)  # Shallow copy
        return new_flappy

    async def play(self, agent):
        self.score.reset()
        self.player.set_mode(PlayerMode.NORMAL)
        while True:
            state = self.get_state()
            action = agent.agent.get_action(state, self)
            self.apply_action(action)
            yield state , action, self
            if self.player.collided(self.pipes, self.floor):
                return
            for i, pipe in enumerate(self.pipes.upper):
                if self.player.crossed(pipe):
                    self.score.add()
                    if self.score.score > self.best_score:
                        self.best_score = self.score.score
                        self.config.fps = 30

            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    self.player.flap()

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()

            pygame.display.update()
            await asyncio.sleep(0)
            self.config.tick()

    async def game_over(self):
        """crashes the player down and shows gameover image"""
        self.config.fps = 300000000000
        self.frames = 300000000000
        self.player.set_mode(PlayerMode.CRASH)
        self.pipes.stop()
        self.floor.stop()

        while True:
            for event in pygame.event.get():
                self.check_quit_event(event)
                if self.is_tap_event(event):
                    if self.player.y + self.player.h >= self.floor.y - 1:
                        return

            self.background.tick()
            self.floor.tick()
            self.pipes.tick()
            self.score.tick()
            self.player.tick()
            self.game_over_message.tick()

            self.config.tick()
            pygame.display.update()
            await asyncio.sleep(0)
            fake_event = pygame.event.Event(pygame.KEYDOWN, {'key': pygame.K_SPACE})
            pygame.event.post(fake_event)
