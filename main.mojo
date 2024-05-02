from python import Python
from population import Population, game_width, game_height, game_scale
from logger import Logger
from time import sleep

alias snake_count: Int = 50
alias mutation_rate: Float32 = 0.8

fn main() raises:
    var pygame = Python.import_module("pygame")
    var population = Population[snake_count, mutation_rate]()
    var run = True
    while run:
        while population.active and run:
            var events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    run = False
                elif event.type == pygame.KEYDOWN and event.type == pygame.K_SPACE:
                    population.logger.notice("Replay requested")
                    population.replay_active = True

            population.update_habitat()
        population.generate_next_habitat()