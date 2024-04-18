from python import Python
from population import Population, game_width, game_height, game_scale
from logger import Logger
from time import sleep

alias snake_count: Int = 50 # Keep this 30 and under to reduce risk of segmentation error when adding `Position` to `self.body_set`

fn main() raises:
    Logger.notice("Starting simulation of " + str(snake_count) + " snakes...")
    var pygame = Python.import_module("pygame")
    var screen = pygame.display.set_mode((game_width * game_scale, game_height * game_scale))
    var population = Population[snake_count]()
    
    var run = True
    while run:
        var count = 0
        population.active = True
        while count < 100 and run and population.active:
            var events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    run = False

            population.update_habitat(screen)
            
            count += 1
            
        Logger.notice("Generation has died. Generating next habitat...")
        population.generate_next_habitat(survival_rate=0.5)