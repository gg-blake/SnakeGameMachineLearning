from python import Python
from population import Vector2D, Vector1D, DTYPE, GAME_WIDTH_OFFSET, GAME_HEIGHT_OFFSET, INITIAL_SCORE, GAME_SCALE, TTL, SPEC
from neural_network import NeuralNetwork
from math import abs, sqrt, clamp
from tensor import Tensor, TensorSpec
from logger import Logger


struct Snake(Hashable):
    var position: Vector2D
    var direction: Vector2D
    var score: Int
    var min_dist: Vector1D
    var neural_network: NeuralNetwork[SPEC]
    var history: List[Vector2D]
    var fitness: Vector1D

    fn __init__(inout self) raises:
        self.position = Vector2D(0, 0)
        self.direction = Vector2D(-1, 0)
        self.neural_network = NeuralNetwork[SPEC]()
        self.score = INITIAL_SCORE
        self.min_dist = 0
        self.history = List[Vector2D]()
        self.fitness = TTL
        for i in range(self.score):
            self.history.append(self.position + Vector2D(self.score - i - 1, 0))

    # Make a Snake instance and transfer ownership of NeuralNetwork
    fn __init__(inout self, owned neural_network: NeuralNetwork[SPEC]):
        self.position = Vector2D(0, 0)
        self.direction = Vector2D(-1, 0)
        self.neural_network = neural_network
        self.score = INITIAL_SCORE
        self.min_dist = 0
        self.history = List[Vector2D]()
        self.fitness = TTL
        for i in range(self.score):
            self.history.append(self.position + Vector2D(self.score - i - 1, 0))
        
    fn __moveinit__(inout self, owned existing: Self):
        self = Self(existing.neural_network)

    fn __contains__(self, point: Vector2D) -> SIMD[DType.bool, 1]:
        for p in self.history:
            if p[] == point:
                return True
        return False

    fn __hash__(self) -> Int:
        return hash(self.neural_network)

    fn __str__(self) -> String:
        var result: String = "Snake("
        result += "\nPosition" + str(self.position) + ", "
        result += "\nScore" + str(self.score) + ", "
        result += "\nMinimum Distance" + str(self.min_dist) + ", "
        result += "\nHistory: ["
        for i in self.history:
            result += str(i[]) + ", "
        result += "]"
        return result

    fn __del__(owned self):
        self.neural_network^.__del__()

    fn reset(inout self):
        self.position = Vector2D(0, 0)
        self.direction = Vector2D(-1, 0)
        self.score = INITIAL_SCORE
        self.min_dist = 0
        self.history = List[Vector2D]()
        self.fitness = TTL
        for i in range(self.score):
            self.history.append(self.position + Vector2D(self.score - i - 1, 0))

    fn is_dead(self) -> Bool:
        return int(self.direction[0]) == 0 and int(self.direction[1]) == 0

    fn update(inout self, screen: PythonObject, fruit_position: Vector2D, stats: Dict[String, Float32]) raises:
        if self.is_dead():
            return

        var torch = Python.import_module("torch")
        var pygame = Python.import_module("pygame")

        var fruit_left = int((fruit_position < self.position)[0])
        var fruit_right = int((fruit_position > self.position)[0])
        var fruit_top = int((fruit_position < self.position)[1])
        var fruit_bottom = int((fruit_position > self.position)[1])
        
        var wall_left = ~int(Snake.in_bounds(self.position + Vector2D(-1, 0)))
        var wall_right = ~int(Snake.in_bounds(self.position + Vector2D(1, 0)))
        var wall_top = ~int(Snake.in_bounds(self.position + Vector2D(0, -1)))
        var wall_bottom = ~int(Snake.in_bounds(self.position + Vector2D(0, 1)))

        var body_left = int((self.position + Vector2D(-1, 0) in self))
        var body_right = int((self.position + Vector2D(1, 0) in self))
        var body_top = int((self.position + Vector2D(0, -1) in self))
        var body_bottom = int((self.position + Vector2D(0, 1) in self))

        var facing_left = int((self.direction[0] == -1))
        var facing_right = int((self.direction[0] == 1))
        var facing_top = int((self.direction[1] == -1))
        var facing_bottom = int((self.direction[1] == 1))

        var fruit_ahead = (fruit_left and facing_left) or (fruit_right and facing_right) or (fruit_top and facing_top) or (fruit_bottom and facing_bottom)
        var fruit_left_side = (fruit_left and facing_top) or (fruit_right and facing_bottom) or (fruit_top and facing_right) or (fruit_bottom and facing_left)
        var fruit_right_side = (fruit_left and facing_bottom) or (fruit_right and facing_top) or (fruit_top and facing_left) or (fruit_bottom and facing_right)
        var fruit_behind = int(SIMD[DType.bool, 1]((not fruit_ahead) and (not fruit_left) and (not fruit_right)))
        var wall_ahead = int(Self.in_bounds(self.position + self.direction))
        var wall_left_side = (wall_left and facing_top) or (wall_right and facing_bottom) or (wall_top and facing_right) or (wall_bottom and facing_left)
        var wall_right_side = (wall_left and facing_bottom) or (wall_right and facing_top) or (wall_top and facing_left) or (wall_bottom and facing_right)
        var body_ahead = (body_left and facing_left) or (body_right and facing_right) or (body_top and facing_top) or (body_bottom and facing_bottom)
        var body_left_side = (body_left and facing_top) or (body_right and facing_bottom) or (body_top and facing_right) or (body_bottom and facing_left)
        var body_right_side = (body_left and facing_bottom) or (body_right and facing_top) or (body_top and facing_left) or (body_bottom and facing_right)

        var no_body_fruit_ahead = fruit_ahead and not body_ahead
        var no_body_fruit_left_side = fruit_left_side and not body_left_side
        var no_body_fruit_right_side = fruit_right_side and not body_right_side

        var no_body_left_fruit_behind = fruit_behind and not body_left_side
        var no_body_right_fruit_behind = fruit_behind and not body_right_side

        var input = torch.tensor([
            body_ahead, body_left_side, body_right_side,
            no_body_fruit_ahead, no_body_fruit_left_side, no_body_fruit_right_side, no_body_left_fruit_behind, no_body_right_fruit_behind,
            wall_ahead, wall_left_side, wall_right_side
        ]).to(torch.float32).unsqueeze(1)

        var output = self.neural_network.feed(input)
        output = torch.flatten(output)
        var sorted: PythonObject
        var indices: PythonObject
        output = torch.argmax(output)

        var direction_num = 0
        if Vector2D(0, -1) == self.direction:
            direction_num = 0
        elif Vector2D(1, 0) == self.direction:
            direction_num = 1
        elif Vector2D(0, 1) == self.direction:
            direction_num = 2
        elif Vector2D(-1, 0) == self.direction:
            direction_num = 3

        if output == 0:
            # left turn
            direction_num += 1
        elif output == 1:
            # right turn
            direction_num -= 1
        
        if direction_num < 0:
            direction_num = 3
        elif direction_num > 3:
            direction_num = 0

        if direction_num == 0:
            # up
            self.direction = Vector2D(0, -1)
        elif direction_num == 1:
            # right
            self.direction = Vector2D(1, 0)
        elif direction_num == 2:
            # down
            self.direction = Vector2D(0, 1)
        elif direction_num == 3:
            # left
            self.direction = Vector2D(-1, 0)

        var old_fitness = self.fitness
        self.move(fruit_position)
        self.draw(screen, old_fitness, stats["best_fitness"])
        #print(self.position)

    @staticmethod
    fn in_bounds(position: Vector2D) -> SIMD[DType.bool, 1]:
        return position[0] >= -GAME_WIDTH_OFFSET and position[0] < GAME_WIDTH_OFFSET  and position[1] >= -GAME_HEIGHT_OFFSET and position[1] < GAME_HEIGHT_OFFSET 

    @staticmethod
    fn distance(point_a: Vector2D, point_b: Vector2D) -> Vector1D:
        return sqrt((point_a[0] - point_b[0])**2 + (point_a[1] - point_b[1])**2)

    fn move(inout self, fruit_position: Vector2D):
        var next_position = self.position + self.direction
        # Death detection
        var active_death = next_position in self or not Snake.in_bounds(self.position)
        var passive_death = self.fitness <= (self.score - INITIAL_SCORE) * TTL
        if active_death or passive_death:
            # Active death: If the snake hits the game bounds or tail
            self.direction = Vector2D.splat(0)
            return

        # Food detection
        if next_position == fruit_position:
            self.score += 1
            self.fitness = (self.score - INITIAL_SCORE + 1) * TTL# + ((self.score - INITIAL_SCORE) * (TTL // 8))
        else:
            var current_distance = Self.distance(self.position, fruit_position)
            var next_distance = Self.distance(next_position, fruit_position)
            if next_distance < current_distance:
                self.fitness += 1
            else:
                self.fitness -= 5

        self.position = next_position
        self.history.append(self.position)
        while len(self.history) > self.score:
            self.history = self.history[1:]

    # Draws visual representation of this Snake object to the running pygame window
    fn draw(borrowed self, screen: PythonObject, old_fitness: Vector1D, best_fitness: Vector1D) raises:
        var pygame = Python.import_module("pygame")
        var font = pygame.font.SysFont('Comic Sans MS', 20)
        # Draw the body
        var count = 0
        var color = (200, 30, 30)
        for b in self.history:
            var body_part = b[]
            var x_position = int(body_part[0]) + GAME_WIDTH_OFFSET
            var y_position = int(body_part[1]) + GAME_WIDTH_OFFSET
            var rect = (x_position * GAME_SCALE, y_position * GAME_SCALE, GAME_SCALE, GAME_SCALE)
            var weight = int(clamp(self.fitness / best_fitness * 200, 0, 200))
            var body_part_color = (200 - weight, weight, 20)
            if self.fitness > best_fitness:
                body_part_color = (255, 187, 0)
            pygame.draw.rect(screen, body_part_color, rect)
            if count == self.score - 1:
                var text_surface = font.render(str(self.fitness), False, (255, 255, 255))
                screen.blit(text_surface, (x_position * GAME_SCALE - GAME_SCALE, y_position * GAME_SCALE))
                
            count += 1
