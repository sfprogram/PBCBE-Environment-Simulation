import math
import random
import numpy as np
import csv
import json

MAX_GENERATIONS = 100


NUM_RAYS = 24
FROG_POP = 20
SNAKE_COUNT = 2
GENOME_SIZE = 5
LIFESPAN = 1200
NUM_DOTS = 100
DOT_RADIUS = 7
CANVAS_WIDTH = 800
CANVAS_HEIGHT = 600
NUM_GRASS_PATCHES = 6
GRASS_RADIUS = 60
NUM_WATER_POOLS = 3
WATER_RADIUS = 40
FLIES_NEEDED = 5
STARVATION_THRESHOLD = 500
STARVATION_IGNORE_GRASS_THRESHOLD = 250

RANDOM_UNIFORM_NEG = 0.3
RANDOM_UNIFORM_POS = 0.3
MUTATION_RATE = 0.1

# Selection method: steady, tournament or GA-SA
SELECTION_METHOD = 'GA-SA'

REPLACEMENT_RATE = 0.5

TOURNAMENT_SIZE = 6

INITIAL_TEMPERATURE = 1.0
TEMPERATURE_DECAY = 0.98


class AgentSim:
    def __init__(self, x, y, genome, radius=10, fov_deg=320):
        self.x, self.y = x, y
        self.radius = radius
        self.angle = random.uniform(0, 2 * math.pi)
        self.wheel_base = 20
        self.ray_count = NUM_RAYS
        self.ray_length = 80
        self.fov = math.radians(fov_deg)
        self.genome = genome
        self.dead = False
        self.fitness = 0.0
        self.ticks_alive = 0

        self.final_state = None

    def _line_intersection_point(self, p1, p2, wall):
        x1, y1 = p1;
        x2, y2 = p2
        x3, y3, x4, y4 = wall
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            return None, None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        if 0 < t < 1 and 0 < u < 1:
            return x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        return None, None

    def _projection_fraction(self, p1, p2, point):
        x1, y1 = p1;
        x2, y2 = p2
        px, py = point
        dx, dy = x2 - x1, y2 - y1
        l2 = dx * dx + dy * dy
        if l2 == 0:
            return None
        return ((px - x1) * dx + (py - y1) * dy) / l2

    def _circle_line_collision(self, cx, cy, r, wall):
        x1, y1, x2, y2 = wall
        px, py = x2 - x1, y2 - y1
        norm = px * px + py * py
        u = ((cx - x1) * px + (cy - y1) * py) / norm
        u = max(min(u, 1), 0)
        dx = x1 + u * px - cx
        dy = y1 + u * py - cy
        return dx * dx + dy * dy < r * r

    def calculate_turn_response(self, inputs):
        max_detections = [0.0] * 5
        ray_angles = []

        start_ang = -self.fov / 2
        for i in range(self.ray_count):
            ray_angle = start_ang + i * (self.fov / (self.ray_count - 1))
            ray_angles.append(ray_angle)

        strongest_directions = [0.0] * 5

        for ray_idx in range(self.ray_count):
            base_idx = ray_idx * 5
            ray_inputs = inputs[base_idx:base_idx + 5]

            for obj_type in range(5):
                detection_strength = ray_inputs[obj_type]
                if detection_strength > max_detections[obj_type]:
                    max_detections[obj_type] = detection_strength
                    strongest_directions[obj_type] = ray_angles[ray_idx]

        total_turn = 0.0
        for obj_type in range(5):
            if max_detections[obj_type] > 0:
                gene_strength = self.genome[obj_type]
                direction = strongest_directions[obj_type]
                detection_strength = max_detections[obj_type]

                if gene_strength >= 0:
                    turn_contribution = gene_strength * direction * detection_strength
                else:

                    abs_gene = abs(gene_strength)

                    proximity_multiplier = 1.0 + (detection_strength * 4.0)

                    if abs(direction) < 0.2:
                        avoidance_direction = 1.0 if random.random() > 0.5 else -1.0
                        turn_contribution = abs_gene * avoidance_direction * proximity_multiplier * 40.0
                    else:
                        base_turn = abs_gene * (-direction) * detection_strength
                        turn_contribution = base_turn * proximity_multiplier

                    if obj_type == 0:
                        turn_contribution *= 2.0

                total_turn += turn_contribution

        return total_turn


class FrogSim(AgentSim):
    def __init__(self, x, y, genome, radius=10):
        super().__init__(x, y, genome, radius, fov_deg=320)
        self.succeeded = False
        self.in_grass = False
        self.flies_eaten = 0
        self.ticks_since_last_food = 0

    def update(self, wall_coords, dots, snakes, grass_patches, water_pools):
        if self.dead or self.succeeded:
            return
        self.ticks_alive += 1
        self.ticks_since_last_food += 1

        if self.flies_eaten < FLIES_NEEDED and self.ticks_since_last_food >= STARVATION_THRESHOLD:
            self.dead = True
            self.final_state = 'starvation'
            return

        self.in_grass = False
        if self.flies_eaten < FLIES_NEEDED:
            for g in grass_patches:
                if math.hypot(self.x - g['x'], self.y - g['y']) < g['radius']:
                    self.in_grass = True
                    break

        if self.flies_eaten >= FLIES_NEEDED:
            for w in water_pools:
                if math.hypot(self.x - w['x'], self.y - w['y']) < w['radius']:
                    self.succeeded = True
                    self.dead = True
                    self.final_state = 'success'
                    return

        inputs = []
        start_ang = self.angle - self.fov / 2
        for i in range(self.ray_count):
            theta = start_ang + i * (self.fov / (self.ray_count - 1))
            ex = self.x + self.ray_length * math.cos(theta)
            ey = self.y + self.ray_length * math.sin(theta)

            min_wall = min_food = min_snake = min_grass = min_water = 1.0
            w_det = f_det = s_det = g_det = w2_det = False

            for wall in wall_coords:
                ix, iy = self._line_intersection_point((self.x, self.y), (ex, ey), wall)
                if ix is not None:
                    d = math.hypot(ix - self.x, iy - self.y) / self.ray_length
                    if d < min_wall:
                        min_wall, w_det = d, True

            if self.flies_eaten < FLIES_NEEDED:
                for dot in dots:
                    t = self._projection_fraction((self.x, self.y), (ex, ey), (dot['x'], dot['y']))
                    if t is not None and 0 <= t <= 1:
                        px, py = self.x + t * (ex - self.x), self.y + t * (ey - self.y)
                        if math.hypot(px - dot['x'], py - dot['y']) < DOT_RADIUS:
                            d = math.hypot(px - self.x, py - self.y) / self.ray_length
                            if d < min_food:
                                min_food, f_det = d, True
                for snake in snakes:
                    if snake.dead: continue
                    t = self._projection_fraction((self.x, self.y), (ex, ey), (snake.x, snake.y))
                    if t is not None and 0 <= t <= 1:
                        px, py = self.x + t * (ex - self.x), self.y + t * (ey - self.y)
                        if math.hypot(px - snake.x, py - snake.y) < snake.radius:
                            d = math.hypot(px - self.x, py - self.y) / self.ray_length
                            if d < min_snake:
                                min_snake, s_det = d, True
                if self.ticks_since_last_food <= STARVATION_IGNORE_GRASS_THRESHOLD:
                    for g in grass_patches:
                        t = self._projection_fraction((self.x, self.y), (ex, ey), (g['x'], g['y']))
                        if t is not None and 0 <= t <= 1:
                            px, py = self.x + t * (ex - self.x), self.y + t * (ey - self.y)
                            if math.hypot(px - g['x'], py - g['y']) < g['radius']:
                                d = math.hypot(px - self.x, py - self.y) / self.ray_length
                                if d < min_grass:
                                    min_grass, g_det = d, True

            if self.flies_eaten >= FLIES_NEEDED:
                for w in water_pools:
                    t = self._projection_fraction((self.x, self.y), (ex, ey), (w['x'], w['y']))
                    if t is not None and 0 <= t <= 1:
                        px, py = self.x + t * (ex - self.x), self.y + t * (ey - self.y)
                        if math.hypot(px - w['x'], py - w['y']) < w['radius']:
                            d = math.hypot(px - self.x, py - self.y) / self.ray_length
                            if d < min_water:
                                min_water, w2_det = d, True

            if self.flies_eaten >= FLIES_NEEDED:
                inputs.extend([max(0.0, 1 - min_wall) if w_det else 0.0,
                               0.0,
                               max(0.0, 1 - min_snake) if s_det else 0.0,
                               0.0,
                               max(0.0, 1 - min_water) if w2_det else 0.0])
            else:
                inputs.extend([max(0.0, 1 - min_wall) if w_det else 0.0,
                               max(0.0, 1 - min_food) if f_det else 0.0,
                               max(0.0, 1 - min_snake) if s_det else 0.0,
                               max(0.0, 1 - min_grass) if g_det else 0.0,
                               max(0.0, 1 - min_water) if w2_det else 0.0])

        if all(abs(v) < 1e-6 for v in inputs):
            turn = 0.0
        else:
            turn = self.calculate_turn_response(inputs)
            turn = max(min(turn, 2.0), -2.0)

        base_speed = 2.0
        ls = base_speed - turn
        rs = base_speed + turn
        v = (ls + rs) / 2.0
        omega = (rs - ls) / self.wheel_base
        self.angle += omega
        self.x += v * math.cos(self.angle)
        self.y += v * math.sin(self.angle)

        for wall in wall_coords:
            if self._circle_line_collision(self.x, self.y, self.radius, wall):
                self.dead = True
                self.final_state = 'wall'
                return

    def calculate_fitness(self, water_pools):
        fitness = 0

        # Phase 1
        fitness += self.ticks_alive * 0.05

        # Phase 2
        fitness += self.flies_eaten * 100
        if self.flies_eaten > 0:
            fitness += 50
        if self.flies_eaten >= FLIES_NEEDED:
            fitness += 200

        # Phase 3
        if self.flies_eaten >= FLIES_NEEDED:
            min_water_dist = min(math.hypot(self.x - w['x'], self.y - w['y'])
                                 for w in water_pools)
            fitness += max(0, 100 - min_water_dist * 0.5)

        # Phase 4
        if self.succeeded:
            fitness += 500
            time_bonus = max(0, (LIFESPAN - self.ticks_alive) * 0.5)
            fitness += time_bonus

        if self.dead and not self.succeeded:
            if self.final_state == 'wall':
                fitness -= 700
            elif self.final_state == 'eaten_by_snake':
                fitness -= 500
            elif self.final_state == 'starvation':
                fitness -= 300

        return fitness


class SnakeSim(AgentSim):
    def __init__(self, x, y, genome, radius=10):
        super().__init__(x, y, genome, radius, fov_deg=180)

    def update(self, wall_coords, frogs, snakes, grass_patches):
        if self.dead:
            return
        self.ticks_alive += 1

        inputs = []
        start_ang = self.angle - self.fov / 2
        for i in range(self.ray_count):
            theta = start_ang + i * (self.fov / (self.ray_count - 1))
            ex = self.x + self.ray_length * math.cos(theta)
            ey = self.y + self.ray_length * math.sin(theta)

            min_wall = min_prey = min_grass = 1.0
            w_det = p_det = g_det = False

            for wall in wall_coords:
                ix, iy = self._line_intersection_point((self.x, self.y), (ex, ey), wall)
                if ix is not None:
                    d = math.hypot(ix - self.x, iy - self.y) / self.ray_length
                    if d < min_wall:
                        min_wall, w_det = d, True

            for f in frogs:
                if f.dead or f.succeeded or f.in_grass:
                    continue
                t = self._projection_fraction((self.x, self.y), (ex, ey), (f.x, f.y))
                if t is not None and 0 <= t <= 1:
                    px, py = self.x + t * (ex - self.x), self.y + t * (ey - self.y)
                    if math.hypot(px - f.x, py - f.y) < f.radius:
                        d = math.hypot(px - self.x, py - self.y) / self.ray_length
                        if d < min_prey:
                            min_prey, p_det = d, True

            for g in grass_patches:
                t = self._projection_fraction((self.x, self.y), (ex, ey), (g['x'], g['y']))
                if t is not None and 0 <= t <= 1:
                    px, py = self.x + t * (ex - self.x), self.y + t * (ey - self.y)
                    if math.hypot(px - g['x'], py - g['y']) < g['radius']:
                        d = math.hypot(px - self.x, py - self.y) / self.ray_length
                        if d < min_grass:
                            min_grass, g_det = d, True

            inputs.extend([
                max(0.0, 1 - min_wall) if w_det else 0.0,
                max(0.0, 1 - min_prey) if p_det else 0.0,
                0.0,
                max(0.0, 1 - min_grass) if g_det else 0.0,
                0.0
            ])

        if all(abs(v) < 1e-6 for v in inputs):
            turn = 0.0
        else:

            turn = self.calculate_turn_response(inputs)
            turn = max(min(turn, 2.0), -2.0)

        base_speed = 3.0
        ls = base_speed - turn
        rs = base_speed + turn
        v = (ls + rs) / 2.0
        omega = (rs - ls) / self.wheel_base
        self.angle += omega
        self.x += v * math.cos(self.angle)
        self.y += v * math.sin(self.angle)

        for wall in wall_coords:
            if self._circle_line_collision(self.x, self.y, self.radius, wall):
                self.dead = True
                self.final_state = 'wall'
                return


class GeneticSimulation:
    def __init__(self, data_writer):
        self.data_writer = data_writer
        self.frog_generation = 0
        self.snake_generation = 0
        self.frog_step = 0

        global wall_coords
        wall_coords = [
            (0, 0, CANVAS_WIDTH, 0),
            (CANVAS_WIDTH, 0, CANVAS_WIDTH, CANVAS_HEIGHT),
            (CANVAS_WIDTH, CANVAS_HEIGHT, 0, CANVAS_HEIGHT),
            (0, CANVAS_HEIGHT, 0, 0)
        ]

        self.grass_patches = []
        self.water_pools = []
        self.dots = []
        self.frogs = []
        self.snakes = []

        self.create_grass_patches()
        self.create_water_pools()
        self.init_frogs()
        self.init_snakes()
        self.spawn_dots()

    def create_grass_patches(self):
        self.grass_patches = []
        min_d = 100
        for _ in range(NUM_GRASS_PATCHES):
            for _ in range(100):
                x = random.randint(GRASS_RADIUS + 50, CANVAS_WIDTH - GRASS_RADIUS - 50)
                y = random.randint(GRASS_RADIUS + 50, CANVAS_HEIGHT - GRASS_RADIUS - 50)
                if all(math.hypot(x - p['x'], y - p['y']) > min_d for p in self.grass_patches + self.water_pools):
                    self.grass_patches.append({'x': x, 'y': y, 'radius': GRASS_RADIUS})
                    break

    def create_water_pools(self):
        self.water_pools = []
        min_d = 150
        for _ in range(NUM_WATER_POOLS):
            for _ in range(100):
                x = random.randint(WATER_RADIUS + 50, CANVAS_WIDTH - WATER_RADIUS - 50)
                y = random.randint(WATER_RADIUS + 50, CANVAS_HEIGHT - WATER_RADIUS - 50)
                if all(math.hypot(x - p['x'], y - p['y']) > min_d for p in self.grass_patches + self.water_pools):
                    self.water_pools.append({'x': x, 'y': y, 'radius': WATER_RADIUS})
                    break

    def init_frogs(self):
        self.frogs = []
        for _ in range(FROG_POP):
            genome = [random.uniform(-1, 1) for _ in range(GENOME_SIZE)]
            x = random.randint(50, CANVAS_WIDTH - 50)
            y = random.randint(50, CANVAS_HEIGHT - 50)
            self.frogs.append(FrogSim(x, y, genome))

    def init_snakes(self):
        self.snakes = []
        for _ in range(SNAKE_COUNT):
            genome = [random.uniform(-1, 1) for _ in range(GENOME_SIZE)]
            x = random.randint(50, CANVAS_WIDTH - 50)
            y = random.randint(50, CANVAS_HEIGHT - 50)
            self.snakes.append(SnakeSim(x, y, genome))

    def spawn_dots(self):
        self.dots = []
        for _ in range(NUM_DOTS):
            x = random.randint(50, CANVAS_WIDTH - 50)
            y = random.randint(50, CANVAS_HEIGHT - 50)
            dx, dy = random.uniform(-1, 1), random.uniform(-1, 1)
            self.dots.append({'x': x, 'y': y, 'dx': dx, 'dy': dy})

    def update(self):
        all_dead = True
        for f in self.frogs:
            f.update(wall_coords, self.dots, self.snakes,
                     self.grass_patches, self.water_pools)
            if not (f.dead or f.succeeded):
                all_dead = False

        for dot in self.dots:
            dot['x'] += dot['dx']
            dot['y'] += dot['dy']
            if not (DOT_RADIUS < dot['x'] < CANVAS_WIDTH - DOT_RADIUS):
                dot['dx'] *= -1
            if not (DOT_RADIUS < dot['y'] < CANVAS_HEIGHT - DOT_RADIUS):
                dot['dy'] *= -1

        for f in self.frogs:
            if f.dead or f.succeeded: continue
            for dot in self.dots[:]:
                if math.hypot(f.x - dot['x'], f.y - dot['y']) < f.radius + DOT_RADIUS:
                    f.flies_eaten += 1
                    f.ticks_since_last_food = 0
                    self.dots.remove(dot)

        for s in self.snakes:
            s.update(wall_coords, self.frogs, [], self.grass_patches)
        for s in self.snakes:
            if s.dead: continue
            for f in self.frogs:
                if f.dead or f.succeeded or f.in_grass: continue
                if math.hypot(s.x - f.x, s.y - f.y) < s.radius + f.radius:
                    s.fitness += 100
                    f.dead = True
                    f.final_state = 'eaten_by_snake'

        self.frog_step += 1

        if all_dead or self.frog_step > LIFESPAN:
            self.evolve_frogs()
            self.evolve_snakes()

    def calculate_final_fitness(self):
        for frog in self.frogs:
            frog.fitness = frog.calculate_fitness(self.water_pools)

    def evolve_frogs(self):

        self.calculate_final_fitness()

        for idx, f in enumerate(self.frogs):
            wall_gene = f.genome[0]
            food_gene = f.genome[1]
            snake_gene = f.genome[2]
            grass_gene = f.genome[3]
            water_gene = f.genome[4]

            self.data_writer.writerow([
                self.frog_generation,
                'frog',
                idx,
                f.fitness,
                f.flies_eaten,
                f.final_state or 'alive',
                wall_gene,
                food_gene,
                snake_gene,
                grass_gene,
                water_gene
            ])

        if SELECTION_METHOD == 'tournament':
            sorted_frogs = sorted(self.frogs, key=lambda f: f.fitness, reverse=True)
            elite_count = max(1, int(0.1 * FROG_POP))
            elites = [f.genome[:] for f in sorted_frogs[:elite_count]]

            new_genomes = elites[:]

            while len(new_genomes) < FROG_POP:
                contenders = random.sample(self.frogs, TOURNAMENT_SIZE)
                winner = max(contenders, key=lambda f: f.fitness)
                child = winner.genome[:]

                for i in range(len(child)):
                    if random.random() < MUTATION_RATE:
                        child[i] += random.uniform(-RANDOM_UNIFORM_NEG, RANDOM_UNIFORM_POS)
                        child[i] = max(min(child[i], 1.0), -1.0)
                new_genomes.append(child)

        elif SELECTION_METHOD == 'GA-SA':

            T = INITIAL_TEMPERATURE * (TEMPERATURE_DECAY ** self.frog_generation)

            fits = [f.fitness for f in self.frogs]
            f_max = max(fits)
            exps = [math.exp((fi - f_max) / T) for fi in fits]
            total = sum(exps)
            probs = [e / total for e in exps]

            new_genomes = []
            for _ in range(FROG_POP):

                parent = np.random.choice(self.frogs, p=probs)
                child = parent.genome[:]

                for i in range(len(child)):
                    if random.random() < MUTATION_RATE * T:
                        child[i] += random.uniform(-RANDOM_UNIFORM_NEG, RANDOM_UNIFORM_POS)
                        child[i] = max(min(child[i], 1.0), -1.0)

                new_genomes.append(child)

        else:
            sorted_frogs = sorted(self.frogs, key=lambda f: f.fitness, reverse=True)
            elite_count = int(REPLACEMENT_RATE * FROG_POP)
            elites = [f.genome[:] for f in sorted_frogs[:elite_count]]

            new_genomes = elites[:]
            while len(new_genomes) < FROG_POP:
                parent = random.choice(elites)
                child = parent[:]
                for i in range(len(child)):
                    if random.random() < MUTATION_RATE:
                        child[i] += random.uniform(-RANDOM_UNIFORM_NEG, RANDOM_UNIFORM_POS)
                        child[i] = max(min(child[i], 1.0), -1.0)
                new_genomes.append(child)

        fitnesses = [f.fitness for f in self.frogs]
        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        worst_fitness = min(fitnesses)

        self.frog_generation += 1
        succ = sum(1 for f in self.frogs if f.succeeded)

        print(f"Frog Gen {self.frog_generation} — Best: {best_fitness:.1f}, Avg: {avg_fitness:.1f}, "
              f"Worst: {worst_fitness:.1f}, Successes: {succ}")

        self.create_grass_patches()
        self.create_water_pools()
        self.spawn_dots()
        self.init_frogs()
        self.frog_step = 0

    def evolve_snakes(self):

        for idx, s in enumerate(self.snakes):
            wall_gene = s.genome[0]
            prey_gene = s.genome[1]
            snake_gene = s.genome[2]
            grass_gene = s.genome[3]
            water_gene = s.genome[4]

            self.data_writer.writerow([
                self.snake_generation,
                'snake',
                idx,
                s.fitness,
                0,
                s.final_state or 'alive',
                wall_gene,
                prey_gene,
                snake_gene,
                grass_gene,
                water_gene
            ])

        if SELECTION_METHOD == 'tournament':
            sorted_snakes = sorted(self.snakes, key=lambda s: s.fitness, reverse=True)
            elite_count = max(1, int(0.2 * SNAKE_COUNT))
            elites = [s.genome[:] for s in sorted_snakes[:elite_count]]

            new_genomes = elites[:]

            tour_size = min(TOURNAMENT_SIZE, SNAKE_COUNT)
            while len(new_genomes) < SNAKE_COUNT:
                contenders = random.sample(self.snakes, tour_size)
                winner = max(contenders, key=lambda s: s.fitness)
                child = winner.genome[:]

                for i in range(len(child)):
                    if random.random() < MUTATION_RATE:
                        child[i] += random.uniform(-RANDOM_UNIFORM_NEG, RANDOM_UNIFORM_POS)
                        child[i] = max(min(child[i], 1.0), -1.0)
                new_genomes.append(child)

        elif SELECTION_METHOD == 'GA-SA':
            T = INITIAL_TEMPERATURE * (TEMPERATURE_DECAY ** self.snake_generation)
            print("Temperature:", T)
            fits = [s.fitness for s in self.snakes]
            f_max = max(fits)
            exps = [math.exp((fi - f_max) / T) for fi in fits]
            total = sum(exps)
            probs = [e / total for e in exps]

            new_genomes = []
            for _ in range(SNAKE_COUNT):
                parent = np.random.choice(self.snakes, p=probs)
                child = parent.genome[:]
                for i in range(len(child)):
                    if random.random() < MUTATION_RATE * T:
                        child[i] += random.uniform(-RANDOM_UNIFORM_NEG, RANDOM_UNIFORM_POS)
                        child[i] = max(min(child[i], 1.0), -1.0)
                new_genomes.append(child)

        else:
            sorted_snakes = sorted(self.snakes, key=lambda s: s.fitness, reverse=True)

            elite_count = max(1, int(REPLACEMENT_RATE * SNAKE_COUNT))
            elites = [s.genome[:] for s in sorted_snakes[:elite_count]]

            new_genomes = elites[:]
            while len(new_genomes) < SNAKE_COUNT:
                parent = random.choice(elites)
                child = parent[:]
                for i in range(len(child)):
                    if random.random() < MUTATION_RATE:
                        child[i] += random.uniform(-RANDOM_UNIFORM_NEG, RANDOM_UNIFORM_POS)
                        child[i] = max(min(child[i], 1.0), -1.0)
                new_genomes.append(child)

        for i, genome in enumerate(new_genomes):
            self.snakes[i].genome = genome

        self.snake_generation += 1
        best = max(s.fitness for s in self.snakes)
        print(f"Snake Gen {self.snake_generation} — Best Fitness {best:.1f}")
        self.init_snakes()


def run_sim():

    params = {
        'MAX_GENERATIONS': MAX_GENERATIONS,
        'SELECTION_METHOD': SELECTION_METHOD,
        'NUM_RAYS': NUM_RAYS,
        'FROG_POP': FROG_POP,
        'SNAKE_COUNT': SNAKE_COUNT,
        'GENOME_SIZE': GENOME_SIZE,
        'LIFESPAN': LIFESPAN,
        'NUM_DOTS': NUM_DOTS,
        'DOT_RADIUS': DOT_RADIUS,
        'CANVAS_WIDTH': CANVAS_WIDTH,
        'CANVAS_HEIGHT': CANVAS_HEIGHT,
        'NUM_GRASS_PATCHES': NUM_GRASS_PATCHES,
        'GRASS_RADIUS': GRASS_RADIUS,
        'NUM_WATER_POOLS': NUM_WATER_POOLS,
        'WATER_RADIUS': WATER_RADIUS,
        'FLIES_NEEDED': FLIES_NEEDED,
        'STARVATION_THRESHOLD': STARVATION_THRESHOLD,
        'STARVATION_IGNORE_GRASS_THRESHOLD': STARVATION_IGNORE_GRASS_THRESHOLD,
        'RANDOM_UNIFORM_NEG': RANDOM_UNIFORM_NEG,
        'RANDOM_UNIFORM_POS': RANDOM_UNIFORM_POS,
        'MUTATION_RATE': MUTATION_RATE
    }

    if SELECTION_METHOD == 'steady':
        params['REPLACEMENT_RATE'] = REPLACEMENT_RATE
    elif SELECTION_METHOD == 'tournament':
        params['TOURNAMENT_SIZE'] = TOURNAMENT_SIZE
    elif SELECTION_METHOD == 'GA-SA':
        params['INITIAL_TEMPERATURE'] = INITIAL_TEMPERATURE
        params['TEMPERATURE_DECAY'] = TEMPERATURE_DECAY

    json_filepath = "training data/GA/" + f"{SELECTION_METHOD}_params.json"
    with open(json_filepath, "w") as pf:
        json.dump(params, pf, indent=2)
    print(f"Parameters saved to {SELECTION_METHOD}_params.json")

    filename = "training data/GA/" + f"{SELECTION_METHOD}_agent_data.csv"
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'generation',
            'species',
            'id',
            'fitness',
            'flies_eaten',
            'final_state',
            'wall_gene',
            'food_gene',
            'snake_gene',
            'grass_gene',
            'water_gene'
        ])

        sim = GeneticSimulation(writer)
        while sim.frog_generation < MAX_GENERATIONS:
            sim.update()
    print(f"Data saved to {filename}")


if __name__ == "__main__":
    run_sim()