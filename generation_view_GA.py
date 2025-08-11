import json
import pandas as pd
import ast
import tkinter as tk
import math
import random

# Selection method: steady, tournament or GA-SA
type = "tournament"

with open("training data/GA/" + f"{type}_params.json") as f:
    params = json.load(f)


NUM_RAYS = params["NUM_RAYS"]
FROG_POP = params["FROG_POP"]
SNAKE_COUNT = params["SNAKE_COUNT"]
GENOME_SIZE = params["GENOME_SIZE"]
NUM_DOTS = params["NUM_DOTS"]
DOT_RADIUS = params["DOT_RADIUS"]
CANVAS_WIDTH = params["CANVAS_WIDTH"]
CANVAS_HEIGHT = params["CANVAS_HEIGHT"]
NUM_GRASS_PATCHES = params["NUM_GRASS_PATCHES"]
GRASS_RADIUS = params["GRASS_RADIUS"]
NUM_WATER_POOLS = params["NUM_WATER_POOLS"]
WATER_RADIUS = params["WATER_RADIUS"]
FLIES_NEEDED = params["FLIES_NEEDED"]
STARVATION_THRESHOLD = params["STARVATION_THRESHOLD"]
STARVATION_IGNORE_GRASS_THRESHOLD = params["STARVATION_IGNORE_GRASS_THRESHOLD"]
RANDOM_UNIFORM_NEG = params["RANDOM_UNIFORM_NEG"]
RANDOM_UNIFORM_POS = params["RANDOM_UNIFORM_POS"]
MUTATION_RATE = params["MUTATION_RATE"]


agent_df = pd.read_csv("training data/GA/" + f"{type}_agent_data.csv")
latest_gen = agent_df["generation"].max()
frogs_df = agent_df[(agent_df["generation"] == latest_gen) & (agent_df["species"] == "frog")]
snakes_df = agent_df[(agent_df["generation"] == latest_gen) & (agent_df["species"] == "snake")]

class AgentSim:
    def __init__(self, canvas, x, y, genome, radius=10, color="black", fov_deg=320):
        self.canvas = canvas
        self.x, self.y = x, y
        self.radius = radius
        self.angle = random.uniform(0, 2 * math.pi)
        self.wheel_base = 20
        self.ray_count = NUM_RAYS
        self.ray_length = 80
        self.fov = math.radians(fov_deg)
        self.ray_lines = []
        self.genome = genome
        self.dead = False
        self.succeeded = False
        self.fitness = 0.0
        self.flies_eaten = 0
        self.ticks_alive = 0

        self.body = canvas.create_oval(
            self.x - radius, self.y - radius,
            self.x + radius, self.y + radius,
            fill=color
        )
        for _ in range(self.ray_count):
            self.ray_lines.append(canvas.create_line(0, 0, 0, 0, fill="black"))

    def _line_intersection_point(self, p1, p2, wall):
        x1, y1 = p1
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
        x1, y1 = p1
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
        u = ((cx - x1) * px + (cy - y1) * py) / float(norm)
        u = max(min(u, 1), 0)
        dx = x1 + u * px - cx
        dy = y1 + u * py - cy
        return dx * dx + dy * dy < r * r

    def move(self, turn_gain):
        turn = max(min(turn_gain, 2.0), -2.0)
        base_speed = 2.0
        left_speed = base_speed - turn
        right_speed = base_speed + turn
        v = (left_speed + right_speed) / 2.0
        omega = (right_speed - left_speed) / self.wheel_base
        self.angle += omega
        self.x += v * math.cos(self.angle)
        self.y += v * math.sin(self.angle)

        self.canvas.coords(
            self.body,
            self.x - self.radius, self.y - self.radius,
            self.x + self.radius, self.y + self.radius
        )

        for wall in self.wall_coords:
            if self._circle_line_collision(self.x, self.y, self.radius, wall):
                self.canvas.itemconfig(self.body, fill="red")
                self.dead = True
                return

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
    def __init__(self, canvas, x, y, genome, radius=10):
        super().__init__(canvas, x, y, genome, radius, color="green", fov_deg=320)
        self.succeeded = False
        self.in_grass = False
        self.flies_eaten = 0
        self.ticks_since_last_food = 0
        self.was_in_grass = False


    def update(self, wall_coords, dots, snakes, grass_patches, water_pools):
        if self.dead or self.succeeded:
            return
        self.ticks_alive += 1
        self.ticks_since_last_food += 1

        if self.flies_eaten < FLIES_NEEDED and self.ticks_since_last_food >= STARVATION_THRESHOLD:
            self.dead = True
            self.canvas.itemconfig(self.body, fill="gray")
            return

        self.in_grass = False
        if self.flies_eaten < FLIES_NEEDED:
            for grass in grass_patches:
                if math.hypot(self.x - grass['x'], self.y - grass['y']) < grass['radius']:
                    self.in_grass = True
                    break

        self.was_in_grass = self.in_grass

        if self.flies_eaten >= FLIES_NEEDED:
            for water in water_pools:
                if math.hypot(self.x - water['x'], self.y - water['y']) < water['radius']:
                    self.succeeded = True
                    self.dead = True
                    self.fitness += 500
                    self.canvas.itemconfig(self.body, fill="purple")
                    return

        inputs = self.cast_rays(wall_coords, dots, snakes, grass_patches, water_pools)

        if all(abs(i) < 1e-6 for i in inputs):
            turn = 0.0
        else:
            turn = self.calculate_turn_response(inputs)
            turn = max(min(turn, 2.0), -2.0)

        base_speed = 2.0
        left_speed = base_speed - turn
        right_speed = base_speed + turn
        v = (left_speed + right_speed) / 2.0
        omega = (right_speed - left_speed) / self.wheel_base
        self.angle += omega
        self.x += v * math.cos(self.angle)
        self.y += v * math.sin(self.angle)

        self.canvas.coords(
            self.body,
            self.x - self.radius, self.y - self.radius,
            self.x + self.radius, self.y + self.radius
        )

        for wall in wall_coords:
            if self._circle_line_collision(self.x, self.y, self.radius, wall):
                self.canvas.itemconfig(self.body, fill="red")
                self.dead = True
                return

    def cast_rays(self, wall_coords, dots, snakes, grass_patches, water_pools):
        inputs = []
        start_angle = self.angle - self.fov / 2.0
        for i, line_id in enumerate(self.ray_lines):
            theta = start_angle + i * (self.fov / (self.ray_count - 1))
            ex = self.x + self.ray_length * math.cos(theta)
            ey = self.y + self.ray_length * math.sin(theta)

            min_wall, min_food, min_snake, min_grass, min_water = 1.0, 1.0, 1.0, 1.0, 1.0
            wall_detected = False
            food_detected = False
            snake_detected = False
            grass_detected = False
            water_detected = False

            for wall in wall_coords:
                ix, iy = self._line_intersection_point((self.x, self.y), (ex, ey), wall)
                if ix is not None:
                    d = math.hypot(ix - self.x, iy - self.y) / self.ray_length
                    if d < min_wall:
                        wall_detected = True
                        min_wall = max(min(d, 1.0), 0.0)

            if self.flies_eaten < FLIES_NEEDED:
                for dot in dots:
                    t = self._projection_fraction((self.x, self.y), (ex, ey), (dot["x"], dot["y"]))
                    if t is not None and 0.0 <= t <= 1.0:
                        px = self.x + t * (ex - self.x)
                        py = self.y + t * (ey - self.y)
                        if math.hypot(px - dot["x"], py - dot["y"]) < DOT_RADIUS:
                            d = math.hypot(px - self.x, py - self.y) / self.ray_length
                            if d < min_food:
                                food_detected = True
                                min_food = max(min(d, 1.0), 0.0)

                for snake in snakes:
                    if snake.dead:
                        continue
                    t = self._projection_fraction((self.x, self.y), (ex, ey), (snake.x, snake.y))
                    if t is not None and 0.0 <= t <= 1.0:
                        px = self.x + t * (ex - self.x)
                        py = self.y + t * (ey - self.y)
                        if math.hypot(px - snake.x, py - snake.y) < snake.radius:
                            d = math.hypot(px - self.x, py - self.y) / self.ray_length
                            if d < min_snake:
                                snake_detected = True
                                min_snake = max(min(d, 1.0), 0.0)

                if self.ticks_since_last_food <= STARVATION_IGNORE_GRASS_THRESHOLD:
                    for grass in grass_patches:
                        t = self._projection_fraction((self.x, self.y), (ex, ey), (grass["x"], grass["y"]))
                        if t is not None and 0.0 <= t <= 1.0:
                            px = self.x + t * (ex - self.x)
                            py = self.y + t * (ey - self.y)
                            if math.hypot(px - grass["x"], py - grass["y"]) < grass["radius"]:
                                d = math.hypot(px - self.x, py - self.y) / self.ray_length
                                if d < min_grass:
                                    grass_detected = True
                                    min_grass = max(min(d, 1.0), 0.0)

            if self.flies_eaten >= FLIES_NEEDED:
                for water in water_pools:
                    t = self._projection_fraction((self.x, self.y), (ex, ey), (water["x"], water["y"]))
                    if t is not None and 0.0 <= t <= 1.0:
                        px = self.x + t * (ex - self.x)
                        py = self.y + t * (ey - self.y)
                        if math.hypot(px - water["x"], py - water["y"]) < water["radius"]:
                            d = math.hypot(px - self.x, py - self.y) / self.ray_length
                            if d < min_water:
                                water_detected = True
                                min_water = max(min(d, 1.0), 0.0)

            input_wall = max(0.0, 1.0 - min_wall) if wall_detected else 0.0
            input_food = max(0.0, 1.0 - min_food) if food_detected else 0.0
            input_snake = max(0.0, 1.0 - min_snake) if snake_detected else 0.0
            input_grass = max(0.0, 1.0 - min_grass) if grass_detected else 0.0
            input_water = max(0.0, 1.0 - min_water) if water_detected else 0.0

            if self.flies_eaten >= FLIES_NEEDED:
                inputs.extend([input_wall, 0.0, input_snake, 0.0,input_water])
            else:
                inputs.extend([input_wall, input_food, input_snake, input_grass, input_water])

            if not any([wall_detected, food_detected, snake_detected, grass_detected, water_detected]):
                color = "black"
            elif wall_detected and min_wall <= min(min_food, min_snake, min_grass, min_water):
                color = "red"
            elif food_detected and min_food <= min(min_wall, min_snake, min_grass, min_water):
                color = "saddle brown"
            elif snake_detected and min_snake <= min(min_wall, min_food, min_grass, min_water):
                color = "orange"
            elif grass_detected and min_grass <= min(min_wall, min_food, min_snake, min_water):
                color = "green"
            elif water_detected:
                color = "cyan"
            else:
                color = "black"

            self.canvas.itemconfig(line_id, fill=color)
            self.canvas.coords(line_id, self.x, self.y, ex, ey)
        return inputs

    def _line_intersection_point(self, p1, p2, wall):
        x1, y1 = p1
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
        x1, y1 = p1
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
        u = ((cx - x1) * px + (cy - y1) * py) / float(norm)
        u = max(min(u, 1), 0)
        dx = x1 + u * px - cx
        dy = y1 + u * py - cy
        return dx * dx + dy * dy < r * r

class SnakeSim(AgentSim):
    def __init__(self, canvas, x, y, genome, radius=10):
        super().__init__(canvas, x, y, genome, radius, color="orange", fov_deg=180)

    def update(self, wall_coords, frogs, snakes, grass_patches):
        if self.dead:
            return
        self.ticks_alive += 1

        inputs = self.cast_rays(wall_coords, frogs, snakes, grass_patches, [])

        if all(abs(i) < 1e-6 for i in inputs):
            turn = 0.0
        else:
            turn = self.calculate_turn_response(inputs)
            turn = max(min(turn, 2.0), -2.0)

        base_speed = 3
        left_speed = base_speed - turn
        right_speed = base_speed + turn
        v = (left_speed + right_speed) / 2.0
        omega = (right_speed - left_speed) / self.wheel_base
        self.angle += omega
        self.x += v * math.cos(self.angle)
        self.y += v * math.sin(self.angle)

        self.canvas.coords(
            self.body,
            self.x - self.radius, self.y - self.radius,
            self.x + self.radius, self.y + self.radius
        )

        for wall in wall_coords:
            if self._circle_line_collision(self.x, self.y, self.radius, wall):
                self.canvas.itemconfig(self.body, fill="red")
                self.dead = True
                return

    def cast_rays(self, wall_coords, frogs, snakes, grass_patches, water_pools):
        inputs = []
        start_angle = self.angle - self.fov / 2.0
        for i, line_id in enumerate(self.ray_lines):
            theta = start_angle + i * (self.fov / (self.ray_count - 1))
            ex = self.x + self.ray_length * math.cos(theta)
            ey = self.y + self.ray_length * math.sin(theta)

            min_wall, min_prey, min_grass = 1.0, 1.0, 1.0
            wall_detected = False
            prey_detected = False
            grass_detected = False

            for wall in wall_coords:
                ix, iy = self._line_intersection_point((self.x, self.y), (ex, ey), wall)
                if ix is not None:
                    d = math.hypot(ix - self.x, iy - self.y) / self.ray_length
                    if d < min_wall:
                        wall_detected = True
                        min_wall = max(min(d, 1.0), 0.0)

            for frog in frogs:
                if frog.dead or frog.succeeded:
                    continue
                if frog.in_grass:
                    continue
                t = self._projection_fraction((self.x, self.y), (ex, ey), (frog.x, frog.y))
                if t is not None and 0.0 <= t <= 1.0:
                    px = self.x + t * (ex - self.x)
                    py = self.y + t * (ey - self.y)
                    if math.hypot(px - frog.x, py - frog.y) < frog.radius:
                        d = math.hypot(px - self.x, py - self.y) / self.ray_length
                        if d < min_prey:
                            prey_detected = True
                            min_prey = max(min(d, 1.0), 0.0)

            for grass in grass_patches:
                t = self._projection_fraction((self.x, self.y), (ex, ey), (grass["x"], grass["y"]))
                if t is not None and 0.0 <= t <= 1.0:
                    px = self.x + t * (ex - self.x)
                    py = self.y + t * (ey - self.y)
                    if math.hypot(px - grass["x"], py - grass["y"]) < grass["radius"]:
                        d = math.hypot(px - self.x, py - self.y) / self.ray_length
                        if d < min_grass:
                            grass_detected = True
                            min_grass = max(min(d, 1.0), 0.0)

            input_wall = max(0.0, 1.0 - min_wall) if wall_detected else 0.0
            input_prey = max(0.0, 1.0 - min_prey) if prey_detected else 0.0
            input_grass = max(0.0, 1.0 - min_grass) if grass_detected else 0.0

            inputs.extend([input_wall, input_prey, 0.0, input_grass, 0.0])

            if not wall_detected and not prey_detected and not grass_detected:
                color = "black"
            elif wall_detected and min_wall < min(min_prey, min_grass):
                color = "red"
            elif prey_detected and min_prey < min(min_wall, min_grass):
                color = "purple"
            elif grass_detected:
                color = "green"
            else:
                color = "black"

            self.canvas.itemconfig(line_id, fill=color)
            self.canvas.coords(line_id, self.x, self.y, ex, ey)
        return inputs

class GeneticSimulation:
    def __init__(self, canvas):
        self.canvas = canvas
        self.frogs = []
        self.snakes = []
        self.dots = []
        self.grass_patches = []
        self.water_pools = []

        self.wall_objs = [
            canvas.create_line(0, 0, CANVAS_WIDTH, 0, fill="black", width=3),
            canvas.create_line(CANVAS_WIDTH, 0, CANVAS_WIDTH, CANVAS_HEIGHT, fill="black", width=3),
            canvas.create_line(CANVAS_WIDTH, CANVAS_HEIGHT, 0, CANVAS_HEIGHT, fill="black", width=3),
            canvas.create_line(0, CANVAS_HEIGHT, 0, 0, fill="black", width=3)
        ]
        self.wall_coords = [canvas.coords(wall) for wall in self.wall_objs]

        self.info_text = canvas.create_text(CANVAS_WIDTH + 10, 10, anchor='nw',
                                            font=('Courier', 10), text='', justify='left')

        self.create_grass_patches()
        self.create_water_pools()
        self.load_agents()
        self.spawn_dots()

    def load_agents(self):

        for _, row in frogs_df.iterrows():

            genome = [
                row['wall_gene'] if 'wall_gene' in row.index else ast.literal_eval(row['wall_params'])[0],
                row['food_gene'] if 'food_gene' in row.index else ast.literal_eval(row['food_params'])[0],
                row['snake_gene'] if 'snake_gene' in row.index else ast.literal_eval(row['snake_params'])[0],
                row['grass_gene'] if 'grass_gene' in row.index else ast.literal_eval(row['grass_params'])[0],
                row['water_gene'] if 'water_gene' in row.index else ast.literal_eval(row['water_params'])[0]
            ]

            x = random.randint(50, CANVAS_WIDTH - 50)
            y = random.randint(50, CANVAS_HEIGHT - 50)
            self.frogs.append(FrogSim(self.canvas, x, y, genome))

        for _, row in snakes_df.iterrows():
            genome = [
                row['wall_gene'] if 'wall_gene' in row.index else ast.literal_eval(row['wall_params'])[0],
                row['food_gene'] if 'food_gene' in row.index else ast.literal_eval(row['food_params'])[0],
                row['snake_gene'] if 'snake_gene' in row.index else ast.literal_eval(row['snake_params'])[0],
                row['grass_gene'] if 'grass_gene' in row.index else ast.literal_eval(row['grass_params'])[0],
                row['water_gene'] if 'water_gene' in row.index else ast.literal_eval(row['water_params'])[0]
            ]

            x = random.randint(50, CANVAS_WIDTH - 50)
            y = random.randint(50, CANVAS_HEIGHT - 50)
            self.snakes.append(SnakeSim(self.canvas, x, y, genome))

    def get_best_frog_genome_info(self):
        if not self.frogs:
            return ""

        best_frog = max(self.frogs, key=lambda f: f.fitness)
        genome_str = f"Best Genome: Wall:{best_frog.genome[0]:.2f} Food:{best_frog.genome[1]:.2f} Snake:{best_frog.genome[2]:.2f} Grass:{best_frog.genome[3]:.2f} Water:{best_frog.genome[4]:.2f}"
        return genome_str

    def update(self):

        all_dead = True
        for frog in self.frogs:
            frog.update(self.wall_coords, self.dots, self.snakes, self.grass_patches, self.water_pools)
            if not (frog.dead or frog.succeeded):
                all_dead = False

        for dot in self.dots:
            dot['x'] += dot['dx']
            dot['y'] += dot['dy']

            if dot['x'] < DOT_RADIUS or dot['x'] > CANVAS_WIDTH - DOT_RADIUS:
                dot['dx'] *= -1
            if dot['y'] < DOT_RADIUS or dot['y'] > CANVAS_HEIGHT - DOT_RADIUS:
                dot['dy'] *= -1
            self.canvas.coords(dot['id'], dot['x'] - DOT_RADIUS, dot['y'] - DOT_RADIUS,
                               dot['x'] + DOT_RADIUS, dot['y'] + DOT_RADIUS)

        for frog in self.frogs:
            if frog.dead or frog.succeeded:
                continue
            for dot in list(self.dots):
                if math.hypot(frog.x - dot['x'], frog.y - dot['y']) < frog.radius + DOT_RADIUS:
                    frog.flies_eaten += 1
                    frog.fitness += 100
                    frog.ticks_since_last_food = 0
                    self.canvas.delete(dot['id'])
                    self.dots.remove(dot)

        for snake in self.snakes:
            snake.update(self.wall_coords, self.frogs, [], self.grass_patches)

        for snake in self.snakes:
            if snake.dead:
                continue
            for frog in self.frogs:
                if frog.dead or frog.succeeded or frog.in_grass:
                    continue
                if math.hypot(snake.x - frog.x, snake.y - frog.y) < snake.radius + frog.radius:
                    snake.fitness += 100
                    frog.dead = True
                    self.canvas.itemconfig(frog.body, fill="gray")

        self.update_hud()

        if all_dead:
            return False
        return True

    def update_hud(self):

        hud_lines = []

        hud_lines.append(f"Generation: {latest_gen}")
        hud_lines.append(f"Type: {type}")
        hud_lines.append("")

        hud_lines.append("FROGS:")
        hud_lines.append("ID  Fitness  Wall   Food   Snake  Grass  Water")
        hud_lines.append("-" * 50)

        sorted_frogs = sorted(enumerate(self.frogs), key=lambda x: x[1].fitness, reverse=True)
        for i, (frog_id, frog) in enumerate(sorted_frogs):
            status = "DEAD" if frog.dead else ("SUCCESS" if frog.succeeded else "ALIVE")
            hud_lines.append(f"{frog_id:2d}  {frog.fitness:7.1f}  {frog.genome[0]:5.2f}  {frog.genome[1]:5.2f} "
                             f"{frog.genome[2]:5.2f}  {frog.genome[3]:5.2f}  {frog.genome[4]:5.2f}  [{status}]")

        hud_lines.append("")
        hud_lines.append("SNAKES:")
        hud_lines.append("ID  Fitness  Wall   Food   Snake  Grass  Water")
        hud_lines.append("-" * 50)

        sorted_snakes = sorted(enumerate(self.snakes), key=lambda x: x[1].fitness, reverse=True)
        for i, (snake_id, snake) in enumerate(sorted_snakes):
            status = "DEAD" if snake.dead else "ALIVE"
            hud_lines.append(
                f"{snake_id:2d}  {snake.fitness:7.1f}  {snake.genome[0]:5.2f}  {snake.genome[1]:5.2f}  {snake.genome[2]:5.2f}  {snake.genome[3]:5.2f}  {snake.genome[4]:5.2f}  [{status}]")

        hud_text = "\n".join(hud_lines)
        self.canvas.itemconfig(self.info_text, text=hud_text)

    def create_grass_patches(self):
        for patch in self.grass_patches:
            self.canvas.delete(patch['id'])
        self.grass_patches = []
        min_distance = 100
        for _ in range(NUM_GRASS_PATCHES):
            for _ in range(100):
                x = random.randint(GRASS_RADIUS + 50, CANVAS_WIDTH - GRASS_RADIUS - 50)
                y = random.randint(GRASS_RADIUS + 50, CANVAS_HEIGHT - GRASS_RADIUS - 50)
                if all(math.hypot(x - p['x'], y - p['y']) > min_distance for p in
                       self.grass_patches + self.water_pools):
                    grass_id = self.canvas.create_oval(
                        x - GRASS_RADIUS, y - GRASS_RADIUS,
                        x + GRASS_RADIUS, y + GRASS_RADIUS,
                        fill="#90EE90", stipple="gray50", outline="#006400"
                    )
                    self.grass_patches.append({'id': grass_id, 'x': x, 'y': y, 'radius': GRASS_RADIUS})
                    break

    def create_water_pools(self):
        for pool in self.water_pools:
            self.canvas.delete(pool['id'])
        self.water_pools = []
        min_distance = 150
        for _ in range(NUM_WATER_POOLS):
            for _ in range(100):
                x = random.randint(WATER_RADIUS + 50, CANVAS_WIDTH - WATER_RADIUS - 50)
                y = random.randint(WATER_RADIUS + 50, CANVAS_HEIGHT - WATER_RADIUS - 50)
                if all(math.hypot(x - p['x'], y - p['y']) > min_distance for p in
                       self.grass_patches + self.water_pools):
                    water_id = self.canvas.create_oval(
                        x - WATER_RADIUS, y - WATER_RADIUS,
                        x + WATER_RADIUS, y + WATER_RADIUS,
                        fill="cyan", outline="blue"
                    )
                    self.water_pools.append({'id': water_id, 'x': x, 'y': y, 'radius': WATER_RADIUS})
                    break

    def init_frogs(self):
        for f in self.frogs:
            self.canvas.delete(f.body)
            for line in f.ray_lines:
                self.canvas.delete(line)
        self.frogs = []
        for _ in range(FROG_POP):
            genome = [random.uniform(-1, 1) for _ in range(GENOME_SIZE)]
            x = random.randint(50, CANVAS_WIDTH - 50)
            y = random.randint(50, CANVAS_HEIGHT - 50)
            self.frogs.append(FrogSim(self.canvas, x, y, genome))

    def init_snakes(self):
        for s in self.snakes:
            self.canvas.delete(s.body)
            for line in s.ray_lines:
                self.canvas.delete(line)
        self.snakes = []
        for _ in range(SNAKE_COUNT):
            genome = [random.uniform(-1, 1) for _ in range(GENOME_SIZE)]
            x = random.randint(50, CANVAS_WIDTH - 50)
            y = random.randint(50, CANVAS_HEIGHT - 50)
            self.snakes.append(SnakeSim(self.canvas, x, y, genome))

    def spawn_dots(self):
        for dot in self.dots:
            self.canvas.delete(dot['id'])
        self.dots = []
        for _ in range(NUM_DOTS):
            x = random.randint(50, CANVAS_WIDTH - 50)
            y = random.randint(50, CANVAS_HEIGHT - 50)
            dx = random.uniform(-1, 1)
            dy = random.uniform(-1, 1)
            did = self.canvas.create_oval(
                x - DOT_RADIUS, y - DOT_RADIUS, x + DOT_RADIUS, y + DOT_RADIUS,
                fill="saddle brown"
            )
            self.dots.append({'id': did, 'x': x, 'y': y, 'dx': dx, 'dy': dy})


def run_sim():
    root = tk.Tk()
    root.title(f"Tournament Generation {latest_gen}")

    info_panel_width = 500
    total_width = CANVAS_WIDTH + info_panel_width

    canvas = tk.Canvas(root, width=total_width, height=CANVAS_HEIGHT, bg="white")
    canvas.pack()

    canvas.create_line(CANVAS_WIDTH, 0, CANVAS_WIDTH, CANVAS_HEIGHT, fill="gray", width=2)

    sim = GeneticSimulation(canvas)

    def loop():
        if sim.update():
            root.after(50, loop)

    loop()
    root.mainloop()

if __name__ == "__main__":
    run_sim()