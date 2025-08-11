import os
import tkinter as tk
import math
import random
import numpy as np
import csv
import time

MAX_GENERATIONS = 50

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

CONSENSUS_THRESHOLD = 0.7
EVOLUTION_STEP_SIZE = 0.1
MIN_SUCCESSFUL_BEHAVIORS = 8
DIRECTION_CHANGE_THRESHOLD = 0.8
CONFIDENCE_WINDOW = 3
MOMENTUM_DECAY = 0.9
MOMENTUM_STRENGTH = 0.15

NO_CONSENSUS_RESET_THRESHOLD = 10
INSUFFICIENT_DATA_RESET_THRESHOLD = 15
GENE_RESET_RANGE_MIN = -1.0
GENE_RESET_RANGE_MAX = 1.0

# GA parameters for snake only
RANDOM_UNIFORM_NEG = 0.3
RANDOM_UNIFORM_POS = 0.3
MUTATION_RATE = 0.1
SELECTION_METHOD = 'tournament'
TOURNAMENT_SIZE = 5


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
        self.final_state = None

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

        self.successful_behaviors = {
            'wall_avoidance': [],
            'food_acquisition': [],
            'snake_avoidance': [],
            'grass_entry': [],
            'water_completion': []
        }

        self.last_wall_detected = False
        self.last_food_detected = False
        self.last_snake_detected = False
        self.last_grass_detected = False
        self.last_water_detected = False

    def _record_successful_behavior(self, behavior_type, gene_index):
        gene_value = self.genome[gene_index]
        direction = 'positive' if gene_value >= 0 else 'negative'
        self.successful_behaviors[behavior_type].append((gene_value, direction))

    def update(self, wall_coords, dots, snakes, grass_patches, water_pools):
        if self.dead or self.succeeded:
            return
        self.ticks_alive += 1
        self.ticks_since_last_food += 1

        if self.flies_eaten < FLIES_NEEDED and self.ticks_since_last_food >= STARVATION_THRESHOLD:
            self.dead = True
            self.final_state = 'starvation'
            self.canvas.itemconfig(self.body, fill="gray")
            return

        currently_in_grass = False
        if self.flies_eaten < FLIES_NEEDED:
            for grass in grass_patches:
                if math.hypot(self.x - grass['x'], self.y - grass['y']) < grass['radius']:
                    currently_in_grass = True
                    break
        self.in_grass = currently_in_grass
        self.was_in_grass = self.in_grass

        if self.flies_eaten >= FLIES_NEEDED:
            for water in water_pools:
                if math.hypot(self.x - water['x'], self.y - water['y']) < water['radius']:
                    self.succeeded = True
                    self.dead = True
                    self.final_state = 'success'
                    self.fitness += 500
                    self.canvas.itemconfig(self.body, fill="purple")
                    self.on_water_success()
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
                self.final_state = 'wall'
                return

    def cast_rays(self, wall_coords, dots, snakes, grass_patches, water_pools):
        inputs = []
        start_angle = self.angle - self.fov / 2.0

        wall_detected_now = False
        food_detected_now = False
        snake_detected_now = False
        grass_detected_now = False
        water_detected_now = False

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
                        wall_detected_now = True
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
                                food_detected_now = True
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
                                snake_detected_now = True
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
                                    grass_detected_now = True
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
                                water_detected_now = True
                                min_water = max(min(d, 1.0), 0.0)

            input_wall = max(0.0, 1.0 - min_wall) if wall_detected else 0.0
            input_food = max(0.0, 1.0 - min_food) if food_detected else 0.0
            input_snake = max(0.0, 1.0 - min_snake) if snake_detected else 0.0
            input_grass = max(0.0, 1.0 - min_grass) if grass_detected else 0.0
            input_water = max(0.0, 1.0 - min_water) if water_detected else 0.0

            if self.flies_eaten >= FLIES_NEEDED:
                inputs.extend([input_wall, 0.0, input_snake, 0.0, input_water])
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

        self._track_detections(wall_detected_now, food_detected_now, snake_detected_now,
                               grass_detected_now, water_detected_now)

        return inputs

    def _track_detections(self, wall_detected, food_detected, snake_detected, grass_detected, water_detected):
        if self.last_wall_detected and not wall_detected:
            self._record_successful_behavior('wall_avoidance', 0)

        if self.last_snake_detected and not snake_detected:
            self._record_successful_behavior('snake_avoidance', 2)

        if grass_detected and self.in_grass:
            self._record_successful_behavior('grass_entry', 3)

        self.last_wall_detected = wall_detected
        self.last_food_detected = food_detected
        self.last_snake_detected = snake_detected
        self.last_grass_detected = grass_detected
        self.last_water_detected = water_detected

    def on_food_eaten(self):
        if self.last_food_detected:
            self._record_successful_behavior('food_acquisition', 1)

    def on_water_success(self):
        self._record_successful_behavior('water_completion', 4)


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
                self.final_state = 'wall'
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


class PopulationConsensus:
    def __init__(self):
        self.consensus_threshold = CONSENSUS_THRESHOLD
        self.evolution_step_size = EVOLUTION_STEP_SIZE
        self.min_successful_behaviors = MIN_SUCCESSFUL_BEHAVIORS

        self.recent_confidence = {
            'wall_gene': [],
            'food_gene': [],
            'snake_gene': [],
            'grass_gene': [],
            'water_gene': []
        }

        self.direction_streak = {
            'wall_gene': {'direction': None, 'count': 0},
            'food_gene': {'direction': None, 'count': 0},
            'snake_gene': {'direction': None, 'count': 0},
            'grass_gene': {'direction': None, 'count': 0},
            'water_gene': {'direction': None, 'count': 0}
        }

        self.no_consensus_counters = {
            'wall_gene': 0,
            'food_gene': 0,
            'snake_gene': 0,
            'grass_gene': 0,
            'water_gene': 0
        }

        self.insufficient_data_counters = {
            'wall_gene': 0,
            'food_gene': 0,
            'snake_gene': 0,
            'grass_gene': 0,
            'water_gene': 0
        }

        self.confidence_window = CONFIDENCE_WINDOW
        self.direction_change_threshold = DIRECTION_CHANGE_THRESHOLD

    def update_stuck_gene_counters(self, consensus_results):
        gene_names = ['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene']

        for gene_name in gene_names:
            if gene_name in consensus_results:
                direction, confidence, count = consensus_results[gene_name]

                if direction == 'no_consensus':
                    self.no_consensus_counters[gene_name] += 1
                    self.insufficient_data_counters[gene_name] = 0
                elif direction in ['insufficient_data', 'no_data']:
                    self.insufficient_data_counters[gene_name] += 1
                    self.no_consensus_counters[gene_name] = 0
                else:
                    self.no_consensus_counters[gene_name] = 0
                    self.insufficient_data_counters[gene_name] = 0

    def check_genes_for_reset(self):

        genes_to_reset = {
            'no_consensus': [],
            'insufficient_data': []
        }

        gene_names = ['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene']
        gene_indices = [0, 1, 2, 3, 4]

        for gene_name, gene_idx in zip(gene_names, gene_indices):
            if self.no_consensus_counters[gene_name] >= NO_CONSENSUS_RESET_THRESHOLD:
                genes_to_reset['no_consensus'].append({
                    'name': gene_name,
                    'index': gene_idx,
                    'stuck_count': self.no_consensus_counters[gene_name]
                })
                self.no_consensus_counters[gene_name] = 0

            if self.insufficient_data_counters[gene_name] >= INSUFFICIENT_DATA_RESET_THRESHOLD:
                genes_to_reset['insufficient_data'].append({
                    'name': gene_name,
                    'index': gene_idx,
                    'stuck_count': self.insufficient_data_counters[gene_name]
                })
                self.insufficient_data_counters[gene_name] = 0

        return genes_to_reset

    def reset_stuck_genes(self, frogs, genes_to_reset):
        total_resets = len(genes_to_reset['no_consensus']) + len(genes_to_reset['insufficient_data'])

        if total_resets == 0:
            return 0

        print(f"\nResetting stuck genes (Total: {total_resets})")

        for gene_info in genes_to_reset['no_consensus']:
            gene_name = gene_info['name']
            gene_idx = gene_info['index']
            stuck_count = gene_info['stuck_count']

            print(f"{gene_name}: no consensus for {stuck_count} generations - randomising")

            for frog in frogs:
                new_value = random.uniform(GENE_RESET_RANGE_MIN, GENE_RESET_RANGE_MAX)
                frog.genome[gene_idx] = new_value

            self.direction_streak[gene_name] = {'direction': None, 'count': 0}
            self.recent_confidence[gene_name] = []

        for gene_info in genes_to_reset['insufficient_data']:
            gene_name = gene_info['name']
            gene_idx = gene_info['index']
            stuck_count = gene_info['stuck_count']

            print(f"{gene_name}: INSUFFICIENT_DATA for {stuck_count} generations - randomising")

            for frog in frogs:
                new_value = random.uniform(GENE_RESET_RANGE_MIN, GENE_RESET_RANGE_MAX)
                frog.genome[gene_idx] = new_value

            self.direction_streak[gene_name] = {'direction': None, 'count': 0}
            self.recent_confidence[gene_name] = []

        return total_resets

    def analyze_population_consensus(self, frogs):
        behavior_types = ['wall_avoidance', 'food_acquisition', 'snake_avoidance', 'grass_entry', 'water_completion']
        gene_names = ['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene']

        consensus_results = {}

        for i, behavior_type in enumerate(behavior_types):
            all_behaviors = []
            for frog in frogs:
                all_behaviors.extend(frog.successful_behaviors[behavior_type])

            gene_name = gene_names[i]

            if len(all_behaviors) >= self.min_successful_behaviors:
                positive_count = sum(1 for _, direction in all_behaviors if direction == 'positive')
                negative_count = sum(1 for _, direction in all_behaviors if direction == 'negative')
                total_count = positive_count + negative_count

                if total_count > 0:
                    positive_ratio = positive_count / total_count
                    negative_ratio = negative_count / total_count

                    if positive_ratio >= self.consensus_threshold:
                        raw_direction = 'positive'
                        raw_confidence = positive_ratio
                    elif negative_ratio >= self.consensus_threshold:
                        raw_direction = 'negative'
                        raw_confidence = negative_ratio
                    else:
                        raw_direction = 'no_consensus'
                        raw_confidence = max(positive_ratio, negative_ratio)

                    if raw_direction in ['positive', 'negative']:
                        confidence_entry = (raw_direction, raw_confidence)
                        self.recent_confidence[gene_name].append(confidence_entry)

                        if len(self.recent_confidence[gene_name]) > self.confidence_window:
                            self.recent_confidence[gene_name].pop(0)

                    final_direction, final_confidence = self._check_trend_override(gene_name, raw_direction,
                                                                                   raw_confidence)

                    consensus_results[gene_name] = (final_direction, final_confidence, total_count)
                else:
                    consensus_results[gene_name] = ('no_data', 0, 0)
            else:
                consensus_results[gene_name] = ('insufficient_data', 0, len(all_behaviors))

        return consensus_results

    def _check_trend_override(self, gene_name, raw_direction, raw_confidence):

        current_streak = self.direction_streak[gene_name]
        recent_history = self.recent_confidence[gene_name]

        if current_streak['direction'] is None:
            if raw_direction in ['positive', 'negative']:
                current_streak['direction'] = raw_direction
                current_streak['count'] = 1
            return raw_direction, raw_confidence

        if raw_direction == current_streak['direction']:
            current_streak['count'] += 1
            return raw_direction, raw_confidence

        if raw_direction in ['positive', 'negative'] and raw_direction != current_streak['direction']:

            if len(recent_history) >= 2:

                opposing_confidences = [conf for dir, conf in recent_history if dir == raw_direction]

                if len(opposing_confidences) >= 2:
                    avg_opposing_confidence = sum(opposing_confidences) / len(opposing_confidences)

                    if (avg_opposing_confidence >= self.direction_change_threshold and
                            raw_confidence >= self.direction_change_threshold):
                        print( f" {gene_name}: Trend override {current_streak['direction']} to {raw_direction} (confidence: {raw_confidence:.2%})")

                        current_streak['direction'] = raw_direction
                        current_streak['count'] = 1

                        self.recent_confidence[gene_name] = [(raw_direction, raw_confidence)]

                        return raw_direction, raw_confidence

            weakened_confidence = 0.3
            return current_streak['direction'], weakened_confidence

        return raw_direction, raw_confidence

    def evolve_population_genes(self, frogs, consensus_results):
        evolved_genomes = []

        for frog in frogs:
            new_genome = frog.genome[:]
            gene_names = ['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene']

            for i, gene_name in enumerate(gene_names):
                if gene_name in consensus_results:
                    direction, confidence, count = consensus_results[gene_name]

                    base_step = self.evolution_step_size
                    confidence_multiplier = confidence

                    if confidence >= 0.8:
                        confidence_multiplier *= 1.5

                    sample_multiplier = min(2.0, 1.0 + (count / 50.0))

                    total_adjustment = base_step * confidence_multiplier * sample_multiplier

                    if direction == 'positive':
                        new_genome[i] = min(1.0, new_genome[i] + total_adjustment)
                    elif direction == 'negative':
                        new_genome[i] = max(-1.0, new_genome[i] - total_adjustment)

            evolved_genomes.append(new_genome)

        return evolved_genomes


class GeneticSimulation:
    def __init__(self, canvas):
        self.canvas = canvas
        self.generation = 0
        self.step = 0
        self.population_consensus = PopulationConsensus()
        self.data_writer = None
        self.csv_file = None

        self.start_time = time.time()
        self.converged = False
        self.convergence_generation = None

        self._setup_csv_logging()

        self.latest_consensus_results = None

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

        self.info_text = canvas.create_text(CANVAS_WIDTH + 10, 10, anchor='nw', font=('Courier', 10), text='', justify='left')

        self.create_grass_patches()
        self.create_water_pools()
        self.init_frogs()
        self.init_snakes()
        self.spawn_dots()

    def update_hud(self, consensus_results=None):
        hud_lines = []

        status_text = "Converged" if self.converged else "Learning"
        hud_lines.append(f"Population status - {status_text.lower()}")
        hud_lines.append(f"Generation: {self.generation}")
        hud_lines.append(f"Step: {self.step}/{LIFESPAN}")
        hud_lines.append("")

        current_consensus = consensus_results or self.latest_consensus_results

        hud_lines.append("Population learning:")
        hud_lines.append("Gene      Value      Trend           Confidence   Behaviour count")
        hud_lines.append("-" * 70)

        gene_names = ['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene']
        gene_display = ['Wall', 'Food', 'Snake', 'Grass', 'Water']

        for i, (gene_name, display_name) in enumerate(zip(gene_names, gene_display)):
            if self.frogs:
                avg_value = np.mean([f.genome[i] for f in self.frogs])
            else:
                avg_value = 0.0

            if current_consensus and gene_name in current_consensus:
                direction, confidence, count = current_consensus[gene_name]

                streak_info = self.population_consensus.direction_streak[gene_name]
                if streak_info['direction']:
                    trend = f"{streak_info['direction']} ({streak_info['count']} gen)"
                else:
                    trend = "no trend"

                conf_percent = f"{confidence * 100:.1f}%"
                behavior_count = str(count)
            else:
                trend = "no trend"
                conf_percent = "0.0%"
                behavior_count = "0"

            hud_lines.append(f"{display_name:<9} {avg_value:+7.3f}    {trend:<20} {conf_percent:<12} {behavior_count}")

        hud_lines.append("")

        hud_lines.append("Frogs")
        hud_lines.append("ID      Wall      Food      Snake     Grass     Water     Status")
        hud_lines.append("-" * 70)

        for i, frog in enumerate(self.frogs):
            status = "DEAD" if frog.dead else ("SUCCESS" if frog.succeeded else "ALIVE")
            hud_lines.append(
                f"{i:2d}      {frog.genome[0]:5.2f}     {frog.genome[1]:5.2f}     "
                f"{frog.genome[2]:5.2f}     {frog.genome[3]:5.2f}     {frog.genome[4]:5.2f}     [{status}]")

        hud_lines.append("")

        if SNAKE_COUNT > 0:
            hud_lines.append("Snakes")
            hud_lines.append("ID      Fitness             Wall      Food      Snake     Grass     Water     Status")
            hud_lines.append("-" * 85)

            for i, snake in enumerate(self.snakes):
                status = "DEAD" if snake.dead else "ALIVE"
                hud_lines.append(
                    f"{i:2d}      {snake.fitness:7.1f}             {snake.genome[0]:5.2f}     {snake.genome[1]:5.2f}     "
                    f"{snake.genome[2]:5.2f}     {snake.genome[3]:5.2f}     {snake.genome[4]:5.2f}     [{status}]")

        hud_text = "\n".join(hud_lines)
        self.canvas.itemconfig(self.info_text, text=hud_text)


    def _setup_csv_logging(self):

        os.makedirs("training data/consensus", exist_ok=True)

        filename = "training data/consensus/consensus_learning_main.csv"

        self.csv_file = open(filename, 'w', newline='')
        self.data_writer = csv.writer(self.csv_file)
        self.data_writer.writerow([
            'generation', 'species', 'id', 'fitness_or_confidence',
            'flies_eaten_or_behavior_count', 'final_state_or_direction',
            'wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene',
            'wall_behaviors', 'food_behaviors', 'snake_behaviors', 'grass_behaviors', 'water_behaviors'
        ])
        print(f"Logging data to: {filename}")

    def check_convergence(self, frogs, tolerance=0.01):

        gene_names = ['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene']

        avg_genes = []
        for i in range(5):
            avg_value = np.mean([f.genome[i] for f in frogs])
            avg_genes.append(avg_value)

        converged_genes = []
        for i, avg_value in enumerate(avg_genes):
            if avg_value >= (1.0 - tolerance) or avg_value <= (-1.0 + tolerance):
                converged_genes.append(gene_names[i])

        if len(converged_genes) == 5:
            return True, converged_genes, avg_genes
        else:
            return False, converged_genes, avg_genes

    def get_best_frog_genome_info(self):
        if not self.frogs:
            return ""

        avg_genes = []
        for i in range(5):
            avg_value = np.mean([f.genome[i] for f in self.frogs])
            avg_genes.append(avg_value)

        genome_str = (f"Gen {self.generation} Avg Genes: Wall:{avg_genes[0]:.3f} Food:{avg_genes[1]:.3f} "
                      f"Snake:{avg_genes[2]:.3f} Grass:{avg_genes[3]:.3f} Water:{avg_genes[4]:.3f}")
        return genome_str

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
                    self.canvas.tag_lower(grass_id)
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

                    self.canvas.tag_lower(water_id)
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

    def update(self):

        all_dead = True
        for f in self.frogs:
            f.update(self.wall_coords, self.dots, self.snakes, self.grass_patches, self.water_pools)
            if not (f.dead or f.succeeded):
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
                    frog.on_food_eaten()
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
                    frog.final_state = 'eaten_by_snake'
                    self.canvas.itemconfig(frog.body, fill="gray")

        self.update_hud()

        if all_dead or self.step > LIFESPAN:
            self.evolve_frogs_by_consensus()

            if self.converged or self.generation >= MAX_GENERATIONS:
                if self.csv_file:
                    self.csv_file.close()
                    print(f"Simulation completed. Data logged to CSV file.")
                return False

            self.create_grass_patches()
            self.create_water_pools()
            self.spawn_dots()
            self.reset_frogs_for_next_generation()
            if SNAKE_COUNT > 0:
                self.evolve_snakes()
            self.step = 0

        self.step += 1
        return True

    def evolve_frogs_by_consensus(self):
        print(f"Generation {self.generation} - pop consensus: ")

        is_converged, converged_genes, avg_genes = self.check_convergence(self.frogs)

        if is_converged and not self.converged:
            self.converged = True
            self.convergence_generation = self.generation
            elapsed_time = time.time() - self.start_time

            print(f"\nPop genes converged")
            print(f"Generation: {self.generation}")
            print(f"Time elapsed: {elapsed_time:.2f} secs ({elapsed_time / 60:.2f} mins)")
            print(f"All genes have converged:")
            for i, gene_name in enumerate(['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene']):
                print(f"  {gene_name}: {avg_genes[i]:+.3f}")
            return

        if len(converged_genes) > 0:
            print(f"Not all converged: {len(converged_genes)}/5 genes converged: {converged_genes}")

        consensus_results = self.population_consensus.analyze_population_consensus(self.frogs)

        self.latest_consensus_results = consensus_results

        self.population_consensus.update_stuck_gene_counters(consensus_results)

        genes_to_reset = self.population_consensus.check_genes_for_reset()

        num_resets = self.population_consensus.reset_stuck_genes(self.frogs, genes_to_reset)

        print("\n======= Consensus analysis")
        for gene_name, (direction, confidence, count) in consensus_results.items():
            gene_index = ['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene'].index(gene_name)
            current_avg = np.mean([f.genome[gene_index] for f in self.frogs])

            streak_info = self.population_consensus.direction_streak[gene_name]
            trend_indicator = f"({streak_info['direction']} trend, {streak_info['count']} gens)" if streak_info[
                'direction'] else "(no trend)"

            print(f"{gene_name}: {direction} (conf: {confidence:.2%}, behaviors: {count}) {trend_indicator}")
            print(f"  Current avg: {current_avg:+.3f}")

        self.record_consensus_data(consensus_results, num_resets)

        evolved_genomes = self.population_consensus.evolve_population_genes(self.frogs, consensus_results)

        for i, genome in enumerate(evolved_genomes):
            if i < len(self.frogs):
                self.frogs[i].genome = genome

    def record_consensus_data(self, consensus_results, num_resets=0):
        for idx, f in enumerate(self.frogs):

            wall_behaviors = len(f.successful_behaviors['wall_avoidance'])
            food_behaviors = len(f.successful_behaviors['food_acquisition'])
            snake_behaviors = len(f.successful_behaviors['snake_avoidance'])
            grass_behaviors = len(f.successful_behaviors['grass_entry'])
            water_behaviors = len(f.successful_behaviors['water_completion'])

            self.data_writer.writerow([
                self.generation,
                'frog',
                idx,
                0,
                f.flies_eaten,
                f.final_state or 'alive',
                f.genome[0], f.genome[1], f.genome[2], f.genome[3], f.genome[4],
                wall_behaviors, food_behaviors, snake_behaviors, grass_behaviors, water_behaviors
            ])

        for gene_name, (direction, confidence, count) in consensus_results.items():
            self.data_writer.writerow([
                self.generation,
                'consensus',
                gene_name,
                confidence,
                count,
                direction,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0
            ])

        if num_resets > 0:
            self.data_writer.writerow([
                self.generation,
                'reset_event',
                'gene_reset',
                num_resets,
                0,
                'genes_randomized',
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0
            ])

    def reset_frogs_for_next_generation(self):

        for frog in self.frogs:
            frog.x = random.randint(50, CANVAS_WIDTH - 50)
            frog.y = random.randint(50, CANVAS_HEIGHT - 50)
            frog.angle = random.uniform(0, 2 * math.pi)
            frog.dead = False
            frog.succeeded = False
            frog.flies_eaten = 0
            frog.ticks_alive = 0
            frog.ticks_since_last_food = 0
            frog.final_state = None
            frog.in_grass = False
            frog.was_in_grass = False

            for behavior_type in frog.successful_behaviors:
                frog.successful_behaviors[behavior_type] = []

            frog.last_wall_detected = False
            frog.last_food_detected = False
            frog.last_snake_detected = False
            frog.last_grass_detected = False
            frog.last_water_detected = False

            self.canvas.itemconfig(frog.body, fill="green")

    def evolve_snakes(self):
        for idx, s in enumerate(self.snakes):
            self.data_writer.writerow([
                self.generation,
                'snake',
                idx,
                s.fitness,
                0,
                s.final_state or 'alive',
                s.genome[0], s.genome[1], s.genome[2], s.genome[3], s.genome[4],
                0, 0, 0, 0, 0
            ])

        if SNAKE_COUNT > 0:
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

            for i, genome in enumerate(new_genomes):
                self.snakes[i].genome = genome
                self.snakes[i].x = random.randint(50, CANVAS_WIDTH - 50)
                self.snakes[i].y = random.randint(50, CANVAS_HEIGHT - 50)
                self.snakes[i].angle = random.uniform(0, 2 * math.pi)
                self.snakes[i].dead = False
                self.snakes[i].fitness = 0.0
                self.snakes[i].ticks_alive = 0
                self.snakes[i].final_state = None
                self.canvas.itemconfig(self.snakes[i].body, fill="orange")

        self.generation += 1


def run_sim():
    root = tk.Tk()
    root.title("Consensus learning frogs")

    info_panel_width = 600
    total_width = CANVAS_WIDTH + info_panel_width

    canvas = tk.Canvas(root, width=total_width, height=CANVAS_HEIGHT, bg="white")
    canvas.pack()

    canvas.create_line(CANVAS_WIDTH, 0, CANVAS_WIDTH, CANVAS_HEIGHT, fill="gray", width=2)

    sim = GeneticSimulation(canvas)

    def loop():
        if sim.update():
            root.after(50, loop)
        else:
            print("Sim finished")

    loop()
    root.mainloop()


if __name__ == "__main__":
    print("Starting sim")
    run_sim()