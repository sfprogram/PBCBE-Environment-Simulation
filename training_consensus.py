import math
import random
import numpy as np
import csv
import json
import time

MAX_GENERATIONS = 100
ENSEMBLE_SEEDS = [424242, 85123, 13579, 73421, 29374]

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

NO_CONSENSUS_RESET_THRESHOLD = 10
INSUFFICIENT_DATA_RESET_THRESHOLD = 15
GENE_RESET_RANGE_MIN = -1.0
GENE_RESET_RANGE_MAX = 1.0

ENSEMBLE_RUNS = len(ENSEMBLE_SEEDS)
ENSEMBLE_SUCCESS_THRESHOLD = 0.6
ENSEMBLE_CONVERGENCE_SIMILARITY = 0.2


# GA parameters for snake only
RANDOM_UNIFORM_NEG = 0.3
RANDOM_UNIFORM_POS = 0.3
MUTATION_RATE = 0.1
SELECTION_METHOD = 'tournament'
REPLACEMENT_RATE = 0.5
TOURNAMENT_SIZE = 5


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
            return

        currently_in_grass = False
        if self.flies_eaten < FLIES_NEEDED:
            for g in grass_patches:
                if math.hypot(self.x - g['x'], self.y - g['y']) < g['radius']:
                    currently_in_grass = True
                    break
        self.in_grass = currently_in_grass

        if self.flies_eaten >= FLIES_NEEDED:
            for w in water_pools:
                if math.hypot(self.x - w['x'], self.y - w['y']) < w['radius']:
                    self.succeeded = True
                    self.dead = True
                    self.final_state = 'success'
                    self.on_water_success()
                    return

        inputs = []
        wall_detected_now = False
        food_detected_now = False
        snake_detected_now = False
        grass_detected_now = False
        water_detected_now = False

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

            if w_det: wall_detected_now = True
            if f_det: food_detected_now = True
            if s_det: snake_detected_now = True
            if g_det: grass_detected_now = True
            if w2_det: water_detected_now = True

            iw = max(0.0, 1 - min_wall) if w_det else 0.0
            if self.flies_eaten >= FLIES_NEEDED:
                inputs.extend([iw, 0.0,
                               max(0.0, 1 - min_snake) if s_det else 0.0,
                               0.0,
                               max(0.0, 1 - min_water) if w2_det else 0.0])
            else:
                inputs.extend([iw,
                               max(0.0, 1 - min_food) if f_det else 0.0,
                               max(0.0, 1 - min_snake) if s_det else 0.0,
                               max(0.0, 1 - min_grass) if g_det else 0.0,
                               max(0.0, 1 - min_water) if w2_det else 0.0])

        self._track_detections(wall_detected_now, food_detected_now, snake_detected_now,
                               grass_detected_now, water_detected_now)

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

        self.confidence_window = 3
        self.direction_change_threshold = 0.8

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

        for gene_info in genes_to_reset['no_consensus']:
            gene_name = gene_info['name']
            gene_idx = gene_info['index']
            stuck_count = gene_info['stuck_count']

            print(f"{gene_name}: no consensus for {stuck_count} generations - randomising")

            for frog in frogs:
                new_value = random.uniform(GENE_RESET_RANGE_MIN, GENE_RESET_RANGE_MAX)
                old_value = frog.genome[gene_idx]
                frog.genome[gene_idx] = new_value

            self.direction_streak[gene_name] = {'direction': None, 'count': 0}
            self.recent_confidence[gene_name] = []

        for gene_info in genes_to_reset['insufficient_data']:
            gene_name = gene_info['name']
            gene_idx = gene_info['index']
            stuck_count = gene_info['stuck_count']

            print(f"{gene_name}: insufficient data for {stuck_count} generations - randomising")

            for frog in frogs:
                new_value = random.uniform(GENE_RESET_RANGE_MIN, GENE_RESET_RANGE_MAX)
                old_value = frog.genome[gene_idx]
                frog.genome[gene_idx] = new_value

            self.direction_streak[gene_name] = {'direction': None, 'count': 0}
            self.recent_confidence[gene_name] = []

        return total_resets

    def get_stuck_gene_status(self):
        gene_names = ['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene']

        any_approaching_reset = False

        for gene_name in gene_names:
            no_consensus_count = self.no_consensus_counters[gene_name]
            insufficient_data_count = self.insufficient_data_counters[gene_name]

            status_parts = []

            if no_consensus_count > 0:
                remaining = NO_CONSENSUS_RESET_THRESHOLD - no_consensus_count
                status_parts.append(f"NO_CONSENSUS: {no_consensus_count}/{NO_CONSENSUS_RESET_THRESHOLD} "
                                    f"({remaining} until reset)")
                if remaining <= 2:
                    any_approaching_reset = True

            if insufficient_data_count > 0:
                remaining = INSUFFICIENT_DATA_RESET_THRESHOLD - insufficient_data_count
                status_parts.append(f"INSUFFICIENT_DATA: {insufficient_data_count}/{INSUFFICIENT_DATA_RESET_THRESHOLD}"
                                    f" ({remaining} until reset)")
                if remaining <= 2:
                    any_approaching_reset = True

            if status_parts:
                print(f"{gene_name}: {', '.join(status_parts)}")
            else:
                print(f"{gene_name}: making progress")

        if any_approaching_reset:
            print("Some genes approaching reset threshold")

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
                        print(f"{gene_name}: Trend override {current_streak['direction']} → {raw_direction} "
                              f"(confidence: {raw_confidence:.2%})")

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
    def __init__(self, data_writer):
        self.data_writer = data_writer
        self.generation = 0
        self.step = 0
        self.population_consensus = PopulationConsensus()

        self.start_time = time.time()
        self.converged = False
        self.convergence_generation = None

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
                    f.on_food_eaten()
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

        self.step += 1

        if all_dead or self.step > LIFESPAN:
            self.evolve_frogs_by_consensus()

            if self.converged:
                return

            if SNAKE_COUNT != 0:
                self.evolve_snakes()
            else:
                self.generation += 1
                if self.generation < MAX_GENERATIONS:
                    print(f"Starting Generation {self.generation}")
                    self.create_grass_patches()
                    self.create_water_pools()
                    self.spawn_dots()
                    self.init_frogs()
                    self.step = 0

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

    def evolve_frogs_by_consensus(self):
        print(f"Generation {self.generation} - analyzing population consensus\n")

        is_converged, converged_genes, avg_genes = self.check_convergence(self.frogs)

        if is_converged and not self.converged:
            self.converged = True
            self.convergence_generation = self.generation
            elapsed_time = time.time() - self.start_time

            print(f"\nConvergence achieved")
            print(f"Generation: {self.generation}")
            print(f"Time elapsed: {elapsed_time:.2f} secs ({elapsed_time/60:.2f} mins)")
            print(f"All genes have converged:")
            for i, gene_name in enumerate(['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene']):
                print(f"  {gene_name}: {avg_genes[i]:+.3f}")
            return

        if len(converged_genes) > 0:
            print(f"Some convergence: {len(converged_genes)}/5 genes converged: {converged_genes}")

        consensus_results = self.population_consensus.analyze_population_consensus(self.frogs)
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

        if self.generation > 0:
            self.population_consensus.get_stuck_gene_status()

        self.record_consensus_data(consensus_results)

        evolved_genomes = self.population_consensus.evolve_population_genes(self.frogs, consensus_results)

        total_changes = 0
        for i, genome in enumerate(evolved_genomes):
            if i < len(self.frogs):
                old_genome = self.frogs[i].genome[:]
                self.frogs[i].genome = genome

                for j in range(len(genome)):
                    total_changes += abs(genome[j] - old_genome[j])


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

        self.generation += 1

        if self.generation < MAX_GENERATIONS:
            self.create_grass_patches()
            self.create_water_pools()
            self.spawn_dots()

            for frog in self.frogs:
                frog.x = random.randint(50, CANVAS_WIDTH - 50)
                frog.y = random.randint(50, CANVAS_HEIGHT - 50)
                frog.dead = False
                frog.succeeded = False
                frog.flies_eaten = 0
                frog.ticks_alive = 0
                frog.ticks_since_last_food = 0
                frog.final_state = None
                for behavior_type in frog.successful_behaviors:
                    frog.successful_behaviors[behavior_type] = []

            self.init_snakes()
            self.step = 0



class EnsembleRunner:
    def __init__(self):
        self.runs_data = []
        self.successful_runs = []

    def run_ensemble_simulation(self, num_runs=ENSEMBLE_RUNS):

        for run_id in range(min(num_runs, len(ENSEMBLE_SEEDS))):
            print(f"\n---------- Starting Ensemble Run {run_id + 1}/{num_runs}")

            seed = ENSEMBLE_SEEDS[run_id]
            random.seed(seed)
            np.random.seed(seed)

            run_data = self._run_single_simulation(run_id, seed)
            self.runs_data.append(run_data)

            random.seed()

        return self._analyze_ensemble_results()

    def _run_single_simulation(self, run_id, seed):
        filename = f"training data/consensus/ensemble_run_{run_id}_seed_{seed}.csv"

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'generation', 'species', 'id', 'fitness_or_confidence',
                'flies_eaten_or_behavior_count', 'final_state_or_direction',
                'wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene',
                'wall_behaviors', 'food_behaviors', 'snake_behaviors', 'grass_behaviors', 'water_behaviors'
            ])

            sim = GeneticSimulation(writer)
            start_time = time.time()

            while sim.generation < MAX_GENERATIONS and not sim.converged:
                sim.update()

            end_time = time.time()

            final_genes = None
            converged = sim.converged
            final_generation = sim.generation

            if len(sim.frogs) > 0:
                final_genes = []
                for gene_idx in range(GENOME_SIZE):
                    avg_gene = np.mean([frog.genome[gene_idx] for frog in sim.frogs])
                    final_genes.append(avg_gene)

            run_data = {
                'run_id': run_id,
                'seed': seed,
                'converged': converged,
                'final_generation': final_generation,
                'final_genes': final_genes,
                'runtime': end_time - start_time,
                'filename': filename,
                'convergence_generation': sim.convergence_generation
            }

            print(f"Run {run_id + 1} completed: {'converged' if converged else 'MAX_GENS'} at gen {final_generation}")
            if final_genes:
                gene_str = ", ".join([f"{g:+.3f}" for g in final_genes])
                print(f"Final genes: [{gene_str}]")

            return run_data

    def _analyze_ensemble_results(self):

        converged_runs = [run for run in self.runs_data if run['converged']]
        success_rate = len(converged_runs) / len(self.runs_data)

        print(f"Convergence rate: {len(converged_runs)}/{len(self.runs_data)} ({success_rate:.1%})")

        if len(converged_runs) < 2:
            print("Insufficient converged runs")
            return self._handle_insufficient_convergence()

        consistent_runs = self._find_consistent_runs(converged_runs)
        consistency_rate = len(consistent_runs) / len(converged_runs) if converged_runs else 0

        print(f"Consistent runs: {len(consistent_runs)}/{len(converged_runs)} ({consistency_rate:.1%})")

        if consistency_rate >= ENSEMBLE_SUCCESS_THRESHOLD:
            consensus_genes = self._compute_consensus_genes(consistent_runs)
            self._report_ensemble_success(consistent_runs, consensus_genes)
            return {
                'success': True,
                'consensus_genes': consensus_genes,
                'consistent_runs': consistent_runs,
                'all_runs': self.runs_data
            }
        else:
            print("Results not reliable")
            return self._handle_high_variance()

    def _find_consistent_runs(self, converged_runs):
        if len(converged_runs) < 2:
            return converged_runs

        consistent_runs = []
        gene_names = ['wall_gene', 'food_gene', 'snake_gene', 'grass_gene', 'water_gene']

        reference_run = converged_runs[0]
        consistent_runs.append(reference_run)

        for run in converged_runs[1:]:
            is_consistent = True

            for gene_idx in range(GENOME_SIZE):
                ref_value = reference_run['final_genes'][gene_idx]
                run_value = run['final_genes'][gene_idx]

                if abs(ref_value - run_value) > ENSEMBLE_CONVERGENCE_SIMILARITY:
                    is_consistent = False
                    print(f"  Run {run['run_id']} inconsistent: {gene_names[gene_idx]} "
                          f"{ref_value:+.3f} vs {run_value:+.3f} (diff: {abs(ref_value - run_value):.3f})")
                    break

            if is_consistent:
                consistent_runs.append(run)

        return consistent_runs

    def _compute_consensus_genes(self, consistent_runs):
        consensus_genes = []

        for gene_idx in range(GENOME_SIZE):
            gene_values = [run['final_genes'][gene_idx] for run in consistent_runs]
            avg_value = np.mean(gene_values)
            std_value = np.std(gene_values)
            consensus_genes.append((avg_value, std_value))

        return consensus_genes

    def _report_ensemble_success(self, consistent_runs, consensus_genes):

        print("\nReliable results achieved")

        gene_names = ['Wall Avoidance', 'Food Seeking', 'Snake Avoidance', 'Grass Seeking', 'Water Seeking']

        print("\n--------------- Pop consensus gene values")
        for i, (gene_name, (avg_val, std_val)) in enumerate(zip(gene_names, consensus_genes)):
            direction = "Attract" if avg_val > 0 else "Avoid"
            confidence = "High" if std_val < 0.1 else "Medium" if std_val < 0.2 else "Low"
            print(f"{gene_name:15}: {avg_val:+.3f} ± {std_val:.3f} → {direction} ({confidence} confidence)")

        print(f"\n---------------- Details")
        for run in consistent_runs:
            print(f"Run {run['run_id']}: Gen {run['final_generation']}, "
                  f"Time: {run['runtime']:.1f}s, Seed: {run['seed']}")

    def _handle_insufficient_convergence(self):

        return {'success': False, 'reason': 'insufficient_convergence', 'all_runs': self.runs_data}

    def _handle_high_variance(self):

        return {'success': False, 'reason': 'high_variance', 'all_runs': self.runs_data}

    def save_ensemble_summary(self, results):
        summary_file = "training data/consensus/ensemble_summary.json"

        summary_data = {
            'ensemble_parameters': {
                'num_runs': len(self.runs_data),
                'success_threshold': ENSEMBLE_SUCCESS_THRESHOLD,
                'convergence_similarity': ENSEMBLE_CONVERGENCE_SIMILARITY
            },
            'results': results,
            'run_details': self.runs_data
        }

        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)

        print(f"Ensemble summary saved to {summary_file}")


def run_ensemble_sim():
    print("======= Start sim")
    print(f"Running {ENSEMBLE_RUNS} sims")

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
        'CONSENSUS_THRESHOLD': CONSENSUS_THRESHOLD,
        'EVOLUTION_STEP_SIZE': EVOLUTION_STEP_SIZE,
        'MIN_SUCCESSFUL_BEHAVIORS': MIN_SUCCESSFUL_BEHAVIORS,
        'ENSEMBLE_RUNS': ENSEMBLE_RUNS,
        'ENSEMBLE_SUCCESS_THRESHOLD': ENSEMBLE_SUCCESS_THRESHOLD,
        'ENSEMBLE_CONVERGENCE_SIMILARITY': ENSEMBLE_CONVERGENCE_SIMILARITY,
        'RANDOM_UNIFORM_NEG': RANDOM_UNIFORM_NEG,
        'RANDOM_UNIFORM_POS': RANDOM_UNIFORM_POS,
        'MUTATION_RATE': MUTATION_RATE
    }

    with open("training data/consensus/ensemble_params.json", "w") as f:
        json.dump(params, f, indent=2)

    ensemble = EnsembleRunner()
    results = ensemble.run_ensemble_simulation()
    ensemble.save_ensemble_summary(results)

    if results['success']:
        print("\nEnsemble success")
        consensus_genes = results['consensus_genes']

        print("\nBest gene values")
        for i, (avg_val, std_val) in enumerate(consensus_genes):
            print(f"Gene {i}: {avg_val:.6f}")

    else:
        print(f"\nEnsemble failed: {results['reason']}")

    return results


if __name__ == "__main__":
    run_ensemble_sim()