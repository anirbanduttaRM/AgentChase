import pygame
import heapq  # For A* pathfinding

# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 10
CELL_SIZE = 60
SCREEN_SIZE = GRID_SIZE * CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
MAX_STEPS = 50  # Rabbit wins if it takes more than 50 steps

# Load Images
rabbit_img = pygame.image.load("rabbit.png")
rabbit_img = pygame.transform.scale(rabbit_img, (CELL_SIZE, CELL_SIZE))

wolf1_img = pygame.image.load("wolf1.png")
wolf1_img = pygame.transform.scale(wolf1_img, (CELL_SIZE, CELL_SIZE))

wolf2_img = pygame.image.load("wolf2.png")
wolf2_img = pygame.transform.scale(wolf2_img, (CELL_SIZE, CELL_SIZE))

burrow_img = pygame.image.load("burrow.png")
burrow_img = pygame.transform.scale(burrow_img, (CELL_SIZE, CELL_SIZE))

# Initialize Screen
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("AI Wolves vs Rabbit")

# Clock
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)

# --- A* Pathfinding ---
def astar_path(start, goal):
    """Finds the shortest path using A* algorithm."""
    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan Distance

    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        neighbors = [
            (current[0] + dx, current[1] + dy)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= current[0] + dx < GRID_SIZE and 0 <= current[1] + dy < GRID_SIZE
        ]

        for neighbor in neighbors:
            temp_g_score = g_score[current] + 1
            if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return []  # No path found

    # --- Rabbit Movement Prediction ---
def predict_rabbit_move(rabbit, prev_pos):
    """Predicts where the rabbit is likely to move next."""
    dx = rabbit.x - prev_pos[0]
    dy = rabbit.y - prev_pos[1]

    predicted_x = rabbit.x + dx if 0 <= rabbit.x + dx < GRID_SIZE else rabbit.x
    predicted_y = rabbit.y + dy if 0 <= rabbit.y + dy < GRID_SIZE else rabbit.y

    return (predicted_x, predicted_y)

# --- Classes ---
class Rabbit:
    def __init__(self):
        self.x = 0
        self.y = GRID_SIZE - 1
        self.steps = 0
        self.prev_x = self.x  # Store previous position
        self.prev_y = self.y

    def move(self, direction):
        self.prev_x, self.prev_y = self.x, self.y  # Update previous position
        if direction == "UP" and self.y > 0:
            self.y -= 1
        elif direction == "DOWN" and self.y < GRID_SIZE - 1:
            self.y += 1
        elif direction == "LEFT" and self.x > 0:
            self.x -= 1
        elif direction == "RIGHT" and self.x < GRID_SIZE - 1:
            self.x += 1
        self.steps += 1

    def draw(self):
        screen.blit(rabbit_img, (self.x * CELL_SIZE, self.y * CELL_SIZE))

def get_intercept_point(rabbit):
    dx = rabbit.x - rabbit.prev_x
    dy = rabbit.y - rabbit.prev_y
    intercept_x = rabbit.x + dx * 2
    intercept_y = rabbit.y + dy * 2
    return (
        max(0, min(GRID_SIZE - 1, intercept_x)),
        max(0, min(GRID_SIZE - 1, intercept_y))
    )

class Wolf:
    def __init__(self, wolf_id):
        if wolf_id == 0:
            self.x, self.y = 0, 0  # Wolf 1 starts top-left
        else:
            self.x, self.y = GRID_SIZE - 1, GRID_SIZE - 1  # Wolf 2 starts bottom-right
        
        self.id = wolf_id
        self.strategy = "CHASE"
        self.commentary = ""

    def move(self, rabbit, other_wolf):
        goal = predict_rabbit_move(rabbit, (rabbit.prev_x, rabbit.prev_y))

        if self.strategy == "CHASE":
            path = astar_path((self.x, self.y), goal)
            if path:
                self.x, self.y = path[0]
            self.commentary = "Taking shortest path."

        elif self.strategy == "FLANK":
    # Move towards a position blocking the rabbit's path
            flank_x = max(0, min(GRID_SIZE - 1, rabbit.x + (1 if rabbit.x < GRID_SIZE // 2 else -1)))
            flank_y = max(0, min(GRID_SIZE - 1, rabbit.y + (1 if rabbit.y < GRID_SIZE // 2 else -1)))
            path = astar_path((self.x, self.y), (flank_x, flank_y))
            if path:
                self.x, self.y = path[0]
            self.commentary = "Flanking to cut off escape."
        elif self.strategy == "INTERCEPT":
            intercept_point = get_intercept_point(rabbit)
            path = astar_path((self.x, self.y), intercept_point)
            if path:
                self.x, self.y = path[0]
            self.commentary = f"Intercepting at ({intercept_point[0]}, {intercept_point[1]})!"


        # Prevent wolves from colliding
        if self.x == other_wolf.x and self.y == other_wolf.y:
            self.x = max(0, self.x - 1)

    def draw(self):
        screen.blit(wolf1_img if self.id == 0 else wolf2_img, (self.x * CELL_SIZE, self.y * CELL_SIZE))
        label = font.render(f"Wolf {self.id + 1}", True, BLACK)
        screen.blit(label, (self.x * CELL_SIZE + 10, self.y * CELL_SIZE - 20))

def is_rabbit_heading_to_goal(rabbit):
    goal_x, goal_y = GRID_SIZE - 1, 0
    dx = rabbit.x - rabbit.prev_x
    dy = rabbit.y - rabbit.prev_y
    moving_towards_x = (dx > 0 and rabbit.x <= goal_x) or (dx < 0 and rabbit.x >= goal_x)
    moving_towards_y = (dy < 0 and rabbit.y >= goal_y) or (dy > 0 and rabbit.y <= goal_y)
    return moving_towards_x or moving_towards_y

class CoordinatorAgent:
    def __init__(self, wolves):
        self.wolves = wolves

    def assign_strategies(self, rabbit):
        distances = sorted(self.wolves, key=lambda w: abs(w.x - rabbit.x) + abs(w.y - rabbit.y))
        rabbit_goal = (GRID_SIZE - 1, 0)

        if rabbit.steps > 15 or is_rabbit_heading_to_goal(rabbit):
            distances[0].strategy = "CHASE"
            distances[1].strategy = "INTERCEPT"
        else:
            distances[0].strategy = "CHASE"
            distances[1].strategy = "FLANK"

# --- Functions ---
def draw_grid():
    for x in range(0, SCREEN_SIZE, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, SCREEN_SIZE), 1)
    for y in range(0, SCREEN_SIZE, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (SCREEN_SIZE, y), 1)
    screen.blit(burrow_img, ((GRID_SIZE - 1) * CELL_SIZE, 0))  # Burrow in top-right

def draw_debug_info(rabbit, wolves):
    font_debug = pygame.font.Font(None, 20)
    y_offset = 10

    step_text = font_debug.render(f"Rabbit Steps: {rabbit.steps}/{MAX_STEPS}", True, BLACK)
    screen.blit(step_text, (10, y_offset))
    y_offset += 20

    for wolf in wolves:
        text = font_debug.render(f"Wolf {wolf.id + 1}: {wolf.commentary}", True, BLACK)
        screen.blit(text, (10, y_offset))
        y_offset += 20

def check_game_over(rabbit, wolves):
    if rabbit.steps > MAX_STEPS:
        return "TIMEOUT"  # Rabbit wins after 50 moves
    for wolf in wolves:
        if wolf.x == rabbit.x and wolf.y == rabbit.y:
            return "CAUGHT"
    if rabbit.x == GRID_SIZE - 1 and rabbit.y == 0:
        return "ESCAPED"
    return None

def show_game_over(result, rabbit):
    screen.fill(WHITE)
    font = pygame.font.Font(None, 50)
    text = font.render(f"Game Over! {'Rabbit Escaped' if result in ['ESCAPED', 'TIMEOUT'] else 'Rabbit Caught'}", True, RED)
    screen.blit(text, (SCREEN_SIZE // 6, SCREEN_SIZE // 2))
    pygame.display.flip()
    pygame.time.delay(5000)
    pygame.quit()
    exit()

# --- Game Setup ---
rabbit = Rabbit()
wolves = [Wolf(i) for i in range(2)]
coordinator = CoordinatorAgent(wolves)

running = True
while running:
    screen.fill(WHITE)
    draw_grid()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            rabbit.move(pygame.key.name(event.key).upper())
            coordinator.assign_strategies(rabbit)
            for wolf in wolves:
                wolf.move(rabbit, wolves[1 - wolf.id])
            if result := check_game_over(rabbit, wolves): show_game_over(result, rabbit)

    rabbit.draw()
    for wolf in wolves: wolf.draw()
    draw_debug_info(rabbit, wolves)
    pygame.display.flip()
    clock.tick(10)

pygame.quit()
