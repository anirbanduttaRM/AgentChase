import pygame
import heapq
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv

# --- Config ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Pygame
pygame.init()

# Constants
GRID_SIZE = 10
CELL_SIZE = 60
SCREEN_SIZE = GRID_SIZE * CELL_SIZE
WHITE, BLACK, RED = (255, 255, 255), (0, 0, 0), (255, 0, 0)
MAX_STEPS = 50

# Load Images
rabbit_img = pygame.transform.scale(pygame.image.load("rabbit.png"), (CELL_SIZE, CELL_SIZE))
wolf1_img = pygame.transform.scale(pygame.image.load("wolf1.png"), (CELL_SIZE, CELL_SIZE))
wolf2_img = pygame.transform.scale(pygame.image.load("wolf2.png"), (CELL_SIZE, CELL_SIZE))
burrow_img = pygame.transform.scale(pygame.image.load("burrow.png"), (CELL_SIZE, CELL_SIZE))

# Initialize Screen
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("AI Wolves vs Rabbit")
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

# --- Rabbit and Wolves ---
class Rabbit:
    def __init__(self):
        self.x = 0
        self.y = GRID_SIZE - 1
        self.steps = 0
        self.prev_x, self.prev_y = self.x, self.y

    def move(self, direction):
        self.prev_x, self.prev_y = self.x, self.y
        if direction == "UP" and self.y > 0: self.y -= 1
        elif direction == "DOWN" and self.y < GRID_SIZE - 1: self.y += 1
        elif direction == "LEFT" and self.x > 0: self.x -= 1
        elif direction == "RIGHT" and self.x < GRID_SIZE - 1: self.x += 1
        self.steps += 1

    def draw(self):
        screen.blit(rabbit_img, (self.x * CELL_SIZE, self.y * CELL_SIZE))

class Wolf:
    def __init__(self, wolf_id):
        self.id = wolf_id
        self.x, self.y = (0, 0) if wolf_id == 0 else (GRID_SIZE - 1, GRID_SIZE - 1)
        self.commentary = ""

    def move_to(self, x, y):
        self.x, self.y = x, y

    def draw(self):
        screen.blit(wolf1_img if self.id == 0 else wolf2_img, (self.x * CELL_SIZE, self.y * CELL_SIZE))
        label = font.render(f"Wolf {self.id + 1}", True, BLACK)
        screen.blit(label, (self.x * CELL_SIZE + 10, self.y * CELL_SIZE - 20))

# --- Gemini Strategy Agent ---
class WolfStrategyAgent:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("models/gemini-1.5-flash")

    def decide_next_moves(self, grid, wolf1_pos, wolf2_pos, rabbit_pos, rabbit_prev_pos, rabbit_steps):
        grid_str = "\n".join(" ".join(row) for row in grid)
        prompt = f"""
        You control two wolves on a 10x10 grid. 'R'=rabbit, 'W'=wolf, '.'=empty.
        - Rabbit at {rabbit_pos}, previously at {rabbit_prev_pos}, steps={rabbit_steps}.
        - Wolf 1 at {wolf1_pos}, Wolf 2 at {wolf2_pos}.
        Grid:
        {grid_str}

        Respond ONLY with JSON like:
        {{
            "wolf1_next_move": [x, y],
            "wolf2_next_move": [x, y],
            "wolf1_commentary": "...",
            "wolf2_commentary": "..."
        }}
        """
        
        print("Sending to Gemini:")
        print(prompt)  # Print the prompt sent to Gemini

        try:
            result = self.model.generate_content(prompt)
            raw = result.text
            json_text = raw[raw.find("{"): raw.rfind("}") + 1]
            data = json.loads(json_text)

            print("\nReceived from Gemini:")
            print(json.dumps(data, indent=4))  # Print the response from Gemini
            
            # Return the AI's decision
            data["wolf1_next_move"] = tuple(data["wolf1_next_move"])
            data["wolf2_next_move"] = tuple(data["wolf2_next_move"])
            return data
        except Exception as e:
            print("Gemini error:", e)
            return {
                "wolf1_next_move": wolf1_pos,
                "wolf2_next_move": wolf2_pos,
                "wolf1_commentary": "Error fallback",
                "wolf2_commentary": "Error fallback"
            }

# --- Game Utilities ---
def draw_grid():
    for x in range(0, SCREEN_SIZE, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (x, 0), (x, SCREEN_SIZE), 1)
    for y in range(0, SCREEN_SIZE, CELL_SIZE):
        pygame.draw.line(screen, BLACK, (0, y), (SCREEN_SIZE, y), 1)
    screen.blit(burrow_img, ((GRID_SIZE - 1) * CELL_SIZE, 0))

def draw_debug_info(rabbit, wolves):
    font_debug = pygame.font.Font(None, 20)
    screen.blit(font_debug.render(f"Rabbit Steps: {rabbit.steps}/{MAX_STEPS}", True, BLACK), (10, 10))
    for i, wolf in enumerate(wolves):
        msg = font_debug.render(f"Wolf {i+1}: {wolf.commentary}", True, BLACK)
        screen.blit(msg, (10, 30 + i * 20))

def check_game_over(rabbit, wolves):
    if rabbit.steps > MAX_STEPS:
        return "TIMEOUT"
    for wolf in wolves:
        if (wolf.x, wolf.y) == (rabbit.x, rabbit.y):
            return "CAUGHT"
    if (rabbit.x, rabbit.y) == (GRID_SIZE - 1, 0):
        return "ESCAPED"
    return None

def show_game_over(result):
    screen.fill(WHITE)
    msg = f"Game Over! {'Rabbit Escaped' if result in ['ESCAPED', 'TIMEOUT'] else 'Rabbit Caught'}"
    screen.blit(pygame.font.Font(None, 50).render(msg, True, RED), (SCREEN_SIZE // 6, SCREEN_SIZE // 2))
    pygame.display.flip()
    pygame.time.delay(5000)
    pygame.quit()
    exit()

def get_grid_state(rabbit, wolves):
    grid = [["." for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    grid[rabbit.y][rabbit.x] = "R"
    for wolf in wolves:
        grid[wolf.y][wolf.x] = "W"
    return grid

# --- Main Game Loop ---
rabbit = Rabbit()
wolves = [Wolf(0), Wolf(1)]
strategy_agent = WolfStrategyAgent(GOOGLE_API_KEY)
running = True

while running:
    screen.fill(WHITE)
    draw_grid()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            rabbit.move(pygame.key.name(event.key).upper())

            # Get grid state and request strategy from Gemini AI
            grid_state = get_grid_state(rabbit, wolves)
            move_data = strategy_agent.decide_next_moves(grid_state, (wolves[0].x, wolves[0].y), (wolves[1].x, wolves[1].y),
                                                        (rabbit.x, rabbit.y), (rabbit.prev_x, rabbit.prev_y), rabbit.steps)

            # Update wolves' positions and commentary
            wolves[0].move_to(*move_data["wolf1_next_move"])
            wolves[1].move_to(*move_data["wolf2_next_move"])
            wolves[0].commentary = move_data["wolf1_commentary"]
            wolves[1].commentary = move_data["wolf2_commentary"]

            # Check if game is over
            result = check_game_over(rabbit, wolves)
            if result:
                show_game_over(result)

    # Draw rabbit and wolves
    rabbit.draw()
    for wolf in wolves:
        wolf.draw()

    draw_debug_info(rabbit, wolves)

    pygame.display.flip()
    clock.tick(30)  # 30 FPS

pygame.quit()
