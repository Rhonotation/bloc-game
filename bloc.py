import os
import pygame
import numpy as np

'''
./noVNC/utils/novnc_proxy --vnc localhost:5900 --listen 6080 &
killall Xvfb x11vnc
Xvfb :1 -screen 0 1024x768x16 &
x11vnc -display :1 -nopw -listen localhost -xkb &
DISPLAY=:1 python bloc.py
'''

# ./noVNC/utils/novnc_proxy --vnc localhost:5900 --listen 6080 &
def configure_pygame_display():
    if os.environ.get("SDL_VIDEODRIVER", "") == "":
        if os.environ.get("DISPLAY", "") == "" and os.environ.get("WAYLAND_DISPLAY", "") == "":
            os.environ["SDL_VIDEODRIVER"] = "dummy"

configure_pygame_display()

moves = {((1,-1),(1,0)):[[-1,1],[1,0]],
            ((1,-1),(1,1)):[[-1,1],[1,1]],
            ((1,-1),(0,0)):[[-1,-1],[0,0]],
            ((1,-1),(0,1)):[[-1,1],[1,0]],
            ((1,1),(1,0)):[[1,1],[1,1]],
            ((1,1),(1,1)):[[1,1],[1,0]],
            ((1,1),(0,0)):[[-1,1],[0,1]],
            ((1,1),(0,1)):[[-1,1],[1,1]],
            ((1,0),(1,0)):[[1,0],[0,1]],
            ((1,0),(0,1)):[[-1,-1],[1,1]],
            ((1,0),(1,1)):[[1,1],[1,0]],
            ((1,0),(0,0)):[[0,0],[1,1]],
            ((1,-1),(0,-1)):[[-1,1],[0,1]],
            ((1,-1),(1,-1)):[[-1,1],[1,1]],
            ((1,1,0),(1,1,1)):[[1,1,1],[1,1,0]],
            ((1,1,-1),(1,1,1)):[[1,1,1],[1,1,1]]}
patterns = list(moves.keys())
# 1 is black 0 is white -1 is unclaimed / empty
class BlocDataStruct:
    global moves, patterns
    def __init__(self, master=0):
        self.board = np.array([[1, 0, -1, 1, 0],[0, 1, -1, 0, 1], [1, 0, -1, 1, 0], [0, 1, -1, 0, 1], [1, 0, -1, 1, 0]])
        self.pieces = np.array([[1, -1, -1, -1, 0], [1, -1, -1, -1, 0], [1, -1, -1, -1, 0], [1, -1, -1, -1, 0], [1, -1, -1, -1, 0]])
        self.cellselected = [0,0]
        self.selected = False
        self.master = master
        self.turn = 1
        self.state_history = {}

    def get_board(self):
        return self.board

    def get_pieces(self):
        return self.pieces

    def get_cellselected(self):
        return self.cellselected

    def get_selected(self):
        return self.selected

    def get_selected_value(self):
        if self.selected:
            row, col = self.cellselected
            return self.board[row][col]
        else:
            raise ValueError("No cell is currently selected.")

    def select_cell(self, row, col):
        if self.pieces[row][col] in [0, 1]:
            self.cellselected = [row, col]
            self.selected = True
        else:
            self.selected = False
    
    def toggle_select_cell(self, row, col):
        vector = [row - self.cellselected[0], col - self.cellselected[1]]
        if vector[0] * vector[1] == 0 and (vector[0] + vector[1] != 0) and self.selected:
            vector = [vector[0] / abs(vector[0] + vector[1]), vector[1] / abs(vector[0] + vector[1])]
            if self.check_legal(self.cellselected, vector / np.linalg.norm(vector)):
                self.full_move(self.cellselected, vector / np.linalg.norm(vector))
                self.selected = False
        elif self.pieces[row][col] in [0, 1]:
            if self.cellselected == [row, col] and self.selected:
                self.selected = False
            else:
                self.cellselected = [row, col]
                self.selected = True
        else:
            self.selected = False
    
    def map_move(self, pieces, board):
        pieces_key = tuple([int(piece) for piece in pieces])
        board_key = tuple([int(cell) for cell in board])
        if pieces_key[0] == 0:
            pieces_key = tuple([{1:0, 0:1, -1:-1}[piece] for piece in pieces])
            board_key = tuple([{1:0, 0:1, -1:-1}[cell] for cell in board])
            unflipped = moves[(pieces_key, board_key)]
            return tuple([tuple([{1:0, 0:1, -1:-1}[piece] for piece in unflipped[0]]), tuple([{1:0, 0:1, -1:-1}[cell] for cell in unflipped[1]])])
        else:
            return moves[(pieces_key, board_key)]

    def raycast(self, start, direction):
        points = [start,
                  [int(start[0] + direction[0]), int(start[1] + direction[1])],
                  [int(start[0] + 2 * direction[0]), int(start[1] + 2 * direction[1])]]
        
        if any(p[0] < 0 or p[1] < 0 for p in points):
            if points[1][0] < 0 or points[1][1] < 0:
                return [False, start, direction, points]
            else:
                points = points[0:2]
                pattern_points = [[self.pieces[point[0]][point[1]] for point in points], [self.board[point[0]][point[1]] for point in points]]
                pattern_tuple = (tuple(pattern_points[0]), tuple(pattern_points[1]))
                return [True, start, direction, pattern_tuple]
        try:
            pattern_points = [[self.pieces[point[0]][point[1]] for point in points], [self.board[point[0]][point[1]] for point in points]]
        except IndexError:
            try:
                # Do the same check for the shortened 2-point list
                if any(p[0] < 0 or p[1] < 0 for p in points[0:2]):
                    raise IndexError
                pattern_points = [[self.pieces[point[0]][point[1]] for point in points[0:2]], [self.board[point[0]][point[1]] for point in points[0:2]]]
            except IndexError:
                return [False, start, direction, points]
            pattern_tuple = (tuple(pattern_points[0]), tuple(pattern_points[1]))
            return [True, start, direction, pattern_tuple]
        
        pattern_tuple = tuple([[int(x) for x in pp] for pp in pattern_points])
        if pattern_tuple in patterns:
            return [True, start, direction, pattern_tuple]
        else:
            return [True, start, direction, tuple([pat[0:2] for pat in pattern_tuple])]
    
    def transform(self, raycasting):
        if raycasting[0]:
            return [raycasting[0], raycasting[1], raycasting[2], self.map_move(raycasting[3][0], raycasting[3][1])]
        else:
            return [False, raycasting[1], raycasting[2], raycasting[3]]
    
    def replace_transform(self, transformed_raycast):
        if transformed_raycast[0]:
            for i in range(len(transformed_raycast[3])):
                self.pieces[int(transformed_raycast[1][0] + i * transformed_raycast[2][0])][int(transformed_raycast[1][1] + i * transformed_raycast[2][1])] = transformed_raycast[3][0][i]
                self.board[int(transformed_raycast[1][0] + i * transformed_raycast[2][0])][int(transformed_raycast[1][1] + i * transformed_raycast[2][1])] = transformed_raycast[3][1][i]
            return True
        else:
            return False
    
    def full_move(self, start, direction):
        if self.pieces[start[0]][start[1]] == self.turn:
            if not self.replace_transform(self.transform(self.raycast(start, direction))):
                return False
            self.turn = 1 - self.turn

            state_key = (self.board.tobytes(), self.pieces.tobytes())
            self.state_history[state_key] = self.state_history.get(state_key, 0) + 1
            self.killall()
            return True
        else:
            return False

    def killall(self):
        for row in range(self.pieces.shape[0]):
            for col in range(self.pieces.shape[1]):
                if self.pieces[row][col] != -1:
                    self.kill([row, col])
    
    def reset(self):
        self.board = np.array([[1, 0, -1, 1, 0],[0, 1, -1, 0, 1], [1, 0, -1, 1, 0], [0, 1, -1, 0, 1], [1, 0, -1, 1, 0]])
        self.pieces = np.array([[1, -1, -1, -1, 0], [1, -1, -1, -1, 0], [1, -1, -1, -1, 0], [1, -1, -1, -1, 0], [1, -1, -1, -1, 0]])
    
    def get_legal_moves(self):
        legal_moves = []
        for row in range(self.pieces.shape[0]):
            for col in range(self.pieces.shape[1]):
                if self.pieces[row][col] in [0, 1]:
                    for direction in [[1, 0], [0, 1], [-1, 0], [0, -1]]:
                        raycast_result = self.raycast([row, col], direction)
                        if raycast_result[0]:
                            transformed = self.transform(raycast_result)
                            if transformed[0]:
                                legal_moves.append(([row, col], direction))
        return legal_moves

    def check_legal(self, start, direction):
        raycast_result = self.raycast(start, direction)
        if raycast_result[0]:
            return True
        return False
    
    def check_win(self):
        current_key = (self.board.tobytes(), self.pieces.tobytes())
        if self.state_history.get(current_key, 0) >= 3:
            return 1 - self.turn
        if np.count_nonzero(self.pieces == 1) == 1 and np.count_nonzero(self.pieces == 0) > 1:
            return 0
        if np.count_nonzero(self.pieces == 0) == 1 and np.count_nonzero(self.pieces == 1) > 1:
            return 1
        if np.count_nonzero(self.pieces == 0) == 1 and np.count_nonzero(self.pieces == 1) == 1:
            return 0.5
        if np.count_nonzero(self.pieces[:,4] == 1) >= 2:
            return 1
        if np.count_nonzero(self.pieces[:,0] == 0) >= 2:
            return 0
        return -1

    def clone(self):
        new_board = BlocDataStruct()
        new_board.board = self.board.copy()
        new_board.pieces = self.pieces.copy()
        new_board.state_history = self.state_history.copy()
        new_board.turn = self.turn
        return new_board

    def kill(self, loc):
        side = self.pieces[loc]
        neighbors = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        neighborcells = [[loc[0] + n[0], loc[1] + n[1]] for n in neighbors]
        trueneighbors = []
        for neighborcell in neighborcells:
            if 0 <= neighborcell[0] < self.pieces.shape[0] and 0 <= neighborcell[1] < self.pieces.shape[1]:
                trueneighbors.append(neighborcell)
        boardcells = [self.board[cell] for cell in trueneighbors]
        piececells = [self.pieces[cell] for cell in trueneighbors]
        pressure = np.count_nonzero(piececells == side) * 2 + np.count_nonzero(boardcells == side)
        if pressure >= 5:
            self.pieces[loc] = -1
            return True
        return False

class PygameBlocDisplay:
    def __init__(self, master):
        pygame.init()
        self.master = master
        self.data = BlocDataStruct(master)
        self.rows, self.cols = self.data.board.shape
        self.tile_size = min(500 // self.cols, 500 // self.rows)
        self.screen = pygame.display.set_mode((self.rows * self.tile_size, self.cols * self.tile_size))
        pygame.display.set_caption("Bloc Game")
        self.clock = pygame.time.Clock()
        self.running = True
        self.game_over = False
        self.font = pygame.font.SysFont(None, 36)

    def draw_grid(self):
        white = (255, 255, 255)
        black = (0, 0, 0)
        gray = (160, 160, 160)
        border = (80, 80, 80)

        for row in range(self.rows):
            for col in range(self.cols):
                board_value = int(self.data.board[row][col])
                if board_value == -1:
                    tile_color = gray
                elif board_value == 0:
                    tile_color = white
                else:
                    tile_color = black

                display_row = self.cols - 1 - col
                display_col = row
                cell_rect = pygame.Rect(
                    display_col * self.tile_size,
                    display_row * self.tile_size,
                    self.tile_size,
                    self.tile_size,
                )
                pygame.draw.rect(self.screen, tile_color, cell_rect)
                pygame.draw.rect(self.screen, border, cell_rect, 1)

                if self.data.selected and self.data.cellselected == [row, col]:
                    pygame.draw.rect(self.screen, (255, 215, 0), cell_rect, 3)

                piece_value = int(self.data.pieces[row][col])
                if piece_value != -1:
                    piece_color = white if piece_value == 0 else black
                    piece_radius = self.tile_size // 3
                    piece_center = (
                        display_col * self.tile_size + self.tile_size // 2,
                        display_row * self.tile_size + self.tile_size // 2,
                    )
                    pygame.draw.circle(self.screen, piece_color, piece_center, piece_radius)
                    pygame.draw.circle(self.screen, border, piece_center, piece_radius, 2)

    def display_to_board(self, pos):
        x, y = pos
        display_col = x // self.tile_size
        display_row = y // self.tile_size
        if display_col < 0 or display_col >= self.rows or display_row < 0 or display_row >= self.cols:
            return None
        board_row = display_col
        board_col = self.cols - 1 - display_row
        return board_row, board_col

    def get_win_message(self):
        result = self.data.check_win()
        if result == 1:
            return "Black wins!"
        if result == 0:
            return "White wins!"
        if result == 0.5:
            return "It's a draw."
        return ""

    def update_game_over_state(self):
        self.game_over = self.data.check_win() != -1

    def draw_game_over_overlay(self):
        if not self.game_over:
            return

        message = self.get_win_message()
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        text_surface = self.font.render(message, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=self.screen.get_rect().center)
        self.screen.blit(text_surface, text_rect)

    def handle_events(self):
        self.update_game_over_state()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and not self.game_over:
                cell = self.display_to_board(event.pos)
                if cell is not None:
                    self.data.toggle_select_cell(*cell)

    def draw(self):
        self.screen.fill((40, 40, 40))
        self.draw_grid()
        self.draw_game_over_overlay()
        pygame.display.flip()

    def run(self):
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(30)


if __name__ == "__main__":
    display = PygameBlocDisplay(None)
    display.run()
    pygame.quit()