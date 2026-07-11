from time import time

import bloc, copy, random, os, heapq
import numpy as np

class MatrixAI:
    def __init__(self, W=None, U=None):
        # Initialize with passed weights, or random weights between -1 and 1
        self.W = W if W is not None else np.random.uniform(-1, 1, (5, 5))
        self.U = U if U is not None else np.random.uniform(-1, 1, (5, 5))

    def evaluate_board(self, A, B, side=1):
        if side == 0:
            # Create a boolean mask where elements are 0 or 1 (ignoring -1)
            mask_A = (A == 0) | (A == 1)
            mask_B = (B == 0) | (B == 1)
            
            # Make a quick copy to avoid modifying the board in-place
            A = A.copy()
            B = B.copy()
            
            # Apply the swap instantly using vector subtraction
            A[mask_A] = 1 - A[mask_A]
            B[mask_B] = 1 - B[mask_B]

        # 1. Multiply W & V against both matrices
        M1 = self.W @ A
        M2 = self.V @ B

        # 2. Add the products
        M_combined = M1 + M2
        
        # 3. Dot with U
        score = np.dot(M_combined.ravel(), self.U.ravel())
        
        return score
    
    def minimax(self, board, side, depth=4, topn=3, alpha=float('-inf'), beta=float('inf')):
        # 1. Base cases: Terminal states or Max Depth reached
        win_status = board.check_win()
        if win_status != -1:
            if win_status == side:
                return float('inf')
            elif win_status == 0.5:
                return 0
            else:
                return float('-inf')
                
        if depth == 0:
            return self.evaluate_board(board.board, board.pieces, side)

        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return 0  # No moves left usually implies a draw or specific game rule

        # 2. Lightweight Move Ordering (Shallow Evaluation)
        # Clone and make the move just once to calculate its immediate strength
        move_candidates = []
        for move in legal_moves:
            next_board = board.clone()
            next_board.full_move(move[0], move[1])
            # Score from the perspective of the current player
            score = self.evaluate_board(next_board.board, next_board.pieces, side)
            move_candidates.append((score, next_board))

        # Sort moves by highest score first and grab your top N
        move_candidates.sort(key=lambda x: x[0], reverse=True)
        best_candidates = move_candidates[:topn]

        # 3. Alpha-Beta Search Loop
        best_val = float('-inf')
        
        for _, next_board in best_candidates:
            # Negamax style: invert the score of the opponent's best response
            value = -self.minimax(next_board, 1 - side, depth - 1, topn, -beta, -alpha)
            
            best_val = max(best_val, value)
            alpha = max(alpha, value)
            
            # Alpha-Beta Pruning cut-off
            if alpha >= beta:
                break
                
        return best_val

class TriMatrixAI(MatrixAI):
    def __init__(self, W=None, V=None, U=None):
        # Initialize with passed weights, or random weights between -1 and 1
        self.W = W if W is not None else np.random.uniform(-1, 1, (5, 5))
        self.U = U if U is not None else np.random.uniform(-1, 1, (5, 5))
        self.V = V if V is not None else np.random.uniform(-1, 1, (5, 5))

    def evaluate_board(self, A, B, side=1):
        if side == 0:
            # Create a boolean mask where elements are 0 or 1 (ignoring -1)
            mask_A = (A == 0) | (A == 1)
            mask_B = (B == 0) | (B == 1)
            
            # Make a quick copy to avoid modifying the board in-place
            A = A.copy()
            B = B.copy()
            
            # Apply the swap instantly using vector subtraction
            A[mask_A] = 1 - A[mask_A]
            B[mask_B] = 1 - B[mask_B]

        # 1. Multiply W & V against both matrices
        M1 = self.W @ A
        M2 = self.V @ B

        # 2. Add the products
        M_combined = M1 + M2
        
        # 3. Dot with U
        score = np.dot(M_combined.ravel(), self.U.ravel())
        
        return score

class BlocAiPopulation:
    def __init__(self, popsize, ai=None, noise=0):
        self.popsize = popsize
        if ai == None:
            self.population = np.array([MatrixAI() for _ in range(popsize)])
        else:
            self.population = np.array([MatrixAI(ai.W + np.random.uniform(-1 * noise, noise, (5, 5)), ai.U + np.random.uniform(-1 * noise, noise, (5, 5))) for _ in range(popsize)])
        self.cache = {}
    
    def halvepopsize(self):
        self.popsize = len(self.population) // 2

    def split(self):
        np.random.shuffle(self.population)
        self.halvepopsize()
        self.groupa = self.population[:len(self.population) // 2]
        self.groupb = self.population[len(self.population) // 2:]
        self.groups = [self.groupa, self.groupb]

    def get_legal_moves_cached(self, matchboard):
        board = matchboard.board
        pieces = matchboard.pieces
        board_key = (tuple(board.flat), tuple(pieces.flat))
        
        if board_key in self.cache:
            return self.cache[board_key]
            
        moves = matchboard.get_legal_moves() # Run the heavy raycast logic ONLY once
        self.cache[board_key] = moves
        return moves
    
    def splitforplay(self):
        self.split()
        self.winners = [-1] * (len(self.population) // 2)
        self.fitnesspairs = [[0,0] for _ in range(len(self.population) // 2)]
        self.turns = [0] * (len(self.population) // 2)

    def play_match(self, matchindex, maxturns=100):
        ai1 = self.groupa[matchindex]
        ai2 = self.groupb[matchindex]
        ais = [ai2, ai1]

        turns = 0
        maxturns = maxturns
        flag = True
        matchboard = bloc.BlocDataStruct()
        while flag:
            print(matchboard.board, matchboard.pieces)
            legalmoves = self.get_legal_moves_cached(matchboard)
            legalmovematrices = []
            for move in legalmoves:
                clone = matchboard.clone()
                clone.full_move(move[0], move[1])
                legalmovematrices.append([move, clone.board, clone.pieces])
            movescores = []
            for legalmove in legalmovematrices:
                movescores.append(ais[matchboard.turn].evaluate_board(legalmove[1], legalmove[2], matchboard.turn))
            moveindex = np.argmax(movescores)
            matchboard.full_move(legalmoves[moveindex][0], legalmoves[moveindex][1])
            turns += 1
            print(turns)
            if turns >= maxturns:
                flag = False
            if matchboard.check_win() != -1:
                flag = False
        self.turns[matchindex] = turns
        if matchboard.check_win() == 0:
            self.fitnesspairs[matchindex][1] += 2
            self.fitnesspairs[matchindex][0] -= 1
        elif matchboard.check_win() == 1:
            self.fitnesspairs[matchindex][0] += 2
            self.fitnesspairs[matchindex][1] -= 1
        elif matchboard.check_win() == 0.5:
            self.fitnesspairs[matchindex][0] += 0.1
            self.fitnesspairs[matchindex][1] += 0.1
        else:
            self.fitnesspairs[matchindex][0] -= 0.1
            self.fitnesspairs[matchindex][1] -= 0.1
    
    def play_minimax_match(self, matchindex, topn=3, maxturns=100, depth=4):
        ai1 = self.groupa[matchindex]
        ai2 = self.groupb[matchindex]
        ais = [ai2, ai1]

        turns = 0
        maxturns = maxturns
        flag = True
        matchboard = bloc.BlocDataStruct()
        while flag:
            legalmoves = self.get_legal_moves_cached(matchboard)
            legalmovematrices = []
            for move in legalmoves:
                clone = matchboard.clone()
                clone.full_move(move[0], move[1])
                legalmovematrices.append([move, clone])
            movescores = []
            for legalmove in legalmovematrices:
                movescores.append(ais[matchboard.turn].minimax(legalmove[1], matchboard.turn, depth=depth, topn=topn))
            moveindex = np.argmax(movescores)
            matchboard.full_move(legalmoves[moveindex][0], legalmoves[moveindex][1])
            turns += 1
            if turns >= maxturns:
                flag = False
            if matchboard.check_win() != -1:
                flag = False
        self.turns[matchindex] = turns
        if matchboard.check_win() == 0:
            self.fitnesspairs[matchindex][1] += 2
            self.fitnesspairs[matchindex][0] -= 1
        elif matchboard.check_win() == 1:
            self.fitnesspairs[matchindex][0] += 2
            self.fitnesspairs[matchindex][1] -= 1
        elif matchboard.check_win() == 0.5:
            self.fitnesspairs[matchindex][0] += 0.1
            self.fitnesspairs[matchindex][1] += 0.1
        else:
            self.fitnesspairs[matchindex][0] -= 0.1
            self.fitnesspairs[matchindex][1] -= 0.1

    def play_comp(self, matchindex, maxturns=100, minimax=False, topn=3, depth=4, rounds=3):
        for _ in range(rounds):
            if minimax:
                self.play_minimax_match(matchindex, maxturns=maxturns, depth=depth, topn=topn)
            else:
                self.play_match(matchindex, maxturns=maxturns)
        if self.fitnesspairs[matchindex][0] > self.fitnesspairs[matchindex][1]:
            self.winners[matchindex] = 0
        elif self.fitnesspairs[matchindex][1] > self.fitnesspairs[matchindex][0]:
            self.winners[matchindex] = 1
        elif self.fitnesspairs[matchindex][0] == self.fitnesspairs[matchindex][1]:
            self.winners[matchindex] = random.choice([0, 1])
    
    def play_all(self, maxturns=100, minimax=False, topn=3, depth=4, rounds=3):
        for i in range(len(self.population) // 2):
            self.match_time = time()
            self.play_comp(i, maxturns=maxturns, minimax=minimax, topn=topn, depth=depth, rounds=rounds)
            print(f"\rProgress: " + "#" * (i + 1) + "-" * (len(self.population) // 2 - i - 1) + f", Time: {time()-self.match_time:.2f}, Total Time: {time()-self.start_time:.2f}. Moves: {self.turns[i]}.", end="")

    def kill(self):
        new_population = []
        for i, winner in enumerate(self.winners):
            if winner == 0:
                new_population.append(self.groupa[i])
            elif winner == 1:
                new_population.append(self.groupb[i])
            else:
                new_population.append(random.choice([self.groupa[i], self.groupb[i]]))
        self.population = np.array(new_population)
    
    def reproduce(self, mutation_rate=0.05):
        new_population = []
        for i in range(len(self.groupa)):
            for _ in range(4):
                parent1 = self.groupa[i]
                parent2 = self.groupb[i]
                child_W = (parent1.W + parent2.W) / 2
                child_U = (parent1.U + parent2.U) / 2
                
                # Apply mutation
                mutation_matrix_W = np.random.uniform(-1 * mutation_rate, mutation_rate, child_W.shape)
                mutation_matrix_U = np.random.uniform(-1 * mutation_rate, mutation_rate, child_U.shape)
                child_W += mutation_matrix_W
                child_U += mutation_matrix_U
                
                new_population.append(MatrixAI(child_W, child_U))
        self.population = np.array(new_population)
    
    def epoch(self, maxturns=100, topn=3, minimax=False, depth=4, rounds=3, mutation_rate=0.05):
        print(f"\rProgress: " + "-" * (len(self.population) // 2), end="")
        self.start_time = time()
        self.cache = {}
        self.splitforplay()
        self.play_all(depth=depth, topn=topn, maxturns=maxturns, minimax=minimax, rounds=rounds)
        self.kill()
        self.split()
        self.reproduce(mutation_rate)
    
    def record(self, epoch, filename):
        with open(filename, 'w') as f:
            f.write(f"Epoch: {epoch}")
            for i in range(len(self.population)):
                ai = self.population[i]
                f.write(f"Matrix Pair {i}; W = {ai.W}\nU = {ai.U}\n")
    
    def report(self, epoch):
        moves = sum(self.turns) / len(self.turns)
        print(f"Epoch {epoch} finished. Time: {time() - self.start_time:.2f}. Time per average AI: {(time() - self.start_time) / (len(self.population)):.2f}. Total time: {time() - self.cumul_time:.2f}. Average moves per game: {moves}")

    def train(self, dir, maxturns=100, minimax=False, topn=3, depth=4, epochs=100, rounds=3, mutation_rate=0.05):
        self.cumul_time = time()
        for i in range(epochs):
            self.epoch(depth=depth, topn=topn, maxturns=maxturns, minimax=minimax, rounds=rounds, mutation_rate=mutation_rate)
            self.record(i, f"{dir}/epoch_{i}.txt")
            self.report(i)

class BlocTriAiPopulation(BlocAiPopulation):
    def __init__(self, popsize, ai=None, noise=0):
        self.popsize = popsize
        if ai == None:
            self.population = np.array([TriMatrixAI() for _ in range(popsize)])
        else:
            self.population = np.array([TriMatrixAI(ai.W + np.random.uniform(-1 * noise, noise, (5, 5)), ai.U + np.random.uniform(-1 * noise, noise, (5, 5)), ai.V + np.random.uniform(-1 * noise, noise, (5, 5))) for _ in range(popsize)])
        self.cache = {}

    def record(self, epoch, filename):
        with open(filename, 'w') as f:
            f.write(f"Epoch: {epoch}")
            for i in range(len(self.population)):
                ai = self.population[i]
                f.write(f"Matrix Pair {i}; W = {ai.W}\nV = {ai.V}\nU = {ai.U}\n")

    def reproduce(self, mutation_rate=0.05):
        new_population = []
        for i in range(len(self.groupa)):
            for _ in range(4):
                parent1 = self.groupa[i]
                parent2 = self.groupb[i]
                child_W = (parent1.W + parent2.W) / 2
                child_V = (parent1.V + parent2.V) / 2
                child_U = (parent1.U + parent2.U) / 2
                
                # Apply mutation
                mutation_matrix_W = np.random.uniform(-1 * mutation_rate, mutation_rate, child_W.shape)
                mutation_matrix_U = np.random.uniform(-1 * mutation_rate, mutation_rate, child_U.shape)
                mutation_matrix_V = np.random.uniform(-1 * mutation_rate, mutation_rate, child_V.shape)
                child_W += mutation_matrix_W
                child_U += mutation_matrix_U
                child_V += mutation_matrix_V
                
                new_population.append(TriMatrixAI(child_W, child_U, child_V))
        self.population = np.array(new_population)

best_ai = MatrixAI(np.array([[ 0.23838392, -0.13419667,  0.0150143,  -0.19654428,  0.14868578],
 [-0.04433206, -0.3393874,   0.01248466,  0.21550079, -0.16186957],
 [-0.00739169,  0.05001475, -0.09486223,  0.12241841, -0.01061351],
 [-0.16331054,  0.06789198,  0.13968752, -0.14723337, -0.00458008],
 [-0.01836799, -0.13853593, -0.20648714, -0.08701489,  0.07076733]]), np.array([[ 0.10236422,  0.00497212,  0.19659075,  0.35880435, -0.0695713 ],
 [-0.39174886, -0.06314057,  0.05832172, -0.03492783,  0.07554218],
 [ 0.10893301, -0.18137953, -0.15379302, -0.26375244,  0.1581135 ],
 [ 0.25531835,  0.04493948,  0.35401408,  0.00881072, -0.1837255 ],
 [-0.20483909,  0.0268084,  -0.2960147,  -0.09777757,  0.05309948]]))

population = BlocTriAiPopulation(64)
trainingdir = "blocaibatchMAT1"
os.makedirs(trainingdir, exist_ok=True)
population.train(dir=trainingdir, maxturns=40, minimax=True, depth=3, epochs=100, rounds=1, mutation_rate=0.05, topn=2)