from time import time

import bloc, copy, random, os
import numpy as np
os.makedirs("blocaibatch2", exist_ok=True)

class MatrixAI:
    def __init__(self, W=None, U=None):
        # Initialize with passed weights, or random weights between -1 and 1
        self.W = W if W is not None else np.random.uniform(-1, 1, (5, 5))
        self.U = U if U is not None else np.random.uniform(-1, 1, (5, 5))

    def evaluate_board(self, A, B):
        # 1. Multiply W against both matrices
        M1 = self.W @ A
        M2 = self.W @ B
        
        # 2. Add the products
        M_combined = M1 + M2
        
        # 3. Dot with U
        score = np.dot(M_combined.ravel(), self.U.ravel())
        
        return score
    
    def minimax(self, A, B, depth):
        if depth == 0:
         return self.evaluate_board(A, B)

class BlocAiPopulation:
    def __init__(self, popsize, ai=None):
        self.popsize = popsize
        if ai == None:
            self.population = np.array([MatrixAI() for _ in range(popsize)])
        else:
            self.population = np.array([copy.deepcopy(ai) for _ in range(popsize)])
        self.cache = {}
    
    def halvepopsize(self):
        self.popsize = len(self.population) // 2

    def split(self):
        np.random.shuffle(self.population)
        self.halvepopsize()
        self.groupa = self.population[:self.popsize // 2]
        self.groupb = self.population[self.popsize // 2:]
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
        self.winners = [-1] * (self.popsize // 2)
        self.fitnesspairs = [[0,0] for _ in range(self.popsize // 2)]

    def play_match(self, matchindex):
        ai1 = self.groupa[matchindex]
        ai2 = self.groupb[matchindex]
        ais = [ai2, ai1]

        turns = 0
        maxturns = 100

        matchboard = bloc.BlocDataStruct()
        while matchboard.check_win() == -1:
            legalmoves = self.get_legal_moves_cached(matchboard)
            legalmovematrices = []
            for move in legalmoves:
                clone = matchboard.clone()
                clone.full_move(move[0], move[1])
                legalmovematrices.append([move, clone.board, clone.pieces])
            movescores = []
            for legalmove in legalmovematrices:
                movescores.append(ais[matchboard.turn].evaluate_board(legalmove[1], legalmove[2]))
            moveindex = np.argmax(movescores)
            matchboard.full_move(legalmoves[moveindex][0], legalmoves[moveindex][1])
            turns += 1
            if turns >= maxturns:
                break
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
        

    def play_comp(self, matchindex, rounds=3):
        for _ in range(rounds):
            self.play_match(matchindex)
        if self.fitnesspairs[matchindex][0] > self.fitnesspairs[matchindex][1]:
            self.winners[matchindex] = 0
        elif self.fitnesspairs[matchindex][1] > self.fitnesspairs[matchindex][0]:
            self.winners[matchindex] = 1
        elif self.fitnesspairs[matchindex][0] == self.fitnesspairs[matchindex][1]:
            self.winners[matchindex] = random.choice([0, 1])
    
    def play_all(self, rounds=3):
        for i in range(self.popsize // 2):
            self.play_comp(i, rounds)

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
            for _ in range(16):
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
    
    def epoch(self, rounds=3, mutation_rate=0.05):
        self.start_time = time()
        self.cache = {}
        self.splitforplay()
        self.play_all(rounds)
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
        print(f"Epoch {epoch} finished. Time: {time() - self.start_time:.2f}. Time per average AI: {(time() - self.start_time) / (len(self.population)):.2f}. Total time: {time() - self.cumul_time:.2f}")

    def train(self, dir, epochs=100, rounds=3, mutation_rate=0.05):
        self.cumul_time = time()
        for i in range(epochs):
            self.epoch(rounds, mutation_rate)
            self.record(i, f"{dir}/epoch_{i}.txt")
            self.report(i)

class BlocAiBeatBestPopulation(BlocAiPopulation):
    def __init__(self, popsize, best_ai):
        super().__init__(popsize)
        self.best_ai = best_ai
        self.fitnesses = [0] * popsize
    
    def play_match(self, matchindex):
        ai1 = self.best_ai
        ai2 = self.population[matchindex]
        ais = [ai2, ai1]

        turns = 0
        maxturns = 100

        matchboard = bloc.BlocDataStruct()
        matchboard.turn = random.choice([0, 1])  # Set the turn to the randomly chosen AI
        while matchboard.check_win() == -1:
            legalmoves = self.get_legal_moves_cached(matchboard)
            legalmovematrices = []
            for move in legalmoves:
                clone = matchboard.clone()
                clone.full_move(move[0], move[1])
                legalmovematrices.append([move, clone.board, clone.pieces])
            movescores = []
            for legalmove in legalmovematrices:
                movescores.append(ais[matchboard.turn].evaluate_board(legalmove[1], legalmove[2]))
            moveindex = np.argmax(movescores)
            matchboard.full_move(legalmoves[moveindex][0], legalmoves[moveindex][1])
            turns += 1
            if turns >= maxturns:
                break
        #print(f"Match {matchindex} finished in {turns} turns.")
        if matchboard.check_win() == 0:
            self.fitnesses[matchindex] += 1
        elif matchboard.check_win() == 1:
            self.fitnesses[matchindex] -= 1
        elif matchboard.check_win() == 0.5:
            self.fitnesses[matchindex] += 0.1
        else:
            self.fitnesses[matchindex] -= 1

    def kill(self):
        """
        Splits the population down the middle based on self.fitnesses,
        keeping the top 50% and discarding the bottom 50%.
        """
        # 1. Zip the bots and their fitness scores together so they stay paired
        zipped_population = list(zip(self.fitnesses, self.population))

        # 2. Sort by fitness score in descending order (highest score first)
        zipped_population.sort(key=lambda pair: pair[0], reverse=True)

        # 3. Calculate the halfway cutoff point
        half_size = len(zipped_population) // 2

        # 4. Extract the survivors from the top half
        survivors = [bot for score, bot in zipped_population[:half_size]]

        # 5. Overwrite self.population with the survivors
        self.population = survivors

    def play_comp(self, matchindex, rounds=3):
        for _ in range(rounds):
            self.play_match(matchindex)

    def play_all(self, rounds=3):
        for i in range(len(self.population)):
            self.play_comp(i, rounds)

    def epoch(self, rounds=10, mutation_rate=0.05):
        self.cache = {}
        self.start_time = time()
        self.play_all(rounds)
        self.kill()
        self.split()
        self.reproduce(mutation_rate)
    
    def train(self, dir, epochs=100, rounds=10, mutation_rate=0.05):
        self.cumul_time = time()
        for i in range(epochs):
            self.epoch(rounds, mutation_rate)
            self.record(i, f"{dir}/epoch_{i}.txt")
            self.report(i)
            self.fitnesses = [0] * (2 * len(self.population))

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

    def split(self):
        np.random.shuffle(self.population)
        self.groupa = self.population[:len(self.population) // 2]
        self.groupb = self.population[len(self.population) // 2:]
        self.groups = [self.groupa, self.groupb]

    def report(self, epoch):
        print(f"Epoch {epoch} finished. Time: {time() - self.start_time:.2f}. Time per average AI: {(time() - self.start_time) / (len(self.population)):.2f}. Total time: {time() - self.cumul_time:.2f}. Average fitness: {np.mean(self.fitnesses):.2f}. Best fitness: {np.max(self.fitnesses):.2f}. Worst fitness: {np.min(self.fitnesses):.2f}")

best_ai = MatrixAI(np.array([[ 0.23838392, -0.13419667,  0.0150143,  -0.19654428,  0.14868578],
 [-0.04433206, -0.3393874,   0.01248466,  0.21550079, -0.16186957],
 [-0.00739169,  0.05001475, -0.09486223,  0.12241841, -0.01061351],
 [-0.16331054,  0.06789198,  0.13968752, -0.14723337, -0.00458008],
 [-0.01836799, -0.13853593, -0.20648714, -0.08701489,  0.07076733]]), np.array([[ 0.10236422,  0.00497212,  0.19659075,  0.35880435, -0.0695713 ],
 [-0.39174886, -0.06314057,  0.05832172, -0.03492783,  0.07554218],
 [ 0.10893301, -0.18137953, -0.15379302, -0.26375244,  0.1581135 ],
 [ 0.25531835,  0.04493948,  0.35401408,  0.00881072, -0.1837255 ],
 [-0.20483909,  0.0268084,  -0.2960147,  -0.09777757,  0.05309948]]))
population = BlocAiPopulation(32, best_ai)
population.train(dir="blocaibatch2", epochs=100, rounds=3, mutation_rate=0.1)