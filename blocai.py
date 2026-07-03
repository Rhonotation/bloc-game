import bloc, copy, random
import numpy as np

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
        
        # 3 & 4. Element-wise product with U, then sum the total scalar score
        score = np.sum(M_combined * self.U)
        
        return score

class BlocAiPopulation:
    def __init__(self, popsize):
        self.popsize = popsize
        self.population = np.array([MatrixAI() for _ in range(popsize)])

    def split(self):
        np.random.shuffle(self.population)
        self.groupa = self.population[:self.popsize // 2]
        self.groupb = self.population[self.popsize // 2:]
        self.groups = [self.groupa, self.groupb]
    
    def splitforplay(self):
        self.split()
        self.winners = [-1] * (self.popsize // 2)
        self.fitnesspairs = [[0,0] for _ in range(self.popsize // 2)]

    def play_match(self, matchindex):
        ai1 = self.groupa[matchindex]
        ai2 = self.groupb[matchindex]
        ais = [ai2, ai1]

        matchboard = bloc.BlocBoard()
        while matchboard.check_winner() == -1:
            legalmoves = matchboard.get_legal_moves()
            legalmovematrices = []
            for move in legalmoves:
                clone = copy.deepcopy(matchboard)
                clone.full_move(move) # This mutates the clone in-place
                legalmovematrices.append([move, clone.board, clone.pieces])
            movescores = []
            for legalmove in legalmovematrices:
                movescores.append(ais[matchboard.turn].evaluate_board(legalmove[1], legalmove[2]))
            moveindex = np.argmax(movescores)
            matchboard.full_move(legalmoves[moveindex])
        if matchboard.check_winner() == 0:
            self.fitnesspairs[matchindex][1] += 2
            self.fitnesspairs[matchindex][0] -= 1
        elif matchboard.check_winner() == 1:
            self.fitnesspairs[matchindex][0] += 2
            self.fitnesspairs[matchindex][1] -= 1
        elif matchboard.check_winner() == 0.5:
            self.fitnesspairs[matchindex][0] += 0.1
            self.fitnesspairs[matchindex][1] += 0.1

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
        self.population = np.array(new_population)
    
    def reproduce(self, mutation_rate=0.05):
        new_population = []
        for i in range(len(self.groupa)):
            for _ in range(2):
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
        self.population = np.array(self.population) + np.array(new_population)
    
    def epoch(self, rounds=3, mutation_rate=0.05):
        self.splitforplay()
        self.play_all(rounds)
        self.kill()
        self.split()
        self.reproduce(mutation_rate)
    
    def train(self, epochs=100, rounds=3, mutation_rate=0.05):
        for _ in range(epochs):
            self.epoch(rounds, mutation_rate)