import math
import random
import time


class Nim():

    def __init__(self, initial=[1, 3, 5, 7]):
        """
        Initialize game board.
        Each game board has
            - `piles`: a list of how many elements remain in each pile
            - `player`: 0 or 1 to indicate which player's turn
            - `winner`: None, 0, or 1 to indicate who the winner is
        """
        self.piles = initial.copy()
        self.player = 0
        self.winner = None

    @classmethod
    def available_actions(cls, piles):
        """
        Nim.available_actions(piles) takes a `piles` list as input
        and returns all of the available actions `(i, j)` in that state.

        Action `(i, j)` represents the action of removing `j` items
        from pile `i` (where piles are 0-indexed).
        """
        actions = set()
        for i, pile in enumerate(piles):
            for j in range(1, pile + 1):
                actions.add((i, j))
        return actions

    @classmethod
    def other_player(cls, player):
        """
        Nim.other_player(player) returns the player that is not
        `player`. Assumes `player` is either 0 or 1.
        """
        return 0 if player == 1 else 1

    def switch_player(self):
        """
        Switch the current player to the other player.
        """
        self.player = Nim.other_player(self.player)

    def move(self, action):
        """
        Make the move `action` for the current player.
        `action` must be a tuple `(i, j)`.
        """
        pile, count = action

        # Check for errors
        if self.winner is not None:
            raise Exception("Game already won")
        elif pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        elif count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        # Update pile
        self.piles[pile] -= count
        self.switch_player()

        # Check for a winner
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class NimAI():

    def __init__(self, alpha=0.5, epsilon=0.1):
        """
        Initialize AI with an empty Q-learning dictionary,
        an alpha (learning) rate, and an epsilon rate.

        The Q-learning dictionary maps `(state, action)`
        pairs to a Q-value (a number).
         - `state` is a tuple of remaining piles, e.g. (1, 1, 4, 4)
         - `action` is a tuple `(i, j)` for an action
        """
        self.q = dict()
        self.alpha = alpha
        self.epsilon = epsilon

    def update(self, old_state, action, new_state, reward):
        """
        Update Q-learning model, given an old state, an action taken
        in that state, a new resulting state, and the reward received
        from taking that action.
        """
        old = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old, reward, best_future)

    def get_q_value(self, state, action):
        """
        Return the Q-value for the state `state` and the action `action`.
        If no Q-value exists yet in `self.q`, return 0.
        """
        try:
            return self.q[tuple(state), action]
        except KeyError:
            return 0

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Update the Q-value for the state `state` and the action `action`
        given the previous Q-value `old_q`, a current reward `reward`,
        and an estiamte of future rewards `future_rewards`.

        Use the formula:

        Q(s, a) <- old value estimate
                   + alpha * (new value estimate - old value estimate)

        where `old value estimate` is the previous Q-value,
        `alpha` is the learning rate, and `new value estimate`
        is the sum of the current reward and estimated future rewards.
        """
        new_q = old_q + self.alpha * ((reward + future_rewards) - old_q)
        self.q[tuple(state), action] = new_q

    def best_future_reward(self, state):
        """
        Given a state, return the maximum Q-value among all possible actions.
        If no actions are available, return 0.
        """

        actions = Nim.available_actions(state)

        if not actions:
            return 0

        best_reward = 0

        for action in actions:
            q_value = self.q.get((tuple(state), action), 0)
            if q_value > best_reward:
                best_reward = q_value

        return best_reward

    def choose_action(self, state, epsilon=True):
        """
        Given a state `state`, return an action `(i, j)` to take.

        If `epsilon` is `False`, then return the best action
        available in the state (the one with the highest Q-value,
        using 0 for pairs that have no Q-values).

        If `epsilon` is `True`, then with probability
        `self.epsilon` choose a random available action,
        otherwise choose the best action available.

        If multiple actions have the same Q-value, any of those
        options is an acceptable return value.
        """

        max_reward = 0
        best_action = None

        available_moves = Nim.available_actions(state)

        for move in available_moves:
            try:
                q = self.q[tuple(state), move]
            except KeyError:
                q = 0

            if q > max_reward:
                max_reward = q
                best_action = move

        if max_reward == 0:
            return random.choice(tuple(available_moves))

        if not epsilon:
            return best_action
        else:
            if random.random() < self.epsilon:
                return random.choice(tuple(available_moves))
            else:
                return best_action


def train(n):
    """
    Train an AI by playing `n` games against itself.
    """

    player = NimAI()

    # Play n games
    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = Nim()

        # Keep track of last move made by either player
        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        # Game loop
        while True:

            # Keep track of current state and action
            state = game.piles.copy()
            action = player.choose_action(game.piles)

            # Keep track of last state and action
            last[game.player]["state"] = state
            last[game.player]["action"] = action

            # Make move
            game.move(action)
            new_state = game.piles.copy()

            # When game is over, update Q values with rewards
            if game.winner is not None:
                player.update(state, action, new_state, -1)
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    1
                )
                break

            # If game is continuing, no rewards yet
            elif last[game.player]["state"] is not None:
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    0
                )

    print("Done training")

    # Return the trained AI
    return player

def play(ai_agent, human_side=None):
    """
    Run a Nim game between a human player and an AI agent.
    `human_side` determines whether the human plays first (0) or second (1).
    """

    # Decide human position randomly if not specified
    if human_side is None:
        human_side = random.choice([0, 1])

    # Initialize game instance
    nim_game = Nim()

    # Continue until the game ends
    while True:

        # Display current pile status
        print("\nPiles:")
        for index, stones in enumerate(nim_game.piles):
            print(f"Pile {index}: {stones}")
        print()

        # Determine all valid moves
        valid_moves = Nim.available_actions(nim_game.piles)
        time.sleep(1)

        # Human player's move
        if nim_game.player == human_side:
            print("Your Turn")
            while True:
                chosen_pile = int(input("Choose Pile: "))
                chosen_count = int(input("Choose Count: "))
                if (chosen_pile, chosen_count) in valid_moves:
                    move = (chosen_pile, chosen_count)
                    break
                print("Invalid move, try again.")

        # AI player's move
        else:
            print("AI's Turn")
            move = ai_agent.choose_action(nim_game.piles, epsilon=False)
            print(f"AI chose to take {move[1]} from pile {move[0]}.")

        # Apply move to the game
        nim_game.move(move)

        # Check if the game has ended
        if nim_game.winner is not None:
            print("\nGAME OVER")
            final_winner = "Human" if nim_game.winner == human_side else "AI"
            print(f"Winner is {final_winner}")
            return
