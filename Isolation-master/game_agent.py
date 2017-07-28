"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass
    
def get_longest_chain(player_location, blank_spaces, length):
    if len(blank_spaces) == 0:
        return float(length)
    r, c = player_location

    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2),  (1, 2), (2, -1),  (2, 1)]

    valid_moves = [(r+dr,c+dc) for dr, dc in directions if (r+dr, c+dc) in blank_spaces]
    current_length = 0
    for move in valid_moves:
        new_blank_spaces = list(blank_spaces)
        new_blank_spaces.remove(move)
        new_length = get_longest_chain(move, new_blank_spaces, length + 1)
        if new_length > current_length:
            current_length = new_length
    return float(current_length)
    
#longest chain possible
def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    This heuristic tries to maximize the difference in the lengths of longest
    possible chains of available moves between our agent and the opponent
    while also taking into account the difference in the number of available
    moves for our agent and the opponent.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")      
        
    location_player = game.get_player_location(player)
    location_opponent = game.get_player_location(game.get_opponent(player))
        
    number_of_blank_spaces = len(game.get_blank_spaces())
    number_of_fields = game.width * game.height
    
    if number_of_blank_spaces < (number_of_fields / 4):
        own_moves = len(game.get_legal_moves(player))
        opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
        return get_longest_chain(location_player, game.get_blank_spaces(), 0) - get_longest_chain(location_opponent, game.get_blank_spaces(), 0) + float(own_moves - opp_moves)
    else:        
        own_moves = len(game.get_legal_moves(player))
        opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
        return float(own_moves - opp_moves)
        
#keep to the center
def custom_score_center(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    This heuristic tries to maximise the distance to the edges of the game
    board. In other words, it tries to keep the agent near the center.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    player_location = game.get_player_location(player);
    distance_on_x = min(game.width - 1 - player_location[0], player_location[0])
    distance_on_y = min(game.height - 1 - player_location[1], player_location[1])

    return float(distance_on_x + distance_on_y)

#no intersect
def custom_score_no_intersect(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    This heuristic tries to minimize the intersection between our legal moves
    and the legal moves of the opponent.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return -float(len(list(set(game.get_legal_moves(player)).intersection(game.get_legal_moves(game.get_opponent(player))))))

#coward
def custom_score_hero(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    This heuristic tries to maximize the distance between our agent and the
    opponent.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    location_player = game.get_player_location(player)
    location_opponent = game.get_player_location(game.get_opponent(player))

    return ((location_player[0]-location_opponent[0]) ** 2 + ((location_player[1]-location_opponent[1]) ** 2)) ** (0.5)
        
#hero
def custom_score_hero(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.
    This heuristic tries to minimize the distance between our agent and the
    opponent.
    
    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")
    
    location_player = game.get_player_location(player)
    location_opponent = game.get_player_location(game.get_opponent(player))

    return -(((location_player[0]-location_opponent[0]) ** 2 + ((location_player[1]-location_opponent[1]) ** 2)) ** (0.5))


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
    
    def max_value(self, game, depth, alpha=None, beta=None):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        #if this is a terminal state or we reached our desired depth, return the utility value
        if not game.get_legal_moves() or depth == 0:
            return self.score(game, self), (-1,-1)
        #go through all legal moves and return the maximum utility
        value = float("-inf")
        best_move = (-1,-1)
        for move in game.get_legal_moves():
            new_game = game.forecast_move(move)
            new_value, _ = self.min_value(new_game, depth - 1, alpha, beta)
            if new_value > value:
                value = new_value
                best_move = move
            if alpha is not None:
                if value >= beta:
                    return value, best_move
                alpha = max(alpha, value)
        return float(value), best_move
    
    def min_value(self, game, depth, alpha=None, beta=None):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        #if this is a terminal state or we reached our desired depth, return the utility value
        if not game.get_legal_moves() or depth == 0:
            return self.score(game, self), (-1,-1)
        #go through all legal moves and return the minimum utility
        value = float("+inf")
        best_move = (-1,-1)
        for move in game.get_legal_moves():
            new_game = game.forecast_move(move)
            new_value, _ = self.max_value(new_game, depth - 1, alpha, beta)
            if new_value < value:
                value = new_value
                best_move = move
            if alpha is not None:
                if value <= alpha:
                    return value, best_move
                beta = min(beta, value)
        return float(value), best_move

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        
        depth = 1
        best_move = (-1,-1)

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.method == 'minimax':
                if self.iterative:
                    while True:
                        value, best_move = self.minimax(game, depth)
                        depth = depth + 1
                else:
                    value, best_move = self.minimax(game, depth)
            else:
                if self.iterative:
                    while True:
                        value, best_move = self.alphabeta(game, depth)
                        depth = depth + 1
                else:
                    value, best_move = self.alphabeta(game, depth)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
               
        if maximizing_player:
            return self.max_value(game, depth)
        else:
            return self.min_value(game, depth)


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if maximizing_player:
            return self.max_value(game, depth, alpha, beta)
        else:
            return self.min_value(game, depth, alpha, beta)