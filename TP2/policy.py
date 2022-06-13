# ##############################################################################
#  Aprendizagem Profunda, TP2 2021/2022
#  :Authors:
#  Gonçalo Martins Lourenço nº55780
#  Joana Soares Faria nº55754
# ##############################################################################
TURN_LEFT, FORWARD, TURN_RIGHT = -1, 0, 1
N, E, S, W = 0, 1, 2, 3


def policy(score, apple, head, tail, direction):
    return policy1(apple[0], head, direction)


def policy1(apple, head, direction):
    """
        This policy goes staight to the apple ignoring direction to the walls, its own body and everything else
        just like an angry bull running towards a red cape
    """

    action = FORWARD

    if apple[0] > head[0]:  # apple is lower than snake's head (S)
        if direction == N:
            action = TURN_LEFT  #pode bater na parede
        elif direction == E:
            action = TURN_RIGHT
        elif direction == S:
            action = FORWARD
        else: # W
            action = TURN_LEFT

    elif apple[0] < head[0]:  # apple is higher than snake's head (N)
        if direction == N:
            action = FORWARD
        elif direction == E:
            action = TURN_LEFT
        elif direction == S: #pode bater na parede
            action = FORWARD
        else: # W
            action = TURN_RIGHT

    elif apple[1] > head[1]:  # apple is at the same height but to the snake's head East(Right)
        if direction == N:
            action = TURN_RIGHT
        elif direction == E:
            action = FORWARD
        elif direction == S:
            action = TURN_LEFT
        else:  # W
            action = TURN_LEFT  #pode bater na parede

    elif apple[1] < head[1]:  # apple is at the same height but to the snake's head West(Left)
        if direction == N:
            action = TURN_LEFT
        elif direction == E:
            action = TURN_LEFT  #pode bater na parede
        elif direction == S:
            action = FORWARD
        else: # W
            action = TURN_RIGHT


    return action

