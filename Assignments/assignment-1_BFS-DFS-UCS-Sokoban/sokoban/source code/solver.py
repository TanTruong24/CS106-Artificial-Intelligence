from ast import NodeTransformer
import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
global posWalls, posGoals


class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""

    def __init__(self):
        self.Heap = []
        self.Count = 0
        self.len = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0


"""Load puzzles and define the rules of sokoban"""


def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n', '') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ':
                layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#':
                layout[irow][icol] = 1  # wall
            elif layout[irow][icol] == '&':
                layout[irow][icol] = 2  # player
            elif layout[irow][icol] == 'B':
                layout[irow][icol] = 3  # box
            elif layout[irow][icol] == '.':
                layout[irow][icol] = 4  # goal
            elif layout[irow][icol] == 'X':
                layout[irow][icol] = 5  # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)])

    # print(layout)
    return np.array(layout)


def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp


def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0])  # e.g. (2, 2)


def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5)))  # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))


def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1))  # e.g. like those above


def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5)))  # e.g. like those above


def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)


def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper():  # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls


"""
# h??m ki???m tra v??? tr?? c?? ph?? h???p hay kh??ng
# n???u (x1,y1) kh??ng c?? trong posBox + posWalls -> true
# ng?????c l???i false
"""


def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1, 0, 'u', 'U'], [1, 0, 'd', 'D'],
                  [0, -1, 'l', 'L'], [0, 1, 'r', 'R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox:  # the move was a push
            action.pop(2)  # drop the little letter
        else:
            action.pop(3)  # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else:
            continue
    # e.g. ((0, -1, 'l'), (0, 1, 'R'))
    return tuple(tuple(x) for x in legalActions)


"""
if isLegalAction(action, posPlayer, posBox):
    -> n???u v??? tr?? ???? ph?? h???p, th??m action v??o
    -> legatAction : c??c v??? tr?? ti???p theo c?? th??? x???y ra
"""


def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer  # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer +
                    action[1]]  # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper():  # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox


"""
posBox.remove(newPosPlayer) # x??a v??? tr?? box hi???n t???i, ch??nh l?? v??? tr?? m???i player
"""


def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0, 1, 2, 3, 4, 5, 6, 7, 8],
                     [2, 5, 8, 1, 4, 7, 0, 3, 6],
                     [0, 1, 2, 3, 4, 5, 6, 7, 8][::-1],
                     [2, 5, 8, 1, 4, 7, 0, 3, 6][::-1]]
    flipPattern = [[2, 1, 0, 5, 4, 3, 8, 7, 6],
                   [0, 3, 6, 1, 4, 7, 2, 5, 8],
                   [2, 1, 0, 5, 4, 3, 8, 7, 6][::-1],
                   [0, 3, 6, 1, 4, 7, 2, 5, 8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1),
                     (box[0], box[1] - 1), (box[0],
                                            box[1]), (box[0], box[1] + 1),
                     (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls:
                    return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls:
                    return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox:
                    return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox:
                    return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls:
                    return True
    return False


"""Implement all approcahes"""


def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])
    exploredSet = set()
    actions = [[0]]
    temp = []
    while frontier:
        node = frontier.pop()
        node_action = actions.pop()
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            print("cost dfs:", len(temp))
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(
                    node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue
                frontier.append(node + [(newPosPlayer, newPosBox)])
                actions.append(node_action + [action[-1]])

    return temp


"""
u -> up , U -> ?????y th??ng l??n
d -> down, D -> ?????y th??ng xu???ng
l -> left, L -> ?????y th??ng qua tr??i
r -> right, R -> ?????y th??ng qua ph???i

node[-1][0] -> posPlayer
node[-1][1] -> posBox
posPlayer -> v??? tr?? c???a ng?????i ch??i
posBox -> v??? tr?? c???a h???p

"""


def breadthFirstSearch(gameState):
    """Implement breadthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = collections.deque([[startState]])  # store states
    actions = collections.deque([[0]])  # store actions
    exploredSet = set()  # t???p h???p theo d??i c??c n??t ???? ??i qua
    temp = []

    # Implement breadthFirs tSearch here
    """
    C??i ?????t: BFS theo c?? ch??? FIFO s??? d???ng Queue
    """
    while frontier:
        # do BFS theo c?? ch??? FIFO n??n s??? d???ng popleft() ????? l???y node c?? gi?? tr??? frontier[0]
        node = frontier.popleft()
        node_action = actions.pop()

        # h??m ki???m tra c??c h???p ???? v??o v??? tr?? kho ch??a?
        # n???u c??c h???p ???? v??o v??? tr?? kho, l???y ???????ng ??i t??? node_action[1:] (do node_action[0] = 0)
        # v?? tho??t v??ng l???p break
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            print("cost bfs:", len(temp))   # in ra s??? b?????c ph???i ??i (chi ph??)
            break
        # n???u node[-1] (v??? tr?? c??c h???p) kh??ng c?? exploredSet th?? th??m node[-1] v??o exploredSet
        # ng?????c l???i - > kh??ng l???p l???i tr???ng th??i ???? ??i, b??? qua
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            # h??m legalActions tr??? v?? c??c v??? tr?? ti???p theo agent c?? th??? x???y ra
            for action in legalActions(node[-1][0], node[-1][1]):
                # updateState : tr??? v??? v??? tr?? m???i c???a agent v?? c??c h???p
                newPosPlayer, newPosBox = updateState(
                    node[-1][0], node[-1][1], action)
                # Ki???m tra xem t???t c??? c??c ?? ?????u ??? kho hay ch??a (t???c l?? v?????t qua tr?? ch??i)
                # n???u ch??a th?? ti???p t???c l???nh ti???p theo, c??n r???i th?? b??? qua th??m ph???n t??? v??o frontier v?? actions
                if isFailed(newPosBox):
                    continue
                # n???u v??? tr?? m???i h???p l???, th??m v??o b??n tr??i frontier
                frontier.append(node + [(newPosPlayer, newPosBox)])
                # th??m tr???ng tr??i v??o ph??a ph???i actions
                actions.appendleft(node_action + [action[-1]])

    return temp


# uniformCostSearch

"""
heapq.heappush(self.Heap, entry)
- > heappush (heap, ele): - H??m n??y ???????c s??? d???ng ????? ch??n ph???n t??? ???????c ????? c???p trong c??c ?????i s??? c???a n?? v??o heap. 
    Th??? t??? ???????c ??i???u ch???nh, do ???? c???u tr??c ?????ng ???????c duy tr??.

heapq.heappop(self.Heap)
-> heappop(heap): - Ch???c n??ng n??y d??ng ????? lo???i b??? v?? tr??? v??? ph???n t??? nh??? nh???t t??? ??????heap. 
    Th??? t??? ???????c ??i???u ch???nh, do ???? c???u tr??c ?????ng ???????c duy tr??.

"""

# tr??? v??? s??? ???????ng ??i kh??ng ph???i ?????y th??ng c???a agent
# -> m???c ????ch nh???m gi???m thi???u ???????ng ??i th???a (kh??ng ph???i h??nh ?????ng ?????y th??ng)


def cost(actions):
    """A cost function"""
    return len([x for x in actions if x.islower()])


def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""
    beginBox = PosOfBoxes(gameState)
    beginPlayer = PosOfPlayer(gameState)

    startState = (beginPlayer, beginBox)
    frontier = PriorityQueue()
    frontier.push([startState], 0)
    exploredSet = set()
    actions = PriorityQueue()
    actions.push([0], 0)
    temp = []

    # Implement uniform cost search here
    while (frontier.isEmpty() == False):
        # h??m .pop() l???y ph???n t??? tr??ng frontier hay actions v???i cost_path nh??? nh???t
        # cost_path nh??? ???????c ??u ti??n
        # l???y v??? tr?? c??c h???p d???a v??o cost_path n??o nh??? h??n ???????c ??u ti??n h??n
        node = frontier.pop()
        # l???y c??c h??nh ?????ng ti???p theo d???a v??o cost_path n??o nh??? h??n ???????c ??u ti??n h??n
        node_action = actions.pop()
        if isEndState(node[-1][-1]):
            temp += node_action[1:]
            print("cost ucs:", len(temp))   # in ????? d??i ???????ng ??i
            break
        if node[-1] not in exploredSet:
            exploredSet.add(node[-1])
            for action in legalActions(node[-1][0], node[-1][1]):
                newPosPlayer, newPosBox = updateState(
                    node[-1][0], node[-1][1], action)
                if isFailed(newPosBox):
                    continue

                cost_path = cost(node_action[1:] + [action[-1]])
                # H??m push ch??n ph???n t??? v??o heap.
                # Th??? t??? ???????c ??i???u ch???nh d???a v??o ????? ??u ti??n (cost_path), cost_path nh??? s??? ???????c ??u ti??n
                # do ???? c???u tr??c ?????ng ???????c duy tr??.
                frontier.push(node + [(newPosPlayer, newPosBox)], cost_path)
                actions.push(node_action + [action[-1]], cost_path)

    return temp


"""Read command"""


def readCommand(argv):
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels, "r") as f:
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args


def get_move(layout, player_pos, level, method):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    if method == 'dfs':
        result = depthFirstSearch(gameState)
    elif method == 'bfs':
        result = breadthFirstSearch(gameState)
    elif method == 'ucs':
        result = uniformCostSearch(gameState)
    else:
        raise ValueError('Invalid method.')
    time_end = time.time()
    print('Runtime of %s level %s: %.2f second.' %
          (method, level, time_end-time_start))
    print(result)
    return result
