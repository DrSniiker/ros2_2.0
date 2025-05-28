import csv
import math

# Tar in cameFrom listan och den nuvarande koordinaten som borde vara mål-koordinaten och
# bygger tillbaka vägen från mål till start genom att följa tidigare koordinater.
def reconstruct_path(cameFrom, current):
    totalPath = [current]

    # Bygger tillbaka vägen från målet till start
    while True:
        for node in cameFrom:
            if node[0] == current:
                previous = node[1]
                break
            else:
                previous = None
        
        if previous == None:
            break
        
        current = previous
        totalPath.insert(0, current)

    return totalPath

# Heuristikfunktionen som tar in två koordinater, a och b, och beräknar avståndet mellan de.
def heuristic(a, b):
    # Manhattan distance as heuristic
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Tar in en av scorlistorna, gScore eller fScore, en koordinat och ett nytt värde som den
# koordinaten ska få och uppdaterar eller lägger till den i listan.
def set_score(score_list, coord, newScore):
    for i in range(len(score_list)):
        if score_list[i][0] == coord:
            score_list[i] = (coord, newScore)
            return

    score_list.append((coord, newScore))

# Tar in en av scorlistorna, gScore eller fScore och en koordinat och returnerar värdet som den
# koordinaten har i listan. Om den inte finns returneras oändligheten.
def get_score(score_list, coord):
    newScore = math.inf

    for node, score in score_list:
        if node == coord:
            newScore = score + 1

    return newScore

# Tar in openSet listan och fScore listan och returnerar den nod i openSet som har lägst fScore.
def get_node_with_lowest_fscore(openSet, fScore):
    minCoords = None
    minScore = math.inf

    for node in openSet:
        for score in fScore:
            if score[0] == node and score[1] < minScore:
                minScore = score[1]
                minCoords = node
    
    return minCoords

# Tar in en koordinat och en karta och returnerar alla grannar till den koordinaten som är gångbara
# (dvs. har värdet 0 i kartan).
def get_neighbors(coord, maze):
    x, y = coord[0], coord[1]
    neighbors = []

    # Kollar alla möjliga håll (N, NÖ, Ö, SÖ, S, SV, V, NV)
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if maze[nx][ny] == 0:  # 0 är en fri cell
            neighbors.append((nx, ny))
    
    return neighbors

# A* funktionen som tar in en startkoordinat, en målkoordinat och en karta i form av en 2D lista
# med ettor och nollor och räknar ut den kortaste vägen på kartan från start till mål med hjälp
# av A* algoritmen.
def a_star(start, goal, maze):
    openSet = [start]  # Lista med hittade koordinater som kan behöva undersökas
    cameFrom = []  # Lista med tuples: (coord, previous_coord)
    gScore = []  # Lista med tuples: (coord, gscore)
    fScore = []  # Lista med tuples: (coord, fscore)

    set_score(gScore, start, 0)
    set_score(fScore, start, heuristic(start, goal))

    while openSet:
        current = get_node_with_lowest_fscore(openSet, fScore)

        if current == goal:
            return reconstruct_path(cameFrom, current)
        
        openSet.remove(current)

        for neighbor in get_neighbors(current, maze):
            tentativeGScore = get_score(gScore, current) + 1

            if tentativeGScore < get_score(gScore, neighbor):
                # Denna väg är bättre än tidigare känd väg, uppdatera vägen
                
                for coord, previous in cameFrom:
                    # Ta bort tidigare koordinat om den finns i cameFrom
                    if coord == neighbor:
                        cameFrom.remove((coord, previous))
                        break
                cameFrom.append((neighbor, current))

                set_score(gScore, neighbor, tentativeGScore)

                set_score(fScore, neighbor, tentativeGScore + heuristic(neighbor, goal))

                if neighbor not in openSet:
                    openSet.append(neighbor)
    
    return None  # Ingen väg hittades

def maze_from_csv(filename):
    maze = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Filtrera bort tomma celler i varje rad
            filtered_row = [cell for cell in row if cell.strip() != '']
            if not filtered_row:
                # Hoppa över tomma rader
                continue
            maze.append([int(cell) for cell in filtered_row])
    return maze

def print_maze_with_path(maze, start, goal, path):
    for i, row in enumerate(maze):
        line = ''
        for j, cell in enumerate(row):
            if (i, j) == start:
                line += 'O'  # Startpunkt
            elif (i, j) == goal:
                line += 'X'  # Målpunkt
            elif path and (i, j) in path:
                line += '*'  # Del av vägen
            else:
                line += '█' if cell == 1 else ' '
        print(line)

start = (2, 2)
goal = (2, 18)
maze = maze_from_csv('maze.csv')

# Skriv ut för att verifiera
for row in maze:
    print(''.join('█' if cell == 1 else ' ' for cell in row))

path = a_star(start, goal, maze)

print(path)
print_maze_with_path(maze, start, goal, path)