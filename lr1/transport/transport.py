# Транспортная задача

from tabulate import tabulate
from copy import deepcopy
from dataclasses import dataclass

# транспортная таблица таблица
@dataclass
class TransportTable():
    grid:   list[list[int]]
    supply: list[int]
    demand: list[int]
    alloc:  list[list[int]]

    def __str__(self) -> str:  
        table:list[list[str]] = [[""] + ["З/П"] + [str(i) for i in self.demand]]

        for i, (r, a, s) in enumerate(zip(self.grid, self.alloc, self.supply)):
            table.append([f"A{i+1}"] + [str(s)] + [f"a[{a_}] {r_}" for (a_, r_) in zip(a, r)])

        headers = [""]*2 + [f"B{i+1}" for i in range(len(self.demand))]

        return tabulate(table, headers=headers, tablefmt="grid") 

#
# методы построения опорного плана
#

# Метод северо западного угла
def tp_nwcm(t: TransportTable) -> list[tuple[str, TransportTable]]:
    res:list[tuple[str, TransportTable]] = []

    grid = deepcopy(t.grid) # дипкопи потому что вложенные списки
    supply = t.supply.copy()
    demand = t.demand.copy()

    rows = len(grid)
    cols = len(grid[0])

    allocation = [[0]*cols for _ in range(rows)]  # матрица распределения
    total = 0

    r, c = 0, 0
    while r < rows and c < cols:
        if supply[r] <= demand[c]:
            allocation[r][c] = supply[r]
            total += supply[r] * grid[r][c]
            demand[c] -= supply[r]
            supply[r] = 0
            r += 1
        else:
            allocation[r][c] = demand[c]
            total += demand[c] * grid[r][c]
            supply[r] -= demand[c]
            demand[c] = 0
            c += 1

        # добавление промежуточных таблиц
        res.append((f"текущая стоимость: {total}", TransportTable(grid, supply, demand, allocation)))

    return res

# метод наименьшей стоимости
def tp_lccm(t: TransportTable) -> list[tuple[str, TransportTable]]:
    res:list[tuple[str, TransportTable]] = []

    grid = deepcopy(t.grid) # дипкопи потому что вложенные списки
    supply = t.supply.copy()
    demand = t.demand.copy()

    rows = len(grid)
    cols = len(grid[0])

    allocation = [[0]*cols for _ in range(rows)]  # матрица распределения
    total = 0
    cells = [(i, j, grid[i][j]) for i in range(rows) for j in range(cols)]
    while any(s > 0 for s in supply) and any(d > 0 for d in demand):
        cells = [(i,j,c) for (i,j,c) in cells if supply[i] > 0 and demand[j] > 0]
        if not cells:
            break

        i_min, j_min, cost = min(cells, key=lambda x: x[2]) # выбор по минимальной смоимости

        qty = min(supply[i_min], demand[j_min])
        allocation[i_min][j_min] = qty
        total += qty * cost

        supply[i_min] -= qty
        demand[j_min] -= qty

        # добавление промежуточных таблиц
        res.append((f"текущая стоимость: {total}", TransportTable(grid, supply, demand, allocation)))
    
    return res

# метод потенцевалов
def 

#
# штуки для решения
#

def tp_check_balance(t: TransportTable) -> bool:
    return sum(t.demand) == sum(t.supply)


@dataclass
class TransportSolution:
    stages:list[tuple[str, TransportTable]] # комментарий / таблица

    def __str__(self) -> str:
        return ""


def tp_solve(t: TransportTable,
          method:str = "nwcm" # nwcm или lccm
          ) -> TransportSolution:
    sol:TransportSolution = TransportSolution([])
    
    # проверка на сбалансированность 
    is_balanced = tp_check_balance(t)
    # если не то добавление фейк поставщика/склада

    # построение опорного плана
    sol.stages += \
             tp_nwcm(t) if method == "nwcm" \
        else tp_lccm(t) if method == "lccm" \
        else []
    t = deepcopy(sol.stages[-1][1]) # последняя таблица
    
    # проверка на вырожденость 
    N:int = 0
    for r in t.alloc:
        for c in r:
            if c > 0:
                N+=1
    is_degenerate = N > len(t.supply) + len(t.demand) - 1

    # проверка на оптимальность методлм потенцивалов

    return sol

# тесты
if __name__ == "__main__":
    table = TransportTable(
        [[3, 1, 7, 4],
         [2, 6, 5, 9],
         [8, 3, 3, 2]],
        [300, 400, 500],
        [250, 350, 400, 200], [])

   
