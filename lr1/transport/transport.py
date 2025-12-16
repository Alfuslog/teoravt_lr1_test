# Транспортная задача

from tabulate import tabulate
from copy import deepcopy
from dataclasses import dataclass, field
import math as m

# транспортная таблица таблица
@dataclass
class TransportTable():
    grid:   list[list[int]]
    supply: list[int]
    demand: list[int]

    # распределение товара
    alloc:  list[list[float]]

    # потенциалы
    u: list[float]
    v: list[float]
    
    def __init__(self,
                 grid:list[list[int]],
                 supply:list[int],
                 demand:list[int],
                 alloc: list[list[float]] | None = None,
                 u: list[float] | None = None,
                 v: list[float] | None = None) -> None:
        self.grid = grid
        self.supply = supply
        self.demand = demand

        rows = len(grid)
        cols = len(grid[0])
        self.alloc = [[0.0]*cols for _ in range(rows)] if alloc is None else alloc
        self.u = [0 for _ in range(rows)] if u is None else u
        self.v = [0 for _ in range(cols)] if v is None else v

    def __str__(self) -> str:  
        table:list[list[str]] = [[""] + ["З/П"] + [str(i) for i in self.demand]]

        for i, (u, r, a, s) in enumerate(zip(self.u, self.grid, self.alloc, self.supply)):
            table.append([f"A{i+1} u{i+1}({u})"] + [str(s)] + [f"a[{a_}] {r_}" for (a_, r_) in zip(a, r)])

        headers = [""]*2 + [f"B{i+1} v{i+1}({v})" for i, (_, v) in enumerate(zip(self.demand, self.v))]

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

    allocation = [[0.0]*cols for _ in range(rows)]  # матрица распределения
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
        res.append((f"текущая стоимость: {total}", TransportTable(grid, supply.copy(), demand.copy(), allocation)))

    return res

# метод наименьшей стоимости
def tp_lccm(t: TransportTable) -> list[tuple[str, TransportTable]]:
    res:list[tuple[str, TransportTable]] = []

    grid = deepcopy(t.grid) # дипкопи потому что вложенные списки
    supply = t.supply.copy()
    demand = t.demand.copy()

    rows = len(grid)
    cols = len(grid[0])

    allocation = [[0.0]*cols for _ in range(rows)]  # матрица распределения
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

#
# штуки для решения
#
def tp_make_balanced(t: TransportTable) -> TransportTable:
    total_supply = sum(t.supply)
    total_demand = sum(t.demand)

    grid = deepcopy(t.grid)
    supply = t.supply.copy()
    demand = t.demand.copy()
    alloc = deepcopy(t.alloc) if t.alloc else [[0.0]*len(demand) for _ in supply]

    if total_supply > total_demand: # фиктивный потребитель
        diff = total_supply - total_demand
        for r in grid:
            r.append(0)
        for r in alloc:
            r.append(0)
        demand.append(diff)

    elif total_demand > total_supply: # фиктивный поставщик
        diff = total_demand - total_supply
        grid.append([0]*len(demand))
        alloc.append([0]*len(demand))
        supply.append(diff)

    return TransportTable(grid, supply, demand, alloc)


def tp_check_balance(t: TransportTable) -> bool:
    return sum(t.demand) == sum(t.supply)


def tp_make_not_degenerate(t: TransportTable) -> tuple[str, TransportTable]:
    m, n = len(t.supply), len(t.demand)

    def basic_cells_count():
        return sum(1 for i in range(m) for j in range(n) if t.alloc[i][j] > 0)

    added = 0
    for i in range(m):
        for j in range(n):
            if basic_cells_count() >= m + n - 1:
                break

            # можно добавить ε только в пустую клетку
            if t.alloc[i][j] == 0:
                t.alloc[i][j] = float("inf")
                added += 1

        if basic_cells_count() >= m + n - 1:
            break

    if basic_cells_count() < m + n - 1:
        return ("Не удалось устранить вырождение", t)

    return (f"Добавлено ε-клеток: {added}", t)

# метод потенцевалов
def tp_check_optimal(t: TransportTable) -> tuple[bool, str, TransportTable]:
    rows = len(t.grid)
    cols = len(t.grid[0])

    u = [float("inf")]*rows
    v = [float("inf")]*cols
    u[0] = 0

    changed = True
    while changed:
        changed = False
        for i in range(rows):
            for j in range(cols):
                if t.alloc[i][j] > 0:
                    if not m.isinf(u[i]) and m.isinf(v[j]):
                        v[j] = t.grid[i][j] - u[i]
                        changed = True
                    elif not m.isinf(v[j]) and m.isinf(u[i]):
                        u[i] = t.grid[i][j] - v[j]
                        changed = True

    # проверка оценок
    d = [[0.0]*cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            if t.alloc[i][j] == 0:
                delta = t.grid[i][j] - (u[i] + v[j])
                d[i][j] = delta
                if delta < 0:
                    return (False, f"План неоптимален, найдено улучшение\nДельты:\n{tabulate(d)}", TransportTable(t.grid, t.supply, t.demand, t.alloc, u, v))

    return (True, f"План оптимален\nДельты\n{tabulate(d)}", TransportTable(t.grid, t.supply, t.demand, t.alloc, u, v))


@dataclass
class TransportSolution:
    stages:list[tuple[str, TransportTable]] # комментарий / таблица

    def __str__(self) -> str:
        return ''.join([ f"\n{s[0]}\n{s[1]}\n" for s in self.stages])


def tp_solve(t: TransportTable, method: str = "nwcm") -> TransportSolution:
    sol = TransportSolution([])
    sol.stages.append(("Начато решение для", t))

    if not tp_check_balance(t):
        t = tp_make_balanced(t)
        sol.stages.append(("Делаем задачу балансированной", t))

    sol.stages += (
        tp_nwcm(t) if method == "nwcm"
        else tp_lccm(t) if method == "lccm"
        else []
    )

    t = deepcopy(sol.stages[-1][1])

    # проверка вырожденности
    N = sum(1 for r in t.alloc for c in r if c > 0) # колво ячеек в решении
    is_degenerate = N < len(t.supply) + len(t.demand) - 1
    if is_degenerate:
        m, t = tp_make_not_degenerate(t)
    sol.stages.append((f"План вырожденнйы!! колво базисных клекток:{N}", t))
    is_optimal, m, t = tp_check_optimal(t)

    sol.stages.append((
        m,
        t
    ))

    return sol

# тесты
if __name__ == "__main__":
    t = TransportTable(
        [[4, 3, 4, 5, 3],
         [2, 4, 5, 7, 8],
         [4, 3, 7, 2, 1]],
        [250, 200, 220],
        [140, 110, 170, 90, 140])
    print(tp_solve(t))

   
