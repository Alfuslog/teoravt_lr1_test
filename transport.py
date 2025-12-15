# таблица
class TransportTable():
    grid:   list[list[int]]
    supply: list[int]
    demand: list[int]

    def __init__(
        self,
        grid:   list[list[int]],
        supply: list[int],
        demand: list[int]
        ) -> None:
        self.grid   = grid
        self.supply = supply
        self.demand = demand

    def __str__(self) -> str:   
        res = "  %s  \n" % ''.join(["{:3d} ".format(i) for i in self.demand])
        # "{:.15f}".format(self._data.eps)
        for i, item in enumerate(self.grid):
            res+= "[ "
            for j in item:
                res += "{:3d} ".format(j)
            res+= f"] {self.supply[i]}\n"
        return res

# Метод северо западного угла
def tp_nwcm(t: TransportTable) -> None:
    grid = t.grid.copy()
    supply = t.supply.copy()
    demand = t.demand.copy()

    startR = 0  # start row
    startC = 0  # start col
    ans = 0

    # loop runs until it reaches the bottom right corner
    while(startR != len(grid) and startC != len(grid[0])):
        # if demand is greater than supply
        if(supply[startR] <= demand[startC]):
            ans += supply[startR] * grid[startR][startC]
            # subtract the value of supply from the demand
            demand[startC] -= supply[startR]
            startR += 1

        # if supply is greater than demand
        else:
            ans += demand[startC] * grid[startR][startC]
            # subtract the value of demand from the supply
            supply[startR] -= demand[startC]
            startC += 1

        print(TransportTable(grid, supply, demand))

    print("The initial feasible basic solution is ", ans)

# метод наименьшей стоимости
def tp_lccm(t: TransportTable) -> None:
    pass


# тесты
if __name__ == "__main__":
    table = TransportTable(
        [[3, 1, 7, 4],
         [2, 6, 5, 9],
         [8, 3, 3, 2]],
        [300, 400, 500],
        [250, 350, 400, 200])


    table = TransportTable(
        [[2, 4, 5, 8, 6],
         [7, 3, 6, 4, 5]],
        [180, 300],
        [110, 140, 220, 190, 120])

    tp_nwcm(table)
