import math
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from typing import List

# ==============================================================================
# ====== МОДУЛИ ДЛЯ РЕШЕНИЯ ТРАНСПОРТНЫХ ЗАДАЧ ======
# ==============================================================================

import sys

class TransportElement:
    def __init__(self, row: int, col: int, quantity: float | str = '*', cost: int = -1, is_base: bool = False):
        self._row = row
        self._col = col
        self._quantity = quantity
        self._cost = cost
        self._is_base:bool = is_base

    @property
    def position_row(self) -> int:
        return self._row

    @property
    def position_col(self) -> int:
        return self._col

    @property
    def quantity(self) -> float:
        return float(self._quantity) if self._quantity!='*' else 0

    @quantity.setter
    def quantity(self, quantity: float) -> None:
        self._quantity = quantity

    @property
    def cost(self) -> int:
        return self._cost

    @cost.setter
    def cost(self, cost: int) -> None:
        self._cost = cost
            
    @property
    def is_base(self) -> bool:
        return self._is_base
            
    @is_base.setter
    def is_base(self, value: bool) -> None:
        self._is_base = value

    def __eq__(self, other) -> bool:
        return (self._row == other.position_row and self._col == other.position_col and
                self._quantity == other.quantity and self._cost == other.cost and self._is_base == other.is_base)

    def __ne__(self, other) -> bool:
        return not self == other


class TransportationTable:
    def __init__(self, rows_amount: int, cols_amount: int):
        self._rows_amount = rows_amount
        self._cols_amount = cols_amount
        self._producers: list[int] = []
        self._producers_sum: int = 0
        self._consumers: list[int] = []
        self._consumers_sum: int = 0
        self._transportations: list[list[TransportElement]] = []

        for i in range(self._rows_amount):
            transportations_row = []
            for j in range(self._cols_amount):
                transportations_row.append(TransportElement(i, j))
            self._transportations.append(transportations_row)

    def get_element(self, index1: int, index2: int) -> TransportElement:
        return self._transportations[index1][index2]

    @property
    def dimension_rows(self) -> int:
        return self._rows_amount

    @property
    def dimension_columns(self) -> int:
        return self._cols_amount

    @property
    def producers(self) -> list[int]:
        return self._producers.copy()

    @producers.setter
    def producers(self, producers: list[int]) -> None:
        if len(producers) == self._rows_amount:
            self._producers = producers.copy()
            self._producers_sum = sum(self._producers)
        else:
            raise Exception('Длина входного массива не совпадает с измерением таблицы')

    @property
    def consumers(self) -> list[int]:
        return self._consumers.copy()

    @consumers.setter
    def consumers(self, consumers: list[int]) -> None:
        if len(consumers) == self._cols_amount:
            self._consumers = consumers.copy()
            self._consumers_sum = sum(self._consumers)
        else:
            raise Exception('Длина входного массива не совпадает с измерением таблицы')

    @property
    def transportations(self) -> list[list[TransportElement]]:
        return self._transportations.copy()

    def check_balance(self) -> bool:
        return self._consumers_sum == self._producers_sum

    def balance(self) -> None:
        if not self.check_balance():
            if self._consumers_sum > self._producers_sum:
                self._rows_amount += 1
                self._producers.append(self._consumers_sum - self._producers_sum)
                transportation_row = []
                for j in range(self._cols_amount):
                    transportation_row.append(TransportElement(self._rows_amount - 1, j, '*', 0))
                self._transportations.append(transportation_row)
            else:
                self._cols_amount += 1
                self._consumers.append(self._producers_sum - self._consumers_sum)
                for i in range(self._rows_amount):
                    self._transportations[i].append(TransportElement(i, self._cols_amount - 1, '*', 0))

    def north_west_corner(self) -> None:
        producers_copy = self._producers.copy()
        consumers_copy = self._consumers.copy()

        for i in range(self._rows_amount):
            j = 0
            while producers_copy[i] != 0:
                difference = producers_copy[i] - consumers_copy[j]

                if difference >= 0:
                    if consumers_copy[j] != 0:
                        self._transportations[i][j].quantity = consumers_copy[j]
                    producers_copy[i] = difference
                    consumers_copy[j] = 0
                else:
                    self._transportations[i][j].quantity = producers_copy[i]
                    consumers_copy[j] = abs(difference)
                    producers_copy[i] = 0

                j += 1

    def _transportations_to_list(self) -> list[TransportElement]:
        transportations_list = []
        for i in range(self._rows_amount):
            for j in range(self._cols_amount):
                if self._transportations[i][j].quantity != '*':
                    transportations_list.append(self._transportations[i][j])
        return transportations_list

    def _get_neighbors(self, current_transportation: TransportElement,
                       transportations_list: list[TransportElement]) -> list[TransportElement]:
        neighbors = [TransportElement(-1, -1, '*'), TransportElement(-1, -1, '*')]
        for transportation in transportations_list:
            if transportation != current_transportation:
                if neighbors[0].quantity != '*' and neighbors[1].quantity != '*':
                    break

                if transportation.position_row == current_transportation.position_row \
                        and neighbors[0].quantity == '*':
                    neighbors[0] = transportation
                elif transportation.position_col == current_transportation.position_col \
                        and neighbors[1].quantity == '*':
                    neighbors[1] = transportation

        return neighbors

    def get_cycle(self, start_transportation: TransportElement) -> list[TransportElement]:
        path = self._transportations_to_list()
        path.insert(0, start_transportation)

        previous_length: int
        while True:
            previous_length = len(path)
            i = 0
            while i < len(path):
                neighbors = self._get_neighbors(path[i], path)
                if neighbors[0].quantity == '*' or neighbors[1].quantity == '*':
                    path.pop(i)
                    break
                i += 1

            if previous_length == len(path) or len(path) == 0:
                break

        if len(path) == 0:
            return []

        cycle = [TransportElement(-1, -1, '*')] * len(path)
        previous_transportation = start_transportation
        for i in range(len(cycle)):
            cycle[i] = previous_transportation
            neighbors = self._get_neighbors(previous_transportation, path)
            previous_transportation = neighbors[i % 2]

        return cycle

    def count_basis(self) -> int:
        basis_amount = 0
        for row in self._transportations:
            for transportation in row:
                if transportation.quantity != '*':
                    basis_amount += 1
        return basis_amount

    def check_degeneracy(self) -> bool:
        return self._rows_amount + self._cols_amount - 1 > self.count_basis()

    def remove_degeneracy(self):
        while self.check_degeneracy():
            zero_added = False
            for i in range(self._rows_amount):
                for j in range(self._cols_amount):
                    if self._transportations[i][j].quantity == '*':
                        zero = TransportElement(i, j, sys.float_info.epsilon, self._transportations[i][j].cost)
                        if len(self.get_cycle(zero)) == 0:
                            self._transportations[i][j] = zero
                            zero_added = True
                            break
                if zero_added:
                    break

    def potential_method(self) -> bool:
        max_reduction = 0
        move = []
        leaving: TransportElement
        is_null = True

        self.remove_degeneracy()

        for i in range(self._rows_amount):
            for j in range(self._cols_amount):
                if self._transportations[i][j].quantity != '*':
                    continue

                trial = TransportElement(i, j, 0, self._transportations[i][j].cost)
                cycle = self.get_cycle(trial)

                if len(cycle) == 0:
                    continue

                reduction = 0
                lowest_quantity = sys.maxsize
                leaving_candidate: TransportElement = None

                plus = True
                for transportation in cycle:
                    if plus:
                        reduction += transportation.cost
                    else:
                        reduction -= transportation.cost
                        if transportation.quantity < lowest_quantity:
                            leaving_candidate = transportation
                            lowest_quantity = transportation.quantity
                    plus = not plus

                if reduction < max_reduction:
                    is_null = False
                    move = cycle
                    leaving = leaving_candidate
                    max_reduction = reduction

        if not is_null:
            quantity = leaving.quantity
            plus = True
            for transportation in move:
                transportation.quantity += quantity if plus else -quantity
                if transportation.quantity == 0:
                    self._transportations[transportation.position_row][transportation.position_col] = \
                        TransportElement(transportation.position_row, transportation.position_col, '*',
                                         self._transportations[transportation.position_row][transportation.position_col].cost)
                else:
                    self._transportations[transportation.position_row][transportation.position_col] = transportation
                plus = not plus

            return False

        return True

    def calculate_transportations_cost(self) -> int:
        transportations_cost = 0
        for row in self._transportations:
            for transportation in row:
                if transportation.quantity != '*':
                    transportations_cost += transportation.cost * transportation.quantity
        return int(transportations_cost)
def print_matrix(matrix):
    for row in matrix:
        print(' '.join(f"{x:5}" for x in row))

def print_transportations_cost(transport):
    for i in range(transport.dimension_rows):
        row = []
        for j in range(transport.dimension_columns):
            row.append(f"{transport.get_element(i, j).cost:5}")
        print(' '.join(row))

def print_transportations_quantity(transport):
    for i in range(transport.dimension_rows):
        row = []
        for j in range(transport.dimension_columns):
            if transport.get_element(i, j).is_base:
                row.append(f"{transport.get_element(i, j).quantity:5.1f}*")
            else:
                row.append(f"{transport.get_element(i, j).quantity:5.1f} ")
        print(' '.join(row))

# ==============================================================================
# ====== КЛАССЫ ДЛЯ СИМПЛЕКС-МЕТОДА ======
# ==============================================================================

class SimplexSolver:
    def __init__(self):
        self.table = []
        self.basis = []
        self.nonBasis = []
        self.solution = []
        self.objCoeffs = []
        self.numConstraints = 0
        self.numVariables = 0
        self.numSlackVariables = 0
        self.isMaximization = True
        self.wasMinimization = False
        self.useDualSimplex = False
        self.dualTable = []
        self.dualBasis = []
        self.dualNonBasis = []
        self.dualSolution = []
        self.dualObjCoeffs = []
        self.originalConstraints = []

    def setUseDualSimplex(self, useDual):
        self.useDualSimplex = useDual

    def removeSpaces(self, s):
        return ''.join(c for c in s if not c.isspace())

    def parseExpression(self, expr):
        expression = self.removeSpaces(expr)
        coeffs = {}
        max_var_index = 0
        is_minimization = False
        if expression.startswith("max"):
            expression = expression[3:]
        elif expression.startswith("min"):
            expression = expression[3:]
            is_minimization = True
        if expression and not (expression[0] in "+-"):
            expression = "+" + expression
        current_term = ""
        current_sign = 1.0
        for i in range(len(expression) + 1):
            if i == len(expression) or expression[i] in "+-":
                if current_term:
                    var_index, coeff = self.processTerm(current_term, current_sign)
                    if var_index > 0:
                        coeffs[var_index] = coeff
                        if var_index > max_var_index:
                            max_var_index = var_index
                    current_term = ""
                if i < len(expression):
                    current_sign = 1.0 if expression[i] == "+" else -1.0
            else:
                current_term += expression[i]
        return coeffs, max_var_index, is_minimization

    def processTerm(self, term, sign):
        term = term.replace("*", "")
        if "x" not in term:
            return 0, 0.0
        x_pos = term.find("x")
        coeff_part = term[:x_pos]
        var_part = term[x_pos + 1:]
        coefficient = 1.0
        if coeff_part:
            try:
                coefficient = float(coeff_part)
            except:
                coefficient = 1.0
        var_index = 1
        digits = "".join(filter(str.isdigit, var_part))
        if digits:
            try:
                var_index = int(digits)
            except:
                var_index = 1
        return var_index, sign * coefficient

    def convertToCanonicalForm(self, constraints, objective):
        all_var_indices = set()
        obj_coeffs, max_obj_var, is_min = self.parseExpression(objective)
        for var in obj_coeffs:
            all_var_indices.add(var)
        constraint_expressions = []
        constraint_types = []
        rhs_values = []
        for constraint in constraints:
            if ">=" in constraint:
                parts = constraint.split(">=")
                constraint_type = 2
            elif "<=" in constraint:
                parts = constraint.split("<=")
                constraint_type = 1
            elif "=" in constraint:
                parts = constraint.split("=")
                constraint_type = 0
            else:
                parts = [constraint, "0"]
                constraint_type = 0
            expr_part = parts[0].strip()
            rhs_part = parts[1].strip() if len(parts) > 1 else "0"
            try:
                rhs_value = float(rhs_part)
            except:
                rhs_value = 0.0
            coeffs, max_var, _ = self.parseExpression(expr_part)
            for var in coeffs:
                all_var_indices.add(var)
            constraint_expressions.append(coeffs)
            constraint_types.append(constraint_type)
            rhs_values.append(rhs_value)

        self.numVariables = max(all_var_indices) if all_var_indices else 0
        self.numSlackVariables = sum(1 for ct in constraint_types if ct != 0)
        objLower = objective.lower()
        self.isMaximization = ("min" not in objLower)
        self.wasMinimization = not self.isMaximization
        if self.wasMinimization:
            for var in obj_coeffs:
                obj_coeffs[var] = -obj_coeffs[var]
            self.isMaximization = True
            print("Задача минимизации преобразована в максимизацию (целевая функция домножена на -1)")

        self.objCoeffs = [0.0] * (self.numVariables + self.numSlackVariables)
        for var, coeff in obj_coeffs.items():
            if 1 <= var <= self.numVariables:
                self.objCoeffs[var - 1] = coeff

        self.numConstraints = len(constraints)
        tempTable = [[0.0] * (self.numVariables + self.numSlackVariables + 1) for _ in range(self.numConstraints)]
        slack_index = 0
        for i in range(self.numConstraints):
            for var, coeff in constraint_expressions[i].items():
                if 1 <= var <= self.numVariables:
                    tempTable[i][var - 1] = coeff
            if constraint_types[i] == 1:
                if slack_index < self.numSlackVariables:
                    tempTable[i][self.numVariables + slack_index] = 1.0
                slack_index += 1
            elif constraint_types[i] == 2:
                if slack_index < self.numSlackVariables:
                    tempTable[i][self.numVariables + slack_index] = -1.0
                slack_index += 1
            tempTable[i][self.numVariables + self.numSlackVariables] = rhs_values[i]

        self.originalConstraints = [row[:] for row in tempTable]
        if not self.useDualSimplex:
            for i in range(self.numConstraints):
                if tempTable[i][self.numVariables + self.numSlackVariables] < 0:
                    for j in range(self.numVariables + self.numSlackVariables + 1):
                        tempTable[i][j] = -tempTable[i][j]

        self.findBasisVariables(tempTable)
        self.buildSimplexTable(tempTable)

    def findBasisVariables(self, tempTable):
        used = [False] * self.numConstraints
        self.basis = []
        if self.useDualSimplex:
            for j in range(self.numVariables + self.numSlackVariables):
                unit_row = -1
                is_unit_column = True
                unit_value = 0.0
                for i in range(self.numConstraints):
                    if abs(abs(tempTable[i][j]) - 1.0) < 1e-10:
                        if unit_row == -1:
                            unit_row = i
                            unit_value = tempTable[i][j]
                        else:
                            is_unit_column = False
                            break
                    elif abs(tempTable[i][j]) > 1e-10:
                        is_unit_column = False
                        break
                if is_unit_column and unit_row != -1 and not used[unit_row]:
                    if unit_value < -0.5:
                        for col in range(self.numVariables + self.numSlackVariables + 1):
                            if col < len(tempTable[unit_row]):
                                tempTable[unit_row][col] = -tempTable[unit_row][col]
                    self.basis.append(j)
                    used[unit_row] = True
            for j in range(self.numVariables + self.numSlackVariables):
                unit_row = -1
                is_unit_column = True
                for i in range(self.numConstraints):
                    if abs(tempTable[i][j] - 1.0) < 1e-10:
                        if unit_row == -1:
                            unit_row = i
                        else:
                            is_unit_column = False
                            break
                    elif abs(tempTable[i][j]) > 1e-10:
                        is_unit_column = False
                        break
                if is_unit_column and unit_row != -1 and not used[unit_row]:
                    self.basis.append(j)
                    used[unit_row] = True
        else:
            for j in range(self.numVariables, self.numVariables + self.numSlackVariables):
                unit_row = -1
                is_unit_column = True
                for i in range(self.numConstraints):
                    if abs(tempTable[i][j] - 1.0) < 1e-10:
                        if unit_row == -1:
                            unit_row = i
                        else:
                            is_unit_column = False
                            break
                    elif abs(tempTable[i][j]) > 1e-10:
                        is_unit_column = False
                        break
                if is_unit_column and unit_row != -1 and not used[unit_row]:
                    self.basis.append(j)
                    used[unit_row] = True
            for j in range(self.numVariables):
                unit_row = -1
                is_unit_column = True
                for i in range(self.numConstraints):
                    if abs(tempTable[i][j] - 1.0) < 1e-10:
                        if unit_row == -1:
                            unit_row = i
                        else:
                            is_unit_column = False
                            break
                    elif abs(tempTable[i][j]) > 1e-10:
                        is_unit_column = False
                        break
                if is_unit_column and unit_row != -1 and not used[unit_row]:
                    self.basis.append(j)
                    used[unit_row] = True

        for i in range(self.numConstraints):
            if not used[i] and len(self.basis) < self.numConstraints:
                for j in range(self.numVariables + self.numSlackVariables):
                    if j not in self.basis:
                        self.basis.append(j)
                        used[i] = True
                        break

        unique_basis = []
        for var in self.basis:
            if var not in unique_basis and len(unique_basis) < self.numConstraints:
                unique_basis.append(var)
        self.basis = unique_basis[:self.numConstraints]
        self.nonBasis = [j for j in range(self.numVariables + self.numSlackVariables) if j not in self.basis]
        print("Базисные переменные: ", end="")
        for var in self.basis:
            print(f"x{var + 1} ", end="")
        print()

    def buildSimplexTable(self, tempTable):
        rows = len(self.basis) + 1
        cols = len(self.nonBasis) + 1
        self.table = [[0.0] * cols for _ in range(rows)]
        self.solution = [0.0] * (self.numVariables + self.numSlackVariables)

        for i in range(len(self.basis)):
            basis_var = self.basis[i]
            constraint_row = -1
            for k in range(self.numConstraints):
                if basis_var < len(tempTable[k]) and abs(abs(tempTable[k][basis_var]) - 1.0) < 1e-10:
                    constraint_row = k
                    break
            if constraint_row == -1:
                max_val = 0
                for k in range(self.numConstraints):
                    if basis_var < len(tempTable[k]) and abs(tempTable[k][basis_var]) > abs(max_val):
                        max_val = tempTable[k][basis_var]
                        constraint_row = k
            if constraint_row != -1 and constraint_row < self.numConstraints:
                for j in range(len(self.nonBasis)):
                    non_basis_var = self.nonBasis[j]
                    if non_basis_var < len(tempTable[constraint_row]):
                        self.table[i][j] = tempTable[constraint_row][non_basis_var]
                b_index = self.numVariables + self.numSlackVariables
                if b_index < len(tempTable[constraint_row]):
                    self.table[i][-1] = tempTable[constraint_row][b_index]
                    self.solution[basis_var] = self.table[i][-1]

        for j in range(len(self.nonBasis)):
            non_basis_var = self.nonBasis[j]
            delta = self.objCoeffs[non_basis_var] if non_basis_var < len(self.objCoeffs) else 0.0
            for i in range(len(self.basis)):
                basis_var = self.basis[i]
                coeff = self.objCoeffs[basis_var] if basis_var < len(self.objCoeffs) else 0.0
                delta -= coeff * self.table[i][j]
            self.table[-1][j] = -delta if self.isMaximization else delta

        current_obj_value = 0.0
        for i in range(len(self.basis)):
            basis_var = self.basis[i]
            if basis_var < len(self.objCoeffs):
                current_obj_value += self.objCoeffs[basis_var] * self.solution[basis_var]
        self.table[-1][-1] = current_obj_value

    def findPivotColumn(self):
        last_row = len(self.table) - 1
        pivot_col = -1
        min_val = 0.0
        for j in range(len(self.table[0]) - 1):
            if self.table[last_row][j] < min_val - 1e-10:
                min_val = self.table[last_row][j]
                pivot_col = j
        return pivot_col

    def findPivotRow(self, pivot_col):
        pivot_row = -1
        min_ratio = float('inf')
        for i in range(len(self.table) - 1):
            if self.table[i][pivot_col] > 1e-10:
                ratio = self.table[i][-1] / self.table[i][pivot_col]
                if ratio + 1e-10 < min_ratio:
                    min_ratio = ratio
                    pivot_row = i
        return pivot_row

    def performPivot(self, pivot_row, pivot_col):
        entering_var = self.nonBasis[pivot_col]
        leaving_var = self.basis[pivot_row]
        old_table = [row[:] for row in self.table]
        pivot_element = old_table[pivot_row][pivot_col]

        for j in range(len(self.table[0])):
            self.table[pivot_row][j] = old_table[pivot_row][j] / pivot_element

        for i in range(len(self.table)):
            if i != pivot_row:
                for j in range(len(self.table[0])):
                    if j != pivot_col:
                        self.table[i][j] = old_table[i][j] - (old_table[i][pivot_col] * old_table[pivot_row][j]) / pivot_element
        for i in range(len(self.table)):
            if i != pivot_row:
                self.table[i][pivot_col] = -old_table[i][pivot_col] / pivot_element

        self.basis[pivot_row] = entering_var
        self.nonBasis[pivot_col] = leaving_var
        self.solution = [0.0] * (self.numVariables + self.numSlackVariables + 1)
        for i in range(len(self.basis)):
            self.solution[self.basis[i]] = self.table[i][-1]

    def solve(self):
        iterations = 0
        max_iterations = 100
        print("Начинаем решение симплекс-методом...")
        self.printTable()
        while iterations < max_iterations:
            pivot_col = self.findPivotColumn()
            if pivot_col == -1:
                print("Оптимальное решение найдено!")
                return True
            pivot_row = self.findPivotRow(pivot_col)
            if pivot_row == -1:
                print("Задача неограничена")
                return False
            print(f"Итерация {iterations + 1}: разрешающий элемент: строка {pivot_row + 1}, столбец {pivot_col + 1} = {self.table[pivot_row][pivot_col]}")
            self.performPivot(pivot_row, pivot_col)
            iterations += 1
            print(f"Таблица после итерации {iterations}:")
            self.printTable()
        print("Превышено максимальное число итераций")
        return False

    def printTable(self):
        print("\n" + "Базис".rjust(8), end="")
        for j, var in enumerate(self.nonBasis):
            print(f"x{var + 1}".rjust(10), end="")
        print("b".rjust(10))
        for i in range(len(self.basis)):
            basis_var = self.basis[i]
            print(f"x{basis_var + 1}".rjust(8), end="")
            for j in range(len(self.nonBasis)):
                val = self.table[i][j]
                print(f"{val:.3f}".rjust(10) if abs(val) > 1e-10 else "0".rjust(10), end="")
            b_val = self.table[i][-1]
            print(f"{b_val:.3f}".rjust(10) if abs(b_val) > 1e-10 else "0".rjust(10))
        print("Дельта".rjust(8), end="")
        for j in range(len(self.nonBasis)):
            val = self.table[-1][j]
            print(f"{val:.3f}".rjust(10) if abs(val) > 1e-10 else "0".rjust(10), end="")
        obj_val = self.table[-1][-1]
        print(f"{obj_val:.3f}".rjust(10) if abs(obj_val) > 1e-10 else "0".rjust(10))
        print()

    def printSolution(self):
        print("\n=== РЕЗУЛЬТАТЫ ===")
        objective_value = sum(self.objCoeffs[i] * self.solution[i] for i in range(self.numVariables))
        if self.wasMinimization:
            objective_value = -objective_value
            print("Исходная задача была минимизацией — значение восстановлено")
        print("Оптимальное решение:")
        for i in range(self.numVariables):
            value = self.solution[i] if i < len(self.solution) else 0.0
            if abs(value) < 1e-10: value = 0.0
            print(f"x{i + 1} = {value:.3f}")
        for i in range(self.numSlackVariables):
            idx = self.numVariables + i
            value = self.solution[idx] if idx < len(self.solution) else 0.0
            if abs(value) < 1e-10: value = 0.0
            print(f"x{idx + 1} (добавочная) = {value:.3f}")
        print(f"Значение целевой функции: {objective_value:.3f} ({'минимум' if self.wasMinimization else 'максимум'})")

    def buildDualProblem(self):
        print("\n=== ПОСТРОЕНИЕ ДВОЙСТВЕННОЙ ЗАДАЧИ ===")
        dual_is_max = not self.isMaximization
        self.dualObjCoeffs = [self.originalConstraints[i][-1] for i in range(self.numConstraints)]
        dual_constraints = []
        for j in range(self.numVariables):
            row = [self.originalConstraints[i][j] for i in range(self.numConstraints)]
            row.append(self.objCoeffs[j])
            dual_constraints.append(row)

        print("Двойственная задача:")
        prefix = "max: " if dual_is_max else "min: "
        terms = []
        for i, c in enumerate(self.dualObjCoeffs):
            sign = "+" if i > 0 and c >= 0 else ""
            terms.append(f"{sign}{c}y{i+1}")
        print(prefix + "".join(terms))

        print("Ограничения:")
        for j, row in enumerate(dual_constraints):
            terms = []
            for i, c in enumerate(row[:-1]):
                sign = "+" if i > 0 and c >= 0 else ""
                terms.append(f"{sign}{c}y{i+1}")
            print("".join(terms) + f" >= {row[-1]}")
        print("Условия неотрицательности двойственных переменных:")
        for i in range(self.numConstraints):
            print(f"y{i + 1} >= 0")

    def solveDualFromPrimal(self):
        print("\n=== РЕШЕНИЕ ДВОЙСТВЕННОЙ ЗАДАЧИ ИЗ ПРЯМОЙ ===")
        if not self.table:
            print("Сначала решите прямую задачу!")
            return
        dual_solution = [0.0] * self.numConstraints
        for i in range(self.numConstraints):
            slack_idx = self.numVariables + i
            for j, nb in enumerate(self.nonBasis):
                if nb == slack_idx:
                    dual_solution[i] = abs(self.table[-1][j])
                    break
        dual_obj = sum(dual_solution[i] * self.originalConstraints[i][-1] for i in range(self.numConstraints))
        print("Решение двойственной задачи:")
        for i in range(self.numConstraints):
            print(f"y{i + 1} = {dual_solution[i]:.3f}")
        print(f"Значение целевой функции двойственной задачи: {dual_obj:.3f}")

    def solvePrimalAndDual(self):
        print("\n=== РЕШЕНИЕ ПРЯМОЙ И ДВОЙСТВЕННОЙ ЗАДАЧ ===")
        print("\n--- РЕШЕНИЕ ПРЯМОЙ ЗАДАЧИ ---")
        if self.solve():
            self.printSolution()
            self.buildDualProblem()
            self.solveDualFromPrimal()

class DualSimplexSolver(SimplexSolver):
    def __init__(self):
        super().__init__()
        self.setUseDualSimplex(True)

    def findDualPivotRow(self):
        pivot_row = -1
        min_val = 0.0
        for i in range(len(self.table) - 1):
            if self.table[i][-1] < min_val - 1e-10:
                min_val = self.table[i][-1]
                pivot_row = i
        return pivot_row

    def findDualPivotColumn(self, pivot_row):
        pivot_col = -1
        min_ratio = float('inf')
        for j in range(len(self.table[0]) - 1):
            if self.table[pivot_row][j] < -1e-10:
                ratio = abs(self.table[-1][j] / self.table[pivot_row][j])
                if ratio < min_ratio - 1e-10:
                    min_ratio = ratio
                    pivot_col = j
        return pivot_col

    def solve(self):
        iterations = 0
        max_iterations = 100
        print("Начинаем решение двойственным симплекс-методом...")
        self.printTable()
        while iterations < max_iterations:
            if all(self.table[i][-1] >= -1e-10 for i in range(len(self.table) - 1)):
                print("Оптимальное решение найдено!")
                return True
            pivot_row = self.findDualPivotRow()
            if pivot_row == -1:
                print("Задача не имеет допустимого решения")
                return False
            pivot_col = self.findDualPivotColumn(pivot_row)
            if pivot_col == -1:
                print("Задача неограничена")
                return False
            print(f"Итерация {iterations + 1}: разрешающий элемент: строка {pivot_row + 1}, столбец {pivot_col + 1} = {self.table[pivot_row][pivot_col]}")
            self.performPivot(pivot_row, pivot_col)
            iterations += 1
            print(f"Таблица после итерации {iterations}:")
            self.printTable()
        print("Превышено максимальное число итераций")
        return False

# ==============================================================================
# ====== ФУНКЦИИ КОНСОЛЬНОГО ИНТЕРФЕЙСА ======
# ==============================================================================

def solve_linear_programming_problem():
    print("\n=== РЕШЕНИЕ ЗАДАЧ ЛИНЕЙНОГО ПРОГРАММИРОВАНИЯ ===")
    while True:
        print("\n=== МЕНЮ ===")
        print("1 - Решить задачу симплекс-методом")
        print("2 - Решить задачу двойственным симплекс-методом")
        print("3 - Решить прямую и двойственную задачу")
        print("0 - Вернуться в главное меню")
        choice = input("Выберите опцию: ").strip()
        if choice == "1":
            solveProblem_lp(False)
        elif choice == "2":
            solveProblem_lp(True)
        elif choice == "3":
            solveProblem_lp(False, True)
        elif choice == "0":
            break
        else:
            print("Неверный выбор.")

def solve_transportation_problem():
    print("\n=== РЕШЕНИЕ ТРАНСПОРТНОЙ ЗАДАЧИ (автоматический ввод данных) ===")
    try:
        # Жёстко заданные данные из вашего варианта
        m = 3
        n = 5
        A = [250, 200, 220]
        B = [140, 110, 170, 90, 140]
        C = [
            [4, 3, 4, 5, 3],
            [2, 4, 5, 7, 8],
            [4, 3, 7, 2, 1]
        ]

        print(f'Размер таблицы: {m} x {n}')
        print('Предложения производителей:', A)
        print('Спрос потребителей:', B)
        print('Тарифы перевозок:')
        print_matrix(C)
        print()

        transport = TransportationTable(m, n)
        transport.producers = A
        transport.consumers = B
        for i in range(m):
            for j in range(n):
                transport.get_element(i, j).cost = C[i][j]

        if not transport.check_balance():
            print('Исходная задача не закрыта')
            transport.balance()

            print('Закрытая задача:')
            print(f'Размер таблицы: {transport.dimension_rows} x {transport.dimension_columns}')
            print('Предложения производителей:', transport.producers)
            print('Спрос потребителей:', transport.consumers)
            print('Тарифы перевозок:')
            print_transportations_cost(transport)
        else:
            print('Исходная задача закрыта')

        print()
        transport.north_west_corner()
        print('Опорный план, полученный методом северо-западного угла:')
        print_transportations_quantity(transport)
        print(f'Стоимость перевозок согласно плану: {transport.calculate_transportations_cost()}')
        print()

        if transport.check_degeneracy():
            print('Полученный план - вырожденный')
            transport.remove_degeneracy()
            print('Новый опорный план:')
            print_transportations_quantity(transport)
            print(f'Стоимость перевозок согласно плану: {transport.calculate_transportations_cost()}')
        else:
            print('Полученный план не является вырожденным')
        print()

        count = 0
        while not transport.potential_method():
            print("Базисные клетки:", transport.basic_cells)
            print("Количество базисных клеток:", len(transport.basic_cells))
            print(f'Оптимизация плана методом потенциалов, итерация №{count + 1}:')
            print_transportations_quantity(transport)
            print(f'Стоимость перевозок согласно плану: {transport.calculate_transportations_cost()}')
            count += 1
            print()

        print('Оптимальный план, полученный методом потенциалов:')
        print_transportations_quantity(transport)
        print(f'Минимальная стоимость перевозок: {transport.calculate_transportations_cost()}')

    except Exception as e:
        print(f"Ошибка: {str(e)}")

    input("\nНажмите Enter для продолжения...")

def solveProblem_lp(useDual=False, solveBoth=False):
    try:
        if solveBoth:
            solver = SimplexSolver()
            n_eq = int(input("Количество ограничений: "))
            obj = input("Целевая функция: ").strip()
            cons = [input(f"Ограничение {i+1}: ").strip() for i in range(n_eq)]
            solver.convertToCanonicalForm(cons, obj)
            solver.solvePrimalAndDual()
        elif useDual:
            solver = DualSimplexSolver()
            n_eq = int(input("Количество ограничений: "))
            obj = input("Целевая функция: ").strip()
            cons = [input(f"Ограничение {i+1}: ").strip() for i in range(n_eq)]
            solver.convertToCanonicalForm(cons, obj)
            solver.solve()
            solver.printSolution()
        else:
            solver = SimplexSolver()
            n_eq = int(input("Количество ограничений: "))
            obj = input("Целевая функция: ").strip()
            cons = [input(f"Ограничение {i+1}: ").strip() for i in range(n_eq)]
            solver.convertToCanonicalForm(cons, obj)
            solver.solve()
            solver.printSolution()
    except Exception as e:
        print(f"Ошибка: {e}")
    input("\nНажмите Enter...")

def check_optimality_interactive():
    print("\n=== ПРОВЕРКА ОПТИМАЛЬНОСТИ ТОЧКИ (3 переменные) ===")
    n = 3
    f_str = input("Коэффициенты f (c1 c2 c3) [Enter для: -1 2 1]: ").strip() or "-1 2 1"
    f_coeff = list(map(float, f_str.split()))
    if len(f_coeff) != n: f_coeff = [-1, 2, 1]
    is_max = input("Максимизация? (y/n) [y]: ").strip().lower() in ("", "y", "yes")
    m = int(input("Количество ограничений [3]: ").strip() or "3")
    constraints, rhs, signs = [], [], []
    for i in range(m):
        row = input(f"Коэффициенты ограничения {i+1} [1 -1 2]: ").strip() or ["1 -1 2", "-2 1 0", "0 1 -1"][i % 3]
        constraints.append(list(map(float, row.split())))
        r = float(input(f"Правая часть ограничения {i+1} [5]: ").strip() or ["5", "3", "4"][i % 3])
        rhs.append(r)
        s = input(f"Знак ограничения {i+1} (<=, >=, =) [<=]: ").strip() or "<="
        signs.append(s)
    x_limits = input("Ограничения на x1,x2,x3 [>=0 >=0 >=0]: ").strip() or ">=0 >=0 >=0"
    x_limits = x_limits.split()
    if len(x_limits) != n: x_limits = [">=0"] * n
    point = input("Точка x (x1 x2 x3) [2 1 0]: ").strip() or "2 1 0"
    point = list(map(float, point.split()))
    if len(point) != n: point = [2, 1, 0]
    # --- двойственная задача ---
    g_coeff = rhs
    dual_constraints = [[constraints[j][i] for j in range(m)] for i in range(n)]
    dual_rhs = f_coeff[:]
    def get_dual_sign(i):
        xl = x_limits[i]
        return (">=" if is_max else "<=") if xl == ">=0" else ("<=" if is_max else ">=") if xl == "<=0" else "="
    dual_signs = [get_dual_sign(i) for i in range(n)]
    def get_y_limit(j):
        cs = signs[j]
        return (">=0" if is_max else "<=0") if cs == "<=" else ("<=" if is_max else ">=0") if cs == ">=" else "свободная"
    y_limits = [get_y_limit(j) for j in range(m)]
    Jx = [i for i, v in enumerate(point) if abs(v) > 1e-10]
    Ix = [i for i in range(m) if sum(constraints[i][j] * point[j] for j in range(n)) < rhs[i] - 1e-10]
    print("\nМНОЖЕСТВА:")
    print(f"J(x) = {{{', '.join(str(i+1) for i in Jx)}}}")
    print(f"I(x) = {{{', '.join(str(i+1) for i in Ix)}}}")
    print(f"Точка x = ({', '.join(f'{v:.4f}' for v in point)})")
    print("\nОБЪЕДИНЕННАЯ СИСТЕМА:")
    for j in Jx:
        terms = [f"{'+' if dual_constraints[j][k] >= 0 and k > 0 else ''}{dual_constraints[j][k]}y{k+1}" for k in range(m)]
        print(f"{''.join(terms)} = {dual_rhs[j]}")
    for i in Ix:
        print(f"y{i+1} = 0")
    # --- решение ---
    A, b = [], []
    for j in Jx:
        A.append(dual_constraints[j][:])
        b.append(dual_rhs[j])
    for i in Ix:
        eq = [0.0] * m
        eq[i] = 1.0
        A.append(eq)
        b.append(0.0)
    y_sol = [0.0] * m
    compatible = True
    if A:
        Ab = [row + [val] for row, val in zip(A, b)]
        rows, cols = len(Ab), m + 1
        for col in range(min(rows, m)):
            pivot = col
            while pivot < rows and abs(Ab[pivot][col]) < 1e-10:
                pivot += 1
            if pivot == rows: continue
            if pivot != col: Ab[col], Ab[pivot] = Ab[pivot], Ab[col]
            piv = Ab[col][col]
            for j in range(col, cols): Ab[col][j] /= piv
            for r in range(rows):
                if r != col and abs(Ab[r][col]) > 1e-10:
                    f = Ab[r][col]
                    for j in range(col, cols): Ab[r][j] -= f * Ab[col][j]
        for i in range(rows):
            if all(abs(Ab[i][j]) < 1e-10 for j in range(m)) and abs(Ab[i][m]) > 1e-10:
                compatible = False
                break
        if compatible:
            determined = [False] * m
            for i in Ix:
                if i < m: y_sol[i] = 0.0; determined[i] = True
            for i in range(rows - 1, -1, -1):
                pc = next((j for j in range(m) if abs(Ab[i][j]) > 1e-10), None)
                if pc is not None and not determined[pc]:
                    val = Ab[i][m] - sum(Ab[i][j] * y_sol[j] for j in range(pc + 1, m) if determined[j])
                    y_sol[pc] = val / Ab[i][pc]
                    determined[pc] = True
            for i in range(m):
                if not determined[i]: y_sol[i] = 0.0
    if not compatible:
        print("\nСИСТЕМА НЕСОВМЕСТНА → точка НЕ оптимальна")
        input("\nНажмите Enter...")
        return
    print("\nРешение:")
    for i, y in enumerate(y_sol):
        print(f"y{i+1} = {y:.6f}")
    var_ok = True
    print("\nПроверка ограничений на y:")
    for i, y in enumerate(y_sol):
        lim = y_limits[i]
        if lim == ">=0" and y < -1e-10:
            print(f"y{i+1} = {y:.6f} нарушает {lim}")
            var_ok = False
        elif lim == "<=0" and y > 1e-10:
            print(f"y{i+1} = {y:.6f} нарушает {lim}")
            var_ok = False
        else:
            print(f"y{i+1} = {y:.6f} {'удовл. ' + lim if lim in ('>=0', '<=0') else '(свободная)'}")
    dual_ok = True
    print("\nПроверка ограничений двойственной задачи:")
    for i in range(n):
        left = sum(dual_constraints[i][j] * y_sol[j] for j in range(m))
        right = dual_rhs[i]
        sign = dual_signs[i]
        sat = (abs(left - right) < 1e-10) if sign == "=" else (left >= right - 1e-10) if sign == ">=" else (left <= right + 1e-10)
        expr = "".join((f"+{c}*{y_sol[j]:.2f}" if c >= 0 and j > 0 else f"{c}*{y_sol[j]:.2f}") for j, c in enumerate(dual_constraints[i])).replace("+-", "-")
        print(f"Огр.{i+1}: {expr} {sign} {right} → {'✅' if sat else '❌'}")
        if not sat: dual_ok = False
    print("\n=== ВЫВОД ===")
    if var_ok and dual_ok:
        print("✅ Точка ОПТИМАЛЬНА")
    else:
        print("❌ Точка НЕ оптимальна")
    input("\nНажмите Enter...")

# ==============================================================================
# ====== GUI ФУНКЦИИ ======
# ==============================================================================

def redirect_print_to_widget(widget):
    class StdoutRedirector:
        def __init__(self, text_widget):
            self.text_space = text_widget
        def write(self, string):
            self.text_space.insert('end', string)
            self.text_space.see('end')
        def flush(self):
            pass
    sys.stdout = StdoutRedirector(widget)

def run_simplex_gui():
    win = tk.Toplevel()
    win.title("Симплекс-метод")
    win.geometry("700x600")
    ttk.Label(win, text="Целевая функция (например: max 2x1 + 3x2):").pack(pady=5)
    obj_entry = ttk.Entry(win, width=70)
    obj_entry.pack(pady=5)
    obj_entry.insert(0, "max 2x1 + 3x2")
    ttk.Label(win, text="Количество ограничений:").pack(pady=5)
    cons_count = ttk.Spinbox(win, from_=1, to=10, width=10)
    cons_count.pack(pady=5)
    cons_count.set(2)
    constraints_frame = ttk.Frame(win)
    constraints_frame.pack(pady=10)
    cons_entries = []
    def update_constraints():
        for w in constraints_frame.winfo_children(): w.destroy()
        cons_entries.clear()
        try: n = int(cons_count.get())
        except: n = 2
        for i in range(n):
            lbl = ttk.Label(constraints_frame, text=f"Ограничение {i+1}:")
            lbl.grid(row=i, column=0, sticky='e', padx=5, pady=2)
            ent = ttk.Entry(constraints_frame, width=50)
            ent.grid(row=i, column=1, padx=5, pady=2)
            ent.insert(0, ["2x1 + x2 <= 10", "x1 + 3x2 <= 15"][i] if i < 2 else "")
            cons_entries.append(ent)
    ttk.Button(win, text="Обновить ограничения", command=update_constraints).pack(pady=10)
    update_constraints()
    output = scrolledtext.ScrolledText(win, width=80, height=20, font=("Courier", 10))
    output.pack(pady=10)
    def solve_and_display():
        output.delete(1.0, tk.END)
        redirect_print_to_widget(output)
        try:
            obj = obj_entry.get().strip()
            cons = [e.get().strip() for e in cons_entries if e.get().strip()]
            if not obj or not cons:
                print("Ошибка: заполните все поля!")
                return
            solver = SimplexSolver()
            solver.convertToCanonicalForm(cons, obj)
            solver.solve()
            solver.printSolution()
        except Exception as e:
            print(f"Ошибка: {e}")
    ttk.Button(win, text="Решить", command=solve_and_display).pack(pady=10)

def run_dual_simplex_gui():
    win = tk.Toplevel()
    win.title("Двойственный симплекс-метод")
    win.geometry("700x600")
    ttk.Label(win, text="Целевая функция (например: min 2x1 + 3x2):").pack(pady=5)
    obj_entry = ttk.Entry(win, width=70)
    obj_entry.pack(pady=5)
    obj_entry.insert(0, "min 2x1 + 3x2")
    ttk.Label(win, text="Количество ограничений:").pack(pady=5)
    cons_count = ttk.Spinbox(win, from_=1, to=10, width=10)
    cons_count.pack(pady=5)
    cons_count.set(2)
    constraints_frame = ttk.Frame(win)
    constraints_frame.pack(pady=10)
    cons_entries = []
    def update_constraints():
        for w in constraints_frame.winfo_children(): w.destroy()
        cons_entries.clear()
        try: n = int(cons_count.get())
        except: n = 2
        for i in range(n):
            lbl = ttk.Label(constraints_frame, text=f"Ограничение {i+1}:")
            lbl.grid(row=i, column=0, sticky='e', padx=5, pady=2)
            ent = ttk.Entry(constraints_frame, width=50)
            ent.grid(row=i, column=1, padx=5, pady=2)
            ent.insert(0, ["2x1 + x2 >= 10", "x1 + 3x2 >= 15"][i] if i < 2 else "")
            cons_entries.append(ent)
    ttk.Button(win, text="Обновить ограничения", command=update_constraints).pack(pady=10)
    update_constraints()
    output = scrolledtext.ScrolledText(win, width=80, height=20, font=("Courier", 10))
    output.pack(pady=10)
    def solve_and_display():
        output.delete(1.0, tk.END)
        redirect_print_to_widget(output)
        try:
            obj = obj_entry.get().strip()
            cons = [e.get().strip() for e in cons_entries if e.get().strip()]
            if not obj or not cons:
                print("Ошибка: заполните все поля!")
                return
            solver = DualSimplexSolver()
            solver.convertToCanonicalForm(cons, obj)
            solver.solve()
            solver.printSolution()
        except Exception as e:
            print(f"Ошибка: {e}")
    ttk.Button(win, text="Решить", command=solve_and_display).pack(pady=10)

def run_primal_dual_gui():
    win = tk.Toplevel()
    win.title("Прямая и двойственная задачи")
    win.geometry("700x600")
    ttk.Label(win, text="Целевая функция (например: max 2x1 + 3x2):").pack(pady=5)
    obj_entry = ttk.Entry(win, width=70)
    obj_entry.pack(pady=5)
    obj_entry.insert(0, "max 2x1 + 3x2")
    ttk.Label(win, text="Количество ограничений:").pack(pady=5)
    cons_count = ttk.Spinbox(win, from_=1, to=10, width=10)
    cons_count.pack(pady=5)
    cons_count.set(2)
    constraints_frame = ttk.Frame(win)
    constraints_frame.pack(pady=10)
    cons_entries = []
    def update_constraints():
        for w in constraints_frame.winfo_children(): w.destroy()
        cons_entries.clear()
        try: n = int(cons_count.get())
        except: n = 2
        for i in range(n):
            lbl = ttk.Label(constraints_frame, text=f"Ограничение {i+1}:")
            lbl.grid(row=i, column=0, sticky='e', padx=5, pady=2)
            ent = ttk.Entry(constraints_frame, width=50)
            ent.grid(row=i, column=1, padx=5, pady=2)
            ent.insert(0, ["2x1 + x2 <= 10", "x1 + 3x2 <= 15"][i] if i < 2 else "")
            cons_entries.append(ent)
    ttk.Button(win, text="Обновить ограничения", command=update_constraints).pack(pady=10)
    update_constraints()
    output = scrolledtext.ScrolledText(win, width=80, height=20, font=("Courier", 10))
    output.pack(pady=10)
    def solve_and_display():
        output.delete(1.0, tk.END)
        redirect_print_to_widget(output)
        try:
            obj = obj_entry.get().strip()
            cons = [e.get().strip() for e in cons_entries if e.get().strip()]
            if not obj or not cons:
                print("Ошибка: заполните все поля!")
                return
            solver = SimplexSolver()
            solver.convertToCanonicalForm(cons, obj)
            solver.solvePrimalAndDual()
        except Exception as e:
            print(f"Ошибка: {e}")
    ttk.Button(win, text="Решить", command=solve_and_display).pack(pady=10)

def run_transport_gui():
    win = tk.Toplevel()
    win.title("Транспортная задача")
    win.geometry("600x500")

    ttk.Label(win, text="Поставщики (m):").pack(pady=2)
    m_ent = ttk.Entry(win, width=10)
    m_ent.pack()
    m_ent.insert(0, "3")

    ttk.Label(win, text="Потребители (n):").pack(pady=2)
    n_ent = ttk.Entry(win, width=10)
    n_ent.pack()
    n_ent.insert(0, "5")

    frame = ttk.Frame(win)
    frame.pack(pady=10)

    def create_fields():
        for w in frame.winfo_children():
            w.destroy()
        try:
            m = int(m_ent.get())
            n = int(n_ent.get())
        except:
            m, n = 3, 5

        ttk.Label(frame, text="Предложения A:").grid(row=0, column=0, sticky='e')
        a_ent = ttk.Entry(frame, width=30)
        a_ent.grid(row=0, column=1)
        a_ent.insert(0, "250 200 220")

        ttk.Label(frame, text="Спрос B:").grid(row=1, column=0, sticky='e')
        b_ent = ttk.Entry(frame, width=30)
        b_ent.grid(row=1, column=1)
        b_ent.insert(0, "140 110 170 90 140")

        ttk.Label(frame, text="Тарифы C (по строкам):").grid(row=2, column=0, sticky='nw')
        c_txt = tk.Text(frame, width=30, height=4)
        c_txt.grid(row=2, column=1)
        c_txt.insert("1.0", "4 3 4 5 3\n2 4 5 7 8\n4 3 7 2 1")

        out = scrolledtext.ScrolledText(frame, width=60, height=15, font=("Courier", 9))
        out.grid(row=4, column=0, columnspan=2, pady=10)

        def solve():
            out.delete(1.0, tk.END)
            redirect_print_to_widget(out)
            try:
                A = list(map(int, a_ent.get().split()))
                B = list(map(int, b_ent.get().split()))
                raw_text = c_txt.get("1.0", tk.END).strip()
                if not raw_text:
                    print("Ошибка: матрица тарифов пуста!")
                    return
                lines = raw_text.splitlines()
                C = []
                for line in lines:
                    stripped = line.strip()
                    if stripped:
                        row = list(map(int, stripped.split()))
                        C.append(row)
                if len(A) != m or len(B) != n or len(C) != m or any(len(row) != n for row in C):
                    print("Ошибка: несоответствие размеров!")
                    return

                transport = TransportationTable(m, n)
                transport.producers = A
                transport.consumers = B
                for i in range(m):
                    for j in range(n):
                        transport.get_element(i, j).cost = C[i][j]

                if not transport.check_balance():
                    print('Исходная задача не закрыта')
                    transport.balance()
                    print('Закрытая задача:')
                    print(f'Размер таблицы: {transport.dimension_rows} x {transport.dimension_columns}')
                    print('Предложения производителей:', transport.producers)
                    print('Спрос потребителей:', transport.consumers)
                    print('Тарифы перевозок:')
                    print_transportations_cost(transport)
                else:
                    print('Исходная задача закрыта')
                print()

                transport.north_west_corner()
                print('Опорный план, полученный методом северо-западного угла:')
                print_transportations_quantity(transport)
                print(f'Стоимость перевозок согласно плану: {transport.calculate_transportations_cost()}')
                print()

                if transport.check_degeneracy():
                    print('Полученный план - вырожденный')
                    transport.remove_degeneracy()
                    print('Новый опорный план:')
                    print_transportations_quantity(transport)
                    print(f'Стоимость перевозок согласно плану: {transport.calculate_transportations_cost()}')
                else:
                    print('Полученный план не является вырожденным')
                print()

                count = 0
                while not transport.potential_method():
                    print(f'Оптимизация плана методом потенциалов, итерация №{count + 1}:')
                    print_transportations_quantity(transport)
                    print(f'Стоимость перевозок согласно плану: {transport.calculate_transportations_cost()}')
                    count += 1
                    print()

                print('Оптимальный план, полученный методом потенциалов: ')
                print_transportations_quantity(transport)
                print(f'Стоимость перевозок согласно плану: {transport.calculate_transportations_cost()}')
            except Exception as e:
                print(f"Ошибка: {e}")

        ttk.Button(frame, text="Решить", command=solve).grid(row=3, column=0, columnspan=2, pady=5)

    ttk.Button(win, text="Создать поля", command=create_fields).pack(pady=5)
    create_fields()
def run_optimality_gui():
    win = tk.Toplevel()
    win.title("Проверка оптимальности точки (3 переменные)")
    win.geometry("650x700")
    ttk.Label(win, text="Коэффициенты f = c1*x1 + c2*x2 + c3*x3:").pack(pady=5)
    f_ent = ttk.Entry(win, width=50)
    f_ent.pack()
    f_ent.insert(0, "-1 2 1")
    max_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(win, text="Максимизация", variable=max_var).pack(pady=5)
    ttk.Label(win, text="Количество ограничений (m):").pack(pady=5)
    m_spin = ttk.Spinbox(win, from_=1, to=5, width=10)
    m_spin.pack()
    m_spin.set(3)
    cons_frame = ttk.Frame(win)
    cons_frame.pack(pady=10)
    cons_widgets = []
    def build_cons():
        for w in cons_frame.winfo_children(): w.destroy()
        cons_widgets.clear()
        try: m = int(m_spin.get())
        except: m = 3
        for i in range(m):
            row = ttk.Frame(cons_frame)
            row.pack(pady=2)
            a = ttk.Entry(row, width=20)
            a.insert(0, ["1 -1 2", "-2 1 0", "0 1 -1"][i % 3])
            a.pack(side='left', padx=2)
            b = ttk.Entry(row, width=8)
            b.insert(0, ["5", "3", "4"][i % 3])
            b.pack(side='left', padx=2)
            s = ttk.Combobox(row, values=["<=", ">=", "="], width=5)
            s.set("<=")
            s.pack(side='left', padx=2)
            cons_widgets.append((a, b, s))
    ttk.Button(win, text="Создать ограничения", command=build_cons).pack(pady=5)
    build_cons()
    ttk.Label(win, text="Ограничения на x1,x2,x3 (>=0, <=0, свободная):").pack(pady=5)
    x_lim = ttk.Entry(win, width=50)
    x_lim.pack()
    x_lim.insert(0, ">=0 >=0 >=0")
    ttk.Label(win, text="Точка x = (x1 x2 x3):").pack(pady=5)
    x_ent = ttk.Entry(win, width=50)
    x_ent.pack()
    x_ent.insert(0, "2 1 0")
    output = scrolledtext.ScrolledText(win, width=80, height=20, font=("Courier", 10))
    output.pack(pady=10)
    def run_check():
        output.delete(1.0, tk.END)
        redirect_print_to_widget(output)
        try:
            f_coeff = list(map(float, f_ent.get().split()))
            is_max = max_var.get()
            m = int(m_spin.get())
            constraints, rhs, signs = [], [], []
            for a_ent, b_ent, s_combo in cons_widgets:
                constraints.append(list(map(float, a_ent.get().split())))
                rhs.append(float(b_ent.get()))
                signs.append(s_combo.get())
            x_limits = x_lim.get().split()
            point = list(map(float, x_ent.get().split()))
            n = 3
            dual_constraints = [[constraints[j][i] for j in range(m)] for i in range(n)]
            dual_rhs = f_coeff[:]
            def get_dual_sign(i):
                xl = x_limits[i]
                return (">=" if is_max else "<=") if xl == ">=0" else ("<=" if is_max else ">=") if xl == "<=0" else "="
            dual_signs = [get_dual_sign(i) for i in range(n)]
            def get_y_limit(j):
                cs = signs[j]
                return (">=0" if is_max else "<=0") if cs == "<=" else ("<=" if is_max else ">=0") if cs == ">=" else "свободная"
            y_limits = [get_y_limit(j) for j in range(m)]
            Jx = [i for i, v in enumerate(point) if abs(v) > 1e-10]
            Ix = [i for i in range(m) if sum(constraints[i][j] * point[j] for j in range(n)) < rhs[i] - 1e-10]
            print("МНОЖЕСТВА:")
            print(f"J(x) = {{{', '.join(str(i+1) for i in Jx)}}}")
            print(f"I(x) = {{{', '.join(str(i+1) for i in Ix)}}}")
            print(f"Точка x = ({', '.join(f'{v:.4f}' for v in point)})\n")
            print("ОБЪЕДИНЕННАЯ СИСТЕМА:")
            for j in Jx:
                terms = [f"{'+' if dual_constraints[j][k] >= 0 and k > 0 else ''}{dual_constraints[j][k]}y{k+1}" for k in range(m)]
                print(f"{''.join(terms)} = {dual_rhs[j]}")
            for i in Ix:
                print(f"y{i+1} = 0")
            A, b = [], []
            for j in Jx:
                A.append(dual_constraints[j][:])
                b.append(dual_rhs[j])
            for i in Ix:
                eq = [0.0] * m
                eq[i] = 1.0
                A.append(eq)
                b.append(0.0)
            y_sol = [0.0] * m
            compatible = True
            if A:
                Ab = [row + [val] for row, val in zip(A, b)]
                rows, cols = len(Ab), m + 1
                for col in range(min(rows, m)):
                    pivot = col
                    while pivot < rows and abs(Ab[pivot][col]) < 1e-10:
                        pivot += 1
                    if pivot == rows: continue
                    if pivot != col: Ab[col], Ab[pivot] = Ab[pivot], Ab[col]
                    piv = Ab[col][col]
                    for j in range(col, cols): Ab[col][j] /= piv
                    for r in range(rows):
                        if r != col and abs(Ab[r][col]) > 1e-10:
                            f = Ab[r][col]
                            for j in range(col, cols): Ab[r][j] -= f * Ab[col][j]
                for i in range(rows):
                    if all(abs(Ab[i][j]) < 1e-10 for j in range(m)) and abs(Ab[i][m]) > 1e-10:
                        compatible = False
                        break
                if compatible:
                    determined = [False] * m
                    for i in Ix:
                        if i < m: y_sol[i] = 0.0; determined[i] = True
                    for i in range(rows - 1, -1, -1):
                        pc = next((j for j in range(m) if abs(Ab[i][j]) > 1e-10), None)
                        if pc is not None and not determined[pc]:
                            val = Ab[i][m] - sum(Ab[i][j] * y_sol[j] for j in range(pc + 1, m) if determined[j])
                            y_sol[pc] = val / Ab[i][pc]
                            determined[pc] = True
                    for i in range(m):
                        if not determined[i]: y_sol[i] = 0.0
            if not compatible:
                print("\nСИСТЕМА НЕСОВМЕСТНА → точка НЕ оптимальна")
                return
            print("\nРешение:")
            for i, y in enumerate(y_sol):
                print(f"y{i+1} = {y:.6f}")
            var_ok = True
            print("\nПроверка ограничений на y:")
            for i, y in enumerate(y_sol):
                lim = y_limits[i]
                if lim == ">=0" and y < -1e-10:
                    print(f"y{i+1} = {y:.6f} нарушает {lim}")
                    var_ok = False
                elif lim == "<=0" and y > 1e-10:
                    print(f"y{i+1} = {y:.6f} нарушает {lim}")
                    var_ok = False
                else:
                    print(f"y{i+1} = {y:.6f} {'удовл. ' + lim if lim in ('>=0', '<=0') else '(свободная)'}")
            dual_ok = True
            print("\nПроверка ограничений двойственной задачи:")
            for i in range(n):
                left = sum(dual_constraints[i][j] * y_sol[j] for j in range(m))
                right = dual_rhs[i]
                sign = dual_signs[i]
                sat = (abs(left - right) < 1e-10) if sign == "=" else (left >= right - 1e-10) if sign == ">=" else (left <= right + 1e-10)
                expr = "".join((f"+{c}*{y_sol[j]:.2f}" if c >= 0 and j > 0 else f"{c}*{y_sol[j]:.2f}") for j, c in enumerate(dual_constraints[i])).replace("+-", "-")
                print(f"Огр.{i+1}: {expr} {sign} {right} → {'✅' if sat else '❌'}")
                if not sat: dual_ok = False
            print("\n=== ВЫВОД ===")
            if var_ok and dual_ok:
                print("✅ Точка ОПТИМАЛЬНА")
            else:
                print("❌ Точка НЕ оптимальна")
        except Exception as e:
            print(f"Ошибка: {e}")
    ttk.Button(win, text="Проверить оптимальность", command=run_check).pack(pady=10)

def main_gui():
    root = tk.Tk()
    root.title("Универсальный решатель задач оптимизации")
    root.geometry("550x350")
    ttk.Label(root, text="Универсальный решатель задач оптимизации", font=("Arial", 14, "bold")).pack(pady=20)
    ttk.Button(root, text="Симплекс-метод", command=run_simplex_gui, width=50).pack(pady=5)
    ttk.Button(root, text="Двойственный симплекс-метод", command=run_dual_simplex_gui, width=50).pack(pady=5)
    ttk.Button(root, text="Решить прямую и двойственную задачу", command=run_primal_dual_gui, width=50).pack(pady=5)
    ttk.Button(root, text="Транспортная задача", command=run_transport_gui, width=50).pack(pady=5)
    ttk.Button(root, text="Проверка оптимальности точки (3 пер.)", command=run_optimality_gui, width=50).pack(pady=5)
    ttk.Button(root, text="Выход", command=root.destroy, width=50).pack(pady=20)
    root.mainloop()

# ==============================================================================
# ====== ЗАПУСК ======
# ==============================================================================

if __name__ == "__main__":
    # Для запуска с GUI (рекомендуется):
    main_gui()

    # Для запуска в консоли (раскомментируйте и закомментируйте main_gui()):
    # main()
