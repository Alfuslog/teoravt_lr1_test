class SimplexSolver:
    def __init__(self):
        self.artificial_vars = []
        self.M = 10 ** 6
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
        if expression.lower().startswith("max"):
            expression = expression[3:]
        elif expression.lower().startswith("min"):
            expression = expression[3:]
            is_minimization = True

        if expression and not (expression[0] in ['+', '-']):
            expression = '+' + expression

        current_term = ""
        current_sign = 1.0

        for i in range(len(expression) + 1):
            if i == len(expression) or expression[i] in ['+', '-']:
                if current_term:
                    var_index, coeff = self.processTerm(current_term, current_sign)
                    if var_index > 0:
                        coeffs[var_index] = coeffs.get(var_index, 0.0) + coeff
                        if var_index > max_var_index:
                            max_var_index = var_index
                    current_term = ""
                if i < len(expression):
                    current_sign = 1.0 if expression[i] == '+' else -1.0
            else:
                current_term += expression[i]

        return coeffs, max_var_index, is_minimization

    def processTerm(self, term, sign):
        term = term.replace('*', '')
        if 'x' not in term:
            return 0, 0.0
        x_pos = term.find('x')
        coeff_part = term[:x_pos]
        var_part = term[x_pos + 1:]
        coefficient = 1.0
        if coeff_part:
            try:
                coefficient = float(coeff_part)
            except:
                coefficient = 1.0
        var_index = 1
        digits = ''.join(filter(str.isdigit, var_part))
        if digits:
            try:
                var_index = int(digits)
            except:
                var_index = 1
        return var_index, sign * coefficient

    def convertToCanonicalForm(self, constraints, objective):
        print("=== НАЧАЛО ПРЕОБРАЗОВАНИЯ В КАНОНИЧЕСКУЮ ФОРМУ ===")
        # Парсим целевую функцию
        obj_coeffs, max_obj_var, is_min = self.parseExpression(objective)
        print(f"Парсинг целевой функции: коэффициенты {obj_coeffs}, макс. индекс {max_obj_var}, минимизация={is_min}")

        # Парсим ограничения
        constraint_expressions = []
        constraint_types = []
        rhs_values = []
        all_var_indices = set(obj_coeffs.keys())

        print("\nПарсинг ограничений:")
        for i, constraint in enumerate(constraints):
            print(f"  Ограничение {i + 1}: '{constraint}'")
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
            print(f"    -> Левая часть: {coeffs}, Правая часть: {rhs_value}, Тип: {('=','<=','>=')[constraint_type] if constraint_type in [0,1,2] else constraint_type}")
            for var in coeffs.keys():
                all_var_indices.add(var)
            constraint_expressions.append(coeffs)
            constraint_types.append(constraint_type)
            rhs_values.append(rhs_value)

        # количество реальных переменных
        self.numVariables = max(all_var_indices) if all_var_indices else 0
        print(f"\nОбнаружено переменных (x1..x{self.numVariables}): {list(range(1, self.numVariables + 1))}")

        # подсчёт дополнительных переменных и подготовка структуры
        num_slack = sum(1 for ct in constraint_types if ct == 1)
        num_surplus = sum(1 for ct in constraint_types if ct == 2)
        num_artificial = sum(1 for ct in constraint_types if ct in (0, 2))  # = и >= требуют artificial
        print("\nОпределение количества дополнительных переменных:")
        print(f"  Количество <= ограничений: {num_slack}")
        print(f"  Количество >= ограничений: {num_surplus}")
        print(f"  Количество = ограничений: {sum(1 for ct in constraint_types if ct == 0)}")
        print(f"  Искусственных переменных потребуется: {num_artificial}")

        # Общее число добавочных переменных (we will allocate exactly the slots we need)
        self.numSlackVariables = num_slack + num_surplus + num_artificial
        total_vars = self.numVariables + self.numSlackVariables
        print(f"  Общее количество дополнительных переменных: {self.numSlackVariables} (всего переменных в таблице = {total_vars})")

        # Тип оптимизации
        self.isMaximization = ("min" not in objective.lower())
        self.wasMinimization = not self.isMaximization
        print(f"Тип задачи: {'Максимизация' if self.isMaximization else 'Минимизация'}")
        if self.wasMinimization:
            for k in obj_coeffs:
                obj_coeffs[k] = -obj_coeffs[k]
            self.isMaximization = True
            print("Задача минимизации преобразована в максимизацию (целевая функция домножена на -1)")

        # Формируем objCoeffs для всей расширенной размерности
        self.objCoeffs = [0.0] * total_vars
        for var, coeff in obj_coeffs.items():
            if 1 <= var <= self.numVariables:
                self.objCoeffs[var - 1] = coeff
        print(f"Коэффициенты целевой функции (после преобразований): {self.objCoeffs[:self.numVariables]}")

        # Подготовка временной таблицы
        self.numConstraints = len(constraints)
        tempTable = [[0.0] * (total_vars + 1) for _ in range(self.numConstraints)]
        print(f"\nИнициализирована временная таблица {self.numConstraints}x{total_vars + 1}")

        # Заполнение таблицы: добавляем реальные и дополнительные переменные
        next_extra = 0
        row_basis = [-1] * self.numConstraints  # для каждой строки — индекс переменной, которая является базисной
        self.artificial_vars = []

        print("\nЗаполнение таблицы:")
        for i in range(self.numConstraints):
            # реальные переменные
            for var, coeff in constraint_expressions[i].items():
                if 1 <= var <= self.numVariables:
                    tempTable[i][var - 1] = coeff

            ctype = constraint_types[i]
            if ctype == 1:  # <= : добавляем slack = +1 и делаем её базисной
                slack_idx = self.numVariables + next_extra
                tempTable[i][slack_idx] = 1.0
                row_basis[i] = slack_idx
                print(f"  Строка {i+1}: добавлена slack x{slack_idx+1} как базис")
                next_extra += 1

            elif ctype == 2:  # >= : добавляем surplus (-1) и искусственную (+1) — искусственная в базисе
                surplus_idx = self.numVariables + next_extra
                tempTable[i][surplus_idx] = -1.0
                next_extra += 1
                artificial_idx = self.numVariables + next_extra
                tempTable[i][artificial_idx] = 1.0
                row_basis[i] = artificial_idx
                self.artificial_vars.append(artificial_idx)
                # ставим "большой" штраф в исходной целевой для искусственных (временно)
                self.objCoeffs[artificial_idx] = -self.M if self.isMaximization else self.M
                print(f"  Строка {i+1}: добавлен surplus x{surplus_idx+1} и artificial x{artificial_idx+1} (artificial базис)")
                next_extra += 1

            elif ctype == 0:  # = : добавляем только искусственную и делаем её базисной
                artificial_idx = self.numVariables + next_extra
                tempTable[i][artificial_idx] = 1.0
                row_basis[i] = artificial_idx
                self.artificial_vars.append(artificial_idx)
                self.objCoeffs[artificial_idx] = -self.M if self.isMaximization else self.M
                print(f"  Строка {i+1}: добавлена artificial x{artificial_idx+1} (базис)")
                next_extra += 1

            # правая часть
            tempTable[i][-1] = rhs_values[i] if 'rhs_values' in locals() else rhs_values[i]  # rhs_values определён выше
            if tempTable[i][-1] < 0:
                # делаем RHS положительной корректировкой строки
                for j in range(len(tempTable[i])):
                    tempTable[i][j] = -tempTable[i][j]
                print(f"  Строка {i+1} домножена на -1 для положительной правой части")

            print(f"    -> Полная строка {i+1}: {tempTable[i]}")

        # Сохраняем
        self.originalConstraints = [row[:] for row in tempTable]

        # Формируем self.basis по row_basis (если есть -1, будем заполнять дальше)
        self.basis = []
        for i in range(self.numConstraints):
            if row_basis[i] != -1:
                self.basis.append(row_basis[i])
            else:
                # попробуем найти единичную колонку среди добавочных/реальных
                found = False
                for j in range(total_vars):
                    col_vals = [round(tempTable[r][j], 10) for r in range(self.numConstraints)]
                    if col_vals.count(1.0) == 1 and all(v == 0.0 or v == 1.0 for v in col_vals):
                        if j not in self.basis:
                            self.basis.append(j)
                            found = True
                            break
                if not found:
                    # запасной план: возьмём первую свободную колонку
                    for j in range(total_vars):
                        if j not in self.basis:
                            self.basis.append(j)
                            break
        # гарантируем размер
        self.basis = self.basis[:self.numConstraints]
        # nonBasis — все остальные столбцы
        self.nonBasis = [j for j in range(total_vars) if j not in self.basis]

        print(f"\nИтоговый начальный базис: {[f'x{b+1}' for b in self.basis]}")
        print(f"Искусственные переменные: {[f'x{v+1}' for v in self.artificial_vars]}")

        # Если есть искусственные — запускаем двухфазный метод
        if self.artificial_vars:
            self.twoPhaseSimplex(tempTable)
        else:
            self.buildSimplexTable(tempTable)

        print("=== КАНОНИЧЕСКАЯ ФОРМА СФОРМИРОВАНА ===")

    def buildSimplexTable(self, tempTable):
        print("\n=== ПОСТРОЕНИЕ СИМПЛЕКС-ТАБЛИЦЫ ===")
        rows = len(self.basis) + 1
        cols = len(self.nonBasis) + 1
        self.table = [[0.0] * cols for _ in range(rows)]
        self.solution = [0.0] * (self.numVariables + self.numSlackVariables)
        print(f"Инициализирована таблица {rows}x{cols}")

        for i in range(len(self.basis)):
            basis_var = self.basis[i]
            # найдем строку в tempTable где этот basis_var является единичным
            row_index = -1
            for r in range(self.numConstraints):
                if basis_var < len(tempTable[r]) and abs(tempTable[r][basis_var] - 1.0) < 1e-10:
                    row_index = r
                    break
            if row_index == -1:
                # если нет единичной — берём любую строку (плохой кейс)
                for r in range(self.numConstraints):
                    if basis_var < len(tempTable[r]) and abs(tempTable[r][basis_var]) > 1e-10:
                        row_index = r
                        break
            if row_index == -1:
                continue
            for j, nb in enumerate(self.nonBasis):
                if nb < len(tempTable[row_index]):
                    self.table[i][j] = tempTable[row_index][nb]
            self.table[i][-1] = tempTable[row_index][-1]
            if basis_var < len(self.solution):
                self.solution[basis_var] = self.table[i][-1]
            print(f"  Строка {i+1} таблицы (basis x{basis_var+1}): {self.table[i]}")

        # заполнить строку оценок
        self.recalculateDeltaRow()
        print("=== СИМПЛЕКС-ТАБЛИЦА ПОСТРОЕНА ===")

    def performPivot(self, pivot_row, pivot_col):
        entering_var = self.nonBasis[pivot_col]
        leaving_var = self.basis[pivot_row]
        print(f"    Pivot: x{leaving_var+1} (out) <-> x{entering_var+1} (in)")
        # классический Gauss-Jordan
        pivot_element = self.table[pivot_row][pivot_col]
        if abs(pivot_element) < 1e-12:
            raise ZeroDivisionError("Разрешающий элемент равен нулю при выполнении pivot")

        # нормализация ведущей строки
        for j in range(len(self.table[0])):
            self.table[pivot_row][j] /= pivot_element

        # зануляем остальные строки
        for i in range(len(self.table)):
            if i == pivot_row:
                continue
            factor = self.table[i][pivot_col]
            if abs(factor) > 1e-12:
                for j in range(len(self.table[0])):
                    self.table[i][j] -= factor * self.table[pivot_row][j]
                # чтобы избежать маленьких чисел
                self.table[i][pivot_col] = 0.0

        # обновляем базис/небазис
        self.basis[pivot_row] = entering_var
        self.nonBasis[pivot_col] = leaving_var

        # обновляем solution (значения переменных)
        total_vars = self.numVariables + self.numSlackVariables
        self.solution = [0.0] * total_vars
        for i in range(len(self.basis)):
            bvar = self.basis[i]
            if i < len(self.table):
                self.solution[bvar] = self.table[i][-1]

        # пересчитать строку дельт
        self.recalculateDeltaRow()

        print(f"    Новый базис: {[f'x{b+1}' for b in self.basis]}")

    def findPivotIndices(self):
        last_row = len(self.table) - 1
        pivot_col = -1
        min_val = 0.0
        print("  Поиск разрешающего столбца (наиболее отрицательная дельта):")
        for j in range(len(self.table[0]) - 1):
            val = self.table[last_row][j]
            nb_var = self.nonBasis[j]
            print(f"    x{nb_var+1}: Δ = {val}")
            if val < min_val - 1e-12:
                min_val = val
                pivot_col = j
        if pivot_col == -1:
            return -1, -1

        pivot_row = -1
        min_ratio = float('inf')
        print("  Поиск разрешающей строки (метод отношения):")
        for i in range(len(self.table) - 1):
            a = self.table[i][pivot_col]
            b = self.table[i][-1]
            if a > 1e-12:
                ratio = b / a
                print(f"    Строка {i+1}: b={b}, a={a}, ratio={ratio}")
                if ratio < min_ratio - 1e-12:
                    min_ratio = ratio
                    pivot_row = i
            else:
                print(f"    Строка {i+1}: a={a} <= 0, пропускаем")
        return pivot_col, pivot_row

    def solve(self):
        print("\n=== НАЧАЛО РЕШЕНИЯ СИМПЛЕКС-МЕТОДОМ ===")
        iter_count = 0
        max_iter = 200
        while iter_count < max_iter:
            print(f"\n--- ИТЕРАЦИЯ {iter_count+1} ---")
            self.printTable()
            pivot_col, pivot_row = self.findPivotIndices()
            if pivot_col == -1:
                print("Оптимум достигнут (все Δ >= 0).")
                return True
            if pivot_row == -1:
                print("Задача неограничена (нет положительных a_ij в столбце).")
                return False
            print(f"Разрешающий элемент в строке {pivot_row+1}, столбце {pivot_col+1}")
            self.performPivot(pivot_row, pivot_col)
            iter_count += 1
        print("Превышен лимит итераций.")
        return False

    def recalculateDeltaRow(self):
        rows = len(self.table)
        cols = len(self.table[0])
        for j in range(cols - 1):
            nb_var = self.nonBasis[j]
            c_j = self.objCoeffs[nb_var] if nb_var < len(self.objCoeffs) else 0.0
            z_j = 0.0
            for i in range(rows - 1):
                bvar = self.basis[i]
                c_b = self.objCoeffs[bvar] if bvar < len(self.objCoeffs) else 0.0
                z_j += c_b * self.table[i][j]
            delta = z_j - c_j
            # для максимизации мы хотим Δ >= 0
            self.table[rows - 1][j] = delta

        # вычисляем значение Z
        zvalue = 0.0
        for i in range(rows - 1):
            bvar = self.basis[i]
            c_b = self.objCoeffs[bvar] if bvar < len(self.objCoeffs) else 0.0
            zvalue += c_b * self.table[i][-1]
        self.table[rows - 1][-1] = zvalue

    def printTable(self):
        if not self.table:
            print("Таблица пуста")
            return
        # заголовки
        header = ["Базис"] + [f"x{v+1}" for v in self.nonBasis] + ["b"]
        print(" | ".join(x.rjust(8) for x in header))
        for i in range(len(self.basis)):
            row = [f"x{self.basis[i]+1}".rjust(8)]
            row += [f"{self.table[i][j]:8.3f}" for j in range(len(self.table[0]) - 1)]
            row.append(f"{self.table[i][-1]:8.3f}")
            print(" | ".join(row))
        # строка дельт
        drow = ["Δ".rjust(8)] + [f"{self.table[-1][j]:8.3f}" for j in range(len(self.table[0]) - 1)] + [f"{self.table[-1][-1]:8.3f}"]
        print(" | ".join(drow))

    def printSolution(self):
        print("\n=== РЕЗУЛЬТАТЫ ===")
        total_vars = self.numVariables + self.numSlackVariables
        for i in range(total_vars):
            val = self.solution[i] if i < len(self.solution) else 0.0
            if abs(val) < 1e-10:
                val = 0.0
            name = f"x{i+1}"
            if i >= self.numVariables:
                name += " (добавочная)"
            print(f"{name} = {val:.6f}")
        # значение целевой
        z = 0.0
        for i in range(self.numVariables):
            if i < len(self.objCoeffs) and i < len(self.solution):
                z += self.objCoeffs[i] * self.solution[i]
        if self.wasMinimization:
            z = -z
        print(f"Z = {z:.6f} ({'максимум' if not self.wasMinimization else 'минимум'})")

    # Предполагается, что twoPhaseSimplex реализует фазу 1 и фазу 2 (упрощённо)
    def twoPhaseSimplex(self, tempTable):
        print("\n=== ДВУХЭТАПНЫЙ СИМПЛЕКС (ФАЗА 1) ===")
        # Сохраним исходную целевую
        original_obj = self.objCoeffs[:]
        # Формируем фазу1: минимизация суммы искусственных -> в форме максимизации делаем с отрицанием
        phase1_obj = [0.0] * len(self.objCoeffs)
        for art in self.artificial_vars:
            phase1_obj[art] = 1.0  # минимизируем сумму => для нашей схемы оставляем +1 и потом будем искать минимум по Δ>0

        self.objCoeffs = phase1_obj
        # Построим таблицу для фазы1 используя существующий базис/небазис
        self.buildSimplexTable(tempTable)
        # Теперь преобразуем в задачу минимизации: в нашем реализации будем искать положительные Δ в строке оценок
        success = self.solvePhase1()
        if not success:
            print("Фаза 1 не дала допустимого решения.")
            return False
        # Проверяем: искусственные должны иметь нулевые значения (или быть вынесены)
        for art in self.artificial_vars:
            if art in self.basis:
                print(f"Искусственная x{art+1} осталась в базисе -> проблема.")
                # Попытка удалить: если значение RHS==0 — можно выкинуть столбец
                idx = self.basis.index(art)
                if abs(self.table[idx][-1]) < 1e-10:
                    # выкидываем столбец artificial из nonBasis/basis/objCoeffs
                    pass
        # Восстанавливаем исходную целевую
        self.objCoeffs = original_obj
        # удаление столбцов искусственных переменных из таблицы и nonBasis (можно реализовать при необходимости)
        # Пересчитываем дельты под исходную цель
        self.updateObjectiveRow()
        print("\n=== ПРОВЕРКА ИСКУССТВЕННЫХ ПЕРЕМЕННЫХ ===")
        for art in self.artificial_vars:
            idx = -1
            if art in self.basis:
                idx = self.basis.index(art)
                val = self.table[idx][-1]
                print(f"  Artificial x{art + 1} в базисе, RHS = {val}")
                if abs(val) < 1e-10:
                    print(f"    Значение почти 0, можно удалить из базиса")
                    # ищем колонку для pivot, чтобы вытеснить artificial
                    for j, nb in enumerate(self.nonBasis):
                        if abs(self.table[idx][j]) > 1e-12:
                            print(f"    Pivot для удаления artificial: строка {idx + 1}, столбец {j + 1}")
                            self.performPivot(idx, j)
                            break
                else:
                    print(f"    Artificial не может быть удалена, задача недопустима!")
                    raise RuntimeError(f"Artificial x{art + 1} осталась в базисе с ненулевым значением")

        print("\nПереход ко ФАЗЕ 2 (исходная целевая):")
        return self.solve()

    def solvePhase1(self):
        print("Решаем фазу 1 (минимизация суммы искусственных):")
        iter_cnt = 0
        max_iter = 200
        while iter_cnt < max_iter:
            # найти столбец с наибольшей положительной дельтой
            last_row = len(self.table) - 1
            pivot_col = -1
            max_d = 0.0
            for j in range(len(self.table[0]) - 1):
                val = self.table[last_row][j]
                if val > max_d + 1e-12:
                    max_d = val
                    pivot_col = j
            if pivot_col == -1:
                print("Фаза 1 оптимальна (нет положительных Δ).")
                return True
            # найдем разрешающую строку
            pivot_row = -1
            min_ratio = float('inf')
            for i in range(len(self.table) - 1):
                a = self.table[i][pivot_col]
                b = self.table[i][-1]
                if a > 1e-12:
                    ratio = b / a
                    if ratio < min_ratio - 1e-12:
                        min_ratio = ratio
                        pivot_row = i
            if pivot_row == -1:
                print("Фаза 1: задача неограничена.")
                return False

            # **Здесь выполняем pivot**
            self.performPivot(pivot_row, pivot_col)

            # ✅ Логирование состояния после pivot
            print("После pivot:")
            self.printTable()

            iter_cnt += 1
        print("Фаза 1: превышен лимит итераций.")
        return False

    def updateObjectiveRow(self):
        # обновляем строку дельт под текущую self.objCoeffs
        self.recalculateDeltaRow()
