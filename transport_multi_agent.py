"""
Multi-Agent Transportation Problem Solver
-----------------------------------------
Three agents each build an initial feasible solution:
    1. NorthWestAgent  -> North-West Corner Method
    2. LeastCostAgent  -> Least Cost Method (Matrix Minima)
    3. VogelAgent      -> Vogel's Approximation Method (VAM)

A CoordinatorAgent runs all three, compares total costs and picks the best.
(Optional) If scipy is installed, the result is also checked against the
true LP optimum so you can show how close the best method got.
"""

import copy

# ----------------------------------------------------------------------
# PROBLEM DATA  (change this matrix to try other problems)
# ----------------------------------------------------------------------
# Rows = Plants (sources), Columns = Warehouses (destinations)
COST = [
    [19, 30, 50, 10],   # Plant P1
    [70, 30, 40, 60],   # Plant P2
    [40,  8, 70, 20],   # Plant P3
]
SUPPLY = [7, 9, 18]            # units available at P1, P2, P3
DEMAND = [5, 8, 7, 14]         # units needed at W1, W2, W3, W4


# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------
def balance_problem(cost, supply, demand):
    """If total supply != total demand, add a dummy row/column with 0 cost."""
    cost = copy.deepcopy(cost)
    supply = supply[:]
    demand = demand[:]
    total_s, total_d = sum(supply), sum(demand)

    if total_s > total_d:                       # add dummy destination
        for row in cost:
            row.append(0)
        demand.append(total_s - total_d)
        print(f"Unbalanced: added dummy destination with demand {total_s - total_d}")
    elif total_d > total_s:                     # add dummy source
        cost.append([0] * len(demand))
        supply.append(total_d - total_s)
        print(f"Unbalanced: added dummy source with supply {total_d - total_s}")

    return cost, supply, demand


def total_cost(cost, allocation):
    total = 0
    for i in range(len(cost)):
        for j in range(len(cost[0])):
            total += cost[i][j] * allocation[i][j]
    return total


def print_allocation(cost, allocation):
    rows, cols = len(cost), len(cost[0])
    header = "        " + "".join(f"{'D' + str(j + 1):>8}" for j in range(cols))
    print(header)
    for i in range(rows):
        line = f"  S{i + 1}    "
        for j in range(cols):
            if allocation[i][j] > 0:
                line += f"{allocation[i][j]:>4}x{cost[i][j]:<3}"
            else:
                line += f"{'-':>8}"
        print(line)
    print("  (format: units x unit-cost)")


# ----------------------------------------------------------------------
# AGENT 1 : NORTH-WEST CORNER
# ----------------------------------------------------------------------
class NorthWestAgent:
    name = "North-West Corner Method"

    def solve(self, cost, supply, demand):
        supply, demand = supply[:], demand[:]
        rows, cols = len(supply), len(demand)
        allocation = [[0] * cols for _ in range(rows)]

        i, j = 0, 0
        while i < rows and j < cols:
            qty = min(supply[i], demand[j])
            allocation[i][j] = qty
            supply[i] -= qty
            demand[j] -= qty

            if supply[i] == 0 and i < rows - 1:
                i += 1
            elif demand[j] == 0:
                j += 1
            else:
                i += 1

        return allocation


# ----------------------------------------------------------------------
# AGENT 2 : LEAST COST METHOD
# ----------------------------------------------------------------------
class LeastCostAgent:
    name = "Least Cost Method"

    def solve(self, cost, supply, demand):
        supply, demand = supply[:], demand[:]
        rows, cols = len(supply), len(demand)
        allocation = [[0] * cols for _ in range(rows)]

        row_done = [False] * rows
        col_done = [False] * cols

        while not all(row_done) and not all(col_done):
            # find the cheapest cell among rows/columns still active
            best_cost = None
            best_i, best_j = -1, -1
            for i in range(rows):
                if row_done[i]:
                    continue
                for j in range(cols):
                    if col_done[j]:
                        continue
                    if best_cost is None or cost[i][j] < best_cost:
                        best_cost = cost[i][j]
                        best_i, best_j = i, j

            qty = min(supply[best_i], demand[best_j])
            allocation[best_i][best_j] = qty
            supply[best_i] -= qty
            demand[best_j] -= qty

            if supply[best_i] == 0:
                row_done[best_i] = True
            else:
                col_done[best_j] = True

        return allocation


# ----------------------------------------------------------------------
# AGENT 3 : VOGEL'S APPROXIMATION METHOD
# ----------------------------------------------------------------------
class VogelAgent:
    name = "Vogel's Approximation Method"

    def penalty(self, values):
        """Difference between the two smallest values (or the value itself if only one)."""
        values = sorted(values)
        if len(values) == 1:
            return values[0]
        return values[1] - values[0]

    def solve(self, cost, supply, demand):
        supply, demand = supply[:], demand[:]
        rows, cols = len(supply), len(demand)
        allocation = [[0] * cols for _ in range(rows)]

        row_done = [False] * rows
        col_done = [False] * cols

        while not all(row_done) and not all(col_done):
            # --- row penalties ---
            row_pen = {}
            for i in range(rows):
                if not row_done[i]:
                    row_pen[i] = self.penalty([cost[i][j] for j in range(cols) if not col_done[j]])

            # --- column penalties ---
            col_pen = {}
            for j in range(cols):
                if not col_done[j]:
                    col_pen[j] = self.penalty([cost[i][j] for i in range(rows) if not row_done[i]])

            # --- pick the largest penalty ---
            max_row = max(row_pen.values())
            max_col = max(col_pen.values())

            if max_row >= max_col:
                i = max(row_pen, key=row_pen.get)
                # cheapest active cell in that row
                j = min((j for j in range(cols) if not col_done[j]), key=lambda j: cost[i][j])
            else:
                j = max(col_pen, key=col_pen.get)
                # cheapest active cell in that column
                i = min((i for i in range(rows) if not row_done[i]), key=lambda i: cost[i][j])

            qty = min(supply[i], demand[j])
            allocation[i][j] = qty
            supply[i] -= qty
            demand[j] -= qty

            if supply[i] == 0:
                row_done[i] = True
            else:
                col_done[j] = True

        return allocation


# ----------------------------------------------------------------------
# COORDINATOR AGENT : runs all agents and picks the best
# ----------------------------------------------------------------------
class CoordinatorAgent:
    def __init__(self, agents):
        self.agents = agents

    def run(self, cost, supply, demand):
        results = []

        for agent in self.agents:
            allocation = agent.solve(cost, supply, demand)
            c = total_cost(cost, allocation)
            results.append((agent.name, allocation, c))

            print("\n" + "=" * 55)
            print(f"AGENT: {agent.name}")
            print("=" * 55)
            print_allocation(cost, allocation)
            print(f"Total Transportation Cost = {c}")

        # compare and choose the minimum cost
        best = results[0]
        for r in results:
            if r[2] < best[2]:
                best = r

        print("\n" + "#" * 55)
        print("COMPARISON SUMMARY")
        print("#" * 55)
        for name, _, c in results:
            marker = "  <-- BEST" if c == best[2] else ""
            print(f"{name:<32} Cost = {c}{marker}")

        print(f"\nCoordinator's decision: {best[0]} gives the lowest cost = {best[2]}")
        return best


# ----------------------------------------------------------------------
# OPTIONAL: verify against the true LP optimum using scipy
# ----------------------------------------------------------------------
def verify_with_lp(cost, supply, demand):
    try:
        from scipy.optimize import linprog
    except ImportError:
        print("\n(scipy not installed - skipping LP verification)")
        return None

    rows, cols = len(supply), len(demand)
    c = [cost[i][j] for i in range(rows) for j in range(cols)]

    A_eq, b_eq = [], []
    for i in range(rows):                       # supply constraints
        row = [0] * (rows * cols)
        for j in range(cols):
            row[i * cols + j] = 1
        A_eq.append(row)
        b_eq.append(supply[i])
    for j in range(cols):                       # demand constraints
        row = [0] * (rows * cols)
        for i in range(rows):
            row[i * cols + j] = 1
        A_eq.append(row)
        b_eq.append(demand[j])

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    print(f"\nTrue optimal cost (LP check) = {round(result.fun)}")
    return round(result.fun)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Original Cost Matrix:")
    for row in COST:
        print("  ", row)
    print("Supply:", SUPPLY)
    print("Demand:", DEMAND)

    cost, supply, demand = balance_problem(COST, SUPPLY, DEMAND)

    coordinator = CoordinatorAgent([NorthWestAgent(), LeastCostAgent(), VogelAgent()])
    best = coordinator.run(cost, supply, demand)

    optimal = verify_with_lp(cost, supply, demand)
    if optimal is not None:
        if best[2] == optimal:
            print("Best agent's solution is already the true optimum.")
        else:
            print(f"Best agent is {best[2] - optimal} above the true optimum "
                  f"(initial solutions may need MODI improvement).")
