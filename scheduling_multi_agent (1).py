"""
Multi-Agent Job Scheduling Solver (Single Machine)
--------------------------------------------------
Four agents each build a job sequence using a different dispatching rule:
    1. FCFSAgent -> First Come First Served   (order of arrival)
    2. SPTAgent  -> Shortest Processing Time  (smallest time first)
    3. EDDAgent  -> Earliest Due Date         (earliest deadline first)
    4. CRAgent   -> Critical Ratio            (lowest due date / processing time first)

A CoordinatorAgent runs all agents, computes the performance measures for each
sequence, and picks the best one for the chosen criterion.

It also checks the answer against the true best sequence (brute force over all
orders), the same way the transportation code checked against the LP optimum.
"""

from itertools import permutations

# ----------------------------------------------------------------------
# PROBLEM DATA  (change these to try other problems)
# ----------------------------------------------------------------------
# Jobs are listed in order of arrival (this is the FCFS order)
JOBS = ["A", "B", "C", "D", "E", "F"]
PROCESSING_TIME = {"A": 6, "B": 2, "C": 8, "D": 3, "E": 9, "F": 4}
DUE_DATE        = {"A": 8, "B": 6, "C": 18, "D": 15, "E": 23, "F": 20}

# Criterion the coordinator uses to pick the winner. Choose one of:
#   "total_tardiness"   - sum of lateness of late jobs
#   "avg_flow_time"     - average time a job spends in the system
#   "max_tardiness"     - the worst single delay
#   "tardy_jobs"        - number of late jobs
CRITERION = "total_tardiness"


# ----------------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------------
def evaluate(sequence, ptime, due):
    """Run the sequence on the machine and return the schedule + measures."""
    clock = 0
    schedule = []
    total_flow = 0
    total_tardiness = 0
    max_tardiness = 0
    tardy_jobs = 0

    for job in sequence:
        start = clock
        finish = start + ptime[job]
        tardiness = max(0, finish - due[job])

        schedule.append((job, start, finish, due[job], tardiness))

        total_flow += finish
        total_tardiness += tardiness
        max_tardiness = max(max_tardiness, tardiness)
        if tardiness > 0:
            tardy_jobs += 1

        clock = finish

    measures = {
        "avg_flow_time": round(total_flow / len(sequence), 2),
        "total_tardiness": total_tardiness,
        "max_tardiness": max_tardiness,
        "tardy_jobs": tardy_jobs,
        "makespan": clock,
    }
    return schedule, measures


def print_schedule(schedule, measures):
    print(f"  {'Job':<5}{'Start':>7}{'Finish':>8}{'Due':>6}{'Tardy':>7}")
    for job, start, finish, due, tardy in schedule:
        print(f"  {job:<5}{start:>7}{finish:>8}{due:>6}{tardy:>7}")
    print(f"  Average flow time = {measures['avg_flow_time']}")
    print(f"  Total tardiness   = {measures['total_tardiness']}")
    print(f"  Max tardiness     = {measures['max_tardiness']}")
    print(f"  Number of tardy jobs = {measures['tardy_jobs']}")


# ----------------------------------------------------------------------
# AGENT 1 : FCFS
# ----------------------------------------------------------------------
class FCFSAgent:
    name = "FCFS (First Come First Served)"

    def sequence(self, jobs, ptime, due):
        return list(jobs)          # keep arrival order


# ----------------------------------------------------------------------
# AGENT 2 : SPT
# ----------------------------------------------------------------------
class SPTAgent:
    name = "SPT (Shortest Processing Time)"

    def sequence(self, jobs, ptime, due):
        return sorted(jobs, key=lambda j: ptime[j])


# ----------------------------------------------------------------------
# AGENT 3 : EDD
# ----------------------------------------------------------------------
class EDDAgent:
    name = "EDD (Earliest Due Date)"

    def sequence(self, jobs, ptime, due):
        return sorted(jobs, key=lambda j: due[j])


# ----------------------------------------------------------------------
# AGENT 4 : CRITICAL RATIO
# ----------------------------------------------------------------------
class CRAgent:
    name = "CR (Critical Ratio)"

    def sequence(self, jobs, ptime, due):
        # ratio = due date / processing time, smallest ratio goes first
        return sorted(jobs, key=lambda j: due[j] / ptime[j])


# ----------------------------------------------------------------------
# COORDINATOR AGENT : runs all agents and picks the best
# ----------------------------------------------------------------------
class CoordinatorAgent:
    def __init__(self, agents, criterion):
        self.agents = agents
        self.criterion = criterion

    def run(self, jobs, ptime, due):
        results = []

        for agent in self.agents:
            seq = agent.sequence(jobs, ptime, due)
            schedule, measures = evaluate(seq, ptime, due)
            results.append((agent.name, seq, schedule, measures))

            print("\n" + "=" * 55)
            print(f"AGENT: {agent.name}")
            print("=" * 55)
            print("  Sequence:", " -> ".join(seq))
            print_schedule(schedule, measures)

        # compare using the chosen criterion (smaller is better)
        best = results[0]
        for r in results:
            if r[3][self.criterion] < best[3][self.criterion]:
                best = r

        print("\n" + "#" * 70)
        print("COMPARISON SUMMARY")
        print("#" * 70)
        print(f"{'Method':<34}{'AvgFlow':>9}{'TotTardy':>10}{'MaxTardy':>10}{'#Tardy':>8}")
        for name, seq, _, m in results:
            marker = "  <-- BEST" if name == best[0] else ""
            print(f"{name:<34}{m['avg_flow_time']:>9}{m['total_tardiness']:>10}"
                  f"{m['max_tardiness']:>10}{m['tardy_jobs']:>8}{marker}")

        print(f"\nCriterion used: {self.criterion}")
        print(f"Coordinator's decision: {best[0]}")
        print(f"Best sequence: {' -> '.join(best[1])}")
        return best


# ----------------------------------------------------------------------
# CHECK: try every possible job order to find the true best
# ----------------------------------------------------------------------
def verify_with_brute_force(jobs, ptime, due, criterion):
    best_value = None
    best_seq = None
    for seq in permutations(jobs):
        _, m = evaluate(seq, ptime, due)
        if best_value is None or m[criterion] < best_value:
            best_value = m[criterion]
            best_seq = seq
    print(f"\nTrue best {criterion} (checked all {len(list(permutations(jobs)))} orders) = {best_value}")
    print(f"True best sequence: {' -> '.join(best_seq)}")
    return best_value


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Jobs (in arrival order):")
    print(f"  {'Job':<5}{'ProcTime':>10}{'DueDate':>9}")
    for j in JOBS:
        print(f"  {j:<5}{PROCESSING_TIME[j]:>10}{DUE_DATE[j]:>9}")

    coordinator = CoordinatorAgent(
        [FCFSAgent(), SPTAgent(), EDDAgent(), CRAgent()],
        CRITERION,
    )
    best = coordinator.run(JOBS, PROCESSING_TIME, DUE_DATE)

    # brute force is fine for small problems (up to about 9 jobs)
    if len(JOBS) <= 9:
        optimal = verify_with_brute_force(JOBS, PROCESSING_TIME, DUE_DATE, CRITERION)
        if best[3][CRITERION] == optimal:
            print("Best agent's sequence is already the true optimum.")
        else:
            print(f"Best agent is {round(best[3][CRITERION] - optimal, 2)} above the true optimum.")
