# 🤖 Autonomous Robot Task Planning (PDDL)

## 📌 Project Overview
This project implements a symbolic AI system for **Automated Planning** in logistics. Using **PDDL (Planning Domain Definition Language)**, we modeled an autonomous agent ("Wall-E") capable of cleaning facilities and managing waste transport under strict topological constraints.

Unlike Machine Learning (which learns from data), this project uses **Logical Reasoning** to generate the optimal sequence of actions (plan) to transition from an initial state to a goal state.

*Developed as part of the Automated Reasoning & Planning module - Artificial Intelligence Specialization at UNIR.*

## 🧠 Key Concepts
* **Symbolic AI:** Knowledge representation using predicates and logical operators.
* **State Space Search:** Finding the shortest path of actions using heuristics.
* **STRIPS Architecture:** Defining actions via Preconditions and Effects.

## 🛠️ Tech Stack
* **Language:** PDDL 2.1 (Planning Domain Definition Language).
* **Solver Engine:** Compatible with **Fast Downward** or **Metric-FF**.
* **Algorithms:** A* Search (A-Star).
* **Heuristic:** Landmark-Cut (lmcut) for optimal cost estimation.

## 📂 Domain Logic (`domain.pddl`)
The domain defines the "physics" of the environment:
* **Types:** `robot`, `location`, `bag` (trash).
* **Predicates:** `(at-robot ?r ?loc)`, `(holding ?r ?bag)`, `(connected ?loc1 ?loc2)`.
* **Actions:**
    1.  `move`: Navigates between connected nodes.
    2.  `pick-up`: Grabs an object (requires empty hand).
    3.  `drop`: Deposits waste (only allowed in specific disposal rooms).
    4.  `clean`: Changes the state of a room from dirty to clean.

## 🧪 Scenarios Tested
We evaluated the planner's robustness across three complexity levels:

1.  **Base Case (`problem_base.pddl`):**
    * Star topology map.
    * Goal: Transport 2 bags and clean rooms.
2.  **High Density (`problem_complex.pddl`):**
    * Multiple trash bags per room.
    * Challenges the planner's capacity to manage inventory (pick/drop cycles).
3.  **Linear Topology (`problem_linear_map.pddl`):**
    * Constrained movement (Room A is only accessible via Room B).
    * Forces the agent to plan deep recursive paths.

## 📊 Results
The logic was tested using the A* algorithm with the *lmcut* heuristic.
* **Performance:** All scenarios were solved in **< 0.1 seconds**.
* **Optimality:** The planner successfully found the minimal sequence of steps (shortest path) for logistical distribution.

## 🚀 How to Run
You can run these files using any PDDL solver or an online editor:

1.  Go to [Planning.domains Editor](http://editor.planning.domains/).
2.  Upload `domain.pddl` and one of the problem files (e.g., `problem_base.pddl`).
3.  Click **Solve**.

Alternatively, using **Fast Downward** locally:
```bash
./fast-downward.py pddl/domain.pddl pddl/problem_complex.pddl --search "astar(lmcut())"
