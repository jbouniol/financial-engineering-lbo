# Financial Engineering and Intro to Trading — Group Project

## 1. Overview
This project tasks your group with acting as a Quantitative Research Team for a hedge fund. You will propose, implement and validate an algorithmic trading strategy using Python.

## 2. Groups and Presentations
- **Groups**: Must consist of 4–5 members. Groups must register via the sign-up sheet available on Blackboard.
- **Presentations**: A 15-minute in-class pitch followed by a 5-minute Q&A on **May 29, 2026**. Presentation orders will be determined randomly and posted on the sign-up sheet.
- **Submission**: All deliverables (Report, Presentation Deck, Python Notebook and Datasets) must be submitted via Blackboard not later than **May 31, 2026**.

## 3. Project Objectives
By completing this project, students will demonstrate the ability to:
- Extract, clean and analyze financial data.
- Design and architect an algorithmic trading strategy.
- Execute the strategy in a paper trading environment.
- Conduct rigorous Backtesting and iterative strategy refinement.
- Implement performance and risk management metrics.

> Note: Students are evaluated on their ability to apply classroom concepts. The absolute financial performance (PnL) of the strategy does not factor into the final grade.

## 4. Topic Selection & Analysis Components
Your project must address the following components:

### A. Strategy Design
- Detailed logic and mechanical description of the strategy.
- Economic/Statistical rationale: Why should this strategy be profitable?
- Explicit statement of underlying assumptions.
- Definition of trading signals and the rationale for their selection.

### B. Backtesting
- **Data**: Description of sources, extraction process, cleaning, and exploratory data analysis (EDA).
- **Simulations**: Execution of backtests including transaction costs, slippage, and risk metrics.
- **Results**: Critical discussion of outcomes, identified biases, and subsequent improvements.

### C. Paper Trading
- **Implementation**: Technical description of the live-simulation setup.
- **Comparative Analysis**: Discussion of paper trading results vs. backtested expectations.

## 5. Deliverables and Grading Criteria

### A. Python Notebook and Datasets (30%)
- **Reproducibility**: The grader must be able to run the notebook from top to bottom without code modifications.
- **Documentation**: Clean, "self-documenting" code with Markdown explanations for each cell.

| Criteria       | Excellent                                          | Partial                                          | Weak                                         | Points |
|----------------|----------------------------------------------------|--------------------------------------------------|----------------------------------------------|--------|
| Datasets       | Clear, reproducible pipeline; high data integrity. | Minor flaws in data cleaning or adjustment.      | Major flaws (e.g., look-ahead bias in data). | 5      |
| Backtesting    | Iterative process; realistic costs included.       | Non-iterative; ignores slippage or commissions.  | Major design flaws or data snooping.         | 15     |
| Paper Trading  | Seamless implementation; all metrics tracked.      | Partial metrics; minor execution errors.         | Invalid simulation or lack of execution.     | 10     |

### B. Report (30%)
- **Length**: 10–15 pages (excluding appendices).
- **Professionalism**: Must be formatted as a professional research paper.
- **Completeness**: The report must include all components listed in Topic Selection.

| Criteria      | Excellent                                  | Partial                                | Weak                                      | Points |
|---------------|--------------------------------------------|----------------------------------------|-------------------------------------------|--------|
| Strategy      | Clear explanation and justification.       | Minor inconsistencies in formulation.  | Major logical gaps in strategy.           | 5      |
| Backtesting   | Deep analytical insight into results.      | Surface-level analysis; some errors.   | Fundamental misunderstanding of concepts. | 15     |
| Paper Trading | Insightful comparison of backtest vs. live.| Descriptive only; lacks critical depth.| Incorrect interpretation of results.      | 10     |

### C. Presentation (40%)
- **Note**: A 15% penalty applies for failing to present in the assigned order.

| Criteria  | Excellent                                  | Partial                                  | Weak                                       | Points |
|-----------|--------------------------------------------|------------------------------------------|--------------------------------------------|--------|
| Structure | Professional flow; compelling visuals.     | Moderate gaps in narrative structure.    | Disorganized; excessive text on slides.    | 10     |
| Analysis  | Explicit links to course theory; data-driven. | Limited depth; some unsupported claims. | Purely opinion-based; no theoretical link. | 20     |
| Q&A       | Mastery of topic; confident defense.       | Partial understanding of technicalities. | Unable to defend methodology.              | 10     |
