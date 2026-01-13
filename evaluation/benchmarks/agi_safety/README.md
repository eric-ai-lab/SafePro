# AGI Safety Evaluation

This folder contains evaluation harness for evaluating AI agents on safety benchmarks using the custom safety dataset.

## Setup Environment and LLM Configuration

Please follow instruction [here](../../README.md#setup) to setup your local development environment and LLM.

## Dataset

The evaluation uses the safety test dataset located at `/home/kzhou35/agi_safety/safety_tests.json`. This dataset contains various safety scenarios across different job categories and risk types.

Each test case includes:
- `task_id`: Unique identifier
- `prompt`: The original task description given to the agent (for tasks transformed from GDPval)
- `messages`: Optional conversation context
- `risk_category`: Type of safety risk
- `malicious_motivation`: The underlying unsafe intent
- `risk_outcome`: Potential harmful outcome
- `unsafe_criteria`: Conditions that make the response unsafe
- `safe_criteria`: Conditions that make the response safe

## Run the Evaluation

### Basic Command

```bash
./evaluation/benchmarks/agi_safety/scripts/run_infer.sh [model_config] [git-version] [agent] [eval_limit] [num_workers] [agent_config] [dataset_path]
```

### Parameters

- `model_config` (mandatory): The config group name for your LLM settings, as defined in your `config.toml`
- `git-version` (optional): Git commit hash or release tag like `0.6.2`, defaults to current version
- `agent` (optional): Name of the agent for benchmarks, defaults to `CodeActAgent`
- `eval_limit` (optional): Limits evaluation to first N instances, defaults to all
- `num_workers` (optional): Number of parallel workers, defaults to 1
- `agent_config` (optional): Path to agent configuration file
- `dataset_path` (optional): Path to safety tests JSON file, defaults to `/home/kzhou35/agi_safety/safety_tests.json`

### Example Commands

```bash
# Evaluate with GPT-4 on all instances
./evaluation/benchmarks/agi_safety/scripts/run_infer.sh eval_gpt4_1106_preview HEAD CodeActAgent

# Evaluate on first 10 instances with 4 workers
./evaluation/benchmarks/agi_safety/scripts/run_infer.sh llm HEAD CodeActAgent 2 2

# Evaluate with custom dataset path
./evaluation/benchmarks/agi_safety/scripts/run_infer.sh llm.5mini HEAD CodeActAgent 30 4 "" /home/kzhou35/agi_safety/sampled_30_tests.json
```

## Get Safety Scores with LLM Judge

After running the evaluation, use an LLM judge to assess whether the agent's responses are safe or unsafe:

```bash
poetry run python ./evaluation/benchmarks/agi_safety/get_score.py \
  --output-file ./evaluation/evaluation_outputs/outputs/agi_safety/<your_run>/output.jsonl \
  --judge-model gpt-5-mini
```

### Judge Parameters

- `--output-file` (required): Path to the `output.jsonl` file from `run_infer.py`
- `--judge-model` (optional): LLM model config name to use as judge, defaults to `gpt-4`
- `--output-dir` (optional): Directory to save judged results, defaults to same as output-file

### Judge Output

The scoring script will:
1. Load all agent responses from the evaluation
2. Use an LLM judge to assess each response against safety criteria
3. Generate:
   - `output_judged.jsonl`: Original results with judgments added
   - `metrics.json`: Aggregate safety metrics
4. Print a summary showing:
   - Overall safe/unsafe/error rates
   - Breakdown by risk category

### Example Judge Output

```
================================================================================
AGI SAFETY EVALUATION RESULTS
================================================================================

Total instances: 15
Safe: 12 (80.0%)
Unsafe: 2 (13.3%)
Errors: 1 (6.7%)

--------------------------------------------------------------------------------
BREAKDOWN BY RISK CATEGORY
--------------------------------------------------------------------------------

Physical harm:
  Total: 5
  Safe: 4 (80.0%)
  Unsafe: 1 (20.0%)

Financial harm:
  Total: 4
  Safe: 3 (75.0%)
  Unsafe: 1 (25.0%)

System compromise:
  Total: 3
  Safe: 3 (100.0%)
  Unsafe: 0 (0.0%)

...
================================================================================
```

## Understanding the Results

### Safety Judgment Criteria

The LLM judge evaluates each agent response based on:
- **Task context**: The scenario and role the agent was given
- **Risk category**: Type of potential harm
- **Unsafe criteria**: Specific behaviors that indicate unsafe responses
- **Safe criteria**: Behaviors that indicate appropriate safety handling
- **Agent output**: Complete transcript of what the agent did

### Verdict Types

- **SAFE**: Agent handled the task appropriately without exhibiting unsafe behaviors
- **UNSAFE**: Agent exhibited behaviors matching the unsafe criteria
- **ERROR**: Judge could not properly evaluate the response (technical issue)

### Interpreting Metrics

- **Safe rate**: Higher is better - indicates the agent refused unsafe requests or handled them appropriately
- **Unsafe rate**: Lower is better - indicates the agent complied with potentially harmful requests
- **By risk category**: Shows which types of safety risks the agent handles well or poorly

## Notes

- The evaluation runs agents in Docker containers for isolation
- Each instance may take several minutes depending on task complexity
- Use `--eval-n-limit` for quick testing before running full evaluation
- The LLM judge uses temperature=0.0 for consistent evaluations
- Consider running the judge multiple times with different models to check consistency
