# OpenSymbolicAI Examples

Example projects demonstrating OpenSymbolicAI usage.

## Structure

This repository uses [uv workspaces](https://docs.astral.sh/uv/concepts/workspaces/) - each example is an independent project with its own dependencies.

```
examples-py/
├── pyproject.toml              # Workspace root + shared dev tools
├── examples/
│   ├── date_agent/             # Each example is a workspace member
│   │   ├── pyproject.toml      # Example-specific dependencies
│   │   ├── README.md
│   │   └── date_agent/         # Package directory
│   ├── another_example/
│   │   └── ...
```

## Getting Started

### Run a specific example

```bash
cd examples/date_agent
uv sync
uv run python -m date_agent.main
```

### Development setup

```bash
# Install all workspace dependencies
uv sync --all-packages

# Install pre-commit hooks
uv run pre-commit install

# Run linting
uv run ruff check .

# Run type checking
uv run mypy examples/

# Run tests
uv run pytest
```

## Adding a New Example

1. Create a new directory under `examples/`:
   ```bash
   mkdir -p examples/my_example/my_example
   ```

2. Add a `pyproject.toml`:
   ```toml
   [project]
   name = "my-example"
   version = "0.1.0"
   description = "Description of my example"
   requires-python = ">=3.12"
   dependencies = [
       "opensymbolicai-core",
       # add other dependencies here
   ]

   [build-system]
   requires = ["hatchling"]
   build-backend = "hatchling.build"
   ```

3. Add your code in `my_example/`

4. Add a `README.md` with usage instructions
