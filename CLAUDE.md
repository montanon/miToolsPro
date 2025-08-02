# miToolsPro Development Guide

## Project Overview

miToolsPro is a comprehensive Python toolkit for data analysis, visualization, and research workflows. The codebase consists of 17 specialized modules with a focus on clean architecture, strong typing, and comprehensive testing.

**Current Status:** Preparing for first release with 77% test coverage across 84 test files.

## Core Principles

### Plan & Review

#### Before Starting Work
- Always in plan mode to make a plan
- After making a plan, make sure to Write the plan to .claude/tasks/TASK_NAME.md
- The plan should be a detailed implementation plan and the reasoning behind them, as well as tasks broken down.
- If the task requires external knowledge, use the 'research' task tool to get the information.
- Don't overengineer the plan or the solution, be concise.
- Once you've written the plan, ask me to review it.
- Don't start working in the task until I approve the plan.

#### While Working
- You should always update the plan as you work.
- After you complete a task of the plan, you should update and append detailed descriptions of the changes you made, so following tasks are informed. 


### Code Quality Standards
- **SOLID & DRY**: Maintain single responsibility, avoid tight coupling, abstract duplicated code
- **Strong Typing**: All code must have comprehensive type annotations (enforced by IDE linter)
- **Self-Documenting Code**: Excellent variable/function/method/class names that clearly express purpose
- **NO Docstrings**: Code should speak for itself through clear naming
- **Modular Design**: Modules can call other module functions but must avoid tight coupling

### Testing Philosophy
- **Test After Implementation**: Only create tests after specific implementation is complete and when explicitly requested
- **Maintain Existing Tests**: When modifying code, ALWAYS update corresponding tests to maintain the robust passing test suite
- **Coverage Goals**: Target 80%+ overall coverage with near 100% coverage for core functionality
- **IDE Integration**: Use IDE's integrated test runner and coverage tools

## Architecture Overview

### Module Structure
```
mitoolspro/
├── plotting/           # Most complex module - matplotlib API compatibility
├── clustering/         # K-means, agglomerative with evaluation tools
├── regressions/        # Econometric models (OLS, Panel, IV, etc.)
├── economic_complexity/# Trade analysis and complexity metrics
├── google_utils/       # Places & YouTube API integrations
├── document/           # PDF/DOCX processing and generation
├── llms/              # OpenAI & Ollama clients with usage tracking
├── networks/          # Interactive network visualization
├── databases/         # SQLAlchemy & SQLite utilities
├── files/             # Excel, PDF, ICS file handlers
├── nlp/               # spaCy & transformers integration
├── scraping/          # Selenium-based web scraping
├── utils/             # General utilities and helper functions
└── ...
```

### Key Design Patterns
- **Abstract Base Classes**: Core functionality interfaces (e.g., `Plotter`, `LLMModel`, `TokensCounter`)
- **Mixin Pattern**: Composable behavior (`ParamsMixIn`, `SetterMixIn` in plotting)
- **Lazy Loading**: Module imports deferred until needed via `__getattr__` in `__init__.py`
- **Factory Pattern**: Dynamic object creation in plotting and modeling
- **Strategy Pattern**: Pluggable algorithms in clustering and regression

## Development Workflow

### Standard Development Cycle
1. **Write Code**: Implement functionality following SOLID/DRY principles
2. **Verify Tests**: Run existing tests to ensure no regressions
3. **Create Tests**: Only when implementation is complete and explicitly requested
4. **Integrate**: Add to module's `__init__.py` and ensure proper imports

### Code Quality Checks
- **Type Checking**: Use IDE's integrated mypy linter
- **Code Style**: Use IDE's integrated ruff linter (no specific rule enforcement)
- **Testing**: Use IDE's integrated test runner with coverage

## Module-Specific Guidelines

### Plotting Module (`mitoolspro.plotting`)
**Complexity Level: HIGH**

The plotting module maintains matplotlib API compatibility while providing type-safe, composable plotting functionality.

**Key Components:**
- `Plotter` (ABC): Base class for all plotters
- `ParamsMixIn`: Parameter validation using Pydantic models
- `SetterMixIn`: Common plotting operations
- `PlotComposer`: Multi-plot composition

**Adding New Plotters:**
1. Extend `Plotter` abstract base class
2. Implement required abstract methods: `_validate_data()`, `plot()`
3. Use mixins for shared functionality
4. Handle matplotlib's flexible data input structures
5. Add comprehensive type annotations
6. Update `plotting/__init__.py` imports

**Important:** The matplotlib API allows various data structures and parameters - maintain this flexibility while ensuring type safety.

### Other Critical Modules

**Regression Models (`mitoolspro.regressions`):**
- Extend base model classes for new econometric models
- Maintain diagnostic tool compatibility
- Follow wrapper pattern for complex workflows

**LLM Integration (`mitoolspro.llms`):**
- Implement `LLMModel` ABC for new providers
- Use `TokensCounter` for usage tracking
- Handle API credentials securely

**Economic Complexity (`mitoolspro.economic_complexity`):**
- Trade data analysis with matrix operations
- Maintain compatibility with standard economic metrics

## Testing Guidelines

### Current Test Status
- **Coverage**: 77% across 84 test files
- **Quality**: Robust, passing test suite
- **Framework**: Standard Python unittest

### When to Create Tests
- **After Implementation**: Only create tests when code is complete
- **When Explicitly Requested**: Don't proactively suggest test creation
- **Before Module Integration**: Ensure functionality works before adding to `__init__.py`

### Maintaining Tests
- **Critical**: Always update tests when modifying underlying code
- **Coverage Goal**: 80%+ overall, near 100% for core functionality
- **Test Structure**: Mirror module structure in `tests/` directory

## Common Development Tasks

### Adding New Modules
1. Create module directory with `__init__.py`
2. Implement core functionality with strong typing
3. Create corresponding test directory
4. Add to main `mitoolspro/__init__.py` lazy loading
5. Ensure loose coupling with other modules

### Extending Existing Modules
1. Follow existing patterns and inheritance hierarchies
2. Update tests for modified functionality
3. Maintain backward compatibility
4. Abstract common code to utils if generally applicable

### Code Organization
- **Utils Module**: Use for general-purpose functions used across modules
- **Avoid Duplication**: Abstract repeated code into reusable functions
- **Loose Coupling**: Modules can call other modules but avoid tight dependencies

## Error Handling

The codebase includes 83 custom exception classes with clear inheritance hierarchy:
- Domain-specific exceptions (e.g., `WebScraperTimeoutError`, `ProjectVersionError`)
- Proper exception propagation and handling
- Graceful degradation where appropriate

## Performance Considerations

### Current Optimizations
- **Lazy Loading**: Modules loaded on-demand
- **Parallel Processing**: `@parallel` decorator for batch operations
- **Memory Efficiency**: Optimized data structures using NumPy/Pandas
- **Caching**: Intelligent memoization for expensive operations

### When Adding New Features
- Consider memory usage for large datasets
- Use vectorized operations with NumPy/Pandas when possible
- Implement parallel processing for CPU-intensive tasks

## Pre-Release Checklist

### Code Quality
- [ ] All new code has comprehensive type annotations
- [ ] No tight coupling between modules
- [ ] Common functionality abstracted appropriately
- [ ] Variable/function/class names clearly express purpose

### Testing
- [ ] All existing tests pass
- [ ] New functionality has corresponding tests
- [ ] Test coverage ≥ 80% overall
- [ ] Core functionality has near 100% coverage

### Integration
- [ ] New modules properly integrated into `__init__.py`
- [ ] Lazy loading works correctly
- [ ] No circular import issues
- [ ] Module interdependencies are clean

## AI Assistant Guidelines

### Code Development
- **No Docstrings**: Do not add docstrings unless explicitly requested
- **Excellent Naming**: Prioritize clear, self-documenting names
- **Type Safety**: Always include comprehensive type annotations
- **Test Maintenance**: When modifying code, always update corresponding tests
- **Architecture Respect**: Follow existing patterns and inheritance hierarchies

### Testing Approach
- **Test After Code**: Only create tests when implementation is complete
- **Maintain Coverage**: Ensure existing tests continue to pass
- **No Proactive Testing**: Don't suggest test creation unless explicitly requested

### Module Integration
- **Follow Patterns**: Use existing module structure and patterns
- **Loose Coupling**: Ensure modules can interact without tight dependencies
- **Utils Usage**: Place general-purpose code in utils module
- **Import Management**: Update `__init__.py` files appropriately

Remember: This codebase has a robust, passing test suite with good coverage. The goal is to maintain this quality while preparing for the first release.