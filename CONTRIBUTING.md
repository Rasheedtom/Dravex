# Contributing to Dravex

## Development Setup

1. Clone the repository
2. Run tests: `make test`
3. Check coverage: `make coverage`

## Code Standards

- Pure Python 3.12, zero runtime dependencies
- Every module must have a docstring
- All public functions require type-appropriate documentation
- Tests required for new features (minimum 80% branch coverage)

## Pull Request Process

1. Create a feature branch from `main`
2. Add tests for new functionality
3. Ensure all tests pass with `make test`
4. Update CHANGELOG.md with your changes
5. Submit PR with clear description

## React workbench

The browser workbench uses Node.js 20 and npm. Before submitting a workbench change, run:

```bash
npm ci
npm test
npm run lint
npm run check
npm run build
```

Keep React changes focused. Hooks must remain unconditional, effects must clean up listeners/timers, controls need accessible labels and keyboard behavior, and derived analysis values should not be duplicated into independent state.
