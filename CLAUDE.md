# Claude Instructions

@README.md
@DESIGN.md
@CONTRIBUTING.md
Before trying to check anything in, run all pre-commit linters: `pre-commit run --all-files`
When running tests, prefer to run all tests and ALWAYS use `pytest` not `python -m pytest`
The project folder is very small. Instead of looking around the structure, just run: `tree -a -I '.git'`
ALWAYS use pyright-compatible types in function arguments, including in tests.
Don't use Any unless it's strictly necessary (like **kwargs: Any).
Keep functions focused and well documented.
When writing new features, consider if they can go in a new file, or if existing files should be refactored.
Only add comments when code is non-obvious
Don't catch exceptions unless you plan to handle them meaningfully
Use `raise NewException(...) from e` instead of logging and re-raising

- When pyright complains about PyTorch Dataset not being Sized, use `cast(Sized, dataset)` from typing
- Prefer Optional[x] over x | None
- Don't do imports inside functions, always import at the top of the file
- Use uv instead of pip
