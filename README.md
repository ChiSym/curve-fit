# SIR Curve Fit

## Set Up Python/Jupyter Notebook

- install poetry (homebrew or pipx)
- authenticate with gcloud
- add the Artifact Repository keyring
- install

```bash
poetry self add keyrings.google-artifactregistry-auth
poetry install --no-root
```

The `poetry.toml` file directs poetry to create a virtual environment
for you in the file `.venv` in the project root.

- In VS Code, use the command `Python: Select Interpreter` to choose
  the project's `.venv` directory (This _should_ be the **Recommended**
  choice.)

You should now be able to open the `blocks.ipynb` notebook and
use the "Run All" command (you may be asked again for the Python
environment to use; again, the `.venv` Poetry installation should
be the **Recommended** choice.)

## Set up the Curve Fit Single Page App

```bash
npm i
npm run dev
```
