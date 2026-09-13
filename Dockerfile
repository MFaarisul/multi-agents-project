FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install third-party dependencies only (cached: invalidated only when the
# lock changes, not when application source changes)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

# Copy the application source and install the project itself into the venv
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
