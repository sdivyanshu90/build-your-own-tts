.PHONY: install lint docs test fixture serve docker
install:
	python3 -m pip install -e '.[dev]'
lint:
	ruff check . && ruff format --check . && mypy
docs:
	python3 scripts/validate_docs.py
test:
	pytest --cov=tts_pipeline --cov-report=term-missing
fixture:
	python3 scripts/create_tiny_fixture.py --output data/tiny
serve:
	tts serve --config configs/development.yaml
docker:
	docker compose build
