.PHONY: help smoke

help:
	@echo "anemll-flash-mlx staging repo"
	@echo "Targets:"
	@echo "  make smoke   Run syntax/import smoke checks"

smoke:
	@./ci/smoke.sh
