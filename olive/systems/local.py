# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from olive.common.config_utils import validate_config
from olive.common.ort_inference import get_ort_available_providers, maybe_register_ep_libraries
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import ModelConfig
from olive.systems.common import AcceleratorConfig, SystemType
from olive.systems.olive_system import OliveSystem

if TYPE_CHECKING:
    from olive.evaluator.metric_result import MetricResult
    from olive.evaluator.olive_evaluator import OliveEvaluator, OliveEvaluatorConfig
    from olive.passes.olive_pass import Pass


logger = logging.getLogger(__name__)


class LocalSystem(OliveSystem):
    system_type = SystemType.Local

    def __init__(
        self,
        accelerators: Union[list[AcceleratorConfig], list[dict[str, Any]]] = None,
        hf_token: bool = None,
    ):
        super().__init__(accelerators, hf_token)

        if accelerators:
            accelerators = [validate_config(accelerator, AcceleratorConfig) for accelerator in accelerators]

            maybe_register_ep_libraries(
                {name: path for accelerator in accelerators for name, path in accelerator.get_ep_path_map().items()}
            )

    def _maybe_prepare_model(self, model_config: ModelConfig) -> None:
        """Optionally run a user-provided prepare_model hook before creating the model."""
        logger.info("prepare_model hook: %s", model_config.config.get("prepare_model"))
        cfg = model_config.config or {}
        prepare = cfg.get("prepare_model")
        if not prepare:
            return

        user_script = prepare.get("user_script")
        fn_name = prepare.get("fn")
        kwargs = prepare.get("kwargs", {}) or {}

        if not user_script or not fn_name:
            logger.warning("prepare_model is missing 'user_script' or 'fn'; skipping.")
            return

        script_path = Path(user_script)
        if not script_path.is_file():
            script_path = Path.cwd() / user_script

        if not script_path.is_file():
            logger.error("prepare_model user_script '%s' not found; skipping.", user_script)
            return

        spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
        if spec is None or spec.loader is None:
            logger.error("Could not load prepare_model script '%s'; skipping.", script_path)
            return

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[call-arg]

        fn = getattr(module, fn_name, None)
        if fn is None:
            logger.error("prepare_model function '%s' not found in '%s'; skipping.", fn_name, script_path)
            return

        logger.info("Running prepare_model function '%s' from '%s' with kwargs=%s", fn_name, script_path, kwargs)
        fn(**kwargs)

    def run_pass(
        self,
        the_pass: "Pass",
        model_config: ModelConfig,
        output_model_path: str,
    ) -> ModelConfig:
        """Run the pass on the model."""
        logger.info("LocalSystem.run_pass called; model_config=%s", model_config.config)
        self._maybe_prepare_model(model_config)
        model = model_config.create_model()
        output_model = the_pass.run(model, output_model_path)
        return ModelConfig.from_json(output_model.to_json())

    def evaluate_model(
        self, model_config: ModelConfig, evaluator_config: "OliveEvaluatorConfig", accelerator: AcceleratorSpec
    ) -> "MetricResult":
        """Evaluate the model."""
        if model_config.type.lower() == "compositemodel":
            raise NotImplementedError

        device = accelerator.accelerator_type if accelerator else Device.CPU
        execution_providers = accelerator.execution_provider if accelerator else None

        self._maybe_prepare_model(model_config)
        model = model_config.create_model()
        evaluator: OliveEvaluator = evaluator_config.create_evaluator(model)
        return evaluator.evaluate(
            model, evaluator_config.metrics, device=device, execution_providers=execution_providers
        )

    def get_supported_execution_providers(self) -> list[str]:
        """Get the available execution providers."""
        return get_ort_available_providers()
