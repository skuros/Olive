# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import TYPE_CHECKING, Any, Union

import importlib.util
import logging
from pathlib import Path

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

    def run_pass(
        self,
        the_pass: "Pass",
        model_config: ModelConfig,
        output_model_path: str,
    ) -> ModelConfig:
        """Run the pass on the model."""
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

    def _maybe_prepare_model(self, model_config: ModelConfig) -> None:
        """Optionally run input_model.prepare_model hook before model creation.

        Expected structure in the model config:

            "input_model": {
                "type": "OnnxModel",
                "model_path": "model/sam2_encoder.onnx",
                "prepare_model": {
                    "user_script": "sam2_1_hiera_small.py",
                    "fn": "export_sam2_to_onnx",
                    "kwargs": {
                        "model_id": "facebook/sam2.1-hiera-small",
                        "onnx_path": "model/sam2_encoder.onnx",
                        "opset": 17
                    }
                }
            }

        When present, this hook will import the specified user_script, locate the
        named function, and invoke it with the given kwargs before the model
        handler is constructed, so that artifacts such as ONNX files can be
        materialized lazily.
        """

        logger = logging.getLogger(__name__)

        prepare = getattr(model_config, "config", {}).get("prepare_model")
        if not prepare:
            return

        script_name = prepare.get("user_script")
        fn_name = prepare.get("fn")
        kwargs = prepare.get("kwargs") or {}

        if not script_name or not fn_name:
            raise ValueError("input_model.prepare_model requires 'user_script' and 'fn' fields.")

        script_path = Path(script_name)
        if not script_path.is_file():
            script_path = Path.cwd() / script_name
        if not script_path.is_file():
            raise FileNotFoundError(f"prepare_model.user_script '{script_name}' not found at {script_path}")

        logger.info("Running input_model.prepare_model: %s:%s", script_path, fn_name)

        spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        fn = getattr(module, fn_name, None)
        if fn is None or not callable(fn):
            raise ValueError(f"Function '{fn_name}' not found or not callable in '{script_name}'.")

        fn(**kwargs)
