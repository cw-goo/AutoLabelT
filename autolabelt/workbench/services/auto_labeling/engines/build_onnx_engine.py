import logging
import os
from functools import lru_cache

import onnx
import onnxruntime as ort

from autolabelt.workbench.views.common.device_manager import get_preferred_device


logger = logging.getLogger(__name__)
CUDA_PROVIDER = "CUDAExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"


@lru_cache(maxsize=1)
def _preload_cuda_libraries():
    preload = getattr(ort, "preload_dlls", None)
    if preload is not None:
        preload()


def create_onnx_session(
    model_path, device_type="auto", *, sess_options=None, cuda_options=None
):
    device = device_type.lower()
    if device == "auto":
        device = get_preferred_device().lower()

    providers = [CPU_PROVIDER]
    prefer_cuda = device in {"gpu", "cuda"}
    if prefer_cuda and CUDA_PROVIDER in ort.get_available_providers():
        _preload_cuda_libraries()
        providers.insert(0, (CUDA_PROVIDER, cuda_options) if cuda_options else CUDA_PROVIDER)

    session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)
    active_providers = session.get_providers()
    if prefer_cuda and CUDA_PROVIDER not in active_providers:
        logger.warning("ONNX CUDA unavailable for %s. Using %s", model_path, active_providers)
    else:
        logger.debug("ONNX providers for %s: %s", model_path, active_providers)
    return session


class OnnxBaseModel:
    def __init__(
        self, model_path, device_type: str = "auto", log_severity_level: int = 3
    ):
        self.sess_opts = ort.SessionOptions()
        self.sess_opts.log_severity_level = log_severity_level
        if "OMP_NUM_THREADS" in os.environ:
            self.sess_opts.inter_op_num_threads = int(
                os.environ["OMP_NUM_THREADS"]
            )

        self.ort_session = create_onnx_session(
            model_path, device_type, sess_options=self.sess_opts
        )
        self.providers = self.ort_session.get_providers()
        self.model_path = model_path

    def get_ort_inference(
        self, blob=None, inputs=None, extract=True, squeeze=False
    ):
        if inputs is None:
            inputs = self.get_input_name()
            outs = self.ort_session.run(None, {inputs: blob})
        else:
            outs = self.ort_session.run(None, inputs)
        if extract:
            outs = outs[0]
        if squeeze:
            outs = outs.squeeze(axis=0)
        return outs

    def get_input_name(self):
        return self.ort_session.get_inputs()[0].name

    def get_input_shape(self):
        return self.ort_session.get_inputs()[0].shape

    def get_output_name(self):
        return [out.name for out in self.ort_session.get_outputs()]

    def get_metadata_info(self, field):
        model = onnx.load(self.model_path)
        metadata = model.metadata_props
        for prop in metadata:
            if prop.key == field:
                return prop.value
        return None
