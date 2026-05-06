"""
Multi-GPU sampling pool for single-process training: dispatch samples across the train GPU
and optional extra CUDA devices using a shared job queue (work-stealing).
"""

from __future__ import annotations

import queue
import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass

from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.util import create
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.AudioFormat import AudioFormat
from modules.util.enum.ImageFormat import ImageFormat
from modules.util.enum.VideoFormat import VideoFormat
from modules.util.sampler_only_lora import SamplerOnlyLoRABatchManager, build_sampler_lora_reuse_key
from modules.util.torch_util import torch_gc

import torch

_SENTINEL = object()


def set_sampler_pool_tqdm_context(worker_idx: int | None, device_index: int | None) -> None:
    """Thread-local context for per-worker tqdm bars (set by SamplerPool workers)."""
    tls = threading.current_thread()
    tls._sampler_tqdm_worker_idx = worker_idx  # noqa: SLF001
    tls._sampler_tqdm_device_index = device_index  # noqa: SLF001


@dataclass
class SamplerJob:
    index: int
    sample_config: SampleConfig
    destination: str
    image_format: ImageFormat
    video_format: VideoFormat
    audio_format: AudioFormat
    on_sample: Callable[[ModelSamplerOutput], None]
    on_job_complete: Callable[[], None]
    batch_marker: str


def parse_extra_sample_cuda_devices(config: TrainConfig, train_device: torch.device) -> list[torch.device]:
    """Extra CUDA devices only; train device is always included separately."""
    if config.multi_gpu:
        return []
    raw = (config.sample_device_indexes or "").strip()
    if not raw:
        return []
    if train_device.type != "cuda":
        print("Sampler pool: train device is not CUDA; multi-GPU sampling disabled.")
        return []

    train_idx = train_device.index if train_device.index is not None else 0
    seen: set[int] = {train_idx}
    extras: list[torch.device] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            idx = int(part)
        except ValueError:
            print(f"Sampler pool: ignoring invalid device index '{part}'")
            continue
        if idx < 0 or idx >= torch.cuda.device_count():
            print(f"Sampler pool: CUDA device {idx} not available (have {torch.cuda.device_count()}); skipping.")
            continue
        if idx in seen:
            continue
        seen.add(idx)
        extras.append(torch.device("cuda", idx))
    return extras


def _copy_stateful(src_obj, dst_obj) -> None:
    if src_obj is None or dst_obj is None:
        return
    if not hasattr(src_obj, "state_dict") or not hasattr(dst_obj, "load_state_dict"):
        return
    try:
        src_sd = src_obj.state_dict()
        dst_obj.load_state_dict(src_sd, strict=False)
    except TypeError:
        # Some wrappers use a custom signature without `strict`.
        dst_obj.load_state_dict(src_sd)


def _copy_embedding_tensors(trainer_model: BaseModel, shadow_model: BaseModel) -> None:
    if not hasattr(trainer_model, "all_embeddings") or not hasattr(shadow_model, "all_embeddings"):
        return
    src_list = trainer_model.all_embeddings()
    dst_list = shadow_model.all_embeddings()

    def collect_base_embeddings(obj) -> dict[str, object]:
        out: dict[str, object] = {}
        if obj is None:
            return out
        if hasattr(obj, "uuid") and (hasattr(obj, "vector") or hasattr(obj, "output_vector")):
            out[getattr(obj, "uuid")] = obj
            return out
        for value in vars(obj).values():
            if hasattr(value, "uuid") and (hasattr(value, "vector") or hasattr(value, "output_vector")):
                out[getattr(value, "uuid")] = value
        return out

    src_map: dict[str, object] = {}
    for src_obj in src_list:
        src_map.update(collect_base_embeddings(src_obj))

    for dst_obj in dst_list:
        for uuid, dst_base in collect_base_embeddings(dst_obj).items():
            src_base = src_map.get(uuid)
            if src_base is None:
                continue
            for attr in ("vector", "output_vector"):
                src_t = getattr(src_base, attr, None)
                dst_t = getattr(dst_base, attr, None)
                if src_t is None or dst_t is None:
                    continue
                if not hasattr(src_t, "shape") or src_t.shape != dst_t.shape:
                    continue
                dst_t.copy_(src_t.to(device=dst_t.device, dtype=dst_t.dtype, non_blocking=False))


def _sync_state_dict_from_trainer(trainer_model: BaseModel, shadow_model: BaseModel) -> None:
    """Copy train-updated weights from trainer model objects into a shadow model."""
    with torch.no_grad():
        # Core modules commonly present across model families.
        for module_name in (
                "transformer", "unet", "vae",
                "text_encoder", "text_encoder_1", "text_encoder_2", "text_encoder_3", "text_encoder_4",
                "depth_estimator",
        ):
            _copy_stateful(getattr(trainer_model, module_name, None), getattr(shadow_model, module_name, None))

        # LoRA adapter wrappers are also stateful and need synchronization.
        for adapter_name in (
                "text_encoder_lora", "text_encoder_1_lora", "text_encoder_2_lora",
                "text_encoder_3_lora", "text_encoder_4_lora",
                "transformer_lora", "unet_lora",
        ):
            _copy_stateful(getattr(trainer_model, adapter_name, None), getattr(shadow_model, adapter_name, None))

        # Embedding training vectors are not always covered by module state dicts.
        _copy_embedding_tensors(trainer_model, shadow_model)


def _load_shadow_model(
        config: TrainConfig,
        model_loader: BaseModelLoader,
) -> BaseModel:
    """Full setup path on CPU to avoid VRAM spike on extra GPUs during init."""
    model_names = config.model_names()
    model = model_loader.load(
        model_type=config.model_type,
        model_names=model_names,
        weight_dtypes=config.weight_dtypes(),
        quantization=config.quantization,
    )
    model.train_config = config
    cpu = torch.device("cpu")
    setup = create.create_model_setup(
        config.model_type,
        cpu,
        cpu,
        config.training_method,
        config.debug_mode,
    )
    setup.setup_optimizations(model, config)
    setup.setup_train_device(model, config)
    setup.setup_model(model, config)
    model.to(torch.device(config.temp_device))
    model.eval()
    torch_gc()
    return model


def _run_one_sample(
        config: TrainConfig,
        temp_device: torch.device,
        train_device: torch.device,
        model: BaseModel,
        sampler: BaseModelSampler,
        lora_mgr: SamplerOnlyLoRABatchManager,
        job: SamplerJob,
) -> None:
    reuse_key = build_sampler_lora_reuse_key(
        config,
        job.sample_config,
        train_device,
        batch_marker=job.batch_marker,
    )
    lora_mgr.acquire(
        model,
        config,
        job.sample_config,
        train_device,
        batch_key=reuse_key,
    )
    model.to(temp_device)
    model.eval()
    sampler.sample(
        sample_config=job.sample_config,
        destination=job.destination,
        image_format=job.image_format,
        video_format=job.video_format,
        audio_format=job.audio_format,
        on_sample=job.on_sample,
        on_update_progress=lambda _s, _t: None,
    )


class _WorkerState:
    __slots__ = ("device", "model", "sampler", "is_primary", "worker_idx")

    def __init__(
            self,
            device: torch.device,
            model: BaseModel,
            sampler: BaseModelSampler,
            is_primary: bool,
            worker_idx: int,
    ):
        self.device = device
        self.model = model
        self.sampler = sampler
        self.is_primary = is_primary
        self.worker_idx = worker_idx


class SamplerPool:
    """
    When ``sample_device_indexes`` is empty or invalid, operates in inline mode on the main thread
    (original behavior). Otherwise spawns one thread per GPU that pulls jobs from a shared queue.
    """

    def __init__(
            self,
            config: TrainConfig,
            train_device: torch.device,
            temp_device: torch.device,
            primary_model: BaseModel,
            primary_sampler: BaseModelSampler,
            model_loader: BaseModelLoader,
    ):
        self.config = config
        self.train_device = train_device
        self.temp_device = temp_device
        self.primary_model = primary_model
        self.primary_sampler = primary_sampler
        self.model_loader = model_loader

        self._inline_only = True
        self._job_queue: queue.Queue | None = None
        self._workers: list[_WorkerState] = []
        self._threads: list[threading.Thread] = []
        self._shutdown = False
        self._inline_pending: list[SamplerJob] = []

        extras = parse_extra_sample_cuda_devices(config, train_device)
        if not extras:
            return

        self._inline_only = False
        self._job_queue = queue.Queue()
        self._workers.append(
            _WorkerState(train_device, primary_model, primary_sampler, True, 0),
        )

        worker_idx = 1
        for dev in extras:
            try:
                shadow = _load_shadow_model(config, model_loader)
                samp = create.create_model_sampler(
                    dev,
                    temp_device,
                    shadow,
                    config.model_type,
                    config.training_method,
                )
                self._workers.append(_WorkerState(dev, shadow, samp, False, worker_idx))
                worker_idx += 1
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
                    print(f"Sampler pool: OOM loading shadow model for {dev}; skipping that device. ({e})")
                else:
                    print(f"Sampler pool: failed to load shadow model for {dev}: {e}")
                    traceback.print_exc()
            except Exception as e:
                print(f"Sampler pool: failed to load shadow model for {dev}: {e}")
                traceback.print_exc()

        # Drop extras that failed: keep at least primary
        self._workers = [self._workers[0]] + [w for w in self._workers[1:] if w is not None]

        if len(self._workers) <= 1:
            self._inline_only = True
            self._job_queue = None
            self._workers = []
            print("Sampler pool: no extra sampling GPUs available; using inline sampling only.")
            return

        for ws in self._workers:
            t = threading.Thread(
                target=_sampler_worker_main,
                args=(self, ws),
                name=f"SamplerPool-{ws.device}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def sync_weights_from(self, trainer_model: BaseModel) -> None:
        if self._inline_only:
            return
        for ws in self._workers:
            if ws.is_primary:
                continue
            _sync_state_dict_from_trainer(trainer_model, ws.model)

    def submit(self, job: SamplerJob) -> None:
        if self._inline_only:
            self._inline_pending.append(job)
            return
        assert self._job_queue is not None
        self._job_queue.put(job)

    def wait_all(self) -> None:
        if self._inline_only:
            mgr = SamplerOnlyLoRABatchManager()
            try:
                td_idx = self.train_device.index if self.train_device.type == "cuda" else None
                if td_idx is None and self.train_device.type == "cuda":
                    td_idx = torch.cuda.current_device()
                set_sampler_pool_tqdm_context(0, td_idx)
                for job in self._inline_pending:
                    try:
                        _run_one_sample(
                            self.config,
                            self.temp_device,
                            self.train_device,
                            self.primary_model,
                            self.primary_sampler,
                            mgr,
                            job,
                        )
                    except Exception:
                        traceback.print_exc()
                        print("Error during sampling, proceeding without sampling")
                    finally:
                        job.on_job_complete()
                        torch_gc()
            finally:
                set_sampler_pool_tqdm_context(None, None)
                mgr.close()
                self._inline_pending.clear()
            return

        assert self._job_queue is not None
        self._job_queue.join()

    def shutdown(self) -> None:
        if self._inline_only or self._shutdown or self._job_queue is None:
            return
        self._shutdown = True
        for _ in self._workers:
            self._job_queue.put(_SENTINEL)
        self._job_queue.join()
        for t in self._threads:
            t.join(timeout=120.0)
        self._threads.clear()
        self._job_queue = None


def _sampler_worker_main(pool: SamplerPool, state: _WorkerState) -> None:
    dev_idx = None
    if state.device.type == "cuda":
        idx = state.device.index
        if idx is not None:
            torch.cuda.set_device(idx)
            dev_idx = idx
        else:
            dev_idx = torch.cuda.current_device()

    lora_mgr = SamplerOnlyLoRABatchManager()
    assert pool._job_queue is not None
    jq = pool._job_queue
    try:
        while True:
            job = jq.get()
            if job is _SENTINEL:
                jq.task_done()
                break
            set_sampler_pool_tqdm_context(state.worker_idx, dev_idx)
            try:
                _run_one_sample(
                    pool.config,
                    pool.temp_device,
                    state.device,
                    state.model,
                    state.sampler,
                    lora_mgr,
                    job,
                )
            except Exception:
                traceback.print_exc()
                print("Error during sampling, proceeding without sampling")
            finally:
                job.on_job_complete()
                torch_gc()
                set_sampler_pool_tqdm_context(None, None)
                jq.task_done()
    finally:
        lora_mgr.close()
