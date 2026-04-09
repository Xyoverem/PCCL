"""Pipeline construct for step-level pipelining across channels.

Usage example:

    with CommunicationOp(name="pipelined_ar", device=DeviceType.CUDA) as op:
        op.tensor(dtype="bfloat16", shape=(tensor_size,))
        pipe = Pipeline(depth=2)
        with pipe.bind(op):
            for step in range(num_steps):
                with pipe.stage():
                    # These ops go to channel = step % depth,
                    # with automatic Stream scoping for overlap.
                    op.tma_reduce(...)
                    op.notify(...)

Key behaviours:
- ``stage()`` returns a context manager that calls ``set_channel(ch)`` and
  enters ``Stream(__pipe_ch{ch})``, where ``ch = iteration % depth``.
- Re-using the same Stream name per channel chains same-channel ops via
  ``_StreamState.last_node_id`` (sequential within a channel).
- Different channels have no implicit dependency (parallel across channels).
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from .decorators import Stream

if TYPE_CHECKING:
    from .decorators import CommunicationOp


class _PipelineStage:
    """Context manager for a single pipeline iteration."""

    def __init__(self, pipeline: "Pipeline"):
        self._pipeline = pipeline
        self._stream: Optional[Stream] = None

    def __enter__(self) -> "_PipelineStage":
        pipe = self._pipeline
        ch = pipe._iteration % pipe.depth
        pipe._iteration += 1

        op = pipe._comm_op
        if op is None:
            raise RuntimeError("Pipeline.stage() must be used inside Pipeline.bind()")
        op.set_channel(ch)
        self._stream = Stream(f"__pipe_ch{ch}")
        self._stream.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stream is not None:
            self._stream.__exit__(exc_type, exc_val, exc_tb)
        return False


class Pipeline:
    """Step-level pipeline: overlaps consecutive iterations across channels.

    Parameters
    ----------
    depth : int
        Number of pipeline stages (== number of channels used).
        ``depth=1`` means no pipelining (all ops on channel 0).
    """

    def __init__(self, depth: int = 2):
        if depth < 1:
            raise ValueError(f"Pipeline depth must be >= 1, got {depth}")
        self.depth = depth
        self._comm_op: Optional["CommunicationOp"] = None
        self._iteration: int = 0

    def bind(self, comm_op: "CommunicationOp") -> "_PipelineBound":
        """Bind this pipeline to a CommunicationOp.  Use as context manager."""
        return _PipelineBound(self, comm_op)

    def stage(self) -> _PipelineStage:
        """Return a context manager for one pipeline iteration.

        Channel assignment: ``channel = iteration_count % depth``.
        """
        return _PipelineStage(self)


class _PipelineBound:
    """Context manager that binds a Pipeline to a CommunicationOp."""

    def __init__(self, pipeline: Pipeline, comm_op: "CommunicationOp"):
        self._pipeline = pipeline
        self._comm_op = comm_op

    def __enter__(self) -> Pipeline:
        self._pipeline._comm_op = self._comm_op
        self._pipeline._iteration = 0
        return self._pipeline

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._pipeline._comm_op = None
        return False
