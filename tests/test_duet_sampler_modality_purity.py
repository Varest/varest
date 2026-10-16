"""DUET's collator requires same-modality micro-batches; the sampler must deliver them.

Upstream LLaVA ends an epoch with the image and text remainders concatenated into
one megabatch. LLaVA tolerates a mixed batch, DUET does not, so a 13B stage-2 run
died at step 5196 of 5198 with

    ValueError: DUET batched training requires same-modality micro-batches

after ~11.5h of training.

Fixing only the modality mixing was not enough. The next run reached step 5197
and hung instead: the epoch ended with a partial batch that four ranks divided
unevenly, so the collective never completed and NCCL's watchdog aborted the job.
Purity and rank-evenness are both required, so both are asserted here.
"""
import pytest

from llmoe.compressor.duet.duet_tr import ModalityLengthGroupedSampler


def micro_batches(sampler, lengths):
    """The stream as the loader cuts it: fixed-stride chunks across all ranks."""
    stride = sampler.per_device_batch_size * sampler.world_size
    order = list(iter(sampler))
    return [[lengths[i] for i in order[s:s + stride]] for s in range(0, len(order), stride)]


def shapes():
    for n_img in (0, 1, 7, 32, 33, 64, 100, 257, 1000):
        for n_txt in (0, 1, 5, 32, 61, 128, 333):
            for per_device, world, accum in ((8, 4, 4), (1, 1, 1), (4, 2, 8), (2, 8, 2)):
                yield n_img, n_txt, per_device, world, accum


@pytest.mark.parametrize("n_img,n_txt,per_device,world,accum", list(shapes()))
def test_every_micro_batch_is_single_modality(n_img, n_txt, per_device, world, accum):
    lengths = [10] * n_img + [-10] * n_txt
    if not lengths:
        pytest.skip("empty dataset is not a training case")
    sampler = ModalityLengthGroupedSampler(
        lengths, per_device_batch_size=per_device, world_size=world,
        gradient_accumulation_steps=accum, seed=0,
    )
    for batch in micro_batches(sampler, lengths):
        signs = {length > 0 for length in batch}
        assert len(signs) == 1, f"mixed micro-batch {batch}"


@pytest.mark.parametrize("n_img,n_txt,per_device,world,accum", list(shapes()))
def test_len_matches_the_stream_and_loses_almost_nothing(n_img, n_txt, per_device, world, accum):
    lengths = [10] * n_img + [-10] * n_txt
    if not lengths:
        pytest.skip("empty dataset is not a training case")
    sampler = ModalityLengthGroupedSampler(
        lengths, per_device_batch_size=per_device, world_size=world,
        gradient_accumulation_steps=accum, seed=0,
    )
    order = list(iter(sampler))
    # __len__ drives the trainer's step count, so it must equal what is emitted.
    assert len(sampler) == len(order)
    assert len(set(order)) == len(order), "an index was repeated"
    # Nothing is invented, and at most one stride of rows is given up.
    assert set(order) <= set(range(len(lengths)))
    # Worst case: one stride-remainder per modality, plus one global-batch
    # remainder for the final alignment.
    stride, glob = per_device * world, per_device * world * accum
    assert len(lengths) - len(order) < 2 * stride + glob


@pytest.mark.parametrize("n_img,n_txt,per_device,world,accum", list(shapes()))
def test_no_short_final_batch_so_ranks_stay_in_step(n_img, n_txt, per_device, world, accum):
    """A ragged epoch tail hangs the job: the ranks divide it unevenly and the
    collective never completes. Both modalities present is the case that broke."""
    lengths = [10] * n_img + [-10] * n_txt
    if not lengths:
        pytest.skip("empty dataset is not a training case")
    sampler = ModalityLengthGroupedSampler(
        lengths, per_device_batch_size=per_device, world_size=world,
        gradient_accumulation_steps=accum, seed=0,
    )
    if not (n_img and n_txt):
        pytest.skip("single-modality path is left as upstream")
    emitted = len(list(iter(sampler)))
    assert emitted % (per_device * world) == 0, "final micro-batch is short"
    assert emitted % (per_device * world * accum) == 0, "final global batch is short"
    for batch in micro_batches(sampler, lengths):
        assert len(batch) == per_device * world


def test_single_modality_datasets_lose_nothing():
    for lengths in ([10] * 100, [-10] * 100):
        sampler = ModalityLengthGroupedSampler(
            lengths, per_device_batch_size=8, world_size=4,
            gradient_accumulation_steps=4, seed=0,
        )
        assert len(sampler) == len(lengths) == len(list(iter(sampler)))


def test_the_shape_that_failed_in_the_13b_run():
    """4 GPUs x micro-batch 8, a 665k-row mix: the run that died at step 5196."""
    lengths = [10] * 624610 + [-10] * 40687
    sampler = ModalityLengthGroupedSampler(
        lengths, per_device_batch_size=8, world_size=4,
        gradient_accumulation_steps=4, seed=0,
    )
    for batch in micro_batches(sampler, lengths):
        assert len({length > 0 for length in batch}) == 1
    # The exact arithmetic of the run: 81 rows given up, 5197 whole steps.
    assert len(sampler) == 665216
    assert len(sampler) % (8 * 4 * 4) == 0
    assert len(sampler) // (8 * 4 * 4) == 5197
