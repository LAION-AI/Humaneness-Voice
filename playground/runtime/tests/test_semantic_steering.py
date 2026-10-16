"""CPU point-aware semantic instrumentation; no live service or model required."""
import pytest
import torch

from semantic_steering import SemanticVectorTransform, mean_direction


def projection():
    bridge = torch.nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        bridge.weight.copy_(torch.tensor([[1., 0.], [0., 2.], [1., 1.]]))
    return bridge


@pytest.mark.parametrize('alpha', [0., .25, .5, .75, 1.])
def test_projected_addition_reaches_decoder_and_all_cfg_branches(alpha):
    bridge = projection()
    global_values = torch.tensor([[1., 2.], [3., 4.]])
    direction = torch.tensor([1., -2., 3.])
    transform = SemanticVectorTransform(direction, alpha=alpha)
    transform.set_batch(1, 2)
    assert transform('semantic_global', global_values) is global_values
    original = bridge(global_values)
    steered = transform('semantic_local', original)
    assert torch.allclose(steered, original + alpha * direction)
    if alpha == 0:
        assert steered is original
    else:
        # A fake acoustic head demonstrates causal influence, not just logging.
        head = torch.ones(3)
        assert not torch.equal(original @ head, steered @ head)
    assert torch.allclose(steered[0] - steered[1], original[0] - original[1])


def test_global_addition_is_before_bridge_not_wrong_width_local_addition():
    bridge = projection()
    values = torch.tensor([[1., 2.], [3., 4.]])
    direction = torch.tensor([.5, -1.])
    transform = SemanticVectorTransform(direction, point='semantic_global', alpha=.5)
    transform.set_batch(1, 2)
    steered_global = transform('semantic_global', values)
    steered_local = bridge(steered_global)
    assert transform('semantic_local', steered_local) is steered_local
    assert torch.allclose(steered_local, bridge(values + .5 * direction))
    assert not torch.equal(steered_local, bridge(values))


def test_rms_conditional_scale_shared_with_cfg_and_finished_rows_unchanged():
    values = torch.tensor([[1., 2., 3.], [4., 5., 6.], [30., 40., 50.], [60., 70., 80.]])
    transform = SemanticVectorTransform(torch.ones(3), alpha=.25, normalization='rms_relative')
    transform.set_batch(2, 2)
    transform.set_active(torch.tensor([True, False]))
    steered = transform('semantic_local', values)
    delta = steered - values
    assert torch.allclose(delta[0], delta[2], atol=1e-5)
    assert torch.equal(steered[[1, 3]], values[[1, 3]])
    assert torch.allclose(delta[0].square().mean().sqrt(),
                          .25 * values[0].square().mean().sqrt())


def test_noop_is_bit_exact_bfloat16_and_consumes_no_randomness():
    original = torch.randn(4, 3).bfloat16()
    state = torch.random.get_rng_state().clone()
    transform = SemanticVectorTransform(torch.ones(3), alpha=0)
    transform.set_batch(2, 2)
    assert transform('semantic_local', original) is original
    assert torch.equal(state, torch.random.get_rng_state())


def test_no_collection_or_hook_duplication():
    transform = SemanticVectorTransform()
    transform.set_batch(1, 1)
    values = torch.ones(1, 2)
    assert transform('semantic_global', values) is values
    assert not hasattr(transform, 'collector')


def test_validation_and_dimension_fail_closed():
    for kwargs in ({'point': 'semantic'}, {'alpha': float('nan')}, {'vector': torch.ones(2, 2)}):
        with pytest.raises(ValueError):
            SemanticVectorTransform(**kwargs)
    transform = SemanticVectorTransform(torch.ones(2))
    with pytest.raises(RuntimeError, match='set_batch'):
        transform('semantic_local', torch.ones(1, 3))
    transform.set_batch(1, 1)
    with pytest.raises(ValueError, match='needs 3 dimensions'):
        transform('semantic_local', torch.ones(1, 3))
    with pytest.raises(ValueError):
        transform.set_active(torch.tensor([1]))
    with pytest.raises(ValueError):
        transform('semantic_global', torch.ones(2, 3))


def test_means_equal_sample_weight_and_disjoint_contrast_groups():
    means = [[1., 2.], [3., 4.], [-1., 0.]]
    assert torch.equal(mean_direction(means, [True, True, False]), torch.tensor([2., 3.]))
    assert torch.equal(mean_direction(means, [True, True, False], subtract_selected=[False, False, True]),
                       torch.tensor([3., 3.]))
    with pytest.raises(ValueError, match='at least one'):
        mean_direction(means, [False, False, False])
    with pytest.raises(ValueError, match='disjoint'):
        mean_direction(means, [True, False, False], subtract_selected=[True, False, False])
