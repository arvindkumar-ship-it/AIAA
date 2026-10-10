import pytest
import torch
from intervention.intervention_predictor import InterventionPredictor, InterventionPredictorModel, ActionFeature


def test_missing_weights_do_not_generate_random_probability(tmp_path):
    predictor = InterventionPredictor(str(tmp_path / 'missing.pt'))
    with pytest.raises(RuntimeError, match='weights'):
        predictor.predict([])
    assert predictor.should_intervene([]) is True
    assert predictor.calibrator is None


def test_incompatible_weights_require_review(tmp_path):
    path = tmp_path / 'bad.pt'
    path.write_text('invalid artifact')
    predictor = InterventionPredictor(str(path))
    assert predictor.weights_loaded is False
    assert predictor.should_intervene([]) is True


def test_calibration_requires_explicit_labels(tmp_path):
    with pytest.raises(ValueError, match='held-out'):
        InterventionPredictor(str(tmp_path / 'missing.pt'), calibrate=True)


def test_real_calibration_input_validation(tmp_path):
    predictor = InterventionPredictor(str(tmp_path / 'missing.pt'))
    for scores, labels in [([0.1, 0.8], [0]), ([0.1] * 4, [0] * 4), ([0.1, 0.2, float('nan'), 0.9], [0, 0, 1, 1])]:
        with pytest.raises(ValueError): predictor._train_calibrator(scores, labels)
    predictor._train_calibrator([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1])
    assert predictor.calibrator is not None  # Fixture proves fitting, not real-world calibration.


def test_compatible_fixture_runs_on_cpu(tmp_path):
    path = tmp_path / 'fixture.pt'
    torch.save(InterventionPredictorModel().state_dict(), path)
    predictor = InterventionPredictor(str(path))
    actions = [ActionFeature('click', {}, {}, {})]
    assert 0 <= predictor.predict(actions) <= 1
    with pytest.raises(ValueError): predictor.predict([])


def test_metrics_require_real_nonempty_pairs(tmp_path):
    predictor = InterventionPredictor(str(tmp_path / 'missing.pt'))
    with pytest.raises(ValueError): predictor.get_metrics([], [])
    result = predictor.get_metrics([0.1, 0.9], [False, True])
    assert result['classification_accuracy'] == 1
    assert result['auc_roc'] == 1
    assert predictor.get_metrics([0.1], [False])['auc_roc'] is None
