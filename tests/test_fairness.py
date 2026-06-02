from pathlib import Path

from mentalhealthiq.fairness import evaluate_fairness


def test_fairness_evaluation_generates_report(tmp_path) -> None:
    model_path = Path('data/models/model.joblib')
    test_path = Path('data/processed/test.csv')
    raw_test_path = Path('data/processed/test_raw.csv')

    assert model_path.exists(), 'Trained model file is required for fairness tests'
    assert test_path.exists(), 'Processed test dataset is required for fairness tests'
    assert raw_test_path.exists(), 'Raw test dataset is required for fairness tests'

    output_dir = tmp_path / 'fairness_reports'
    all_metrics, disparities = evaluate_fairness(
        model_path=model_path,
        test_path=test_path,
        raw_test_path=raw_test_path,
        output_dir=output_dir
    )

    assert 'Gender' in all_metrics
    assert 'Age_Group' in all_metrics
    assert 'Race' in all_metrics
    assert 'Income' in all_metrics
    assert output_dir.joinpath('fairness_report.csv').exists()
    assert isinstance(disparities, dict)
    assert hasattr(all_metrics['Gender'], 'to_dict')
