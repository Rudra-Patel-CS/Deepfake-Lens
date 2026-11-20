import torch
from ai.models.deepfake_detector import DeepfakeDetector


def test_chunked_inference_matches_full_sequence():
    torch.manual_seed(19)
    model = DeepfakeDetector(pretrained=False).eval()
    rgb = torch.randn(1, 3, 3, 32, 32)
    fft = torch.randn(1, 3, 1, 32, 32)
    with torch.inference_mode():
        expected = model(rgb, fft)
        model.inference_frame_batch = 1
        actual = model(rgb, fft)
    torch.testing.assert_close(actual['logit'], expected['logit'], atol=1e-5, rtol=1e-4)
    for branch in expected['branch_logits']:
        torch.testing.assert_close(actual['branch_logits'][branch], expected['branch_logits'][branch], atol=1e-5, rtol=1e-4)


def test_lightweight_api_creates_report_directory(tmp_path, monkeypatch):
    import io
    import json
    import sys
    from types import SimpleNamespace
    from fastapi import UploadFile
    from starlette.datastructures import Headers
    from ai import api
    model_dir = tmp_path / 'models'
    (model_dir / 'full_v3').mkdir(parents=True)
    (model_dir / 'full_v3' / 'best.pt').touch()
    monkeypatch.setattr(api, 'MODEL_DIR', model_dir)
    monkeypatch.setattr(api, 'REPORT_DIR', tmp_path / 'reports')
    monkeypatch.setattr(api, 'UPLOAD_DIR', tmp_path / 'uploads')
    monkeypatch.setattr(api, 'LIGHTWEIGHT', True)
    result = {'fake_score': 0.1, 'overlays': [], 'branch_contributions': {'method': 'disabled_on_free_hosting'}, 'suspicion_timeline': {}}
    monkeypatch.setitem(sys.modules, 'scripts.render_predict', SimpleNamespace(run=lambda *args, **kwargs: result))
    upload = UploadFile(file=io.BytesIO(b'test'), filename='sample.mp4', headers=Headers({'content-type': 'video/mp4'}))
    response = api.analyze_video(upload)
    report = tmp_path / 'reports' / 'web_analyses' / response['job_id'] / 'forensic_report.json'
    assert json.loads(report.read_text())['fake_score'] == 0.1
    assert not list((tmp_path / 'uploads').iterdir())
    assert not api.ANALYSIS_LOCK.locked()
