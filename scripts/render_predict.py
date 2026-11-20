"""Inference-only CPU path for memory-constrained hosting; no training/plotting imports."""
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from ai.models.deepfake_detector import DeepfakeDetector
from ai.preprocessing.face_detector import get_face_cropper
from ai.preprocessing.fft_transform import face_to_fft
from ai.utils.config import MODEL_DIR, IMG_SIZE


def run(video, run_name, output_dir=None):
    torch.set_num_threads(1)
    checkpoint = torch.load(MODEL_DIR / run_name / 'best.pt', map_location='cpu', weights_only=True, mmap=True)
    config = checkpoint['config']
    if config.get('branches') != 'full':
        raise ValueError('Render mode requires the full three-branch checkpoint')
    length = int(config['seq_len'])
    model = DeepfakeDetector(sequence_length=length, pretrained=False).eval()
    model.load_state_dict(checkpoint['model'])
    del checkpoint
    model.inference_frame_batch = 1
    crop = get_face_cropper('opencv')
    capture = cv2.VideoCapture(str(video))
    faces = []
    try:
        if not capture.isOpened():
            raise ValueError('Cannot decode this video')
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if fps <= 0 or count <= 0:
            raise ValueError('Video must have valid frame rate and duration metadata')
        if count / fps > 60:
            raise ValueError('Free hosting accepts videos up to 60 seconds')
        for index in range(min(int(count), int(fps * 60))):
            success, frame = capture.read()
            if not success:
                break
            if index % 30 == 0:
                face = crop(frame)
                if face is not None:
                    faces.append(face)
                    if len(faces) >= max(length, 20):
                        break
    finally:
        capture.release()
    if not faces:
        raise ValueError('No faces detected in this video')
    selected = [faces[i] for i in np.linspace(0, len(faces)-1, length, dtype=int)]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    rgb = torch.stack([transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in selected])
    fft = torch.stack([transforms.ToTensor()(face_to_fft(f)) for f in selected])
    with torch.inference_mode():
        outputs = model(rgb.unsqueeze(0), fft.unsqueeze(0))
    return {
        'video': str(video), 'run_name': run_name, 'device': 'cpu', 'device_name': 'CPU',
        'face_detector': 'opencv', 'fake_score': float(torch.sigmoid(outputs['logit'])[0]),
        'branch_scores': {k: float(torch.sigmoid(v)[0]) for k,v in outputs['branch_logits'].items()},
        'sequence_length': length, 'sampled_frame_count': length, 'overlays': [],
        'branch_contributions': {'method': 'disabled_on_free_hosting'},
        'suspicion_timeline': {k: torch.sigmoid(v).tolist() for k,v in outputs['per_frame_logits'].items()},
    }
