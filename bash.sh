python siren_psf.py --batch_size 400000 --epochs 50 --n_sample 3 --model_class SirenNet &&
python siren_psf.py --batch_size 15000 --epochs 50 --n_sample 3 --model_class PsfSirenNet &&
python siren_psf.py --batch_size 3000 --epochs 50 --n_sample 5 --model_class PsfSirenNet &&
python siren_psf.py --batch_size 400000 --epochs 50 --accumulate_grad_batches 5 --n_sample 3 --model_class SirenNet &&
python siren_psf.py --batch_size 15000 --epochs 1420 --accumulate_grad_batches 142 --n_sample 3 --model_class PsfSirenNet