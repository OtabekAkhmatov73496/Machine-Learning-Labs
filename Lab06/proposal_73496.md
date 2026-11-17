# Individual Project Proposal

**Name:** Akhmatov Otabek
**Student ID:** 73496
**Course:** Lab 6 – Training pipelines and project proposal

---

## Problem

Recognize short spoken command words (for example "yes", "no", "stop", "go") on a low-power device such as a smart-home controller or wearable, without sending audio to the cloud. Running on-device protects privacy and works offline, but it requires a model that is both small and fast.

## Dataset

- **Google Speech Commands v2**: about 105,000 one-second audio clips of 35 spoken words, recorded by thousands of different speakers.
- **Classes used:** the 10 standard command words (yes, no, up, down, left, right, on, off, stop, go), plus an "unknown" class made from the other words and a "silence" class made from the background-noise recordings.
- **Input features:** log-mel spectrograms computed from each 1-second clip.
- **Split:** the official `validation_list.txt` and `testing_list.txt` files supplied with the dataset, which keep each speaker in only one split and prevent data leakage.
- **License:** CC BY 4.0 (to be verified and cited in the final report).

## Success Metric

Test accuracy ≥ 90% on the 10 command classes on the official test split. Because of the added "unknown" and "silence" classes, I will also report macro F1 as a secondary metric.

## Deployment Constraints

- **Latency:** ≤ 50 ms per 1-second clip on CPU, including the spectrogram computation.
- **Model size:** ≤ 5 MB on disk.

## Planned Optimization

**Post-training int8 quantization.** A baseline float32 CNN on spectrograms is expected to take roughly 10 to 12 MB. Quantizing the weights to int8 should reduce the file to about a quarter of that size (about 3 MB) and speed up CPU inference. If accuracy drops by more than 2 percentage points, I will switch to quantization-aware training. Structured pruning is a possible extra step if the size or latency target is still not met.

## Feasibility and Evaluation Plan

- Train the baseline with the YAML-driven pipeline from this lab (fixed seed, checkpoints, MLflow logging of params, metrics and artifacts).
- Measure model size with `os.path.getsize` on the saved float32 and int8 model files.
- Measure latency as the mean and 95th percentile over 1000 single-clip inferences with `torch.no_grad()`, including the spectrogram step.
- Log accuracy, macro F1, size and latency for both models in MLflow so the trade-off can be compared directly.

## Risks

- Background noise and different accents can reduce accuracy, so I will use light augmentation (time shift and added noise).
- The "unknown" class is large and can dominate training, so I will balance it by sampling.
- Quantization can hurt accuracy on small models, which is why quantization-aware training is the fallback.
