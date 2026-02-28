# hotcb

A live control plane for ML training: callbacks, optimizer control, and loss control — with step-indexed ledger and deterministic replay.

- Works with bare PyTorch, PyTorch Lightning, and HuggingFace Trainer
- Enable/disable/load callbacks live during training
- Change optimizer LR, weight decay, and clipping mid-run
- Toggle loss terms and update loss weights on the fly
- Every mutation recorded in an applied ledger; fully replayable
