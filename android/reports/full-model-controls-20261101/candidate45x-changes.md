# Candidate45x changes and causality

TX remains compiled out; engagement remains disabled. Cooling optimization is deferred.

Candidate45v/candidate45w preserve the same native and converted-model numerical implementation. Candidate45w adds selected-seed diagnostic tracing. Their original-reference and timing results retain their actual APK identities.

On the physical Fold 3, the candidate45w seed7 trace isolates the road model's remaining final-output failure. Cutting captured `model/p_add_2` into the original host graph produces exact outputs. Cutting the next captured `model/p_layer_norm` into that same graph reproduces all 47 mismatched final output elements (maximum absolute error 0.03125). No non-finite outputs are involved. See `road45w-cut-add2-seed7.json` and `road45w-cut-layernorm-seed7.json`.

On the captured input, the original host's ordered FP32 Welford LayerNorm produces all 4608 FP16 elements exactly; the prior MNN parallel reduction differs in four half elements. The second normalization also matches ordered Welford exactly on its captured input, whereas the prior MNN path differs in nine half elements. Both local differences are within intermediate tolerances, so local tolerance success alone does not establish full-model correctness. See `road45w-layernorm-welford-seed7.json` and `road45w-layernorm1-welford-seed7.json`.

The converter now tags only original FP16, last-axis LayerNormalization with FP32 stash, one output and constant matching FP16 scale/bias. The pinned MNN High/buffer implementation selects an ordered GPU Welford kernel only for this tag, one input/output, unpacked layout, matching gamma width, group1, and no RMS normalization. Correctly rounded division and positive-normal square root reuse the strict OpenCL reference's arithmetic. Explicit following FP16 Casts and original external model inputs/outputs remain intact. Other operators keep their existing paths.

Original models, goldens and rtol=1e-3/atol=1e-4 remain unchanged. The complete untapped original-reference suite on the actual APK is required before acceleration or sustained workload qualification. Diagnostic tensor tracing cannot qualify inference speed.
