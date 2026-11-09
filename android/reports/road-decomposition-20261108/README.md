# Linux converter and emulator follow-up

## 9 November: residual precision and selective consumers

**PARTIAL / NOT QUALIFIED.** Supplying the original pre-cast FP32 residual
`InsertedPrecisionFreeCast_model/p_add_3` and original `model/p_linear_9`
simultaneously to the host LiteRT suffix with the original FP16 rounding after
`p_add_4` clears the suffix output tolerance failures on seeds 42 / 7 / 123
(baseline for that suffix: **44 / 78 / 5**). This is an intervention with
reference values, not executable inference using real accelerator inputs.
The two-output ONNX tap preserved public outputs bit-exactly for these seeds;
intervention can still change execution. Supplying the rounded half residual
instead gave **61 / 81 / 73** failures, demonstrating that using the wrong
side of the rounding boundary changes the result.

The original graph sends the FP32 residual directly to the skip add while a
cast FP16 value feeds the second layer normalization. The diagnostic rounding
patcher can now select specific consumers, and a synthetic test confirms the
untouched branch remains unrounded. Restoring that selective layer-normalization
input boundary in the suffix that already rounds `p_add_4` yielded **52 / 84 /
15** output mismatches; all three seeds regress against its **44 / 78 / 5**
baseline. The derived diagnostic SHA256 is
`c619bb317cfc34665148759a2fd4ff4474617940c209763678cb1a285d48ab35`.
Applying only the selective normalization input boundary, without the `p_add_4`
rounding, also regressed all three seeds: **46 / 34 / 69** against the **42 /
25 / 62** baseline. Its SHA256 is
`bc0b990f45a5fded8d36770ecc139dd5a8a61d6ec8f1f8899be6a347d677a01f`.
Do not promote these graphs. The errors may interact across multiple boundaries;
the converted prefix, recurrent state, replay and actual GPU remain unqualified.

The intervention and selective rounding helpers passed focused host tests. No
new APK, device benchmark or production backend change was made. The 50 ms
deadline, vehicle TX prohibition and engagement block remain unchanged.

To reproduce with fresh output paths, define `VENV` and `BASE` as in the next
section and use the separately hashed `suffix-history-add4-half.tflite`:

```sh
"$VENV/bin/python" android/scripts/intervene_litert_suffix.py --sections "$BASE/rewritten-sections" --conversion "$BASE/rewritten-suffix" --model "$BASE/suffix-history-add4-half.tflite" --derived-sha256 9466b986590338782ee9d290338b3ecd4a6040d2300b0fd7e0e5b44442cb10fc --onnx-tensor InsertedPrecisionFreeCast_model/p_add_3 --litert-tensor InsertedPrecisionFreeCast_model/p_add_3 --onnx-tensor model/p_linear_9 --litert-tensor model/p_linear_9 --output "$NEW_INTERVENTION_DIR"
"$VENV/bin/python" android/scripts/patch_litert_half_boundaries.py --source "$BASE/suffix-history-add4-half.tflite" --source-sha256 9466b986590338782ee9d290338b3ecd4a6040d2300b0fd7e0e5b44442cb10fc --tensor InsertedPrecisionFreeCast_model/p_add_3 --producer add --consumer-output model/p_node_layer_norm_1_layernorm_mean --consumer-output model/p_node_layer_norm_1_layernorm_centered --output "$NEW_SUFFIX"
"$VENV/bin/python" android/scripts/validate_litert_dense_sections_host.py --sections "$BASE/rewritten-sections" --prefix "$BASE/rewritten-prefix/converted/prefix_float32.tflite" --suffix "$NEW_SUFFIX" --output "$NEW_REPORT"
```

## 9 November: policy interventions and output-boundary rounding

**PARTIAL / NOT QUALIFIED.** Two policy projections pass the original tolerance on all three seeds when tested on identical inputs and rounded at the original output boundary. Their results are not always bit-exact. The first eight history slices of the first projection are bit-exact against the individually tapped reference; the current slice differs. These host CPU checks do not establish complete-model or GPU parity.

The new [intervention helper](../../scripts/intervene_litert_suffix.py) feeds an individually tapped original-ONNX value into one downstream point of a separate diagnostic LiteRT suffix, with the original section inputs in both cases. Public outputs of every tapped ONNX graph remained bit-exact in these tests, although tapping or changing the LiteRT graph can still affect execution. Road output tolerance failures for seeds **42 / 7 / 123** were:

| Intervention | Baseline | Reference value substituted |
| --- | --- | --- |
| `model/p_index` or `model/p_linear` | 42 / 25 / 62 | 43 / 47 / 52 |
| `model/p_linear_1` | 42 / 25 / 62 | 19 / 42 / 52 |
| `model/p_layer_norm` | 42 / 25 / 62 | 40 / 36 / 67 |
| `model/p_layer_norm_1` | 42 / 25 / 62 | 42 / 28 / 71 |
| `model/p_add_4` or `model/p_select` | 42 / 25 / 62 | 0 / 0 / 0 |

Thus the mismatch is not repaired merely by correcting the first policy input, and supplying the exact later transformer result removes the remaining suffix failures in this limited test. It does **not** prove that a single operator causes the differences, nor does it address the converted vision prefix, recurrent sequences or replay. Restoring the original FP16 boundary after `p_add_4` alone yielded **44 / 78 / 5** failures, so that diagnostic is rejected because two seeds regress. Its SHA256 is `9466b986590338782ee9d290338b3ecd4a6040d2300b0fd7e0e5b44442cb10fc`.

Fourteen focused host tests passed for the intervention, tensor isolation, trace and rounding helpers. No APK or active model was changed; this follow-up did not rerun device or emulator inference. The Fold 3 was visible to ADB, but no hardware experiment was started for a host candidate that already fails accuracy. Candidate73 remains the separate accurate but slow GPU baseline; camera freshness, controls timing and the 50 ms end-to-end target remain open. TX and physical engagement remain disabled.

To reproduce an intervention, choose a new output directory (the helper refuses to overwrite one):

```sh
VENV=android/build/candidates/candidate74-runtime/follow-up/litert-mac-converter-venv
BASE=android/build/candidates/candidate74-runtime/follow-up/dense-split-mac-20261109
"$VENV/bin/python" android/scripts/intervene_litert_suffix.py --sections "$BASE/rewritten-sections" --conversion "$BASE/rewritten-suffix" --model "$BASE/suffix-history-half.tflite" --derived-sha256 537b6b345d335c2930d21cd7ba8df0b01b267519e1f83a6e44ea9a91e3872a05 --onnx-tensor model/p_add_4 --litert-tensor InsertedPrecisionFreeCast_model/p_add_4 --output "$NEW_INTERVENTION_DIR"
"$VENV/bin/python" android/scripts/patch_litert_half_boundaries.py --source "$BASE/suffix-history-half.tflite" --source-sha256 537b6b345d335c2930d21cd7ba8df0b01b267519e1f83a6e44ea9a91e3872a05 --tensor InsertedPrecisionFreeCast_model/p_add_4 --producer add --output "$NEW_SUFFIX"
"$VENV/bin/python" android/scripts/validate_litert_dense_sections_host.py --sections "$BASE/rewritten-sections" --prefix "$BASE/rewritten-prefix/converted/prefix_float32.tflite" --suffix "$NEW_SUFFIX" --output "$NEW_REPORT"
```

## 9 November: recurrent-history rounding follow-up

**PARTIAL / NOT QUALIFIED.** The original policy casts stored feature history to FP16 before use, but the converted diagnostic suffix omitted that boundary. Restoring it makes the first eight history slices of `model/p_index` bit-exact on all three diagnostic seeds; the current vision slice is still different. With original section inputs, road-output mismatch counts change from **42 / 27 / 62** to **42 / 25 / 62** for seeds 42 / 7 / 123. Rounding the current vision input as well gives **41 / 47 / 52**, a regression on seed 7, and is rejected. The reference tap preserved public outputs in these checks, although tapping can still alter intermediate computation.

Full host recomposition with the rounded-history suffix fails: **701 / 715 / 682** road-output elements and **26 / 16 / 6** feature-state elements exceed the unchanged tolerances. The image-history and desire-state outputs pass on all three seeds. The converted prefix also introduces error, so the suffix-only comparison does not qualify the complete model. These are host CPU diagnostics, not the required GPU, recurrent and replay suite. Candidate73's existing 795/795 parity result is separate; its measured road inference time exceeds the deadline.

The diagnostic suffix SHA256 `537b6b345d335c2930d21cd7ba8df0b01b267519e1f83a6e44ea9a91e3872a05` compiled GPU-only on the Fold 3 in a compile-only shell probe. This does not establish inference, numerical parity, or speed. The guarded experiment used installed APK SHA256 `d06b27a435e1c2fa947066c7cc71efcaf810a053a57834c603d86c1722c74272`; it recorded 32 seconds of normal cooldown and no owned app processes, cameras or forwards after cleanup. The APK and active model were not changed. Receipts and comparisons are in `android/build/candidates/candidate74-runtime/follow-up/dense-split-mac-20261109/`, including `suffix-history-half-host.json`, `hybrid-history-half-host.json`, and `suffix-history-half-fold-experiment.json`.

The updated tracing and rounding helpers passed 24 focused host checks; another 69 host checks covered controls, thermal handling, experiment isolation and Panda restrictions. Reproduce the history experiment with fresh output paths:

```sh
VENV=android/build/candidates/candidate74-runtime/follow-up/litert-mac-converter-venv
BASE=android/build/candidates/candidate74-runtime/follow-up/dense-split-mac-20261109
"$VENV/bin/python" android/scripts/patch_litert_half_boundaries.py --source "$BASE/suffix-norm-silu-proj-round.tflite" --source-sha256 32399f224d492ff2bf2da8f67de866d65da5146a0346bc9d86cee90da6431328 --producer reshape --tensor slice_5 --output "$NEW_SUFFIX"
"$VENV/bin/python" android/scripts/validate_litert_dense_sections_host.py --sections "$BASE/rewritten-sections" --prefix "$BASE/rewritten-prefix/converted/prefix_float32.tflite" --suffix "$NEW_SUFFIX" --output "$NEW_SUFFIX_REPORT"
"$VENV/bin/python" android/scripts/validate_litert_hybrid_host.py --sections "$BASE/rewritten-sections" --prefix "$BASE/rewritten-prefix/converted/prefix_float32.tflite" --suffix "$NEW_SUFFIX" --output "$NEW_FULL_REPORT"
```

Vehicle TX and physical engagement remain disabled. Camera freshness and identity, complete-cycle controls timing, mounted calibration and physical Panda checks remain open; no sustained combined run was advanced.

## 9 November: policy projection follow-up

**PARTIAL / NOT QUALIFIED.** The policy `MatMul_16` fully connected operation matches the isolated ONNX operation on identical, losslessly converted inputs for all three seeds. Its activation input is not bit-exact despite passing tolerance, and the tapped layer has 17 / 27 / 30 mismatches. The test explicitly transposes weights for ONNX `transB=0`, refuses lossy dtype conversions, and checks that reference taps preserve public outputs; the CPU result does not establish GPU arithmetic.

The original graph also rounds the output of the preceding `MatMul_13` projection to FP16. Restoring that boundary in the norm-plus-SiLU diagnostic suffix reduces original-input suffix mismatch counts to **42 / 27 / 62**, compared with **45 / 29 / 79** for the earlier norm-only suffix. The artifact SHA256 is `32399f224d492ff2bf2da8f67de866d65da5146a0346bc9d86cee90da6431328`. Additional vision-concat rounding yielded **41 / 47 / 53**; rounding three more projections yielded **61 / 49 / 29**. Both regress at least one seed and are rejected. These are isolated host CPU diagnostics, not full-model parity or GPU qualification; the active model and APK are unchanged.

Evidence is in `android/build/candidates/candidate74-runtime/follow-up/dense-split-mac-20261109/`, including `suffix-policy-gemm-half-v1-{42,7,123}/isolation.json`, `suffix-norm-silu-proj-round-host.json`, `suffix-policy-vision-half-host.json`, and `suffix-policy-projections-half-host.json`. The [Gemm isolation helper](../../scripts/isolate_litert_suffix_gemm.py) now validates converted weight layout and lossless input types. Vehicle TX remains disabled.

This follow-up passed 20 host LiteRT and trace checks and 69 host controls, thermal, experiment-isolation and Panda checks. The dedicated Android 15 emulator passed both existing compiled-out-TX and invalid-USB-descriptor rejection tests using unchanged APK `d06b27a435e1c2fa947066c7cc71efcaf810a053a57834c603d86c1722c74272`. Its `emulator-safety-policy-followup-experiment.json` receipt records the cooldown and cleanup; the emulator was then stopped. These checks do not establish full-model accuracy, camera freshness or control timing.

The `32399f...` diagnostic suffix also compiled GPU-only on the Fold 3. This is a compile-only ADB-shell probe, not inference, parity or performance evidence. `suffix-policy-proj-fold-experiment.json` records the same installed APK hash, 32 seconds of normal cooldown, and no owned app processes, cameras or forwards after cleanup.

To reproduce the projection experiment using the norm and SiLU diagnostic graph from the preceding follow-up, set `VENV` and `BASE` as below and choose fresh output paths:

```sh
VENV=android/build/candidates/candidate74-runtime/follow-up/litert-mac-converter-venv
BASE=android/build/candidates/candidate74-runtime/follow-up/dense-split-mac-20261109
"$VENV/bin/python" android/scripts/patch_litert_half_boundaries.py --source "$BASE/suffix-norm-silu-round.tflite" --source-sha256 1b60306d0b7f74826ed8968c88899f433451cc7ce13796b88b7fab0d5b00332a --tensor gemm_output_reshape_arg_fc_float32 --output "$NEW_SUFFIX"
"$VENV/bin/python" android/scripts/validate_litert_dense_sections_host.py --sections "$BASE/rewritten-sections" --prefix "$BASE/rewritten-prefix/converted/prefix_float32.tflite" --suffix "$NEW_SUFFIX" --output "$NEW_REPORT"
```

## 9 November: policy normalization and minimally tapped traces

**PARTIAL / NOT QUALIFIED.** Adding all mapped outputs to the ONNX policy graph perturbs its public road output beyond the original tolerance: 120 / 96 / 86 elements differ for seeds 42 / 7 / 123 compared with the untapped graph. The previous broad trace's apparent first failure at `fused model/node_linear_2` is therefore not a reliable localization. Both trace scripts now report whether their taps preserve the original public outputs; matching public outputs is necessary but does not prove that every intermediate remains unchanged.

The [single-tensor trace](../../scripts/trace_litert_suffix_single.py) and [Gemm isolation](../../scripts/isolate_litert_suffix_gemm.py) make smaller changes to the reference graph and record the impact on public outputs. Isolated `model/node_linear_2` and `model/node_linear_3` use identical captured inputs, weights and bias and match the ONNX CPU operator within tolerance on all three seeds. They do not establish GPU operator behavior or complete-model parity.

A separate demonstrated difference is the missing FP16 rounding between the first two policy vector norms and their consumers. A diagnostic graph inserts FP32-to-FP16-to-FP32 casts after both relevant LiteRT SQRT operations. For the first norm, the original half tensor and its expanded value now match bit-for-bit for all three captured seeds; the tapped original public outputs stay within tolerance. The untouched suffix still fails: road-output mismatch counts are **91 / 30 / 84 before, 45 / 29 / 79 after** with original prefix inputs. With the separately modified byte-normalization prefix, full host composition still has **719 / 730 / 669** road-output mismatches and **26 / 16 / 6** feature-history mismatches. These are three-seed host CPU diagnostics, not the 795-comparison qualification, recurrence/replay, GPU parity or timing.

The diagnostic suffix has SHA256 `8316a96bc8d801dc4ffdad6401a360e88a088d36cf4275b74e42c32f31dc3f80`, derived from converted suffix `4ff3a75f88f6cee8e7fbcb5eb6ab198865a8b50df7705e2a579e1e08f509968a`. The source models and live backend are unchanged. Full accuracy, the 50 ms deadline, camera freshness and controls timing remain blockers; TX and engagement remain disabled.

The corrected suffix **compiled** with the GPU-only LiteRT probe on the Fold 3; this was a compile-only test, not inference, accuracy or timing validation. The experiment receipt `suffix-norm-round-fold-experiment.json` records the installed APK SHA256 `d06b27a435e1c2fa947066c7cc71efcaf810a053a57834c603d86c1722c74272`, 32 seconds of normal-temperature cooldown, and released processes, camera and forwarding after the run.

In seed 42's individually tapped host trace, the corrected suffix has no tolerance failures at `linear_6`, `div_1`, `relu_6` or `linear_8`, whereas the earlier uncorrected `linear_8` comparison had 61 failures. A later policy projection, `model/p_linear_1`, still has 19 failures after an upstream activation that passes tolerance but is not bit-exact. Adding another FP16 round trip after that activation produced **42 / 38 / 76** final suffix mismatches for seeds 42 / 7 / 123, compared with **45 / 29 / 79** without it; the seed-7 regression rejects this additional change. Its separate diagnostic artifact SHA256 is `1b60306d0b7f74826ed8968c88899f433451cc7ce13796b88b7fab0d5b00332a`. These local traces do not qualify the complete model.

Host checks: 17 LiteRT, trace and provenance tests and 57 controls, reporting, experiment-isolation and Panda tests passed. Evidence is under `android/build/candidates/candidate74-runtime/follow-up/dense-split-mac-20261109/`: `suffix-all-taps-perturbation.json`, `suffix-gemm{2,3,6,7,8,9,10,20}-check*/isolation.json`, `suffix-norm-round-trace{42,7,123}.json`, `suffix-norm-round-host.json` and `hybrid-byte-norm-round-host.json`. The `*-check*` records identify the actual seed. The unchanged installed APK is not a new build.

Reproduce the diagnostic rewrite and original-input suffix comparison with fresh output paths:

```sh
VENV=android/build/candidates/candidate74-runtime/follow-up/litert-mac-converter-venv
BASE=android/build/candidates/candidate74-runtime/follow-up/dense-split-mac-20261109
"$VENV/bin/python" android/scripts/patch_litert_half_boundaries.py --source "$BASE/rewritten-suffix/converted/suffix_float32.tflite" --source-sha256 4ff3a75f88f6cee8e7fbcb5eb6ab198865a8b50df7705e2a579e1e08f509968a --producer sqrt --tensor InsertedPrecisionFreeCast_model/linalg_vector_norm --tensor InsertedPrecisionFreeCast_model/linalg_vector_norm_1 --output "$NEW_SUFFIX"
"$VENV/bin/python" android/scripts/validate_litert_dense_sections_host.py --sections "$BASE/rewritten-sections" --prefix "$BASE/rewritten-prefix/converted/prefix_float32.tflite" --suffix "$NEW_SUFFIX" --output "$NEW_REPORT"
```

## 9 November: normalization and trace diagnostics

**PARTIAL / NOT QUALIFIED.** Host tracing now resolves the actual LiteRT producer rather than relying on the conversion correspondence's stale operator index. For three captured seeds the traced prefix's first observed tolerance failure is inherited, and isolating the suspected convolution with identical inputs passes tolerance. Intermediate taps can change fusion and do not prove complete-model behavior.

The conversion changes division by 63.75 into multiplication by a rounded reciprocal. On the three captured image inputs, respectively 386,936 / 387,148 / 386,987 out of 786,432 normalization values differ bitwise from the tapped ONNX reference. The diagnostic byte lookup restores bitwise agreement for all values in those inputs and a synthetic test covers all 256 byte values. Its validity is restricted to integral 0–255 inputs; it is not installed or selected by the driving runtime. The complete composed host diagnostic still fails: road output mismatches for seeds 42 / 7 / 123 were 719 / 731 / 668 before and 725 / 728 / 670 after the lookup. Its numerical effect is therefore insufficient to qualify the model.

The initial lookup graph failed Fold 3 GPU-only compilation because the delegate requires one-dimensional gather indices. A flattened-input lookup graph compiled GPU-only on the Fold 3, using fp32 precision and no accepted CPU partition. **This is compilation, not full-graph execution or parity evidence.** A separate GPU-only primitive executed the flattened lookup for all 256 byte values with exact output and no non-finite values; its graph SHA256 was `0129c4cd625f287740245f9196f05517226e409d1b4f724f49688630273febee`. It does not establish full-prefix or full-model parity or speed. The diagnostic prefix graph hash is `82326c5186aee6febd80a0f9071d20258b85b84cb04632d8ca9065f1edaa505c`; the unchanged installed APK hash was `d06b27a435e1c2fa947066c7cc71efcaf810a053a57834c603d86c1722c74272`. Both experiments used the measured cool baseline and cleanup; the failed unflattened attempt is retained. The source converter graph hash is `c92924c345f227c7874f22fdb57f0afe0cdf394a74e12eaf6abe5ae514691c3f`.

The modified [normalization patcher](../../scripts/patch_litert_byte_normalization.py), [host comparison](../../scripts/validate_litert_byte_normalization.py), [prefix trace](../../scripts/trace_litert_prefix.py), and [convolution isolation](../../scripts/isolate_litert_prefix_conv.py) preserve source graphs and reject unexpected contracts. Evidence is under `android/build/candidates/candidate74-runtime/follow-up/dense-split-mac-20261109/`, including `byte-normalization-host.json`, `byte-flat-host.json`, `hybrid-byte-lookup-host.json`, the two `prefix-byte-*-fold-experiment.json` receipts, and `normalization-primitive-fold-experiment.json`. The normalizing rewrite is diagnostic only. Full-model LiteRT parity, three-camera freshness, controls complete-cycle timing and the 50 ms end-to-end deadline remain blocked. Vehicle TX and engagement remain disabled.

Focused checks this turn: 15 LiteRT, provenance and recomposition tests and 57 controls, reporting, experiment isolation and Panda tests passed on the host. The unchanged APK was not rebuilt. Earlier candidate73's 795-comparison pass applies only to its existing ncnn implementation, which is too slow for the requested deadline.

Reproduce the lookup with a fresh output path, then run the host comparison with a separate fresh report:

```sh
VENV=android/build/candidates/candidate74-runtime/follow-up/litert-mac-converter-venv
BASE=android/build/candidates/candidate74-runtime/follow-up/dense-split-mac-20261109
"$VENV/bin/python" android/scripts/patch_litert_byte_normalization.py --source "$BASE/rewritten-prefix/converted/prefix_float32.tflite" --source-sha256 c92924c345f227c7874f22fdb57f0afe0cdf394a74e12eaf6abe5ae514691c3f --flatten-indices --output "$NEW_MODEL"
"$VENV/bin/python" android/scripts/validate_litert_byte_normalization.py --sections "$BASE/rewritten-sections" --source "$BASE/rewritten-prefix/converted/prefix_float32.tflite" --modified "$NEW_MODEL" --output "$NEW_REPORT"
```

## Mac continuation: rounding and composed hybrid checks

**PARTIAL; no new runtime is qualified.** The [candidate73 Fold 3 ncnn results](../dispatch-recovery-20261108/README.md) already pass **795/795 original-reference comparisons**, including recurrent and replay checks, without CPU fallback. Its unprofiled dispatch524288 road inference still took 309.712 ms median and 352.664 ms p99, before camera delivery. This corrects the less precise wording below: full accelerator parity is established for that specific ncnn artifact, but not for the LiteRT or hybrid alternatives and not for a faster qualified pipeline.

The diagnostic [flatbuffer patcher](../../scripts/patch_litert_half_boundaries.py) inserts explicit float32-to-half-to-float32 casts after selected dense operations in a separately hashed **copy** of the converted graph. It does not modify original models or promote a backend. A synthetic LiteRT test checks the numerical rounding and rejects invalid or duplicate targets. With original captured suffix inputs, three restored boundaries produced 153 / 64 / 103 road-output mismatches for seeds 42 / 7 / 123; 24 restored boundaries produced 130 / 66 / 146. The unmodified converted suffix had 91 / 30 / 84. Thus neither change fixes parity. The three-boundary graph compiled GPU-only on the Fold 3, but compilation is not inference or proof that GPU execution preserves the casts. The experiment retained the pre-existing installed APK SHA256 `d06b27a435e1c2fa947066c7cc71efcaf810a053a57834c603d86c1722c74272`, a normal cool baseline and clean teardown.

The [composed hybrid host check](../../scripts/validate_litert_hybrid_host.py) now feeds the converted prefix's *actual* output into the ordered-half dense diagnostic and then the converted suffix. The prefix passes tolerance in isolation, yet its differences lead to 1 / 2 / 6 dense tolerance failures. Even the original ONNX suffix fed from this prefix produces 5 / 90 / 20 road-output failures against the original-prefix version. The entire converted CPU composition has 719 / 731 / 668 road-output and 26 / 17 / 6 feature-history mismatches against the original goldens. This is a host diagnostic, not a GPU hybrid or replay result; it shows that both inherited prefix error and suffix conversion require attention. No runtime switch, arithmetic change or relaxed tolerance has been promoted.

The single-position and paired spatial accumulation implementations were previously tested in candidates70 and 72. The [candidate72 report](../pair-reuse-20261108/README.md) records 795/795 accuracy for candidate72 and a lower isolated median in that comparison, with a worse p99. Candidate73 retains the paired implementation. Those earlier reports are not a fresh identical-input, controlled one-versus-two experiment.

Artifacts and receipts are under `android/build/candidates/candidate74-runtime/follow-up/dense-split-mac-20261109/`: `suffix-three-half-boundaries-host.json`, `suffix-24-half-boundaries-host.json`, `suffix-three-half-fold-experiment.json` and `hybrid-composed-inherited-host.json`. Run the new host checks with the existing isolated converter environment and fresh output paths:

```sh
CONVERTER_VENV=android/build/candidates/candidate74-runtime/follow-up/litert-mac-converter-venv
"$CONVERTER_VENV/bin/python" -m unittest android.tests.test_litert_half_boundaries android.tests.test_litert_targeted_round android.tests.test_litert_dense_sections android.tests.test_litert_road_provenance android.tests.test_road_dense_recomposition
"$CONVERTER_VENV/bin/python" android/scripts/validate_litert_hybrid_host.py --sections "$SECTIONS" --prefix "$PREFIX_TFLITE" --suffix "$SUFFIX_TFLITE" --output "$NEW_REPORT"
"$CONVERTER_VENV/bin/python" android/scripts/patch_litert_half_boundaries.py --source "$SUFFIX_TFLITE" --source-sha256 "$SUFFIX_SHA256" --tensor model/linear_1_fc_float32 --tensor model/linear_35_fc_float32 --tensor model/linear_68_fc_float32 --output "$NEW_DIAGNOSTIC_TFLITE"
```

**Focused validation:** 11 LiteRT and road provenance tests and 67 scheduling, reporting, experiment-isolation and Panda checks passed on the host. The Fold 3 and dedicated Android 15 emulator each passed two instrumentation tests for the compiled-out TX flag and invalid USB descriptor rejection; both used APK SHA256 `d06b27a435e1c2fa947066c7cc71efcaf810a053a57834c603d86c1722c74272`. The experiment receipts `fold-safety-half-followup-experiment.json` and `emulator-safety-half-followup-experiment.json` record more than 30 seconds of normal cooldown and no owned app processes, camera or new forwards after the runs. The emulator was stopped. This is safety-path coverage of the existing APK, not a newly built or qualified model runtime.

Vehicle TX and physical engagement remain disabled. No combined driving workload was advanced: LiteRT/hybrid accuracy, the 50 ms end-to-end deadline, camera freshness and identity, and controls timing remain open. Cooling optimization remains deferred and thermal safeguards remain active.

## Mac continuation: split LiteRT investigation

**PARTIAL; not qualified.** The existing rank-reduced, reference-validated road graph was split around its first ordered-half dense layer. Host-only recomposition passed **636/636 road comparisons**, including recurrence and the pinned replay; it does not establish GPU parity. Its derived prefix and suffix converted independently with onnx2tf 2.6.5. With identical original-section inputs, host LiteRT CPU prefix outputs passed tolerance for three seeds, including exact packed image history, while suffix `outputs` failed with **91, 30 and 84 mismatched elements** for seeds 42, 7 and 123. Host LiteRT is diagnostic, and this was not full converted-model validation.

On the Fold 3, each converted section separately passed a **GPU-only compilation** probe with fp32 precision and no CPU partition. The first attempt using the unmodified rank-five split failed GPU compilation; that failure is preserved. These probes did not run inference, check GPU arithmetic, combine sections, or measure latency. The Fold had installed APK SHA256 `d06b27a435e1c2fa947066c7cc71efcaf810a053a57834c603d86c1722c74272`. Both successful probes used the guarded experiment wrapper, more than 32 seconds of normal cool baseline, and verified camera/process/forwarding cleanup.

A host trace compared 213 mapped tensors against a tapped ONNX graph for seed 42. As demonstrated above, the extensive taps change public outputs beyond tolerance, so the apparent first failure is not reliable. The first observed tolerance failure was `fused model/node_linear_2` (`Gemm` / LiteRT `FULLY_CONNECTED`), with 6 mismatched values and maximum absolute error 0.000282. Earlier converted dense layers removed FP16 output boundaries. For `model/linear_1`, casting the captured float32 result to half on the host reproduced the tapped ONNX tensor exactly; this is evidence for a missing rounding boundary, not proof that it is the only error. Tapping may alter graph optimization. A broad explicit-rounding graph and a targeted three-layer graph both failed host suffix output comparisons; the latter had **118, 63 and 108** mismatched values for the three seeds. Neither is accepted or used for GPU timing.

The Fold 3 and Android 15 emulator each passed two focused TX/USB rejection instrumentation tests. The emulator used installed APK SHA256 `e4e6902dc2955005622b5b9358e4aa507afdcea99f74b904ada515c5b2a17db9`; the wrapper verified teardown and the emulator was stopped. The Fold test used the APK hash listed above and also passed guarded cooldown and cleanup. The two devices had different pre-existing APKs; no final APK was built or installed, and these results are not matched-artifact validation. Twelve focused host tests passed. Vehicle TX and engagement remain disabled. Controls timing, camera freshness and physical camera identity, a combined on-device LiteRT/hybrid model, its complete 795 accelerator comparisons, and the unchanged 50 ms sensor-to-publication deadline remain unresolved. No extended workload was run because the preceding accuracy gate failed.

Artifacts, hashes, conversion receipts, isolated source copies and cleanup records are under `android/build/candidates/candidate74-runtime/follow-up/dense-split-mac-20261109/`. To recreate the host split and checks, use fresh paths and the isolated converter environment created with Python 3.12 and `onnx2tf==2.6.5`, `ai-edge-litert==2.1.2`, `onnxruntime==1.26.0`:

```sh
.venv/bin/python android/scripts/decompose_litert_road_dense.py \
  --source android/build/candidates/candidate74-runtime/follow-up/lowrank-sqrt-20261109/road-low-rank.onnx \
  --receipt android/build/candidates/candidate74-runtime/follow-up/lowrank-sqrt-20261109/repair.json \
  --output "$NEW_SECTIONS"
.venv/bin/python android/scripts/convert_litert_isolated.py --source "$NEW_SECTIONS/prefix.onnx" \
  --source-sha256 "$PREFIX_SHA256" --converter "$CONVERTER_VENV/bin/onnx2tf" --output "$NEW_PREFIX_CONVERSION"
.venv/bin/python android/scripts/convert_litert_isolated.py --source "$NEW_SECTIONS/suffix.onnx" \
  --source-sha256 "$SUFFIX_SHA256" --converter "$CONVERTER_VENV/bin/onnx2tf" --output "$NEW_SUFFIX_CONVERSION"
"$CONVERTER_VENV/bin/python" android/scripts/validate_litert_dense_sections_host.py \
  --sections "$NEW_SECTIONS" --prefix "$NEW_PREFIX_TFLITE" --suffix "$NEW_SUFFIX_TFLITE" --output "$NEW_HOST_REPORT"
"$CONVERTER_VENV/bin/python" -m unittest android.tests.test_litert_targeted_round \
  android.tests.test_litert_dense_sections android.tests.test_road_dense_decomposition android.tests.test_road_decomposition
```

The source hashes are recorded in the new sections' `manifest.json`. The two guarded Fold compile commands and their exact binary, model, library and installed-APK hashes are in `rewritten-prefix-fold-experiment.json` and `rewritten-suffix-fold-experiment.json` in the artifact directory. The added trace and rounding tools are diagnostic only and must not select a production backend.

## Continuation: ordered-half dense recomposition

The pinned original road graph can be split into a prefix, its original first half-precision Gemm, and a suffix without discarding the crossing image-history output. A host-only recomposition replaces that Gemm with the eight-lane ordered-half implementation: all **636 road** original-reference comparisons pass across the three seeds, recurrent sequences and replay. The unchanged host driver reference adds **159/159**, for **795/795 host comparisons**. This does not qualify an Android accelerator or imply the rest of the full model runs on the GPU.

For each of the three initial seeds, the Fold 3 GPU-only probe saved its actual 2,048 dense output values. The inputs match the corresponding original host prefix exactly, the weights and bias match the pinned model, and the saved device results match the original dense reference exactly. Recombining each saved device output with the host prefix and suffix passes all four original road outputs for each seed. These three mixed host/device cases are separate from the host-only 795 checks; recurrent and replay execution were not run on the device. The three single kernel event times were 19.881, 19.820 and 20.755 ms, excluding initialization, transfer, remaining model, camera and publication. They do not meet or establish the 50 ms end-to-end target.

Each Fold experiment used the previously installed `com.samsung.sunnypilot.fold3` APK SHA256 `d06b27a435e1c2fa947066c7cc71efcaf810a053a57834c603d86c1722c74272`. The standalone GPU probe SHA256 was `b02168d2495d77b0272700056cb3a53d79745a9a6d27e9a67e8bdf89519035ed`; its kernel SHA256 was `0f8b9ad3d901c3d5dfe7b671d2346ee5d62b0a00e7933a88a20a65966f60472e`. Each run had more than 32 seconds of normal cool baseline and passed camera, process and forwarding cleanup. The Android 15 emulator rejected the GPU-only probe because its vendor OpenCL library is absent; it produced no captured result and did not fall back to CPU. Its installed APK SHA256 was `e4e6902dc2955005622b5b9358e4aa507afdcea99f74b904ada515c5b2a17db9`. No APK was changed or installed.

The emulator subsequently passed two focused instrumentation tests: physical transmission cannot be enabled and an invalid USB descriptor cannot open a Panda. The device cleanup passed and the emulator was stopped. The focused host suite ran 26 tests successfully, with three skips; `git diff --check` passed. These tests do not exercise physical Panda or vehicle behavior.

Reproduce the host reference with a new output directory:

```sh
.venv/bin/python android/scripts/decompose_road_dense.py --output "$NEW_SECTIONS" --include-driver
.venv/bin/python -m unittest android.tests.test_road_dense_decomposition \
  android.tests.test_road_dense_recomposition android.tests.test_road_dense_opencl_fixture
```

The exact guarded device commands, fixture and result hashes, and three-seed mixed recomposition are in `android/build/candidates/candidate74-runtime/follow-up/road-dense-opencl-20261109/`; the complete host report and hashed derived graphs are in `android/build/candidates/candidate74-runtime/follow-up/ordered-dense-recompose-full-20261109/`. These are diagnostic artifacts only. Road execution on a device, full accelerator parity, controls cycle timing, three-camera freshness and identity, mounted calibration, and physical Panda checks remain unqualified. Vehicle TX and engagement remain disabled.

Status: **PARTIAL / NOT QUALIFIED**. The Linux SNPE converter works on the Chromebox. Subsequent Fold 3 diagnostics are recorded below. TX and engagement remain disabled; original tolerances and the 50 ms sensor-to-publication deadline are unchanged. No candidate74 APK or speed improvement is claimed.

## Continuation: 9 November — captured road dense GPU probe

The pinned first road Gemm (`model/node_linear`, 1024 inputs and 2048 outputs) uses eight ordered half-FMA lanes with half-rounded reductions and bias. The new standalone OpenCL diagnostic in `android/scripts/road_dense_ordered_half.cl` preserves those boundaries and compensates for FP32 double-rounding at half ties. On the Fold 3 it reproduced **all 2048 captured original outputs exactly for each of seeds 42, 7 and 123**, with no non-finite results or CPU fallback. The inputs and reference outputs come from captured original-model prefixes; trace extraction can alter other graph behavior. This test does not establish complete-model accuracy or integration.

One profiled execution on seed 123 recorded **19.959 ms** for this one GPU kernel. A separate run with one work item per accumulation lane (`road_dense_ordered_half_parallel.cl`) also matched all 2048 values exactly but recorded **28.237 ms** for its first event on the same seed and fixture, so that variant was not promoted. These single observations exclude compilation, transfers, synchronization with other sections, camera and output publication; they do not establish sustained latency, p99 or a statistically robust comparison. They indicate that this implementation still needs performance work, not an end-to-end benchmark.

The first Fold attempt failed at executable loading due to a dynamic C++ runtime mismatch; it ran no kernel. Rebuilding with `-static-libstdc++` fixed that diagnostic loader. On the Android 15 emulator the same probe failed to load the absent `/vendor/lib64/libOpenCL.so`, and did not fall back to CPU. All experiments used the guarded cooldown and cleanup wrapper, retained hashes and stopped the test processes. The Fold APK hash was `d06b27a435e1c2fa947066c7cc71efcaf810a053a57834c603d86c1722c74272`; the emulator APK hash was `e4e6902dc2955005622b5b9358e4aa507afdcea99f74b904ada515c5b2a17db9`. No APK was modified or installed.

Evidence, per-run kernel/binary and fixture hashes, and cleanup receipts are under `android/build/candidates/candidate74-runtime/follow-up/road-dense-opencl-20261109/`. The host suite ran 18 tests with three skips. The diagnostic is not wired into LiteRT or SNPE: the full road-model conversion still fails parity, the 795-comparison suite has not passed for it, and camera freshness, controls timing and 50 ms end-to-end operation remain unverified. Vehicle TX and engagement remain disabled.

Rebuild the standalone executable with the installed NDK, then invoke it only through the guarded experiment wrapper; replace serial and output paths for a new run:

```sh
"$HOME/Library/Android/sdk/ndk/27.2.12479018/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android24-clang++" \
  -std=c++17 -O2 -static-libstdc++ android/scripts/road_dense_opencl_probe.cpp -ldl -o "$NEW_PROBE"
.venv/bin/python -m unittest android.tests.test_road_dense_opencl_fixture \
  android.tests.test_litert_road_provenance android.tests.test_litert_conversion_isolation \
  android.tests.test_snpe_tools android.tests.test_litert_tools
.venv/bin/python android/scripts/experiment.py --serial "$FOLD_SERIAL" \
  --package com.samsung.sunnypilot.fold3 --report "$NEW_EXPERIMENT_JSON" -- \
  .venv/bin/python android/scripts/validate_road_dense_opencl.py --adb "$HOME/Library/Android/sdk/platform-tools/adb" \
  --serial "$FOLD_SERIAL" --probe "$NEW_PROBE" --kernel android/scripts/road_dense_ordered_half.cl \
  --fixtures "$CAPTURED_FIXTURES_DIR" --output "$NEW_RESULTS_DIR"
```

## Continuation: 9 November — LiteRT road delegation

The diagnostic road rewrite packs attention and fixed history operations to lower ranks and folds 20 finite, nonnegative float32 constant square roots. The resulting ONNX graph SHA256 `83ee22a19192d10972c03c3ea32a59185df49d55f42f886faad5e2e2efb119bf` passes all **636** original road comparisons, including recurrent and replay cases, at the unchanged tolerances. This is not the complete 795-comparison suite. The conversion runs on an isolated copy and produces LiteRT graph SHA256 `b92df1a7c769383ac61d270085c13d08e1329c40c7cc56d507d017d601756ecb`.

| Check | Result | Evidence and limits |
| --- | --- | --- |
| Fold 3 LiteRT FP32 full-graph compile | PASS, compile only | All 553 operations assigned to one OpenCL GPU partition; full non-CPU acceleration reported. No inference, accuracy or speed was measured. The separate diagnostic ran with `com.samsung.sunnypilot.fold3` APK SHA256 `d06b27a435e1c2fa947066c7cc71efcaf810a053a57834c603d86c1722c74272` installed. |
| Emulator full-graph compile | FAIL | All 553 operations were selected for the OpenGL delegate, but shader creation failed with `GL_INVALID_ENUM`. The test used installed APK SHA256 `e4e6902dc2955005622b5b9358e4aa507afdcea99f74b904ada515c5b2a17db9`; it did not fall back to CPU. |
| Converted LiteRT road, Linux CPU diagnostic | FAIL | Road output mismatches are 860 / 780 / 660 and feature-history mismatches are 50 / 22 / 10 for seeds 42 / 7 / 123. Image and desire histories match; no unexpected non-finite results. These are not Fold 3 GPU results. |
| First dense boundary, Linux CPU diagnostic | FAIL | Captured-input mean has zero tolerance violations and max absolute error 0.0000581; the subsequent FC value has 884 / 2048 violations. With the exact captured original input, the separately isolated widened-FP32 dense calculation has 872 violations, while the ordered-half diagnostic matches exactly. This is evidence of a precision-boundary problem, not proof that upstream differences or the Fold 3 GPU are innocent. |

The physical test passed the required cooldown and post-test cleanup; the app was stopped, cameras released and forwarding cleared. The emulator test was also cleaned up, and the emulator was shut down. The focused host suite ran 16 tests with three skips and `git diff --check` passed. This run did not retest controls, camera freshness, complete GPU inference, the full 795 comparisons, or vehicle hardware. TX and physical engagement remain disabled.

Machine-readable evidence and artifact hashes: `android/build/candidates/candidate74-runtime/follow-up/lowrank-sqrt-20261109/` plus the adjacent `lowrank-sqrt-20261109-{fold,emulator}-experiment.json` receipts. The exact device commands and diagnostic library hashes are in each compile `results.json`. Reproduce the rewrite with a new output directory:

```sh
.venv/bin/python android/scripts/prepare_litert_road.py \
  --source android/build/candidates/candidate74-runtime/snpe-road-float-transport/road-packed.onnx \
  --source-sha256 d46b41796da73cb708b4ffbc5f39fa9ac111ed302391bd4238f8192dd0ec4f1e \
  --source-receipt android/build/candidates/candidate74-runtime/snpe-road-float-transport/repair.json \
  --original "$PINNED_ORIGINAL_ROAD_ONNX" --reference android/app/build/model-reference \
  --fold-constant-sqrt --output "$NEW_REWRITE_DIR"
.venv/bin/python -m unittest android.tests.test_litert_road_provenance \
  android.tests.test_litert_conversion_isolation android.tests.test_snpe_tools android.tests.test_litert_tools
```

Convert with `android/scripts/convert_litert_isolated.py` on the prepared Linux environment, then use `android/scripts/validate_litert_road_host.py` for CPU diagnostic parity and `android/scripts/experiment.py` with `android/scripts/validate_litert_full_compile.py` for guarded device compilation. `android/scripts/trace_litert_road.py` accepts `--dense-fixtures` to compare the captured dense input. None of these commands promotes or qualifies the converted backend.

## Continuation: 9 November — physical precision probes

The following are standalone ADB-shell diagnostics, not APK runtime integration or road qualification. The Fold 3 had candidate73 APK SHA256 `d06b27a435e1c2fa947066c7cc71efcaf810a053a57834c603d86c1722c74272` installed throughout. Each physical experiment verified a clean baseline, at least 30 seconds of NORMAL thermal status with skin temperature below 32°C, unchanged APK, released cameras and no experiment forwards after cleanup. The app was stopped. Original tolerances remain `rtol=1e-3`, `atol=1e-4`.

| Diagnostic | Result | Evidence |
| --- | --- | --- |
| LiteRT default GPU precision | FAIL | FP16 Conv: 5 mismatches, max error 0.00048828125; FP32 Conv: 4 mismatches, max error 0.0003943667; cast passes. |
| LiteRT explicit FP32 GPU | PASS, primitives only | All three Conv/cast checks pass with no mismatches; maximum errors respectively 0.0000581145, 0, 0.0000000298. Full non-CPU acceleration reported. |
| LiteRT FP16 with FP32 accumulation | PASS, primitives only | All three checks pass with no mismatches; maximum errors respectively 0.000244140625, 0, 0.0001280457. Full non-CPU acceleration reported. |
| SNPE section execution | FAIL | Captured-seed vision output mismatches: 1573 / 1575 / 1572 of 1576; policy output mismatches: 952 / 942 / 952 of 2576. Image, desire and feature state outputs pass. GPU was requested but actual assignments are pending diagnostic decoding. |
| LiteRT full-road compile, Fold 3 | FAIL | 354 of 451 operations assigned to the GPU, with 97 unsupported including rank-five history operations. GPU-only compilation rejected the graph; no inference was performed. See the provenance correction below. |
| LiteRT full-road compile, emulator | FAIL | OpenGL shader generation emitted `no_type`; no inference was performed. The emulator was stopped after cleanup. Installed emulator APK SHA256 was `e4e6902dc2955005622b5b9358e4aa507afdcea99f74b904ada515c5b2a17db9`. |

The first SNPE attempt failed before execution because `adb shell` split a graph/output argument; the harness now quotes the complete remote command and has a focused argument-boundary regression test. The first explicit-precision build also failed before inference because its C++ runtime was missing; the successful probe is statically linked to that runtime. Both failures and their retries are retained.

### Verified-source packed-history continuation

The Mac-derived packed graph `d46b41796da73cb708b4ffbc5f39fa9ac111ed302391bd4238f8192dd0ec4f1e` records the pinned source hash and 636 passing road reference comparisons in `android/build/candidates/candidate74-runtime/snpe-road-float-transport/repair.json`. An isolated conversion produced LiteRT graph `7c7133c688ef5e1674168f5eeb186fa2c82e10764cc42014c9afe9cf1fba3654`; the converter changed its disposable input copy while the source stayed unchanged. The receipt is `android/build/candidates/candidate74-runtime/follow-up/packed-mac-conversion-receipt.json`.

The converted graph **fails** the three-seed host original-reference check: road output mismatches were 860, 780 and 660; feature-history mismatches were 50, 22 and 10. Image and desire histories were exact, with no non-finite outputs. Evidence: `packed-mac-host-parity.json`. This CPU diagnostic cannot qualify a GPU.

On the Fold 3, the same graph's **GPU-only compile failed**: 457 operations were assigned to the GPU and 152 remained unsupported; these included rank-five Slice/Reshape, Shape, invalid Transpose and other constraints. No inference or CPU fallback was accepted. The standalone diagnostic ran while `ai.sunnypilot.fold3` APK SHA256 `e5347db807f6551886d7de706d6ebb939c51dcd25c5c0ade1cd57b7fb24941ee` was installed; the separate `com.samsung.sunnypilot.fold3` package remained `d06b27a435e1c2fa947066c7cc71efcaf810a053a57834c603d86c1722c74272`. Evidence: `fold-litert-packed-20261109` and its experiment receipt. Cleanup passed with no app processes, camera ownership or forwards, after 32 seconds of normal cool baseline. The tests were outside either APK and do not qualify their model runtime.

The Android 15 emulator's OpenGL compile of this same graph also failed with 152 unsupported operations and no inference; its installed `com.samsung.sunnypilot.fold3` APK hash was `e4e6902dc2955005622b5b9358e4aa507afdcea99f74b904ada515c5b2a17db9`. Its experiment cleanup passed and the emulator was stopped. See `emulator-litert-packed-20261109` and its experiment receipt. The focused conversion, SNPE and LiteRT harness suite ran 13 tests successfully with three skips; `git diff --check` passed. No new APK was built or installed.

To repeat conversion, copy the Mac-derived ONNX to an isolated Linux work directory and run the following there, substituting paths without overwriting prior receipts:

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 onnx2tf-venv/bin/python convert_litert_isolated.py \
  --source original-packed.onnx \
  --source-sha256 d46b41796da73cb708b4ffbc5f39fa9ac111ed302391bd4238f8192dd0ec4f1e \
  --converter "$PWD/onnx2tf-venv/bin/onnx2tf" --output new-conversion
onnx2tf-venv/bin/python validate_litert_road_host.py \
  --model new-conversion/converted/original-packed_float32.tflite \
  --reference repo/android/app/build/model-reference --output new-host-parity.json --packed-history
```

The LiteRT probe selects the pinned 2.3.0 GPU precision through its opaque-options API and the [v2.3.0 GPU options serialization](https://github.com/google-ai-edge/LiteRT/blob/v2.3.0/litert/c/options/litert_gpu_options.cc). It always rejects CPU partitions and remains diagnostic. Probe source is `android/app/src/main/cpp/litert_runtime_probe.cpp`; the full-graph compile-only runner is `android/scripts/validate_litert_full_compile.py`. Original model and precision boundaries are unchanged. None of the passing primitives establishes recurrent, replay, 795-comparison, 50 ms, camera, controls or vehicle readiness.

Evidence under `android/build/candidates/candidate74-runtime/follow-up/`: `fold-litert-20261109`, `fold-litert-fp32-20261109` (loader failure), `fold-litert-fp32-static-20261109`, `fold-litert-mixed-20261109`, `fold-snpe-road-20261109` (argument failure), `fold-snpe-road-20261109-retry`, `fold-litert-fullcompile-20261109`, `emulator-litert-fullcompile-20261109`, and their corresponding `*-experiment.json` files. These include exact commands and hashes. Build the precision probe with:

```sh
"$HOME/Library/Android/sdk/ndk/28.2.13676358/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android28-clang++" \
  -std=c++17 -O2 -static-libstdc++ -Iandroid/build/native/vendor-runtimes/litert-headers \
  android/app/src/main/cpp/litert_runtime_probe.cpp \
  -Landroid/build/native/vendor-runtimes/litert-android -lLiteRt \
  -o android/build/candidates/candidate74-runtime/follow-up/litert-precision-compile-probe
```

## Continuation: 9 November — full LiteRT execution

Status: **PARTIAL / NOT QUALIFIED**. The Chromebox has a separate onnx2tf 2.6.5 environment with its pinned LiteRT 2.1.2 converter dependency; the existing LiteRT 2.3.0 environment and runtime are unchanged. Its TensorFlow-free direct converter produced both sections and the complete observed Chromebox road graph. The unmodified converted full graph fails CPU invocation on negative GatherND indices. The contiguous GatherND-to-Slice rewrite removes the failure. ONNX CPU comparison of the observed Chromebox graph and its rewritten graph is bit-exact for all four outputs across all three captured seeds; it does not establish provenance from the pinned Mac model or accelerator parity.

**Provenance correction:** the Mac's pinned original is SHA256 `65a08adc31d5c456219687d99b7bf5e44d61dae2d49ea67850e76105c7248cce`, as also recorded in the model-reference manifest. The Chromebox source used in these earlier conversions instead hashes to `bc713de900d8a4369a41dcff09a0d62a77d55a66374c2f2e70a3b1995844dd8c` and differs in graph structure. Thus the earlier ONNX rewrite comparison and the phone/emulator full-compile results above are diagnostics for that observed graph only, not proof for the pinned model. We observed onnx2tf mutate a separate staged ONNX input in place; this establishes a concrete provenance hazard, but does not establish when the earlier file changed. Original reference captures themselves matched between machines. Earlier artifacts remain intact. New conversions must use `android/scripts/convert_litert_isolated.py` with the expected input hash; it protects the source with a disposable copy and records input mutation and output hashes.

The rewritten full LiteRT graph executes all four outputs on Linux CPU across three seeds. At the original tolerances, `outputs` has 1894 / 1787 / 2000 mismatches and `next_state_feat_q` has 388 / 372 / 425. Image and desire history have zero mismatches; no non-finite values occurred. No recurrent or replay qualification was attempted after these failures. The input layout adapters and hashes are recorded by `android/scripts/validate_litert_road_host.py`. The host CPU run is diagnostic and never qualifies a backend.

Earlier converted vision prefixes pass through input scaling, first convolution/GELU and mean within tolerance. The first tested dense output (`model/linear`) fails with 884 mismatches, consistent with the SNPE ordered-half accumulation diagnosis. The capture was generated from a derived SNPE diagnostic graph, and trace partitioning can change behavior; full-graph parity remains the acceptance gate. A custom precision-preserving operation or validated split remains to be implemented and tested. No SNPE repair is claimed.

The emulator rerun passed the three existing GPU-only LiteRT primitive probes with candidate73 APK SHA256 `d06b27a435e1c2fa947066c7cc71efcaf810a053a57834c603d86c1722c74272`. `experiment.py` recorded 33.8 seconds of normal cool baseline, no owned camera/processes/forwards afterward, and an unchanged APK. The emulator was then shut down. The Fold 3 was visible on ADB but was not used, per the current emulator-only request. No APK was built and no 50 ms or full-model GPU qualification was established.

Evidence: `android/build/candidates/candidate74-runtime/follow-up/onnx2tf-20261109/` and `emulator-litert-20261109-experiment.json`. The converted graphs remain on the Chromebox under `/home/dey/sunnypilot-fold3/work/onnx2tf-20261109/`; converted full Slice model SHA256 is `322edc9a80fd64f72e346b1a5a6869cdc02b7338213020746d2dbad579d31172`. The observed Chromebox source SHA256 is `bc713de900d8a4369a41dcff09a0d62a77d55a66374c2f2e70a3b1995844dd8c`, not the pinned Mac original. Converter package versions are in `requirements.txt` in that evidence folder.

Reproduce the full-graph host comparison on the Chromebox (use a new output path each time):

```sh
cd /home/dey/sunnypilot-fold3
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 onnx2tf-venv/bin/python \
  repo/android/scripts/validate_litert_road_host.py \
  --model work/onnx2tf-20261109/full-road-slice/full-road-slice_float32.tflite \
  --reference repo/android/app/build/model-reference \
  --output work/onnx2tf-20261109/full-road-slice-parity-NEW.json
```

TX and engagement remain disabled; the 50 ms deadline and all original tolerances remain unchanged. Thermal safeguards remain active.

## Latest continuation: LiteRT repair and Chromebox preparation

This section supersedes the older status table below; earlier failures remain recorded.

| Check | Result | Evidence / limitation |
|---|---|---|
| LiteRT explicit-half Conv with FP32 external transport | PASS, emulator primitive | All 7 nodes delegated to OpenGL; original FP16 input, weight, bias and output boundaries retained; zero mismatches and zero maximum error |
| LiteRT GPU probes | PASS, emulator | 3/3 Conv/cast probes at original tolerances; standalone diagnostics, not full-road or physical GPU qualification |
| SNPE road CPU diagnostic | FAIL accuracy | Both sections execute; vision and policy outputs diverge. Recurrent queue outputs match. CPU is explicitly diagnostic, never a fallback qualification |
| SNPE first dense layer | ROOT CAUSE PROVEN, repair pending | Identical captured inputs: 872 mismatches, max error 0.00205767. Output half-casting still leaves 901 mismatches. SNPE matches widened FP32 accumulation; ordered eight-lane half accumulation exactly matches all 2,048 original captured values |
| Mac host suite | PASS | 280 tests, 3 Linux-only skips; original goldens/tolerances and 795-comparison suite preserved |
| Linux host suite | FAIL / architecture blockers | 280 tests, 4 failures, 2 errors, 27 skips after fixing native compiler runtime lookup. Original ARM driver references differ on x86, and ARM-generated NhwcFusedConv graphs cannot run on x86. Goldens were not regenerated |
| Emulator NativeBridge instrumentation | PASS | 8/8 tests on unchanged candidate73 APK; TX restrictions, USB rejection and IPC covered |
| Final emulator isolation | PASS | No app/test processes, owned cameras or forwards; NORMAL thermal status; emulator subsequently shut down |
| Chromebox continuation tools | PASS installed | Isolated QAIRT/converter, LiteRT, matching host Python packages, JDK21, Gradle8.9, SDK35, NDK28.2, CMake, emulator/image, private GCC/Clang and Linux IPC/params bindings; Gradle and ARM64 cross-compilation probes pass |
| Chromebox emulator acceleration | BLOCKED | /dev/kvm is not writable by the current user; no permissions or system services changed |
| Full-model qualification / 50 ms | BLOCKED | No new APK, full SNPE/LiteRT accelerator parity or physical timing pass; no combined workload advancement |

New evidence is under `android/build/candidates/candidate74-runtime/follow-up/`: `host-tests-complete.log`, `emulator-litert-float-transport/results.json`, `emulator-nativebridge.log`, `emulator-final-cleanup.json`, `snpe-vision-trace-results/`, `snpe-vision-dense-trace-results/`, `snpe-dense-isolated-original-capture/` and `snpe-dense-rounding-diagnosis.json`. Linux receipts and remaining failures are collected under `follow-up/chromebox/`.

The working LiteRT FP16 Conv graph uses lossless FP32 external transport around explicit half casts, NHWC conversion and float Conv with exactly lifted half weights/bias. This avoids the emulator shader generator's external-half bug. It does not establish full-model intermediate accumulation accuracy. Unsupported geometry is rejected explicitly.

The SNPE repair requires preserving demonstrated intermediate half accumulation, for example a supported custom GPU operation or an accurately validated split backend. Global FP16 conversion currently fails runtime initialization; global FP32 conversion and a final output cast are not accepted substitutes. See [Chromebox continuation commands](../../CHROMEBOX.md).

## Current results

| Check | Result | Evidence / limitation |
|---|---|---|
| Linux SNPE converter | PASS | QAIRT 2.47.0.260601, isolated Python 3.10.22 on Ubuntu 26.04.1 x86_64; help, nine primitives and repaired vision/policy DLC conversion succeed |
| Vendor-supported host | NOT SUPPORTED | Ubuntu 26.04 is outside the published 22.04/24.04 host matrix; successful conversion is observed, not vendor certification |
| Build cache cleanup | PASS | Regenerable intermediates, .cxx and project .gradle removed; exact APK archive, original models, goldens and earlier evidence preserved |
| Host checks | PASS | 275 tests; fresh untouched section recomposition passes all 795 original-reference comparisons |
| Repaired road graph | PASS on host | All 636 road comparisons including recurrence/replay; packed history, constant GatherND lowering, proven Expand-to-Tile and lossless float image transport |
| SNPE mixed GPU primitives on Fold3, before disconnection | PASS | 9/9 at original tolerances; detailed logs identify GPU execution for every layer |
| SNPE GPU_FLOAT16 primitives on Fold3 | FAIL | 7/9 pass; FP32 and FP16 convolution probes have 4 and 5 mismatches respectively |
| Initial SNPE road sections on Fold3 | FAIL | Rank-five StridedSlice and constant GatherND initialization failures; retained evidence |
| Repaired SNPE sections | CONVERTED / DEVICE BLOCKED | Subsequent Expand type conflict repaired by byte-copy Tile; uint8 Concat failure addressed by proven lossless transport. Both final DLCs convert; accelerator outputs still unverified |
| LiteRT 2.3.0 emulator GPU | PARTIAL | FP32 convolution and FP32→FP16→FP32 cast match original goldens, GPU-only assignment confirmed |
| LiteRT native FP16 Conv | FAIL | Runtime rejects FP16 Conv preparation before delegation |
| LiteRT explicit-half Conv variant | FAIL on emulator | Host parity passes. All five nodes assigned OpenGL but shader generation fails: `convert<half> requires one argument, but 0 was passed`. No CPU result accepted |
| SNPE emulator GPU | BLOCKED | Static probe loads and reports GPU_FLOAT16 unavailable, exit 5. Initial missing libc++_shared loader failure retained; corrected by static C++ probe linkage |
| Experiment isolation | PASS | Processes stopped, owned remote directories removed, no camera ownership or forwarding, stable provider PIDs; 30 continuous seconds NORMAL thermal status before each experiment |
| Full alternative-runtime model qualification / speed | BLOCKED | No 795-comparison SNPE/LiteRT device pass, no 50 ms qualification; long combined workloads not advanced |

All new runtime experiments are standalone ADB-shell diagnostics, **not APK backend integration**. The emulator's installed candidate73 APK remained `d06b27a435e1c2fa947066c7cc71efcaf810a053a57834c603d86c1722c74272`. Earlier physical primitive tests were bound to installed APK `e5347db807f6551886d7de706d6ebb939c51dcd25c5c0ade1cd57b7fb24941ee`. Shell diagnostics cannot qualify either application's accelerator interface. No Fold3 experiments were attempted after the user requested emulator-only work.

## Repairs and evidence

`prepare_snpe_road.py` reuses the existing proven transformations. Float image transport changes only losslessly representable byte data movement, retaining FP16 arithmetic and cast boundaries; it is not global FP32 inference. Its original-reference pass includes exact uint8 state restoration. `prepare_snpe_sections.py` records explicit transport layouts and input dtypes. `validate_snpe_sections.py` verifies DLC public tensor contracts and requests **all** public output tensors explicitly. Converter results and diagnostic graphs are separately hashed. Host correctness cannot establish SNPE arithmetic correctness.

Evidence root: `android/build/candidates/candidate74-runtime/`:

- `converter-host.json`, `snpe-linux-requirements.txt`, `litert-linux-requirements.txt`, `linux-toolchain-artifacts.txt`.
- `cache-cleanup.json`, `native-cache-cleanup.json`, `host-tests-final.log`, `final-host-recomposition/host-recomposition.json`.
- `snpe-primitives/`, `snpe-primitives-decoded/`, `snpe-primitives-assignment.json`, `snpe-sections/`; earlier failures remain intact.
- `snpe-road-float-transport/repair.json`, `road-float-transport-dlc/conversion.json`, `snpe-float-transport-fixtures/`.
- `emulator-litert-experiment.json`, `emulator-litert-explicit-half-experiment.json`, their corresponding directories and logs.
- `emulator-snpe-experiment.json` (loader failure), `emulator-snpe-static-experiment.json`, `emulator-snpe-static-probe.json` (runtime unavailable).

## Reproduce this follow-up

Chromebox workspace: `/home/dey/sunnypilot-fold3`. No sudo, Docker changes, system package installation or service changes were used. SDK ZIP SHA256 is `d3497e110eae82c35a9152a93c0a18bbede402aaf9faa7a97c8079eb0f522b01`. The private libc++/libc++abi/libunwind packages were downloaded with `apt download` and extracted with `dpkg-deb -x` under `sysroot`; exact package hashes are retained in `linux-toolchain-artifacts.txt`. User-managed uv 0.13.7 supplies Python 3.10.22; converter dependencies are frozen separately from the Python 3.12.16 LiteRT environment.

```sh
# Chromebox, after the pinned ZIP, private libraries and venv are staged:
converter-venv/bin/python setup_snpe_linux.py --root /home/dey/sunnypilot-fold3 --check-only
converter-venv/bin/python convert_snpe_linux.py --root /home/dey/sunnypilot-fold3 \
  --input work/road-float-transport --output work/road-float-transport-dlc-NEW

# Mac: use new output directories, preserving earlier artifacts.
.venv/bin/python android/scripts/prepare_snpe_road.py --float-image-transport \
  --output android/build/candidates/candidate74-runtime/snpe-road-float-transport-NEW
.venv/bin/python android/scripts/prepare_snpe_sections.py \
  --sections android/build/candidates/candidate74-runtime/snpe-road-float-transport \
  --output android/build/candidates/candidate74-runtime/snpe-float-transport-fixtures-NEW
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
  .venv/bin/python -m unittest discover -s android/tests
```

Exact guarded emulator commands, artifact hashes and results are recorded in each `*-experiment.json`. Native diagnostics use the NDK 28.2 ARM64 compiler; the corrected SNPE probe adds `-static-libstdc++`. LiteRT uses its CompiledModel C API with GPU-only options and verifies full non-CPU acceleration before execution. Registration of a CPU provider does not qualify fallback; any rejected GPU graph remains failed.

Cooling optimization stays deferred. Controls timing, camera latency/identity, mounted calibration, Panda and physical vehicle checks remain open. The previous measurements and failures below are historical and are not overwritten by conversion or emulator success.

---

# Road decomposition and alternative-runtime preparation

Status: **PARTIAL / NOT QUALIFIED**. Candidate73 is preserved. This work does not claim faster inference or a completed candidate74 APK. The original accuracy tolerances, 50 ms sensor-to-publication deadline, TX prohibition and thermal safeguards are unchanged.

## Results

| Check | Result | Evidence / limit |
|---|---|---|
| Original road graph dependency cut | PASS | 272 vision nodes, 105 policy/state nodes; original nodes and weights retained |
| Host recomposition | PASS | 636 road comparisons, all four public outputs, three seeds, recurrent sequences and pinned 60-frame replay |
| Untouched driver host regression | PASS | 159 comparisons; combined host total **795**, `rtol=1e-3`, `atol=1e-4` |
| Focused/full host suite | PASS | 271 tests, including crossing edges, state, FP16 boundary preservation, corrupt downloads and invalid HTTP ranges |
| Official runtime distributions | PASS | QAIRT/SNPE 2.47.0.260601 and LiteRT 2.3.0 fetched selectively; hashes retained and primary runtime hashes pinned |
| Fold3 SNPE GPU availability | PASS, capability only | Native probe reports `GPU_FLOAT16` available in **ADB shell**, not app context; no model executed |
| Probe artifact integrity and teardown | PASS | Uploaded binary/library hashes verified; owned remote directory removed; experiment receipt records camera/provider/process/forward cleanup |
| SNPE primitive/section/full-model execution | BLOCKED | DLC conversion environment unavailable locally; no converted DLC or native model adapter is qualified |
| LiteRT conversion and GPU execution | NOT TESTED | Official AARs available; no original-model TFLite conversion, CompiledModel adapter or no-fallback proof yet |
| Split GPU latency / full original-node GPU attribution | NOT TESTED | Host section timings are CPU wall times, not GPU profiles; converted fused-layer provenance remains to be completed |
| One-versus-two accumulator follow-up | NOT TESTED | Earlier candidate evidence retained; no new comparison claimed |
| Candidate74 APK / emulator rerun | BLOCKED / NOT BUILT | Native model adapters and converted artifacts remain pending; host free space approximately 1.1 GiB, below reliable APK/emulator build needs |
| Combined 30 s / 2 min / 30 min | BLOCKED | No new accelerator accuracy, timing or three-camera freshness qualification |

Detailed machine evidence: `android/build/candidates/candidate74-runtime/`. In particular `road-sections/manifest.json`, `road-sections/host-recomposition.json`, `host-tests.log`, `fold-snpe-probe.json`, `fold-snpe-experiment.json`, `final-cleanup.json` and `artifacts.json`.

Final cleanup passed after 30 continuous seconds of normal thermal status. Ripwire quality-delta did not return a result and was cancelled; no clean static-analysis result is claimed.

## Exact cut and arithmetic

The split is an ancestor-closure cut at `model/vision_outputs`, not a hand-picked layer range. It exports every value required by the remaining graph or by a public output. The crossing tensors are **`model/vision_outputs` (FP16, 1×1576)** and **`next_state_img_q` (uint8, 2×5×6×128×256)**. Policy receives the current vision value and its original side/state inputs. Public outputs produced by vision are retained alongside policy outputs. Every invocation recomputes vision; there is no stale-feature reuse.

The manifest records original node names, inputs/outputs, section membership, tensor shapes/dtypes, operator counts, original model hash and separately hashed extracted graphs. No Cast, transpose, accumulation, precision or rounding rewrite is introduced. Host recomposition includes real state feedback, with existing original goldens and tolerances. This proves extraction correctness only; it does not establish accelerator accuracy or speed.

## Runtime findings and blockers

The official SDK ZIP is approximately 2.35 GB. The preparation script uses validated HTTP ranges and ZIP CRC checking to extract selected headers, Android libraries and the converter entry point without storing the whole archive. The downloaded `libSNPE.so` hash is `6d43b43550e737aedda14d1ac3a85632b60525cd4399e518f260ecd02730059c`.

Running the selected `snpe-onnx-to-dlc --help` locally fails with `No module named 'qti'`. The entry point imports `qti.aisw` converter packages and native converter libraries. The SDK directory listing contains Linux aarch64/x86_64 and Windows native converter builds, not a macOS arm64 build. Docker is installed but its daemon is not running. Thus **SDK download access is available; a functioning converter execution environment is the blocker**. No unsupported macOS conversion or different model was substituted. Next work is to establish a supported Linux conversion environment, extract its matching converter dependencies, generate explicit-precision primitive DLCs, then validate GPU-only assignment and outputs before sections or full models.

LiteRT core AAR SHA256: `9508c6e9142646f0f2bc1fe68b3925fb470ee7ddd29114540339a04a8ab189dc`; GPU AAR SHA256: `173a4c3813601f7e9542f4e6a49e237775e2f9ce1e0948f35407098bffb1824f`. Their presence is not evidence of a converted model or GPU-only execution. No TensorFlow, ONNX-to-TFLite, ai-edge-torch or LiteRT conversion package is currently present in the host venv. Do not promote a requested GPU option without operator assignment and all original-reference checks.

The Fold3 probe was bound to installed APK `e5347db807f6551886d7de706d6ebb939c51dcd25c5c0ade1cd57b7fb24941ee`, unchanged throughout its experiment. This **is not candidate73**. The probe runs independently of app inference; it cannot validate either APK's accelerator interface. Candidate73's archived APK remains `d06b27a435e1c2fa947066c7cc71efcaf810a053a57834c603d86c1722c74272`.

## Timing interpretation

Candidate73 remains the measured baseline: unprofiled road median 309.712 ms, p99 352.664 ms at diagnostic dispatch524288. Its separately instrumented GPU layers total about 190.84 ms, including convolution 117.29 ms and dense 45.77 ms. These are retained earlier measurements, not a new candidate74 benchmark. GPU layer time and host/internal submission waits overlap and must not be added. Host handling below 2 ms does not explain the full inference cost. New host section timings include CPU execution and ORT boundary handling and cannot be compared as Fold3 speedups.

Controls complete-cycle, three-camera identity/freshness, mounted calibration and disconnected Panda/vehicle checks remain separate blockers. Cooling optimization is deferred. No controls or live perception implementation was changed by this host decomposition/capability experiment; TX and engagement remain disabled.

## Reproduce

```sh
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk \
  .venv/bin/python -m unittest discover -s android/tests
.venv/bin/python android/scripts/decompose_road.py --include-driver \
  --output android/build/candidates/candidate74-runtime/road-sections
.venv/bin/python android/scripts/prepare_vendor_runtimes.py \
  --output android/build/native/vendor-runtimes

"$ANDROID_HOME/ndk/28.2.13676358/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android28-clang++" \
  -std=c++17 -O2 -Iandroid/build/native/vendor-runtimes/snpe/include/SNPE \
  android/app/src/main/cpp/snpe_runtime_probe.cpp -ldl \
  -o android/build/candidates/candidate74-runtime/snpe-runtime-probe

.venv/bin/python android/scripts/experiment.py \
  --adb "$ANDROID_HOME/platform-tools/adb" --serial RFCR81Y1TBJ \
  --report android/build/candidates/candidate74-runtime/fold-snpe-experiment.json -- \
  .venv/bin/python android/scripts/probe_snpe_runtime.py \
  --adb "$ANDROID_HOME/platform-tools/adb" --serial RFCR81Y1TBJ \
  --probe android/build/candidates/candidate74-runtime/snpe-runtime-probe \
  --library android/build/native/vendor-runtimes/snpe/lib/aarch64-android/libSNPE.so \
  --output android/build/candidates/candidate74-runtime/fold-snpe-probe.json
```

## Primary references

- [Flowpilot SNPE runner](https://github.com/flowdriveai/flowpilot/blob/59f5cf3ff0fb5ca7b483b146f3bf4fee30239410/android/src/main/java/ai.flow.android/vision/SNPEModelRunner.java): GPU_FLOAT16 selection, initialization caching and reusable tensors; architectural reference, not proof of current-model parity or absence of CPU partitions.
- [Qualcomm SDK download documentation](https://docs.qualcomm.com/bundle/publicresource/topics/80-70015-15B/snpe-download.html) and [Google's pinned QAIRT download](https://github.com/google-ai-edge/LiteRT/blob/main/third_party/qairt/workspace.bzl).
- [Jetlink Android architecture](https://github.com/zoompilot/jetlink/blob/9be48b18c1fa1f6f8b88ef2e14fb4aa85c54a83b/android/README.md): conversion/layout reference only; its parity criteria are not adopted.
- [Official LiteRT GPU interfaces](https://ai.google.dev/edge/litert/next/gpu).
- [Phr00t Thneed runner](https://github.com/phr00t/flowpilot/blob/84f25d0626bfbada0706b46b4b7f68a605b9ff8c/android/src/main/java/ai.flow.android/vision/THNEEDModelRunner.java): architectural reference only. No bundled models or opaque binaries imported.
