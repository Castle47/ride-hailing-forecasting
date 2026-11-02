# Sources and third-party notices

This is a standalone curated project, not a fork or a vendored copy of TSLib.

- `models/Autoformer.py`, `layers/AutoCorrelation.py`, `layers/Autoformer_EncDec.py` and shared embeddings originate from the official Autoformer / supervised PatchTST code lineage. Autoformer: https://github.com/thuml/Autoformer (MIT).
- `models/PatchTST.py`, `layers/PatchTST_backbone.py`, `layers/PatchTST_layers.py`, `layers/RevIN.py` originate from https://github.com/yuqinie98/PatchTST, supervised implementation, reference commit `204c21efe0b39603ad6e2ca640ef5896646ab1a9`. Original project Apache-2.0 license retained in `LICENSE`.
- `models/TimeXer.py` and `layers/TimeXer_Embed.py` are project adaptations of https://github.com/thuml/TimeXer, originally retrieved 2026-10-22. The original model source hash and embedding hash are recorded in `UPSTREAM.json`. Multi-target selection and validation are project extensions.
- Project-authored CEEMDAN fusion code: `layers/CEEMDAN_decomp.py`, `layers/CEEMDAN_PatchTST_backbone.py`, `models/CEEMDANPatchTST.py`. CEEMDAN is an existing published algorithm; the implementation uses the external `EMD-signal` / PyEMD package under its own license (Apache-2.0).
- Project-authored DLL-TimeXer: `models/DLLTimeXer.py`, based on the user's chapter-five design. Shared adapter, data pipeline, training/inference scripts and new tests were written for this standalone project.

Modifications to adapted Autoformer: input-device allocations, explicit inverse FFT length for odd sequences, bounded top-k for tiny windows, optional time markers in embeddings. Original copyrights and license terms remain applicable. Original algorithms and papers are not claimed as this project's inventions.

Autoformer MIT copyright notice is preserved in `LICENSE.Autoformer`. The standalone TimeXer repository has no root license file as checked on 2026-11-02; its license is not inferred from other THUML repositories. The project Apache license does not override third-party rights or grant a separate license to these upstream portions.
