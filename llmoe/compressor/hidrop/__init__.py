from llmoe.compressor.hidrop.hidrop import (
    BACKENDS,
    HiDropCompressor,
    HiDropBackend,
    LlavaHiDropBackend,
    QwenHiDropBackend,
    backend_for_training,
    detect_backend,
    training_compressor_args,
    make_hf_prompt_upstream_tokenization_compatible,
    resolve_checkpoint_dir,
)
from llmoe.compressor.hidrop.data import (
    LlavaHiDropCollator,
    LlavaJsonDataset,
    QwenHiDropCollator,
    tokenize_prompt_with_image_span,
    resolve_training_data,
)
from llmoe.compressor.hidrop.config import (
    CONFIG_METADATA_KEYS,
    DTopKResult,
    DifferentiableTopK,
    HIDROP_EVAL_ARGS,
    HIDROP_IMPLEMENTATION_VERSION,
    HIDROP_PROFILES,
    HIDROP_UPSTREAM_COMMIT,
    LLAVA_REFERENCE_DEPTH,
    QWEN_PROFILES,
    HiDropConfig,
    HiDropDebugger,
    QwenHiDropConfig,
    ReleasedCurve,
    average_retained_tokens,
    dtopk,
    derive_keep_ratios,
    rank_normalize_scores,
    scale_layer,
    stage_layer_weights,
    visual_token_counts_by_layer,
    visual_token_ratios_by_layer,
)
from llmoe.compressor.hidrop.hidrop_llama import HiDropForwardContext, HiDropLlamaModel, VisualTokenSpan
from llmoe.compressor.hidrop.hidrop_llava_next import (
    LLAVA_NEXT_HIDROP_IMPLEMENTATION_VERSION,
    LLAVA_NEXT_LAYOUT_VERSION,
    LlavaNextVisualLayout,
    ResolvedHiDropSchedule,
    resolve_scaled_schedule,
    resolve_visual_layout,
)
from llmoe.compressor.hidrop.hidrop_next_tr import (
    HiDropTrainingLlavaNextForConditionalGeneration,
    LlavaNextHiDropCollator,
    LlavaNextParquetDataset,
    configure_llava_next_trainable_parameters,
    resolve_llava_next_data_root,
)
from llmoe.compressor.hidrop.hidrop_qwen import (
    HiDropQwenForwardContext,
    HiDropQwenModel,
    QwenVisualTokenSpan,
    resolve_mrope_section,
)
from llmoe.compressor.hidrop.hidrop_tr import (
    HiDropTrainingLlavaForConditionalGeneration,
    HiDropTrainingModel,
    finetune_hidrop,
    hidrop_ft_fn_cd,
    hidrop_tr_fn_cd,
    train_hidrop,
)
from llmoe.compressor.hidrop.provenance import (
    RELEASED_TRAINING_SEMANTICS,
    validate_inference_checkpoint_provenance,
)
