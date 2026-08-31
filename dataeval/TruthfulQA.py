"""
Minimal stub for TruthfulQA dataset module.
This file exists so `pipeline.generate` can import it. It raises a clear
error if `get_dataset` is invoked, since the dataset isn't bundled here.
"""
import _settings


def get_dataset(tokenizer, split='validation'):
    raise NotImplementedError("TruthfulQA dataset is not available in this workspace. Use --dataset coqa or add TruthfulQA data module.")


def _generate_config(tokenizer):
    # Return a minimal generation config used only if TruthfulQA is selected.
    return dict(eos_token_id=[getattr(tokenizer, 'eos_token_id', None)], bad_words_ids=[])


if __name__ == '__main__':
    print('TruthfulQA stub module. Not intended to be executed directly.')
