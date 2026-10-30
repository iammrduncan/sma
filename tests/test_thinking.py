"""SMA reads a model's thinking default from its chat template (sma.thinking)."""

from sma.thinking import thinking_default

# The shape of LFM3's template: it thinks unless told not to, and marks the assistant's turn with
# transformers' {% generation %} tag.
LFM3_LIKE = (
    "{%- set enable_thinking = enable_thinking | default(true) -%}"
    "{%- for m in messages -%}<|im_start|>{{ m.role }}\n"
    "{%- if m.role == 'assistant' -%}{%- generation -%}{{ m.content }}{%- endgeneration -%}{%- else -%}{{ m.content }}{%- endif -%}"
    "<|im_end|>\n{%- endfor -%}"
    "{%- if add_generation_prompt -%}<|im_start|>assistant\n"
    "{%- if enable_thinking -%}<think>{%- else -%}<think></think>{%- endif -%}{%- endif -%}"
)


def test_a_template_with_a_generation_block_still_tells_its_thinking_default() -> None:
    # On a Spark, LFM3's template failed here (unknown tag 'generation'), so its KernelBench, SWE-bench,
    # and BrowseComp-Plus requests ran with the quality server's thinking off.
    assert thinking_default(LFM3_LIKE) == {"enable_thinking": True, "how": "the chat template thinks unless told not to"}
