
# pytorch踩坑记录

## pytorch导出大模型，显示ValueError: The generation config instance is invalid

- 错误：

```bash
During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/xx/llama_factory/LLaMA-Factory/src/export_model.py", line 9, in <module>
    main()
  File "/home/xx/llama_factory/LLaMA-Factory/src/export_model.py", line 5, in main
    export_model()
  File "/home/xx/llama_factory/LLaMA-Factory/src/llmtuner/train/tuner.py", line 67, in export_model
    model.save_pretrained(
  File "/home/xx/miniconda3/envs/llama_factory/lib/python3.10/site-packages/transformers/modeling_utils.py", line 2364, in save_pretrained
    model_to_save.generation_config.save_pretrained(save_directory)
  File "/home/xx/miniconda3/envs/llama_factory/lib/python3.10/site-packages/transformers/generation/configuration_utils.py", line 560, in save_pretrained
    raise ValueError(
ValueError: The generation config instance is invalid -- `.validate()` throws warnings and/or exceptions. Fix these issues to save the configuration.

Thrown during validation:
[UserWarning('`do_sample` is set to `False`. However, `temperature` is set to `0.9` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `temperature`.'), UserWarning('`do_sample` is set to `False`. However, `top_p` is set to `0.6` -- this flag is only used in sample-based generation modes. You should set `do_sample=True` or unset `top_p`.')]
```

- 解决方案：
删除模型目录下的 generation_config.json。 ref: https://github.com/hiyouga/LLaMA-Factory/issues/2545
