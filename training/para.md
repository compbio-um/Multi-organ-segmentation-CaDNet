## Parameter Recommendation

The optimal CaDNet hyperparameters may vary depending on the complexity of the segmentation task, the number of target classes, and the anatomical structures of interest. Based on our empirical evaluation, we provide the following parameter recommendations as practical starting points.

### Recommended Settings for Different Tasks

For **complex multi-organ segmentation tasks with a relatively large number of target classes**, such as 16-class multi-organ segmentation, we recommend using a stronger random-convolution intervention and a deeper feature alignment module:

```json
{
    "interm_channel_randconv": 16,
    "kernel_sizes_randconv": [
        1,
        3,
        5
    ],
    "n_layer_randconv": 4,
    "channel_list_align": [
        16,
        8,
        4,
        1
    ],
    "kernel_sizes_align": [
        3,
        3,
        3,
        3
    ],
    "strides_align": [
        1,
        1,
        1,
        1
    ],
    "paddings_align": [
        1,
        1,
        1,
        1
    ]
}
```

A deeper alignment module can provide greater representational capacity for complex multi-organ segmentation, where different organs exhibit diverse shapes, sizes, and appearance characteristics.

For **single-organ or relatively less complex segmentation tasks**, a shallower alignment module is generally sufficient. For example:

```json
{
    "interm_channel_randconv": 16,
    "kernel_sizes_randconv": [
        1,
        3,
        5
    ],
    "n_layer_randconv": 3,
    "channel_list_align": [
        16,
        1
    ],
    "kernel_sizes_align": [
        3,
        3
    ],
    "strides_align": [
        1,
        1
    ],
    "paddings_align": [
        1,
        1
    ]
}
```

This lightweight configuration was found to be effective for tasks such as **MSD Spleen and ACDC**.

### Empirical Parameter Selection

The following configurations achieved the best performance in our experiments:

| Dataset / Task | `interm_channel_randconv` | `kernel_sizes_randconv` | `n_layer_randconv` | `channel_list_align` |
| -------------- | ------------------------: | ----------------------- | -----------------: | -------------------- |
| LGE            |                         8 | [1, 3, 5]               |                  2 | [16, 8, 4, 1]        |
| bSSFP          |                         8 | [1, 3, 5]               |                  2 | [16, 8, 4, 1]        |
| AMOS-CT        |                        16 | [1, 3, 5]               |                  4 | [16, 8, 4, 1]        |
| AMOS-MRI       |                        16 | [1, 3, 5]               |                  4 | [16, 8, 4, 1]        |
| MSD Liver      |                         8 | [1, 3, 5]               |                  3 | [16, 8, 4, 1]        |
| MSD Spleen     |                        16 | [1, 3, 5]               |                  3 | [16, 1]              |
| ACDC           |                        16 | [1, 3, 5]               |                  3 | [16, 1]              |



### General Guidelines

Based on these experiments, the following empirical guidelines can be used:

* **Complex multi-organ segmentation:**
  `interm_channel_randconv = 16`, `n_layer_randconv = 4`, and `channel_list_align = [16, 8, 4, 1]`.

* **Moderate-complexity organ segmentation:**
  `interm_channel_randconv = 8–16`, `n_layer_randconv = 2–3`, and `channel_list_align = [16, 8, 4, 1]`.

* **Single-organ or relatively simple segmentation:**
  `interm_channel_randconv = 16`, `n_layer_randconv = 3`, and `channel_list_align = [16, 1]`.

These settings are intended as empirical recommendations based on our experiments. Task-specific hyperparameter tuning may further improve performance for individual datasets.
