# Evaluating Hallucinations in AI-Generated Storyboards for Human-Centered Design

![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![License](https://img.shields.io/badge/license-MIT-green)
---

## Overview

This is an image splitter built to take an image, split it, and return the results. It is specifically optimised for storyboard images which have inner smaller panels, and so, usually a border. This tool is optimised to detect those borders, and subtract the panel.

---

## Requirements

### Hardware
- A CUDA-capable GPU is **strongly recommended**. The pipeline can run on CPU, but inference with Qwen3-VL-2B will be significantly slower.
- Minimum 8GB RAM recommended.

### Software

| Library | Purpose |
|---|---|
| `streamlit` | Web interface |
| `opencv-python` | Image processing and contour detection |
| `Pillow` | Image handling |
| `torch` | Model inference backend |
| `transformers` | Qwen3-VL model loading |
| `qwen-vl-utils` | Vision input preprocessing (`process_vision_info`) |
| `google-generativeai` | Gemini API access |
| `numpy` | Array operations |

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Markxth/Splitter.git
cd Splitter

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your Gemini API key
export GEMINI_API_KEY=your_key_here

# 4. Run the app
streamlit run main.py
```

> **Note:** On first run, `load_model()` will download the Qwen3-VL-2B-Instruct weights (~4.3GB). These are cached locally after the first download.

---

## Usage

1. Launch the app with `streamlit run main.py`
2. Upload a storyboard image via the interface
3. The pipeline will automatically segment the image into panels. Then, choose which image you wish to have analysed.
4. Configure your analysis instructions in the provided text fields
5. Results are displayed per panel in the interface

---

## Function references for those who wish to edit the code

### `load_model()`

Loads the Qwen3-VL-2B-Instruct vision-language model and its associated processor. Decorated with `@st.cache_resource` to ensure the model is only loaded once per session.

**Returns:**

| Name | Type | Description |
|---|---|---|
| `model` | `AutoModelForImageTextToText` | The loaded Qwen3-VL model |
| `processor` | `AutoProcessor` | The associated tokenizer and image processor |

> **Note:** Uses `torch.bfloat16` on CUDA and `torch.float32` on CPU to avoid dtype errors on CPU-only machines. To use a larger model variant, change `model_id` to e.g. `"Qwen/Qwen3-VL-7B-Instruct"`.

---

### `run_qwen(panel, processor, model, text_input)`

Runs inference on a single panel image using the Qwen3-VL model. Formats the input according to the HuggingFace chat template format and returns the model's text output.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `panel` | `np.ndarray` | Grayscale image of a single panel |
| `processor` | `AutoProcessor` | Processor returned by `load_model()` |
| `model` | `AutoModelForImageTextToText` | Model returned by `load_model()` |
| `text_input` | `str` | The prompt/instruction to pass alongside the image |

**Returns:** `list[str]` — A single-element list containing the model's decoded output string.

> **Note:** Video inputs are intentionally excluded. `max_new_tokens` is set to 600 and can be adjusted depending on expected output length.

> **Attribution:** The base inference code is adapted from the Qwen2-VL model cards on HuggingFace, with modifications to variable names and to remove video handling. See [References](#references).

---

### `splitter(image)`

The main panel segmentation function. Takes a raw uploaded image and returns a list of cropped panel images in reading order (top to bottom, left to right).

**Pipeline steps:**
1. Decodes raw image bytes into a NumPy array
2. Converts to grayscale
3. Applies Gaussian blur to reduce noise
4. Applies adaptive Gaussian thresholding to produce a binary image
5. Detects external contours
6. Filters contours by area (between 6% and 80% of total image area) to exclude noise and full-page elements
7. Sorts panels into reading order via `split_sort()`
8. Adds padding around each panel (default: 3px)
9. Applies `vertical_split_sort()` to catch panels that were not split vertically

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `image` | file-like object | Raw uploaded image file (e.g. from Streamlit's `file_uploader`) |

**Returns:** `list[np.ndarray]` — Ordered list of cropped panel images.

---

### `split_sort(contours, image)`

Sorts detected contour bounding boxes into reading order by grouping them into rows and sorting each row left to right.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `contours` | `list` | List of contours returned by `cv.findContours()` |
| `image` | `np.ndarray` | The source image, used to compute the row-grouping tolerance |

**Returns:** `list[tuple]` — Bounding boxes `(x, y, w, h)` sorted in reading order.

> **Note:** The row-grouping tolerance is set to 1% of image height (`image.shape[0] * 0.01`). This can be adjusted for images with tighter or looser panel spacing.

---

### `vertical_split_sort(panel, image_area, image_y)`

Handles cases where panels were not split vertically by the initial segmentation. Detects bright horizontal dividing lines within a panel and splits it at those boundaries.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `panel` | `np.ndarray` | Grayscale image of a single panel |
| `image_area` | `int` | Total pixel area of the original image (height × width) |
| `image_y` | `int` | Unused. Reserved for future use. |

**Returns:** `list[np.ndarray]` or `None` — List of sub-panels if splits were found, `None` otherwise.

**Logic:**
1. If the panel is smaller than 25% of the total image area, it is returned as-is (assumed to already be a valid panel)
2. Computes mean brightness per row along the vertical axis
3. Identifies rows where brightness exceeds 200 (configurable)
4. Filters to rows where at least 95% of pixels are above the threshold, to distinguish full dividing lines from partial bright elements (e.g. object outlines)
5. Clusters consecutive bright rows and takes the midpoint of each cluster as the split point
6. Slices the panel at each split point and returns the sub-panels

   
---

### `text_analysis(text_original, text_analysis_instructions)`

Passes extracted panel text to the Gemini API for analysis, using user-defined instructions.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `text_original` | `str` | The text extracted from the panel |
| `text_analysis_instructions` | `str` | User-defined instructions for the analysis task |

**Returns:** `str` — The Gemini model's response.

> **Note:** A commented-out HuggingFace implementation is included for users who wish to replace Gemini with a local model.

---

### `comparison(session_panels, embed_text, comparison_text, text_original)`

Compares each panel image against its analysis result and the original text using the Gemini API, returning a per-panel response.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `session_panels` | `list[tuple]` | List of `(index, image)` pairs for the current session |
| `embed_text` | `dict` | Dictionary mapping panel index to its analysis result string |
| `comparison_text` | `str` | User-defined instructions for the comparison task |
| `text_original` | `str` | The original source text to compare against |

**Returns:** `list[str]` — One response string per panel.

---

## Limitations

- Running Qwen3-VL-2B on CPU is functional but slow. GPU use is strongly recommended for practical throughput.
- Panel segmentation assumes panels are separated by a line - both white and dark work.
- The vertical splitting threshold may need tuning depending on image quality and style.
- Gemini API usage is subject to Google's free tier rate limits. The `time.sleep(4)` call in `comparison()` is commented out but can be re-enabled if rate limit errors occur.
- The vertical split sort also can be tuned depending on context.

---

## References
```bibtex
@misc{qwen3technicalreport,
    title={Qwen3 Technical Report}, 
    author={Qwen Team},
    year={2025},
    eprint={2505.09388},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2505.09388}, 
}

@article{Qwen2.5-VL,
    title={Qwen2.5-VL Technical Report},
    author={Bai, Shuai and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Song, Sibo and Dang, Kai and Wang, Peng and Wang, Shijie and Tang, Jun and Zhong, Humen and Zhu, Yuanzhi and Yang, Mingkun and Li, Zhaohai and Wan, Jianqiang and Wang, Pengfei and Ding, Wei and Fu, Zheren and Xu, Yiheng and Ye, Jiabo and Zhang, Xi and Xie, Tianbao and Cheng, Zesen and Zhang, Hang and Yang, Zhibo and Xu, Haiyang and Lin, Junyang},
    journal={arXiv preprint arXiv:2502.13923},
    year={2025}
}

@article{Qwen2VL,
    title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
    author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},
    journal={arXiv preprint arXiv:2409.12191},
    year={2024}
}

@misc{wada2026zinamultimodalfinegrainedhallucination,
      title={ZINA: Multimodal Fine-grained Hallucination Detection and Editing}, 
      author={Yuiga Wada and Kazuki Matsuda and Komei Sugiura and Graham Neubig},
      year={2026},
      eprint={2506.13130},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2506.13130}, 
}
```

- Qwen3-VL Model Card: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct
- Qwen2-VL Inference Code (base for `run_qwen`): https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
- QwenLM GitHub: https://github.com/QwenLM/Qwen3-VL
- `process_vision_info` source: https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-utils/src/qwen_vl_utils/vision_process.py
- Google Gemini API: https://ai.google.dev/

---

## License

 Copyright © 2026 Co-Intelligence Lab

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
