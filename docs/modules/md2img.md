# Markdown/HTML to Image Rendering (`md2img`)

## Overview

Converts markdown and HTML documents to rendered PNG images. Useful for converting text documents into visual format for vision-language models.

## Key Modules

### `markdown_api_server.py`

Server for rendering markdown to images.

````python
from md2img.markdown_api_server import MarkdownRenderer

renderer = MarkdownRenderer(headless=True)

# Render markdown string to image
image = renderer.render_markdown("""
# Hello World

This is a **bold** and *italic* example.

- List item 1
- List item 2

```python
print("Code block")
````

""")

# Save image

image.save("output.png")

# Or get bytes

image_bytes = renderer.render_markdown_bytes(markdown_str)

````

**Features**:
- Markdown to HTML conversion
- CSS styling support
- Code highlighting
- Browser automation (headless mode)

### `html_api_server.py`

Server for rendering HTML to images.

```python
from md2img.html_api_server import HTMLRenderer

renderer = HTMLRenderer(headless=True)

# Render HTML string to image
html = """
<html>
<head>
    <style>
        body { font-family: Arial; padding: 20px; }
        h1 { color: blue; }
    </style>
</head>
<body>
    <h1>Hello World</h1>
    <p>This is HTML content.</p>
</body>
</html>
"""

image = renderer.render_html(html)
image.save("output.png")
````

**Features**:

- Direct HTML rendering
- Custom CSS styling
- Browser automation
- Screenshot capture

## API Usage

### Markdown API

```bash
# Start markdown server
bash start_api.sh

# Render markdown via HTTP
curl -X POST http://localhost:5000/render/markdown \
    -H "Content-Type: application/json" \
    -d '{
        "content": "# Hello\n\nThis is **markdown**.",
        "width": 800,
        "height": 600
    }' \
    > output.png
```

**Endpoint**: `POST /render/markdown`

**Request Parameters**:

- `content` (str): Markdown content
- `width` (int, optional): Image width in pixels (default: 1024)
- `height` (int, optional): Image height in pixels (default: 768)

**Response**: PNG image bytes

### HTML API

```bash
# Start HTML server
bash start_html_api.sh

# Render HTML via HTTP
curl -X POST http://localhost:5001/render/html \
    -H "Content-Type: application/json" \
    -d '{
        "content": "<h1>Hello</h1><p>HTML content</p>",
        "width": 800,
        "height": 600
    }' \
    > output.png
```

**Endpoint**: `POST /render/html`

**Request Parameters**:

- `content` (str): HTML content
- `width` (int, optional): Image width (default: 1024)
- `height` (int, optional): Image height (default: 768)

**Response**: PNG image bytes

## Integration with Vision Models

Use rendered images with Qwen VL or other VL models:

```python
from md2img.markdown_api_server import MarkdownRenderer
from PIL import Image
import requests

# Render markdown to image
renderer = MarkdownRenderer()
image = renderer.render_markdown("""
# Analysis

## Problem
Analyze this document...

## Solution
The approach is...
""")

# Use with VL model
from recurrent.interface import RAgent

class DocumentAnalysisAgent(RAgent):
    def action(self):
        # Render documents to images
        doc_images = []
        for doc in self.documents:
            img = self.renderer.render_markdown(doc)
            doc_images.append(img)

        prompts = ["Analyze this document" for _ in doc_images]
        return prompts, doc_images, {"input_pad_to": 2048}
```

## Configuration

### Requirements

```
pip install -r requirements_api.txt
```

Contents typically include:

- Flask or FastAPI for HTTP server
- Playwright or Selenium for browser automation
- Markdown2 or markdown library
- Pillow for image processing

### Environment Variables

```bash
# Port for markdown server
export MD2IMG_PORT=5000

# Port for HTML server
export HTML2IMG_PORT=5001

# Browser executable path (optional)
export BROWSER_PATH=/usr/bin/chromium
```

## Performance Considerations

1. **Batch Processing**: Render multiple documents concurrently
2. **Caching**: Cache rendered images if documents are reused
3. **Resolution**: Adjust width/height based on document complexity
4. **Memory**: Browser processes consume significant memory with many concurrent renders

## Example: Document to Vision Model Pipeline

```python
from md2img.markdown_api_server import MarkdownRenderer
from recurrent.generation_manager import LLMGenerationManager
from recurrent.interface import RAgent
from transformers import AutoTokenizer, AutoProcessor

class DocumentQAAgent(RAgent):
    def __init__(self, tokenizer, config, processor):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.renderer = MarkdownRenderer()

    def start(self, gen_batch, timing_raw):
        self.gen_batch = gen_batch
        self.documents = gen_batch.non_tensor_batch["documents"]
        self.step = 0
        self.rendered_images = []

        # Pre-render all documents
        for doc in self.documents:
            img = self.renderer.render_markdown(doc)
            self.rendered_images.append(img)

    def action(self):
        if self.step == 0:
            # First turn: ask about document
            prompts = ["Summarize this document."] * len(self.gen_batch)
        else:
            # Follow-up questions
            prompts = ["What are the key points?"] * len(self.gen_batch)

        return prompts, self.rendered_images, {"input_pad_to": 2048}

    def update(self, gen_output):
        self.step += 1
        return gen_output

    def done(self):
        return self.step >= 2

    def end(self):
        import torch
        final_mask = torch.ones(len(self.gen_batch), dtype=torch.bool)
        sample_index = torch.arange(len(self.gen_batch))
        return final_mask, sample_index

# Use in pipeline
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat")
processor = AutoProcessor.from_pretrained("Qwen/Qwen-VL-Chat")

manager = LLMGenerationManager(
    tokenizer=tokenizer,
    actor_rollout_wg=ray_actors,
    config=config,
    agent_cls=DocumentQAAgent,
    processor=processor
)
```

## Limitations

1. **Font Support**: Browser-dependent font rendering
2. **JavaScript**: Server-side rendering can't execute JavaScript
3. **Performance**: Rendering is slower than direct text processing
4. **Quality**: Image compression may affect text readability

## Troubleshooting

**Issue**: Images are blurry

- **Solution**: Increase width/height parameters

**Issue**: Server crashes with OOM

- **Solution**: Limit concurrent renders, increase memory, or use smaller pages

**Issue**: Fonts not rendering correctly

- **Solution**: Ensure required fonts are installed on system

## See Also

- [Vision Processing](./vision-processing.md) - Using images with VL models
- [Recurrent Agent Interface](./recurrent-interface.md) - Custom agent implementation
- [README API](../../md2img/README_API.md) - Detailed API documentation
