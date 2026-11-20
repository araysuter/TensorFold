"""Public request boundary for the pinned, loopback Inkling MLX adapter."""
import base64
import binascii


def validate_request(body, name, context, benchmark=False):
    if not isinstance(body, dict) or body.get('model', name) not in (name, 'default_model'):
        raise ValueError('Only the configured model is available')
    if any(body.get(k) not in (None, 'default_model') for k in ('adapters', 'draft_model')):
        raise ValueError('Client-selected adapters and drafters are unsupported')
    maximum = body.get('max_completion_tokens')
    if maximum is None:
        maximum = body.get('max_tokens', 4096)
    if type(maximum) is not int or not 1 <= maximum < context:
        raise ValueError('Invalid output budget')
    if body.get('ignore_eos', False) and not benchmark:
        raise ValueError('ignore_eos is available only in the isolated benchmark worker')
    prompt = body.get('prompt')
    if prompt is not None:
        if not isinstance(prompt, (str, list)) or not prompt:
            raise ValueError('Expected text or one token array')
        if isinstance(prompt, list) and len(prompt) + maximum > context:
            raise ValueError('Token prompt plus output exceeds context')
        if isinstance(prompt, list) and any(type(t) is not int or t < 0 for t in prompt):
            raise ValueError('Invalid token array')
    images = 0
    messages = body.get('messages', [])
    if not isinstance(messages, list):
        raise ValueError('Expected messages')
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError('Invalid message')
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                raise ValueError('Invalid message content')
            if part.get('type') == 'text':
                if not isinstance(part.get('text'), str):
                    raise ValueError('Invalid text part')
                continue
            if part.get('type') != 'image_url':
                raise ValueError('Only text and image_url parts are supported')
            spec = part.get('image_url')
            url = spec.get('url') if isinstance(spec, dict) else spec
            if not isinstance(url, str) or not url.startswith(('data:image/png;base64,', 'data:image/jpeg;base64,', 'data:image/webp;base64,')):
                raise ValueError('Images must be inline PNG, JPEG or WebP data URLs')
            try:
                payload = base64.b64decode(url.split(',', 1)[1], validate=True)
            except (ValueError, binascii.Error):
                raise ValueError('Invalid image encoding') from None
            if not 0 < len(payload) <= 4 * 1024**2:
                raise ValueError('Image exceeds the 4 MiB limit')
            # Verify the decoded image and cap expansion before handing it to MLX.
            from PIL import Image
            import io
            try:
                with Image.open(io.BytesIO(payload)) as image:
                    if image.width * image.height > 4_194_304:
                        raise ValueError('Image exceeds the 4 megapixel limit')
                    image.verify()
            except Image.DecompressionBombError:
                raise ValueError('Unsafe image dimensions') from None
            images += 1
    if images > 4:
        raise ValueError('At most four images per request')
    if not messages and prompt is None:
        raise ValueError('A prompt or messages are required')
    return maximum
